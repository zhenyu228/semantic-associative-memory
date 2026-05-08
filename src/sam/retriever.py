from __future__ import annotations

from collections import deque

from sam.embedding import EmbeddingProvider
from sam.graph import GraphBuilder
from sam.models import MemoryNode, RetrievalHit
from sam.store import MemoryStore
from sam.text import cosine_similarity, extract_keywords


RETRIEVAL_METHOD_NAMES = {
    "embedding_topk": "Embedding Top-k",
    "raptor_style": "RAPTOR-style",
    "graphrag_style": "GraphRAG-style",
    "hipporag_style": "HippoRAG-style",
    "sam": "SAM 动态联想检索",
}


class Retriever:
    """实现纯向量检索与联想图检索。"""

    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider,
        graph_builder: GraphBuilder,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.graph_builder = graph_builder

    def retrieve(
        self,
        query: str,
        mode: str = "associative",
        top_k: int = 4,
        seed_k: int = 1,
        hops: int = 2,
        candidate_doc_ids: list[str] | None = None,
    ) -> list[RetrievalHit]:
        mode = _normalize_mode(mode)
        if mode not in RETRIEVAL_METHOD_NAMES:
            raise ValueError(f"未知检索模式：{mode}")
        query_embedding = self.embedding_provider.embed(query)
        candidates = self.store.get_nodes(candidate_doc_ids)
        vector_hits = self._vector_hits(query_embedding, candidates, top_k=max(top_k, seed_k))
        if mode == "embedding_topk":
            hits = vector_hits[:top_k]
            self._finalize_retrieval(query, mode, hits, {"top_k": top_k})
            return hits

        if mode == "raptor_style":
            hits = self._raptor_style_hits(query, query_embedding, candidates, top_k)
            self._finalize_retrieval(query, mode, hits, {"top_k": top_k})
            return hits

        if mode == "graphrag_style":
            seed_nodes = [hit.node for hit in vector_hits[: max(seed_k, 2)]]
            self.graph_builder.build_edges_on_demand(seed_nodes)
            hits = self._graphrag_style_hits(query, query_embedding, candidates, top_k)
            self._finalize_retrieval(query, mode, hits, {"top_k": top_k, "seed_k": max(seed_k, 2)})
            return hits

        if mode == "hipporag_style":
            seed_nodes = [hit.node for hit in vector_hits[: max(seed_k, 2)]]
            self.graph_builder.build_edges_on_demand(seed_nodes)
            hits = self._hipporag_style_hits(query_embedding, candidates, top_k)
            self._finalize_retrieval(query, mode, hits, {"top_k": top_k, "seed_k": max(seed_k, 2)})
            return hits

        seed_nodes = [hit.node for hit in vector_hits[:seed_k]]
        self.graph_builder.build_edges_on_demand(seed_nodes)
        hits = self._associative_hits(
            query_embedding=query_embedding,
            candidates=candidates,
            vector_hits=vector_hits[:top_k],
            seed_hits=vector_hits[:seed_k],
            top_k=top_k,
            hops=hops,
        )
        self._finalize_retrieval(query, mode, hits, {"top_k": top_k, "seed_k": seed_k, "hops": hops})
        return hits

    def explain_retrieval(
        self,
        query: str,
        top_k: int = 4,
        seed_k: int = 1,
        hops: int = 2,
        candidate_doc_ids: list[str] | None = None,
    ) -> dict[str, object]:
        hits = self.retrieve(
            query=query,
            mode="associative",
            top_k=top_k,
            seed_k=seed_k,
            hops=hops,
            candidate_doc_ids=candidate_doc_ids,
        )
        return {
            "query": query,
            "hits": [
                {
                    "node_id": hit.node.id,
                    "title": hit.node.metadata.get("title", hit.node.id),
                    "score": round(hit.score, 4),
                    "path": hit.path,
                    "reason": hit.reason,
                }
                for hit in hits
            ],
        }

    def _vector_hits(
        self,
        query_embedding: list[float],
        candidates: list[MemoryNode],
        top_k: int,
    ) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for node in candidates:
            similarity = cosine_similarity(query_embedding, node.embedding)
            usage_score = min(0.12, node.usage_count * 0.02)
            confidence_score = node.confidence * 0.04
            score = similarity + usage_score + confidence_score
            hits.append(
                RetrievalHit(
                    node=node,
                    score=score,
                    similarity_score=similarity,
                    graph_score=0.0,
                    usage_score=usage_score,
                    confidence_score=confidence_score,
                    path=[node.id],
                    reason=f"向量相似度={similarity:.3f}",
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _raptor_style_hits(
        self,
        query: str,
        query_embedding: list[float],
        candidates: list[MemoryNode],
        top_k: int,
    ) -> list[RetrievalHit]:
        """RAPTOR-style 多层摘要树检索。

        这里不声称复现 RAPTOR 官方实现，只模拟它的核心对照思想：
        先把叶子 chunk 聚成若干语义簇，再同时考虑 chunk 与簇摘要的相关性。
        """

        clusters = self._semantic_clusters(candidates)
        query_keywords = set(extract_keywords(query, limit=12))
        cluster_scores: dict[str, float] = {}
        cluster_keywords: dict[str, set[str]] = {}
        for cluster_id, nodes in clusters.items():
            summary_text = " ".join(
                f"{node.metadata.get('title', '')} {' '.join(node.keywords[:6])} {node.summary}"
                for node in nodes
            )
            summary_embedding = self.embedding_provider.embed(summary_text)
            keywords = set(extract_keywords(summary_text, limit=18))
            keyword_score = _overlap_ratio(query_keywords, keywords)
            cluster_scores[cluster_id] = 0.72 * cosine_similarity(query_embedding, summary_embedding) + 0.28 * keyword_score
            cluster_keywords[cluster_id] = keywords

        hits: list[RetrievalHit] = []
        for node in candidates:
            cluster_id = self._cluster_id(node)
            node_similarity = cosine_similarity(query_embedding, node.embedding)
            cluster_score = cluster_scores.get(cluster_id, 0.0)
            keyword_score = _overlap_ratio(query_keywords, set(node.keywords) | cluster_keywords.get(cluster_id, set()))
            score = 0.58 * node_similarity + 0.32 * cluster_score + 0.10 * keyword_score
            hits.append(
                RetrievalHit(
                    node=node,
                    score=score,
                    similarity_score=node_similarity,
                    graph_score=cluster_score,
                    usage_score=0.0,
                    confidence_score=node.confidence * 0.03,
                    path=[f"summary::{cluster_id}", node.id],
                    reason=f"RAPTOR-style：先命中摘要簇 {cluster_id}，再下钻到叶子记忆节点",
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _graphrag_style_hits(
        self,
        query: str,
        query_embedding: list[float],
        candidates: list[MemoryNode],
        top_k: int,
    ) -> list[RetrievalHit]:
        """GraphRAG-style 实体图局部检索。"""

        query_terms = set(extract_keywords(query, limit=16))
        candidate_ids = {node.id for node in candidates}
        edges_by_node: dict[str, list[float]] = {node.id: [] for node in candidates}
        for edge in self.store.get_edges():
            if edge.source_id in candidate_ids and edge.target_id in candidate_ids:
                edges_by_node.setdefault(edge.source_id, []).append(edge.weight)
                edges_by_node.setdefault(edge.target_id, []).append(edge.weight)

        hits: list[RetrievalHit] = []
        for node in candidates:
            entities = {str(entity).lower() for entity in node.metadata.get("entities", [])}
            title_terms = set(extract_keywords(str(node.metadata.get("title", "")), limit=8))
            entity_score = _overlap_ratio(query_terms, set(node.keywords) | title_terms | entities)
            neighborhood_score = min(1.0, sum(edges_by_node.get(node.id, [])) / 3.0)
            similarity = cosine_similarity(query_embedding, node.embedding)
            score = 0.38 * similarity + 0.34 * entity_score + 0.22 * neighborhood_score + node.confidence * 0.03
            hits.append(
                RetrievalHit(
                    node=node,
                    score=score,
                    similarity_score=similarity,
                    graph_score=entity_score + neighborhood_score,
                    usage_score=0.0,
                    confidence_score=node.confidence * 0.03,
                    path=[node.id],
                    reason=(
                        "GraphRAG-style：结合实体/关键词命中、局部图邻域强度和文本相似度排序，"
                        f"实体得分={entity_score:.3f}，邻域得分={neighborhood_score:.3f}"
                    ),
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _hipporag_style_hits(
        self,
        query_embedding: list[float],
        candidates: list[MemoryNode],
        top_k: int,
    ) -> list[RetrievalHit]:
        """HippoRAG-style Personalized PageRank 图激活检索。"""

        candidate_ids = {node.id for node in candidates}
        nodes_by_id = {node.id: node for node in candidates}
        priors = {
            node.id: max(0.0, cosine_similarity(query_embedding, node.embedding))
            for node in candidates
        }
        prior_sum = sum(priors.values()) or 1.0
        priors = {node_id: value / prior_sum for node_id, value in priors.items()}

        adjacency: dict[str, list[tuple[str, float]]] = {node.id: [] for node in candidates}
        for edge in self.store.get_edges():
            if edge.source_id in candidate_ids and edge.target_id in candidate_ids:
                adjacency[edge.source_id].append((edge.target_id, edge.weight))

        ranks = dict(priors)
        damping = 0.78
        for _ in range(18):
            next_ranks = {node_id: (1.0 - damping) * priors[node_id] for node_id in candidate_ids}
            for node_id, neighbors in adjacency.items():
                if not neighbors:
                    continue
                total_weight = sum(weight for _, weight in neighbors) or 1.0
                for target_id, weight in neighbors:
                    next_ranks[target_id] += damping * ranks.get(node_id, 0.0) * weight / total_weight
            ranks = next_ranks

        hits: list[RetrievalHit] = []
        for node_id, rank in ranks.items():
            node = nodes_by_id[node_id]
            similarity = cosine_similarity(query_embedding, node.embedding)
            score = 0.52 * rank + 0.40 * similarity + node.confidence * 0.03
            hits.append(
                RetrievalHit(
                    node=node,
                    score=score,
                    similarity_score=similarity,
                    graph_score=rank,
                    usage_score=0.0,
                    confidence_score=node.confidence * 0.03,
                    path=[node.id],
                    reason=f"HippoRAG-style：以查询相似度作为个性化先验，在知识图上执行 PPR，rank={rank:.4f}",
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _associative_hits(
        self,
        query_embedding: list[float],
        candidates: list[MemoryNode],
        vector_hits: list[RetrievalHit],
        seed_hits: list[RetrievalHit],
        top_k: int,
        hops: int,
    ) -> list[RetrievalHit]:
        candidate_ids = {node.id for node in candidates}
        nodes_by_id = {node.id: node for node in candidates}
        best_paths: dict[str, tuple[list[str], float, str]] = {}
        queue: deque[tuple[str, list[str], float, int, str]] = deque()
        for vector_hit in vector_hits:
            best_paths[vector_hit.node.id] = ([vector_hit.node.id], 0.0, "向量候选节点")

        for seed_hit in seed_hits:
            queue.append((seed_hit.node.id, [seed_hit.node.id], 0.0, 0, "向量种子节点"))
            best_paths[seed_hit.node.id] = ([seed_hit.node.id], 0.0, "向量种子节点")

        while queue:
            current_id, path, graph_score, depth, reason = queue.popleft()
            if depth >= hops:
                continue
            edges = self.store.get_edges_for([current_id])
            for edge in edges:
                if edge.relation_type == "context_cooccurrence" and depth > 0:
                    continue
                next_id = edge.target_id if edge.source_id == current_id else edge.source_id
                if next_id not in candidate_ids or next_id in path:
                    continue
                next_graph_score = graph_score + edge.weight / (depth + 1)
                next_path = [*path, next_id]
                next_reason = f"{reason} -> {edge.relation_type}({edge.reason})"
                previous = best_paths.get(next_id)
                if previous is None or next_graph_score > previous[1]:
                    best_paths[next_id] = (next_path, next_graph_score, next_reason)
                    queue.append((next_id, next_path, next_graph_score, depth + 1, next_reason))

        hits: list[RetrievalHit] = []
        for node_id, (path, graph_score, reason) in best_paths.items():
            node = nodes_by_id[node_id]
            similarity = cosine_similarity(query_embedding, node.embedding)
            usage_score = min(0.14, node.usage_count * 0.02)
            confidence_score = node.confidence * 0.04
            # 联想节点允许相似度较低，但必须由图路径补足。
            score = 0.68 * similarity + 0.24 * graph_score + usage_score + confidence_score
            hits.append(
                RetrievalHit(
                    node=node,
                    score=score,
                    similarity_score=similarity,
                    graph_score=graph_score,
                    usage_score=usage_score,
                    confidence_score=confidence_score,
                    path=path,
                    reason=reason,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        seed_ids = {hit.node.id for hit in seed_hits}
        seed_results = [hit for hit in hits if hit.node.id in seed_ids]
        other_results = [hit for hit in hits if hit.node.id not in seed_ids]
        # 联想检索以向量种子作为“当前被激活记忆”，不能在扩展后把种子挤掉。
        # 否则图扩展会变成替代检索，而不是开题报告中的“种子激活 + 语义扩散”。
        return [*seed_results, *other_results][:top_k]

    def _finalize_retrieval(
        self,
        query: str,
        mode: str,
        hits: list[RetrievalHit],
        metadata: dict[str, object],
    ) -> None:
        accessed_at = _now()
        hit_node_ids = [hit.node.id for hit in hits]
        activated_edges = self._activated_edges_from_paths(hits)
        self.store.increment_usage(hit_node_ids, accessed_at=accessed_at)
        self.store.activate_edges(activated_edges, activated_at=accessed_at)

        refreshed_nodes = {node.id: node for node in self.store.get_nodes(hit_node_ids)}
        for hit in hits:
            if hit.node.id in refreshed_nodes:
                hit.node = refreshed_nodes[hit.node.id]

        dynamic_metadata = {
            **metadata,
            "dynamic_update": {
                "accessed_at": accessed_at,
                "updated_node_ids": hit_node_ids,
                "activated_edges": [
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation_type": relation_type,
                    }
                    for source_id, target_id, relation_type in activated_edges
                ],
            },
        }
        self.store.log_retrieval(query, mode, hits, dynamic_metadata)

    def _activated_edges_from_paths(self, hits: list[RetrievalHit]) -> list[tuple[str, str, str]]:
        path_pairs: set[tuple[str, str]] = set()
        for hit in hits:
            path = [str(node_id) for node_id in hit.path]
            for left, right in zip(path, path[1:], strict=False):
                path_pairs.add((left, right))
        if not path_pairs:
            return []

        edge_by_pair: dict[tuple[str, str], tuple[str, str, str, float]] = {}
        involved_node_ids = {node_id for pair in path_pairs for node_id in pair}
        for edge in self.store.get_edges_for(involved_node_ids):
            pair = (edge.source_id, edge.target_id)
            if pair not in path_pairs:
                continue
            previous = edge_by_pair.get(pair)
            if previous is None or edge.weight > previous[3]:
                edge_by_pair[pair] = (edge.source_id, edge.target_id, edge.relation_type, edge.weight)
        return [
            (source_id, target_id, relation_type)
            for source_id, target_id, relation_type, _ in edge_by_pair.values()
        ]

    def _semantic_clusters(self, nodes: list[MemoryNode]) -> dict[str, list[MemoryNode]]:
        clusters: dict[str, list[MemoryNode]] = {}
        for node in nodes:
            clusters.setdefault(self._cluster_id(node), []).append(node)
        return clusters

    def _cluster_id(self, node: MemoryNode) -> str:
        query_id = node.metadata.get("query_id")
        entities = node.metadata.get("entities", [])
        if entities:
            return f"{query_id or node.metadata.get('dataset', 'global')}::{str(entities[0]).lower()}"
        if node.keywords:
            return f"{query_id or node.metadata.get('dataset', 'global')}::{node.keywords[0]}"
        return f"{query_id or node.metadata.get('dataset', 'global')}::misc"


def _normalize_mode(mode: str) -> str:
    aliases = {
        "vector": "embedding_topk",
        "associative": "sam",
    }
    return aliases.get(mode, mode)


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _now() -> str:
    from sam.models import utc_now_iso

    return utc_now_iso()
