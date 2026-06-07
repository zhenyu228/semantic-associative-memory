from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from sam.embedding import EmbeddingProvider
from sam.graph import GraphBuilder
from sam.models import MemoryEvent, MemoryNode, RetrievalHit, utc_now_iso
from sam.reranker import PathReranker
from sam.store import MemoryStore
from sam.text import cosine_similarity, extract_keywords


RETRIEVAL_METHOD_NAMES = {
    "embedding_topk": "Embedding Top-k",
    "raptor_style": "RAPTOR",
    "graphrag_style": "GraphRAG",
    "hipporag_style": "HippoRAG",
    "sam": "SAM 动态联想检索",
    "sam_full": "SAM-full",
    "sam_no_multipath": "SAM-no-multipath",
    "sam_no_memory_state": "SAM-no-memory-state",
    "sam_no_graph": "SAM-no-graph",
    "sam_static_graph": "SAM-static-graph",
    "sam_no_summary": "SAM-no-summary",
    "sam_with_summary": "SAM-with-summary",
    "sam_no_feedback": "SAM-no-feedback",
    "sam_vector_anchor": "SAM-vector-anchor",
    "sam_adaptive_anchor": "SAM-adaptive-anchor",
    "sam_with_analogy": "SAM-with-analogy",
    "sam_no_lexical_activation": "SAM-no-lexical-activation",
    "sam_with_lexical_activation": "SAM-with-lexical-activation",
}


@dataclass(frozen=True, slots=True)
class SamRetrievalConfig:
    """SAM 消融实验开关。"""

    use_graph: bool = True
    build_graph_on_demand: bool = True
    use_multipath: bool = True
    use_memory_state: bool = True
    update_dynamic_state: bool = True
    use_summary_nodes: bool = False
    min_vector_keep: int = 1
    adaptive_vector_anchor: bool = False
    adaptive_anchor_keep: int = 2
    adaptive_anchor_threshold: float = 0.75
    use_analogy: bool = False
    analogy_top_k: int = 2
    use_lexical_activation: bool = False
    use_feedback: bool = True
    use_consolidated_memory: bool = False


SAM_RETRIEVAL_CONFIGS = {
    "sam": SamRetrievalConfig(),
    "sam_full": SamRetrievalConfig(),
    "sam_no_multipath": SamRetrievalConfig(use_multipath=False),
    "sam_no_memory_state": SamRetrievalConfig(use_memory_state=False),
    "sam_no_graph": SamRetrievalConfig(use_graph=False, build_graph_on_demand=False),
    "sam_static_graph": SamRetrievalConfig(
        build_graph_on_demand=False,
        update_dynamic_state=False,
        use_feedback=False,
    ),
    "sam_no_summary": SamRetrievalConfig(use_summary_nodes=False),
    "sam_with_summary": SamRetrievalConfig(use_summary_nodes=True),
    "sam_no_feedback": SamRetrievalConfig(use_feedback=False),
    "sam_vector_anchor": SamRetrievalConfig(min_vector_keep=2),
    "sam_adaptive_anchor": SamRetrievalConfig(adaptive_vector_anchor=True),
    "sam_with_analogy": SamRetrievalConfig(use_analogy=True, use_consolidated_memory=True),
    "sam_no_lexical_activation": SamRetrievalConfig(use_lexical_activation=False),
    "sam_with_lexical_activation": SamRetrievalConfig(use_lexical_activation=True),
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
        self.path_reranker = PathReranker.from_env()

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
        if mode in SAM_RETRIEVAL_CONFIGS and not SAM_RETRIEVAL_CONFIGS[mode].use_summary_nodes:
            candidates = [
                node for node in candidates if node.metadata.get("node_type") != "query_summary"
            ]
        seed_candidates = (
            [node for node in candidates if node.metadata.get("node_type") != "query_summary"]
            if mode in SAM_RETRIEVAL_CONFIGS
            else candidates
        )
        vector_hits = self._vector_hits(query_embedding, seed_candidates, top_k=max(top_k, seed_k))
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

        sam_config = SAM_RETRIEVAL_CONFIGS[mode]
        if sam_config.use_lexical_activation:
            vector_hits = self._sam_activation_hits(
                query=query,
                query_embedding=query_embedding,
                candidates=seed_candidates,
                top_k=max(top_k, seed_k),
            )
        seed_nodes = [hit.node for hit in vector_hits[:seed_k]]
        if sam_config.use_graph and sam_config.build_graph_on_demand:
            self.graph_builder.build_edges_on_demand(seed_nodes, candidates)
        hits = self._associative_hits(
            query=query,
            query_embedding=query_embedding,
            candidates=candidates,
            vector_hits=vector_hits[:top_k],
            seed_hits=vector_hits[:seed_k],
            top_k=top_k,
            hops=hops,
            config=sam_config,
        )
        self._finalize_retrieval(
            query,
            mode,
            hits,
            {
                "top_k": top_k,
                "seed_k": seed_k,
                "hops": hops,
                "sam_config": _sam_config_payload(sam_config),
            },
            update_dynamic_state=sam_config.update_dynamic_state,
        )
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
                    metadata={
                        "score_breakdown": {
                            "similarity": round(similarity, 4),
                            "usage": round(usage_score, 4),
                            "confidence": round(confidence_score, 4),
                        }
                    },
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _sam_activation_hits(
        self,
        *,
        query: str,
        query_embedding: list[float],
        candidates: list[MemoryNode],
        top_k: int,
    ) -> list[RetrievalHit]:
        """SAM 初始激活：在 embedding 相似度外补充关键词、标题和实体线索。"""

        query_terms = set(extract_keywords(query, limit=24))
        hits: list[RetrievalHit] = []
        for node in candidates:
            similarity = cosine_similarity(query_embedding, node.embedding)
            lexical_score = _lexical_activation_score(query, query_terms, node)
            usage_score = min(0.12, node.usage_count * 0.02)
            confidence_score = node.confidence * 0.04
            lexical_component = 0.24 * lexical_score
            score = similarity + lexical_component + usage_score + confidence_score
            hits.append(
                RetrievalHit(
                    node=node,
                    score=score,
                    similarity_score=similarity,
                    graph_score=0.0,
                    usage_score=usage_score,
                    confidence_score=confidence_score,
                    path=[node.id],
                    reason=(
                        f"SAM 初始激活：向量相似度={similarity:.3f}，"
                        f"词项/实体激活={lexical_score:.3f}"
                    ),
                    metadata={
                        "lexical_activation_score": lexical_score,
                        "score_breakdown": {
                            "similarity": round(similarity, 4),
                            "lexical_activation": round(lexical_component, 4),
                            "lexical_activation_score": round(lexical_score, 4),
                            "usage": round(usage_score, 4),
                            "confidence": round(confidence_score, 4),
                        }
                    },
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
        """RAPTOR 多层摘要树检索。"""

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
                    reason=f"RAPTOR：先命中摘要簇 {cluster_id}，再下钻到叶子记忆节点",
                    metadata={
                        "score_breakdown": {
                            "node_similarity": round(node_similarity, 4),
                            "cluster_score": round(cluster_score, 4),
                            "keyword_score": round(keyword_score, 4),
                        }
                    },
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
        """GraphRAG 实体图局部检索。"""

        query_terms = set(extract_keywords(query, limit=16))
        candidate_ids = {node.id for node in candidates}
        edges_by_node: dict[str, list[float]] = {node.id: [] for node in candidates}
        for edge in self.store.get_edges_for(candidate_ids):
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
                        "GraphRAG：结合实体/关键词命中、局部图邻域强度和文本相似度排序，"
                        f"实体得分={entity_score:.3f}，邻域得分={neighborhood_score:.3f}"
                    ),
                    metadata={
                        "score_breakdown": {
                            "similarity": round(similarity, 4),
                            "entity_score": round(entity_score, 4),
                            "neighborhood_score": round(neighborhood_score, 4),
                        }
                    },
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
        """HippoRAG Personalized PageRank 图激活检索。"""

        candidate_ids = {node.id for node in candidates}
        nodes_by_id = {node.id: node for node in candidates}
        priors = {
            node.id: max(0.0, cosine_similarity(query_embedding, node.embedding))
            for node in candidates
        }
        prior_sum = sum(priors.values()) or 1.0
        priors = {node_id: value / prior_sum for node_id, value in priors.items()}

        adjacency: dict[str, list[tuple[str, float]]] = {node.id: [] for node in candidates}
        for edge in self.store.get_edges_for(candidate_ids):
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
                    reason=f"HippoRAG：以查询相似度作为个性化先验，在知识图上执行 PPR，rank={rank:.4f}",
                    metadata={
                        "score_breakdown": {
                            "pagerank": round(rank, 4),
                            "similarity": round(similarity, 4),
                        }
                    },
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _associative_hits(
        self,
        query: str,
        query_embedding: list[float],
        candidates: list[MemoryNode],
        vector_hits: list[RetrievalHit],
        seed_hits: list[RetrievalHit],
        top_k: int,
        hops: int,
        config: SamRetrievalConfig,
    ) -> list[RetrievalHit]:
        candidate_ids = {node.id for node in candidates}
        nodes_by_id = {node.id: node for node in candidates}
        best_paths: dict[str, tuple[list[str], float, str]] = {}
        path_signals: dict[str, list[dict[str, object]]] = {}
        analogy_signals = (
            self._analogy_support_signals(
                query=query,
                candidate_ids=candidate_ids,
                top_k=config.analogy_top_k,
            )
            if config.use_analogy
            else {}
        )
        queue: deque[tuple[str, list[str], float, int, str]] = deque()
        expanded_edge_node_ids: set[str] = {hit.node.id for hit in seed_hits}
        for vector_hit in vector_hits:
            best_paths[vector_hit.node.id] = ([vector_hit.node.id], 0.0, "向量候选节点")
            path_signals.setdefault(vector_hit.node.id, []).append(
                {
                    "path": [vector_hit.node.id],
                    "graph_score": 0.0,
                    "reason": vector_hit.reason or "向量候选节点",
                    "depth": 0,
                    "lexical_activation_score": vector_hit.metadata.get("lexical_activation_score", 0.0),
                }
            )

        for seed_hit in seed_hits:
            queue.append((seed_hit.node.id, [seed_hit.node.id], 0.0, 0, "向量种子节点"))
            best_paths[seed_hit.node.id] = ([seed_hit.node.id], 0.0, "向量种子节点")
            path_signals.setdefault(seed_hit.node.id, []).append(
                {
                    "path": [seed_hit.node.id],
                    "graph_score": 0.0,
                    "reason": seed_hit.reason or "向量种子节点",
                    "depth": 0,
                    "lexical_activation_score": seed_hit.metadata.get("lexical_activation_score", 0.0),
                }
            )

        while config.use_graph and queue:
            current_id, path, graph_score, depth, reason = queue.popleft()
            if depth >= hops:
                continue
            if config.build_graph_on_demand and current_id not in expanded_edge_node_ids:
                current_node = nodes_by_id.get(current_id)
                if current_node is not None:
                    self.graph_builder.build_edges_on_demand([current_node], candidates)
                expanded_edge_node_ids.add(current_id)
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
                if config.use_multipath or next_id not in path_signals:
                    path_signals.setdefault(next_id, []).append(
                        {
                            "path": next_path,
                            "graph_score": next_graph_score,
                            "edge_weight": edge.weight,
                            "edge_activation_count": edge.activation_count,
                            "relation_type": edge.relation_type,
                            "reason": next_reason,
                            "depth": depth + 1,
                            **_edge_quality_signal(edge.metadata),
                        }
                    )
                previous = best_paths.get(next_id)
                if previous is None or next_graph_score > previous[1]:
                    best_paths[next_id] = (next_path, next_graph_score, next_reason)
                    queue.append((next_id, next_path, next_graph_score, depth + 1, next_reason))

        if not config.use_multipath:
            path_signals = {
                node_id: [
                    {
                        "path": path,
                        "graph_score": graph_score,
                        "reason": reason,
                        "depth": max(0, len(path) - 1),
                    }
                ]
                for node_id, (path, graph_score, reason) in best_paths.items()
            }

        for support_node_id, analogy_signal in analogy_signals.items():
            if support_node_id not in nodes_by_id:
                continue
            consolidated_node_id = str(analogy_signal.get("consolidated_node_id") or "")
            path = (
                [consolidated_node_id, support_node_id]
                if consolidated_node_id and consolidated_node_id in nodes_by_id
                else [support_node_id]
            )
            graph_score = float(analogy_signal.get("graph_score", 0.0))
            reason = (
                f"类比案例 {analogy_signal.get('case_id')} 复用历史支持证据"
            )
            path_signals.setdefault(support_node_id, []).append(
                {
                    "path": path,
                    "graph_score": graph_score,
                    "relation_type": "analogy_case_reuse",
                    "edge_activation_count": analogy_signal.get("edge_activation_count", 0),
                    "reason": reason,
                    "depth": max(1, len(path) - 1),
                    "analogy_case_id": analogy_signal.get("case_id"),
                }
            )
            previous = best_paths.get(support_node_id)
            if previous is None or graph_score > previous[1]:
                best_paths[support_node_id] = (path, graph_score, reason)

        hits: list[RetrievalHit] = []
        for node_id, (path, graph_score, reason) in best_paths.items():
            node = nodes_by_id[node_id]
            similarity = cosine_similarity(query_embedding, node.embedding)
            signals = path_signals.get(node_id, [])
            path_score = self.path_reranker.score(
                similarity=similarity,
                graph_score=graph_score,
                signals=signals,
                node=node,
                use_multipath=config.use_multipath,
                use_memory_state=config.use_memory_state,
            )
            score_breakdown = dict(path_score.breakdown)
            total_score = path_score.total
            lexical_activation_score = max(
                (
                    float(signal.get("lexical_activation_score", 0.0))
                    for signal in signals
                ),
                default=0.0,
            )
            if lexical_activation_score > 0.0:
                score_breakdown["initial_lexical_activation_score"] = round(lexical_activation_score, 4)
            hit_metadata: dict[str, object] = {
                "score_breakdown": score_breakdown,
                "reranker_profile": path_score.profile,
                "path_support_score": round(path_score.path_support_score, 4),
                "edge_memory_score": round(path_score.edge_memory_score, 4),
                "recency_score": round(path_score.recency_score, 4),
                "candidate_path_count": len(signals),
                "candidate_paths": _top_path_signals(signals),
                "ablation_config": _sam_config_payload(config),
            }
            reason_text = (
                f"{reason}；多路径支持={path_score.path_support_score:.3f}，"
                f"历史边激活={path_score.edge_memory_score:.3f}，近期访问={path_score.recency_score:.3f}"
            )
            if node_id in analogy_signals:
                analogy_signal = analogy_signals[node_id]
                analogy_component = min(
                    0.16,
                    float(analogy_signal.get("match_score", 0.0)) * 0.12,
                )
                total_score += analogy_component
                score_breakdown["analogy_component"] = round(analogy_component, 4)
                hit_metadata.update(
                    {
                        "analogy_case_id": analogy_signal.get("case_id"),
                        "analogy_support_node_id": node_id,
                        "analogy_match_score": round(
                            float(analogy_signal.get("match_score", 0.0)),
                            4,
                        ),
                        "analogy_prompt_hint": analogy_signal.get("prompt_hint", ""),
                    }
                )
                reason_text = f"{reason_text}；类比案例={analogy_signal.get('case_id')}"
            hits.append(
                RetrievalHit(
                    node=node,
                    score=total_score,
                    similarity_score=similarity,
                    graph_score=graph_score,
                    usage_score=path_score.usage_score,
                    confidence_score=path_score.confidence_score,
                    path=path,
                    reason=reason_text,
                    metadata=hit_metadata,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        anchor_count, anchor_reason = _resolve_anchor_policy(
            hits=hits,
            seed_count=len(seed_hits),
            top_k=top_k,
            config=config,
        )
        for hit in hits:
            hit.metadata["adaptive_anchor_count"] = anchor_count
            hit.metadata["adaptive_anchor_reason"] = anchor_reason
        anchor_ids = {
            hit.node.id
            for hit in vector_hits[:anchor_count]
        }
        anchor_results_by_id = {hit.node.id: hit for hit in hits if hit.node.id in anchor_ids}
        anchor_results = [
            anchor_results_by_id[vector_hit.node.id]
            for vector_hit in vector_hits
            if vector_hit.node.id in anchor_results_by_id
        ]
        other_results = [hit for hit in hits if hit.node.id not in anchor_ids]
        # 联想检索以向量候选作为“当前被激活记忆”，需要保留少量锚点。
        # Bad case 显示：若完全按图扩展重排，噪声路径可能把原本有效的向量证据挤出 top-k。
        ranked = [*anchor_results, *other_results]
        final_excluded_node_types = {"query_summary", "consolidated_memory"}
        document_hits = [
            hit
            for hit in ranked
            if hit.node.metadata.get("node_type") not in final_excluded_node_types
        ]
        if len(document_hits) >= top_k:
            return document_hits[:top_k]
        summary_hits = [hit for hit in ranked if hit.node.metadata.get("node_type") == "query_summary"]
        return [*document_hits, *summary_hits][:top_k]

    def _finalize_retrieval(
        self,
        query: str,
        mode: str,
        hits: list[RetrievalHit],
        metadata: dict[str, object],
        update_dynamic_state: bool = True,
    ) -> None:
        accessed_at = _now()
        hit_node_ids = [hit.node.id for hit in hits]
        activated_edges = self._activated_edges_from_paths(hits)
        if update_dynamic_state:
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
                "enabled": update_dynamic_state,
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
        self.store.log_memory_events(
            _retrieval_events(
                query=query,
                mode=mode,
                hits=hits,
                activated_edges=activated_edges,
                metadata=metadata,
            )
        )

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

    def _analogy_support_signals(
        self,
        *,
        query: str,
        candidate_ids: set[str],
        top_k: int,
    ) -> dict[str, dict[str, object]]:
        from sam.analogy import AnalogyEngine

        engine = AnalogyEngine(self.store, self.embedding_provider, self.graph_builder)
        signals: dict[str, dict[str, object]] = {}
        for match in engine.retrieve_cases(query, top_k=top_k):
            support_node_ids = [
                str(node_id)
                for node_id in match.metadata.get("support_node_ids", [])
                if str(node_id) in candidate_ids
            ]
            if not support_node_ids:
                continue
            match_score = float(match.score)
            graph_score = min(
                1.0,
                0.72 * match_score
                + 0.18 * float(match.metadata.get("consolidated_confidence") or 0.0)
                + 0.10,
            )
            for support_node_id in support_node_ids:
                previous = signals.get(support_node_id)
                if previous is not None and float(previous.get("match_score", 0.0)) >= match_score:
                    continue
                signals[support_node_id] = {
                    "support_node_id": support_node_id,
                    "case_id": match.case_id,
                    "match_score": match_score,
                    "graph_score": graph_score,
                    "consolidated_node_id": match.metadata.get("consolidated_node_id"),
                    "prompt_hint": match.prompt_hint,
                    "edge_activation_count": 1,
                }
        return signals


def _normalize_mode(mode: str) -> str:
    aliases = {
        "vector": "embedding_topk",
        "associative": "sam",
    }
    return aliases.get(mode, mode)


def _sam_config_payload(config: SamRetrievalConfig) -> dict[str, object]:
    return {
        "use_graph": config.use_graph,
        "build_graph_on_demand": config.build_graph_on_demand,
        "use_multipath": config.use_multipath,
        "use_memory_state": config.use_memory_state,
        "update_dynamic_state": config.update_dynamic_state,
        "use_summary_nodes": config.use_summary_nodes,
        "min_vector_keep": config.min_vector_keep,
        "adaptive_vector_anchor": config.adaptive_vector_anchor,
        "adaptive_anchor_keep": config.adaptive_anchor_keep,
        "adaptive_anchor_threshold": config.adaptive_anchor_threshold,
        "use_analogy": config.use_analogy,
        "analogy_top_k": config.analogy_top_k,
        "use_lexical_activation": config.use_lexical_activation,
        "use_feedback": config.use_feedback,
        "use_consolidated_memory": config.use_consolidated_memory,
    }


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _lexical_activation_score(query: str, query_terms: set[str], node: MemoryNode) -> float:
    if not query_terms:
        return 0.0
    title = str(node.metadata.get("title", ""))
    entities = " ".join(str(entity) for entity in node.metadata.get("entities", []))
    node_text = " ".join(
        [
            title,
            node.summary,
            " ".join(node.keywords[:16]),
            entities,
        ]
    )
    node_terms = set(extract_keywords(node_text, limit=48))
    title_terms = set(extract_keywords(title, limit=16))
    entity_terms = set(extract_keywords(entities, limit=16))
    exact_title_score = 1.0 if _normalized_phrase(title) in _normalized_phrase(query) else 0.0
    keyword_score = _overlap_ratio(query_terms, node_terms)
    title_score = _overlap_ratio(query_terms, title_terms)
    entity_score = _overlap_ratio(query_terms, entity_terms)
    return min(
        1.0,
        0.62 * keyword_score
        + 0.25 * exact_title_score
        + 0.06 * title_score
        + 0.07 * entity_score,
    )


def _normalized_phrase(text: str) -> str:
    return " ".join(extract_keywords(text, limit=32))


def _top_path_signals(signals: list[dict[str, object]], limit: int = 4) -> list[dict[str, object]]:
    ordered = sorted(signals, key=lambda signal: float(signal.get("graph_score", 0.0)), reverse=True)
    return [
        {
            "path": signal.get("path", []),
            "graph_score": round(float(signal.get("graph_score", 0.0)), 4),
            "relation_type": signal.get("relation_type"),
            "edge_activation_count": signal.get("edge_activation_count", 0),
            "edge_quality": signal.get("edge_quality"),
            "similarity": signal.get("similarity"),
            "shared_entities": signal.get("shared_entities", []),
            "keyword_overlap": signal.get("keyword_overlap", []),
            "reason": signal.get("reason", ""),
        }
        for signal in ordered[:limit]
    ]


def _edge_quality_signal(metadata: dict[str, object]) -> dict[str, object]:
    score = metadata.get("score_breakdown", metadata)
    if not isinstance(score, dict):
        return {}
    return {
        "edge_quality": score.get("edge_quality"),
        "similarity": score.get("similarity"),
        "shared_entities": score.get("shared_entities", []),
        "keyword_overlap": score.get("keyword_overlap", []),
    }


def _resolve_anchor_policy(
    *,
    hits: list[RetrievalHit],
    seed_count: int,
    top_k: int,
    config: SamRetrievalConfig,
) -> tuple[int, str]:
    base_anchor_count = max(seed_count, config.min_vector_keep)
    if not config.adaptive_vector_anchor:
        return min(top_k, base_anchor_count), "fixed"

    expanded_support_scores = [
        float(hit.metadata.get("path_support_score", 0.0))
        for hit in hits
        if len(hit.path) > 1
    ]
    average_path_support = (
        sum(expanded_support_scores) / len(expanded_support_scores)
        if expanded_support_scores
        else 0.0
    )
    if average_path_support < config.adaptive_anchor_threshold:
        return (
            min(top_k, max(base_anchor_count, config.adaptive_anchor_keep)),
            "weak_graph_paths",
        )
    return min(top_k, base_anchor_count), "strong_graph_paths"


def _now() -> str:
    return utc_now_iso()


def _retrieval_events(
    query: str,
    mode: str,
    hits: list[RetrievalHit],
    activated_edges: list[tuple[str, str, str]],
    metadata: dict[str, object],
) -> list[MemoryEvent]:
    created_at = utc_now_iso()
    query_id = str(metadata.get("query_id")) if metadata.get("query_id") else None
    events: list[MemoryEvent] = []
    for rank, hit in enumerate(hits, start=1):
        events.append(
            MemoryEvent(
                event_type="node_retrieved",
                query_id=query_id,
                query=query,
                mode=mode,
                node_id=hit.node.id,
                path=[str(node_id) for node_id in hit.path],
                score=hit.score,
                created_at=created_at,
                metadata={
                    "rank": rank,
                    "title": hit.node.metadata.get("title"),
                    "score_breakdown": hit.metadata.get("score_breakdown", {}),
                },
            )
        )
    for edge_key in activated_edges:
        events.append(
            MemoryEvent(
                event_type="edge_traversed",
                query_id=query_id,
                query=query,
                mode=mode,
                edge_key=edge_key,
                created_at=created_at,
                metadata={"source_id": edge_key[0], "target_id": edge_key[1], "relation_type": edge_key[2]},
            )
        )
    return events
