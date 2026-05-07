from __future__ import annotations

from collections import deque

from sam.embedding import EmbeddingProvider
from sam.graph import GraphBuilder
from sam.models import MemoryNode, RetrievalHit
from sam.store import MemoryStore
from sam.text import cosine_similarity


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
        if mode not in {"vector", "associative"}:
            raise ValueError("mode 只能是 vector 或 associative")
        query_embedding = self.embedding_provider.embed(query)
        candidates = self.store.get_nodes(candidate_doc_ids)
        vector_hits = self._vector_hits(query_embedding, candidates, top_k=max(top_k, seed_k))
        if mode == "vector":
            hits = vector_hits[:top_k]
            self.store.increment_usage([hit.node.id for hit in hits])
            self.store.log_retrieval(query, mode, hits, {"top_k": top_k})
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
        self.store.increment_usage([hit.node.id for hit in hits])
        self.store.log_retrieval(query, mode, hits, {"top_k": top_k, "seed_k": seed_k, "hops": hops})
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
