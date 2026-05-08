from __future__ import annotations

from itertools import combinations

from sam.models import MemoryEdge, MemoryNode, utc_now_iso
from sam.store import MemoryStore
from sam.text import cosine_similarity


GENERIC_EDGE_KEYWORDS = {
    "film",
    "american",
    "british",
    "directed",
    "drama",
    "comedy",
    "series",
    "book",
    "books",
    "novel",
    "music",
    "album",
    "football",
    "team",
    "home",
    "they",
    "their",
    "known",
    "based",
    "people",
    "city",
}


class GraphBuilder:
    """按需构建语义边，避免在写入阶段进行全量建图。"""

    def __init__(
        self,
        store: MemoryStore,
        similarity_threshold: float = 0.18,
        keyword_overlap_threshold: int = 2,
    ) -> None:
        self.store = store
        self.similarity_threshold = similarity_threshold
        self.keyword_overlap_threshold = keyword_overlap_threshold

    def build_edges_on_demand(self, seed_nodes: list[MemoryNode]) -> list[MemoryEdge]:
        """只围绕被检索激活的种子节点补边。"""

        all_nodes = self.store.get_nodes()
        candidates: dict[tuple[str, str, str], MemoryEdge] = {}
        for seed in seed_nodes:
            for other in all_nodes:
                if seed.id == other.id:
                    continue
                edge = self._maybe_create_edge(seed, other)
                if edge:
                    candidates[edge.key] = edge
                    reverse = MemoryEdge(
                        source_id=edge.target_id,
                        target_id=edge.source_id,
                        relation_type=edge.relation_type,
                        weight=edge.weight,
                        reason=edge.reason,
                        created_at=edge.created_at,
                        updated_at=edge.updated_at,
                        activation_count=edge.activation_count,
                        last_activated_at=edge.last_activated_at,
                        metadata=edge.metadata,
                    )
                    candidates[reverse.key] = reverse
        for edge in candidates.values():
            self.store.upsert_edge(edge)
        return list(candidates.values())

    def bootstrap_context_edges(self, nodes: list[MemoryNode]) -> list[MemoryEdge]:
        """为同一公开基准上下文内的文档建立轻量初始边。

        这不是全量知识图谱构建，只是保留数据集天然给出的同题上下文关系，
        便于后续按需激活时沿候选上下文扩展。
        """

        created: list[MemoryEdge] = []
        for left, right in combinations(nodes, 2):
            if left.metadata.get("query_id") != right.metadata.get("query_id"):
                continue
            now = utc_now_iso()
            edge = MemoryEdge(
                source_id=left.id,
                target_id=right.id,
                relation_type="context_cooccurrence",
                weight=0.08,
                reason="同一公开多跳问答样本中的候选上下文，保留跨文档推理的候选关系",
                created_at=now,
                updated_at=now,
                activation_count=0,
                last_activated_at=None,
                metadata={"query_id": left.metadata.get("query_id")},
            )
            reverse = MemoryEdge(
                source_id=right.id,
                target_id=left.id,
                relation_type=edge.relation_type,
                weight=edge.weight,
                reason=edge.reason,
                created_at=now,
                updated_at=now,
                activation_count=0,
                last_activated_at=None,
                metadata=edge.metadata,
            )
            self.store.upsert_edge(edge)
            self.store.upsert_edge(reverse)
            created.extend([edge, reverse])
        return created

    def _maybe_create_edge(self, seed: MemoryNode, other: MemoryNode) -> MemoryEdge | None:
        keyword_overlap = sorted(
            (set(seed.keywords) & set(other.keywords)) - GENERIC_EDGE_KEYWORDS
        )
        shared_entities = sorted(
            set(seed.metadata.get("entities", [])) & set(other.metadata.get("entities", []))
        )
        similarity = cosine_similarity(seed.embedding, other.embedding)

        relation_type: str | None = None
        weight = 0.0
        reason = ""
        metadata: dict[str, object] = {
            "similarity": round(similarity, 4),
            "keyword_overlap": keyword_overlap,
            "shared_entities": shared_entities,
        }
        if shared_entities:
            relation_type = "shared_entity"
            weight = min(0.95, 0.55 + 0.12 * len(shared_entities))
            reason = f"共享实体：{', '.join(shared_entities[:4])}"
        elif len(keyword_overlap) >= self.keyword_overlap_threshold:
            relation_type = "keyword_overlap"
            weight = min(0.85, 0.35 + 0.08 * len(keyword_overlap))
            reason = f"关键词重叠：{', '.join(keyword_overlap[:4])}"
        elif similarity >= self.similarity_threshold:
            relation_type = "embedding_similarity"
            weight = min(0.8, similarity)
            reason = f"语义相似度达到阈值：{similarity:.3f}"

        if relation_type is None:
            return None
        now = utc_now_iso()
        return MemoryEdge(
            source_id=seed.id,
            target_id=other.id,
            relation_type=relation_type,
            weight=weight,
            reason=reason,
            created_at=now,
            updated_at=now,
            activation_count=0,
            last_activated_at=None,
            metadata=metadata,
        )
