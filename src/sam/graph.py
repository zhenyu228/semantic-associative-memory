from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

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


@dataclass(slots=True)
class EdgeScore:
    """一次候选建边的可解释打分结果。"""

    relation_type: str | None
    weight: float
    reason: str
    score_breakdown: dict[str, object]


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
        self.edge_creation_log: list[dict[str, object]] = []

    def build_edges_on_demand(
        self,
        seed_nodes: list[MemoryNode],
        candidate_nodes: list[MemoryNode] | None = None,
    ) -> list[MemoryEdge]:
        """只围绕被检索激活的种子节点补边。"""

        all_nodes = candidate_nodes if candidate_nodes is not None else self.store.get_nodes()
        candidates: dict[tuple[str, str, str], MemoryEdge] = {}
        for seed in seed_nodes:
            for other in all_nodes:
                if seed.id == other.id:
                    continue
                edge = self._maybe_create_edge(seed, other)
                if edge:
                    existing = self.store.get_edge(*edge.key)
                    candidates[edge.key] = edge
                    self._record_edge_event(edge, seed, other, "created" if existing is None else "updated")
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
                    reverse_existing = self.store.get_edge(*reverse.key)
                    candidates[reverse.key] = reverse
                    self._record_edge_event(reverse, other, seed, "created" if reverse_existing is None else "updated")
        self.store.upsert_edges(candidates.values())
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
            created.extend([edge, reverse])
        self.store.upsert_edges(created)
        return created

    def bootstrap_summary_edges(self, summary_nodes: list[MemoryNode]) -> list[MemoryEdge]:
        """连接摘要记忆节点和其覆盖的原始文档节点。"""

        created: list[MemoryEdge] = []
        for summary_node in summary_nodes:
            child_node_ids = [
                str(node_id)
                for node_id in summary_node.metadata.get("child_node_ids", [])
            ]
            if not child_node_ids:
                continue
            now = utc_now_iso()
            for child_id in child_node_ids:
                summary_to_child = MemoryEdge(
                    source_id=summary_node.id,
                    target_id=child_id,
                    relation_type="summary_parent",
                    weight=0.32,
                    reason="摘要记忆节点覆盖该候选文档，用于层级联想扩展",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                    metadata={
                        "query_id": summary_node.metadata.get("query_id"),
                        "summary_node_id": summary_node.id,
                        "child_node_id": child_id,
                        "score_breakdown": {
                            "hierarchy_score": 0.32,
                            "relation_source": "query_summary_memory",
                        },
                    },
                )
                child_to_summary = MemoryEdge(
                    source_id=child_id,
                    target_id=summary_node.id,
                    relation_type="summary_child",
                    weight=0.32,
                    reason="候选文档归属于该摘要记忆节点，用于从局部证据回到上下文摘要",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                    metadata=summary_to_child.metadata,
                )
                created.extend([summary_to_child, child_to_summary])
        self.store.upsert_edges(created)
        return created

    def _maybe_create_edge(self, seed: MemoryNode, other: MemoryNode) -> MemoryEdge | None:
        edge_score = self._score_candidate_edge(seed, other)
        if edge_score.relation_type is None:
            return None
        now = utc_now_iso()
        return MemoryEdge(
            source_id=seed.id,
            target_id=other.id,
            relation_type=edge_score.relation_type,
            weight=edge_score.weight,
            reason=edge_score.reason,
            created_at=now,
            updated_at=now,
            activation_count=0,
            last_activated_at=None,
            metadata={
                **edge_score.score_breakdown,
                "score_breakdown": edge_score.score_breakdown,
            },
        )

    def _score_candidate_edge(self, seed: MemoryNode, other: MemoryNode) -> EdgeScore:
        shared_entities = sorted(
            set(seed.metadata.get("entities", [])) & set(other.metadata.get("entities", []))
        )
        keyword_overlap = sorted(
            (set(seed.keywords) & set(other.keywords)) - GENERIC_EDGE_KEYWORDS
        )
        similarity = cosine_similarity(seed.embedding, other.embedding)

        entity_score = self._entity_score(shared_entities)
        keyword_score = self._keyword_score(keyword_overlap)
        semantic_score = self._semantic_score(similarity)
        score_breakdown: dict[str, object] = {
            "entity_score": round(entity_score, 4),
            "keyword_score": round(keyword_score, 4),
            "semantic_score": round(semantic_score, 4),
            "similarity": round(similarity, 4),
            "keyword_overlap": keyword_overlap,
            "shared_entities": shared_entities,
            "thresholds": {
                "similarity_threshold": self.similarity_threshold,
                "keyword_overlap_threshold": self.keyword_overlap_threshold,
            },
        }

        if shared_entities:
            return EdgeScore(
                relation_type="shared_entity",
                weight=entity_score,
                reason=f"共享实体：{', '.join(shared_entities[:4])}",
                score_breakdown=score_breakdown,
            )
        if len(keyword_overlap) >= self.keyword_overlap_threshold:
            return EdgeScore(
                relation_type="keyword_overlap",
                weight=keyword_score,
                reason=f"关键词重叠：{', '.join(keyword_overlap[:4])}",
                score_breakdown=score_breakdown,
            )
        if similarity >= self.similarity_threshold:
            return EdgeScore(
                relation_type="embedding_similarity",
                weight=semantic_score,
                reason=f"语义相似度达到阈值：{similarity:.3f}",
                score_breakdown=score_breakdown,
            )
        return EdgeScore(
            relation_type=None,
            weight=0.0,
            reason="未达到任一按需建边阈值",
            score_breakdown=score_breakdown,
        )

    def _entity_score(self, shared_entities: list[str]) -> float:
        return min(0.95, 0.55 + 0.12 * len(shared_entities)) if shared_entities else 0.0

    def _keyword_score(self, keyword_overlap: list[str]) -> float:
        return min(0.85, 0.35 + 0.08 * len(keyword_overlap)) if keyword_overlap else 0.0

    def _semantic_score(self, similarity: float) -> float:
        return min(0.8, similarity) if similarity >= self.similarity_threshold else similarity

    def _record_edge_event(
        self,
        edge: MemoryEdge,
        source_node: MemoryNode,
        target_node: MemoryNode,
        action: str,
    ) -> None:
        self.edge_creation_log.append(
            {
                "created_at": edge.updated_at,
                "action": action,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "source_title": source_node.metadata.get("title", source_node.id),
                "target_title": target_node.metadata.get("title", target_node.id),
                "relation_type": edge.relation_type,
                "weight": round(edge.weight, 4),
                "reason": edge.reason,
                "score_breakdown": edge.metadata.get("score_breakdown", edge.metadata),
            }
        )

    def write_edge_creation_log(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.edge_creation_log, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target
