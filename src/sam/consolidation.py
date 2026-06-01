from __future__ import annotations

from dataclasses import dataclass

from sam.embedding import EmbeddingProvider
from sam.models import EvaluationQuery, MemoryEdge, MemoryEvent, MemoryNode, RetrievalHit, utc_now_iso
from sam.store import MemoryStore
from sam.text import extract_keywords, stable_id


@dataclass(frozen=True, slots=True)
class ConsolidationRecord:
    """一次记忆巩固产生的长期记忆节点。"""

    node_id: str
    support_node_ids: list[str]


class MemoryConsolidator:
    """把一次成功检索沉淀为可复用的长期记忆。"""

    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider

    def consolidate_query(
        self,
        *,
        query: EvaluationQuery,
        mode: str,
        hits: list[RetrievalHit],
        support_node_ids: set[str],
        answer_status: str,
    ) -> ConsolidationRecord | None:
        support_hits = [hit for hit in hits if hit.node.id in support_node_ids]
        if not support_hits:
            return None

        now = utc_now_iso()
        support_ids = [hit.node.id for hit in support_hits]
        node_id = stable_id("consolidated", f"{query.id}:{mode}:{','.join(sorted(support_ids))}")
        evidence_titles = [
            str(hit.node.metadata.get("title") or hit.node.id)
            for hit in support_hits
        ]
        evidence_lines = [
            f"- {title}: {hit.node.summary or hit.node.text[:240]}"
            for title, hit in zip(evidence_titles, support_hits, strict=True)
        ]
        text = "\n".join(
            [
                f"问题：{query.question}",
                f"答案：{query.answer}",
                f"检索方法：{mode}",
                f"答案状态：{answer_status}",
                "支持证据：",
                *evidence_lines,
            ]
        )
        existing = self.store.get_node(node_id)
        node = MemoryNode(
            id=node_id,
            text=text,
            summary=f"{query.question} -> {query.answer}",
            keywords=extract_keywords(text, limit=12),
            tags=["consolidated_memory", mode, query.dataset],
            source="sam_consolidation",
            created_at=existing.created_at if existing else now,
            last_accessed_at=existing.last_accessed_at if existing else None,
            usage_count=existing.usage_count if existing else 0,
            confidence=min(0.96, 0.70 + 0.06 * len(support_hits)),
            embedding=self.embedding_provider.embed(text),
            metadata={
                "node_type": "consolidated_memory",
                "query_id": query.id,
                "dataset": query.dataset,
                "mode": mode,
                "answer": query.answer,
                "answer_status": answer_status,
                "support_node_ids": support_ids,
                "support_titles": evidence_titles,
                "consolidation_source": "feedback_support_hit",
            },
        )
        self.store.upsert_node(node)
        self._update_support_nodes(support_hits, node_id)
        self.store.upsert_edges(self._consolidation_edges(node_id, support_ids, now))
        self.store.log_memory_events(
            [
                MemoryEvent(
                    event_type="memory_consolidated",
                    query_id=query.id,
                    query=query.question,
                    mode=mode,
                    node_id=node_id,
                    path=[node_id, *support_ids],
                    score=node.confidence,
                    created_at=now,
                    metadata={
                        "answer": query.answer,
                        "answer_status": answer_status,
                        "support_node_ids": support_ids,
                        "support_titles": evidence_titles,
                    },
                )
            ]
        )
        return ConsolidationRecord(node_id=node_id, support_node_ids=support_ids)

    def _update_support_nodes(self, support_hits: list[RetrievalHit], consolidated_id: str) -> None:
        for hit in support_hits:
            node = self.store.get_node(hit.node.id) or hit.node
            consolidated_by = [
                str(item)
                for item in node.metadata.get("consolidated_by", [])
            ]
            if consolidated_id not in consolidated_by:
                consolidated_by.append(consolidated_id)
            node.metadata = {
                **node.metadata,
                "consolidated_by": consolidated_by,
                "consolidation_count": int(node.metadata.get("consolidation_count", 0)) + 1,
            }
            node.confidence = min(0.99, node.confidence + 0.02)
            self.store.upsert_node(node)

    def _consolidation_edges(
        self,
        consolidated_id: str,
        support_node_ids: list[str],
        now: str,
    ) -> list[MemoryEdge]:
        edges: list[MemoryEdge] = []
        for support_id in support_node_ids:
            metadata = {
                "consolidated_node_id": consolidated_id,
                "support_node_id": support_id,
                "score_breakdown": {
                    "feedback_support": 1.0,
                    "relation_source": "memory_consolidation",
                },
            }
            edges.append(
                MemoryEdge(
                    source_id=consolidated_id,
                    target_id=support_id,
                    relation_type="consolidates_support",
                    weight=0.72,
                    reason="成功检索后的长期记忆巩固节点指向支持证据",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                    metadata=metadata,
                )
            )
            edges.append(
                MemoryEdge(
                    source_id=support_id,
                    target_id=consolidated_id,
                    relation_type="support_consolidated_by",
                    weight=0.72,
                    reason="支持证据被成功检索巩固为长期记忆",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                    metadata=metadata,
                )
            )
        return edges
