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
    evidence_node_ids: list[str] | None = None


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
        evidence_hits = support_hits or _structural_evidence_hits(hits)
        if not evidence_hits:
            return None

        now = utc_now_iso()
        support_ids = [hit.node.id for hit in support_hits]
        evidence_ids = [hit.node.id for hit in evidence_hits]
        consolidation_source = (
            "feedback_support_hit"
            if support_hits
            else "structural_activation"
        )
        node_id = stable_id("consolidated", f"{query.id}:{mode}:{','.join(sorted(support_ids))}")
        if not support_ids:
            node_id = stable_id("consolidated", f"{query.id}:{mode}:structural:{','.join(sorted(evidence_ids))}")
        evidence_titles = [
            str(hit.node.metadata.get("title") or hit.node.id)
            for hit in evidence_hits
        ]
        evidence_lines = [
            f"- {title}: {hit.node.summary or hit.node.text[:240]}"
            for title, hit in zip(evidence_titles, evidence_hits, strict=True)
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
        evidence_original_doc_ids = [
            str(hit.node.metadata.get("original_doc_id"))
            for hit in evidence_hits
            if hit.node.metadata.get("original_doc_id")
        ]
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
            confidence=min(0.96, 0.70 + 0.06 * len(support_hits)) if support_hits else 0.62,
            embedding=self.embedding_provider.embed(text),
            metadata={
                "node_type": "consolidated_memory",
                "query_id": query.id,
                "dataset": query.dataset,
                "mode": mode,
                "answer": query.answer,
                "answer_status": answer_status,
                "support_node_ids": support_ids,
                "support_titles": evidence_titles if support_hits else [],
                "evidence_node_ids": evidence_ids,
                "evidence_original_doc_ids": evidence_original_doc_ids,
                "evidence_titles": evidence_titles,
                "consolidation_source": consolidation_source,
            },
        )
        self.store.upsert_node(node)
        if support_hits:
            self._update_support_nodes(support_hits, node_id)
        self.store.upsert_edges(
            self._consolidation_edges(
                node_id,
                support_ids if support_hits else evidence_ids,
                now,
                relation_source=consolidation_source,
            )
        )
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
                        "support_titles": evidence_titles if support_hits else [],
                        "evidence_node_ids": evidence_ids,
                        "evidence_titles": evidence_titles,
                        "consolidation_source": consolidation_source,
                    },
                )
            ]
        )
        return ConsolidationRecord(node_id=node_id, support_node_ids=support_ids, evidence_node_ids=evidence_ids)

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
        *,
        relation_source: str = "feedback_support_hit",
    ) -> list[MemoryEdge]:
        edges: list[MemoryEdge] = []
        for support_id in support_node_ids:
            metadata = {
                "consolidated_node_id": consolidated_id,
                "support_node_id": support_id,
                "score_breakdown": {
                    "feedback_support": 1.0,
                    "relation_source": relation_source,
                },
            }
            if relation_source == "structural_activation":
                metadata["score_breakdown"] = {
                    "structural_activation": 1.0,
                    "relation_source": relation_source,
                }
            edges.append(
                MemoryEdge(
                    source_id=consolidated_id,
                    target_id=support_id,
                    relation_type="consolidates_support" if relation_source != "structural_activation" else "consolidates_evidence",
                    weight=0.72 if relation_source != "structural_activation" else 0.58,
                    reason="成功检索后的长期记忆巩固节点指向支持证据"
                    if relation_source != "structural_activation"
                    else "检索激活后的结构性长期记忆节点指向实际激活证据",
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
                    relation_type="support_consolidated_by" if relation_source != "structural_activation" else "evidence_consolidated_by",
                    weight=0.72 if relation_source != "structural_activation" else 0.58,
                    reason="支持证据被成功检索巩固为长期记忆"
                    if relation_source != "structural_activation"
                    else "实际激活证据被沉淀为结构性长期记忆",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                    metadata=metadata,
                )
            )
        return edges


def _structural_evidence_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """选择可用于结构性巩固的实际激活证据。

    结构性巩固不声称这些节点是 gold support，只记录本次检索中被激活、
    可用于后续类比分析的候选证据节点。
    """

    evidence_hits = [
        hit
        for hit in hits
        if hit.node.metadata.get("node_type") != "query_summary"
        and hit.node.metadata.get("node_type") != "consolidated_memory"
    ]
    return evidence_hits[:3]
