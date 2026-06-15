from __future__ import annotations

from dataclasses import dataclass
import math

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


@dataclass(frozen=True, slots=True)
class InsightRecord:
    """多个长期记忆重构后形成的高层洞察节点。"""

    node_id: str
    source_consolidated_node_ids: list[str]
    evidence_node_ids: list[str]


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
            embedding=_consolidated_embedding(evidence_hits),
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
                "embedding_source": "evidence_centroid",
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


class InsightMemoryBuilder:
    """把多个单次巩固记忆重构为高层洞察记忆。"""

    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider

    def build_from_consolidated_memories(
        self,
        *,
        dataset: str | None = None,
        min_consolidated_count: int = 2,
        max_source_memories: int | None = None,
    ) -> list[InsightRecord]:
        consolidated_nodes = [
            node
            for node in self.store.get_nodes()
            if node.metadata.get("node_type") == "consolidated_memory"
            and (dataset is None or node.metadata.get("dataset") == dataset)
        ]
        consolidated_nodes = sorted(consolidated_nodes, key=lambda node: (node.created_at, node.id))
        if max_source_memories is not None:
            consolidated_nodes = consolidated_nodes[:max_source_memories]
        if len(consolidated_nodes) < min_consolidated_count:
            return []

        insight = self._build_insight_node(consolidated_nodes, dataset=dataset)
        self.store.upsert_node(insight)
        self._update_source_nodes(insight, consolidated_nodes)
        edges = self._insight_edges(insight, consolidated_nodes)
        self.store.upsert_edges(edges)
        self.store.log_memory_events(
            [
                MemoryEvent(
                    event_type="insight_memory_created",
                    query_id=None,
                    query="; ".join(_source_questions(consolidated_nodes)[:3]),
                    mode="memory_reconstruction",
                    node_id=insight.id,
                    path=[
                        insight.id,
                        *insight.metadata["source_consolidated_node_ids"],
                    ],
                    score=insight.confidence,
                    created_at=insight.created_at,
                    metadata={
                        "dataset": insight.metadata.get("dataset"),
                        "source_consolidated_node_ids": insight.metadata[
                            "source_consolidated_node_ids"
                        ],
                        "evidence_node_ids": insight.metadata["evidence_node_ids"],
                        "shared_keywords": insight.metadata["shared_keywords"],
                        "insight_source": insight.metadata["insight_source"],
                    },
                )
            ]
        )
        return [
            InsightRecord(
                node_id=insight.id,
                source_consolidated_node_ids=insight.metadata["source_consolidated_node_ids"],
                evidence_node_ids=insight.metadata["evidence_node_ids"],
            )
        ]

    def _build_insight_node(
        self,
        consolidated_nodes: list[MemoryNode],
        *,
        dataset: str | None,
    ) -> MemoryNode:
        now = utc_now_iso()
        source_ids = [node.id for node in consolidated_nodes]
        evidence_ids = _unique_strings(
            evidence_id
            for node in consolidated_nodes
            for evidence_id in node.metadata.get("evidence_node_ids", [])
        )
        evidence_nodes = self.store.get_nodes(evidence_ids)
        evidence_titles = _unique_strings(
            str(node.metadata.get("title") or node.id)
            for node in evidence_nodes
        )
        shared_keywords = _top_keywords(consolidated_nodes, limit=10)
        source_questions = _source_questions(consolidated_nodes)
        source_answers = _unique_strings(
            str(node.metadata.get("answer"))
            for node in consolidated_nodes
            if node.metadata.get("answer")
        )
        dataset_name = dataset or str(consolidated_nodes[0].metadata.get("dataset") or "unknown")
        node_id = stable_id(
            "insight",
            f"{dataset_name}:{','.join(source_ids)}",
        )
        existing = self.store.get_node(node_id)
        question_lines = [f"- {question}" for question in source_questions[:5]]
        evidence_lines = [
            f"- {title}: {node.summary or node.text[:200]}"
            for title, node in zip(evidence_titles, evidence_nodes, strict=False)
        ][:6]
        keyword_text = "、".join(shared_keywords[:6]) if shared_keywords else "跨证据关联"
        text = "\n".join(
            [
                f"高层洞察：{dataset_name} 中 {len(consolidated_nodes)} 个已巩固记忆围绕 {keyword_text} 形成可复用证据模式。",
                "覆盖问题：",
                *question_lines,
                "可回溯证据：",
                *evidence_lines,
            ]
        )
        summary = f"{dataset_name} 高层洞察：{keyword_text}"
        return MemoryNode(
            id=node_id,
            text=text,
            summary=summary,
            keywords=extract_keywords(text, limit=14),
            tags=["insight_memory", dataset_name, "memory_reconstruction"],
            source="sam_insight_reconstruction",
            created_at=existing.created_at if existing else now,
            last_accessed_at=existing.last_accessed_at if existing else None,
            usage_count=existing.usage_count if existing else 0,
            confidence=min(0.98, 0.74 + 0.03 * len(consolidated_nodes)),
            embedding=_centroid_embedding([*consolidated_nodes, *evidence_nodes]),
            metadata={
                "node_type": "insight_memory",
                "dataset": dataset_name,
                "source_consolidated_node_ids": source_ids,
                "source_query_ids": _unique_strings(
                    str(node.metadata.get("query_id"))
                    for node in consolidated_nodes
                    if node.metadata.get("query_id")
                ),
                "source_answers": source_answers,
                "evidence_node_ids": evidence_ids,
                "evidence_titles": evidence_titles,
                "shared_keywords": shared_keywords,
                "source_consolidated_count": len(consolidated_nodes),
                "evidence_count": len(evidence_ids),
                "insight_source": "consolidated_memory_cluster",
                "reconstruction_level": "insight",
                "traceability": "insight_memory -> consolidated_memory -> evidence_memory",
                "embedding_source": "consolidated_and_evidence_centroid",
            },
        )

    def _update_source_nodes(
        self,
        insight: MemoryNode,
        consolidated_nodes: list[MemoryNode],
    ) -> None:
        evidence_ids = insight.metadata["evidence_node_ids"]
        nodes = [*consolidated_nodes, *self.store.get_nodes(evidence_ids)]
        for node in nodes:
            insight_ids = [
                str(item)
                for item in node.metadata.get("insight_ids", [])
            ]
            if insight.id not in insight_ids:
                insight_ids.append(insight.id)
            node.metadata = {
                **node.metadata,
                "insight_ids": insight_ids,
                "insight_count": int(node.metadata.get("insight_count", 0)) + 1,
            }
            self.store.upsert_node(node)

    def _insight_edges(
        self,
        insight: MemoryNode,
        consolidated_nodes: list[MemoryNode],
    ) -> list[MemoryEdge]:
        now = insight.created_at
        edges: list[MemoryEdge] = []
        for node in consolidated_nodes:
            metadata = {
                "insight_node_id": insight.id,
                "source_consolidated_node_id": node.id,
                "score_breakdown": {
                    "memory_reconstruction": 1.0,
                    "source_confidence": node.confidence,
                },
            }
            edges.append(
                MemoryEdge(
                    source_id=insight.id,
                    target_id=node.id,
                    relation_type="insight_summarizes_memory",
                    weight=0.82,
                    reason="高层洞察节点汇总多个已巩固长期记忆",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                    metadata=metadata,
                )
            )
            edges.append(
                MemoryEdge(
                    source_id=node.id,
                    target_id=insight.id,
                    relation_type="memory_summarized_by_insight",
                    weight=0.82,
                    reason="已巩固长期记忆被重构为高层洞察",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                    metadata=metadata,
                )
            )

        for evidence_id in insight.metadata["evidence_node_ids"]:
            metadata = {
                "insight_node_id": insight.id,
                "evidence_node_id": evidence_id,
                "score_breakdown": {
                    "traceability": 1.0,
                    "insight_confidence": insight.confidence,
                },
            }
            edges.append(
                MemoryEdge(
                    source_id=insight.id,
                    target_id=evidence_id,
                    relation_type="insight_traces_evidence",
                    weight=0.76,
                    reason="高层洞察可回溯到底层证据记忆",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                    metadata=metadata,
                )
            )
            edges.append(
                MemoryEdge(
                    source_id=evidence_id,
                    target_id=insight.id,
                    relation_type="evidence_supports_insight",
                    weight=0.76,
                    reason="底层证据支撑高层洞察记忆",
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


def _consolidated_embedding(hits: list[RetrievalHit]) -> list[float]:
    """用已检索证据的向量合成长期记忆向量，避免巩固阶段再次请求在线 embedding。"""

    valid_hits = [hit for hit in hits if hit.node.embedding]
    if not valid_hits:
        return []
    dimensions = min(len(hit.node.embedding) for hit in valid_hits)
    if dimensions <= 0:
        return []
    weighted = [0.0] * dimensions
    total_weight = 0.0
    for hit in valid_hits:
        weight = max(0.05, float(hit.score))
        total_weight += weight
        for index, value in enumerate(hit.node.embedding[:dimensions]):
            weighted[index] += float(value) * weight
    if total_weight <= 0.0:
        return [0.0] * dimensions
    vector = [value / total_weight for value in weighted]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _centroid_embedding(nodes: list[MemoryNode]) -> list[float]:
    """用已有节点向量生成洞察记忆向量，避免重构阶段再次请求在线 embedding。"""

    valid_nodes = [node for node in nodes if node.embedding]
    if not valid_nodes:
        return []
    dimensions = min(len(node.embedding) for node in valid_nodes)
    if dimensions <= 0:
        return []
    vector = [0.0] * dimensions
    total_weight = 0.0
    for node in valid_nodes:
        weight = max(0.05, float(node.confidence))
        total_weight += weight
        for index, value in enumerate(node.embedding[:dimensions]):
            vector[index] += float(value) * weight
    if total_weight <= 0.0:
        return [0.0] * dimensions
    vector = [value / total_weight for value in vector]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _unique_strings(values: object) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _top_keywords(nodes: list[MemoryNode], limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for node in nodes:
        for keyword in node.keywords:
            text = str(keyword).strip()
            if not text:
                continue
            counts[text] = counts.get(text, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [keyword for keyword, _ in ranked[:limit]]


def _source_questions(consolidated_nodes: list[MemoryNode]) -> list[str]:
    questions: list[str] = []
    for node in consolidated_nodes:
        first_line = node.text.splitlines()[0] if node.text else node.summary
        question = first_line.removeprefix("问题：").strip()
        if question:
            questions.append(question)
    return _unique_strings(questions)
