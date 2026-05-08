from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    """返回统一的 UTC 时间字符串，便于实验复现和日志排序。"""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class MemoryNode:
    """记忆节点，对应开题报告中的“记忆笔记”。"""

    id: str
    text: str
    summary: str
    keywords: list[str]
    tags: list[str]
    source: str
    created_at: str
    usage_count: int
    confidence: float
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "summary": self.summary,
            "keywords": self.keywords,
            "tags": self.tags,
            "source": self.source,
            "created_at": self.created_at,
            "usage_count": self.usage_count,
            "confidence": self.confidence,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class MemoryEdge:
    """语义边，记录两个记忆节点之间的联想关系。"""

    source_id: str
    target_id: str
    relation_type: str
    weight: float
    reason: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source_id, self.target_id, self.relation_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "reason": self.reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RetrievalHit:
    """检索结果，包含可解释排序信号和联想路径。"""

    node: MemoryNode
    score: float
    similarity_score: float
    graph_score: float
    usage_score: float
    confidence_score: float
    path: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node.to_dict(),
            "score": self.score,
            "similarity_score": self.similarity_score,
            "graph_score": self.graph_score,
            "usage_score": self.usage_score,
            "confidence_score": self.confidence_score,
            "path": self.path,
            "reason": self.reason,
        }


@dataclass(slots=True)
class EvaluationQuery:
    """公开数据集中的一个评测查询。"""

    id: str
    dataset: str
    question: str
    answer: str
    supporting_doc_ids: list[str]
    candidate_doc_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetDocument:
    """被写入记忆库的文档片段。"""

    id: str
    dataset: str
    title: str
    text: str
    source: str
    tags: list[str]
    keywords: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
