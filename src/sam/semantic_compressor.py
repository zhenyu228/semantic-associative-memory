"""高层语义压缩模块。

该模块只负责把一组已经激活的底层 memory items 压缩为高层 summary memory。
底层建边仍由非 LLM 信号完成，LLM 只在这里承担语义抽象职责。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from sam.llm import ChatClient, create_chat_client
from sam.models import MemoryNode, utc_now_iso


class SemanticCompressor(Protocol):
    """高层语义压缩接口。"""

    def compress(
        self,
        nodes: list[MemoryNode],
        *,
        group_id: str,
        chunk_token_size: int = 512,
    ) -> "CompressionResult":
        """把一组底层记忆节点压缩为一个高层记忆结果。"""


@dataclass(slots=True)
class CompressionResult:
    """一次高层语义压缩的结果。"""

    group_id: str
    summary: str
    source_node_ids: list[str]
    input_token_count: int
    output_token_count: int
    created_at: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_memory_node(self, *, embedding: list[float] | None = None) -> MemoryNode:
        """转换为可写入 MemoryStore 的 summary memory node。"""

        node_id = f"summary_{_safe_id(self.group_id)}"
        return MemoryNode(
            id=node_id,
            text=self.summary,
            summary=self.summary,
            keywords=list(self.metadata.get("keywords", [])),
            tags=["summary_memory"],
            source="semantic_compressor",
            created_at=self.created_at,
            last_accessed_at=None,
            usage_count=0,
            confidence=float(self.metadata.get("confidence", 0.85)),
            embedding=embedding or [],
            metadata={
                "node_type": "summary_memory",
                "group_id": self.group_id,
                "source_node_ids": self.source_node_ids,
                "input_token_count": self.input_token_count,
                "output_token_count": self.output_token_count,
                **self.metadata,
            },
        )


class ExtractiveSemanticCompressor:
    """不依赖 API 的抽取式高层压缩器。"""

    def __init__(self, *, max_sentences: int = 4) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences 必须大于 0")
        self.max_sentences = max_sentences

    def compress(
        self,
        nodes: list[MemoryNode],
        *,
        group_id: str,
        chunk_token_size: int = 512,
    ) -> CompressionResult:
        if not nodes:
            raise ValueError("nodes 不能为空")
        snippets = [_first_sentence(node.summary or node.text) for node in nodes]
        selected = [item for item in snippets if item][: self.max_sentences]
        summary = " ".join(selected).strip()
        keywords = _merge_keywords(nodes)
        return CompressionResult(
            group_id=group_id,
            summary=summary,
            source_node_ids=[node.id for node in nodes],
            input_token_count=sum(_token_count(node.text) for node in nodes),
            output_token_count=_token_count(summary),
            created_at=utc_now_iso(),
            metadata={
                "compressor": "extractive",
                "chunk_token_size": chunk_token_size,
                "source_node_count": len(nodes),
                "keywords": keywords,
            },
        )


class LLMSemanticCompressor:
    """使用聊天模型的高层语义压缩器。"""

    def __init__(
        self,
        chat_client: ChatClient | None = None,
        *,
        max_tokens: int = 300,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        self.chat_client = chat_client or create_chat_client()
        self.max_tokens = max_tokens

    def compress(
        self,
        nodes: list[MemoryNode],
        *,
        group_id: str,
        chunk_token_size: int = 512,
    ) -> CompressionResult:
        if not nodes:
            raise ValueError("nodes 不能为空")
        prompt = _compression_prompt(nodes, group_id=group_id)
        summary = self.chat_client.complete(prompt, max_tokens=self.max_tokens).strip()
        keywords = _merge_keywords(nodes)
        return CompressionResult(
            group_id=group_id,
            summary=summary,
            source_node_ids=[node.id for node in nodes],
            input_token_count=sum(_token_count(node.text) for node in nodes),
            output_token_count=_token_count(summary),
            created_at=utc_now_iso(),
            metadata={
                "compressor": "llm",
                "chunk_token_size": chunk_token_size,
                "source_node_count": len(nodes),
                "llm_max_tokens": self.max_tokens,
                "keywords": keywords,
            },
        )


def _compression_prompt(nodes: list[MemoryNode], *, group_id: str) -> list[dict[str, object]]:
    evidence_lines: list[str] = []
    for index, node in enumerate(nodes, start=1):
        title = str(node.metadata.get("title") or node.metadata.get("original_doc_id") or node.id)
        evidence_lines.append(
            f"[{index}] title={title}\nsummary={node.summary}\ntext={node.text[:1200]}"
        )
    user = (
        "请把下面一组底层记忆压缩成一个高层语义记忆。"
        "要求：保留核心研究对象、方法、证据关系和限制；不要编造未出现的信息。\n\n"
        f"group_id: {group_id}\n\n"
        + "\n\n".join(evidence_lines)
    )
    return [
        {"role": "system", "content": "你是一个科研记忆系统中的高层语义压缩模块。"},
        {"role": "user", "content": user},
    ]


def _first_sentence(text: str) -> str:
    clean = " ".join(text.strip().split())
    if not clean:
        return ""
    parts = re.split(r"(?<=[。！？.!?])\s+", clean, maxsplit=1)
    return parts[0].strip()


def _merge_keywords(nodes: list[MemoryNode]) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for node in nodes:
        for keyword in node.keywords:
            normalized = keyword.strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                keywords.append(normalized)
    return keywords[:12]


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip())
    return normalized.strip("_") or "memory"


def _token_count(text: str) -> int:
    return len(text.split())
