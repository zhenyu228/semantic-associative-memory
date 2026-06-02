from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from sam.llm import ChatClient, create_chat_client
from sam.models import EvaluationQuery
from sam.text import extract_keywords


@dataclass(slots=True)
class QueryPlan:
    """一次检索前的查询规划结果。"""

    retrieval_query: str
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    reason: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "retrieval_query": self.retrieval_query,
            "keywords": self.keywords,
            "entities": self.entities,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class QueryPlanner(Protocol):
    """查询规划接口，用于把原始问题改写成更适合检索的文本。"""

    def plan(self, query: EvaluationQuery) -> QueryPlan:
        raise NotImplementedError


class HeuristicQueryPlanner:
    """轻量查询规划器。

    该规划器只使用问题和元信息，不把多选题全部选项塞进检索文本，避免 NovelQA
    中干扰选项反向污染召回。
    """

    def plan(self, query: EvaluationQuery) -> QueryPlan:
        keywords = extract_keywords(query.question, limit=10)
        entities = _extract_surface_entities(query.question)
        metadata_terms = _metadata_terms(query.metadata)
        query_parts = [
            query.question,
            " ".join(keywords),
            " ".join(metadata_terms),
        ]
        retrieval_query = " ".join(part.strip() for part in query_parts if part.strip())
        return QueryPlan(
            retrieval_query=retrieval_query,
            keywords=keywords,
            entities=entities,
            reason="使用问题关键词、显式实体和数据集任务元信息构造检索文本，避免引入全部候选选项。",
            metadata={
                "planner": "heuristic",
                "used_metadata_fields": [
                    field
                    for field in ["aspect", "complexity", "question_type"]
                    if query.metadata.get(field)
                ],
                "option_text_included": False,
            },
        )


class ChatQueryPlanner:
    """基于聊天模型的查询规划器。

    模型只返回 JSON；如果返回格式不可解析，自动退回启发式规划，保证实验流程不中断。
    """

    def __init__(self, client: ChatClient | None = None) -> None:
        self.client = client or create_chat_client("azure_openai")
        self.fallback = HeuristicQueryPlanner()

    def plan(self, query: EvaluationQuery) -> QueryPlan:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是长文本问答系统的查询规划器。请把原始问题改写为适合检索的查询，"
                    "提取关键词和实体。不要把所有候选选项原文都加入 retrieval_query。"
                    "只输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": query.question,
                        "answer_format": query.metadata.get("answer_format"),
                        "aspect": query.metadata.get("aspect"),
                        "complexity": query.metadata.get("complexity"),
                        "options": query.metadata.get("options"),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            content = self.client.complete(messages, max_tokens=500)
            return self._parse_plan(content, query)
        except Exception as exc:
            fallback = self.fallback.plan(query)
            fallback.metadata["planner"] = "chat_fallback"
            fallback.metadata["fallback_reason"] = str(exc)
            return fallback

    def _parse_plan(self, content: str, query: EvaluationQuery) -> QueryPlan:
        payload = json.loads(_strip_json_fence(content))
        retrieval_query = str(payload.get("retrieval_query", "")).strip()
        if not retrieval_query:
            raise ValueError("聊天模型没有返回 retrieval_query")
        keywords = _string_list(payload.get("keywords"))
        entities = _string_list(payload.get("entities"))
        reason = str(payload.get("reason", ""))
        return QueryPlan(
            retrieval_query=retrieval_query,
            keywords=keywords or extract_keywords(retrieval_query, limit=10),
            entities=entities,
            reason=reason,
            metadata={
                "planner": "chat",
                "model_query_id": query.id,
            },
        )


def create_query_planner(name: str | None) -> QueryPlanner | None:
    """按命令行参数创建查询规划器。"""

    planner_name = (name or "disabled").strip().lower()
    if planner_name in {"disabled", "none", "off"}:
        return None
    if planner_name in {"heuristic", "local"}:
        return HeuristicQueryPlanner()
    if planner_name in {"gpt54", "azure_openai", "chat"}:
        return ChatQueryPlanner()
    raise ValueError(f"未知 query planner: {name}")


def _metadata_terms(metadata: dict[str, object]) -> list[str]:
    terms: list[str] = []
    for field in ["aspect", "complexity", "question_type"]:
        value = metadata.get(field)
        if isinstance(value, str) and value.strip():
            terms.append(value.strip())
    return terms


def _extract_surface_entities(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9_\-]*(?:\s+[A-Z][A-Za-z0-9_\-]*)*\b", text)
    seen: dict[str, None] = {}
    for candidate in candidates:
        if candidate.lower() in {"what", "which", "why", "where", "when", "who", "how"}:
            continue
        seen.setdefault(candidate, None)
    return list(seen.keys())[:8]


def _strip_json_fence(content: str) -> str:
    clean = content.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
