from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from sam.llm import ChatClient, create_chat_client
from sam.models import MemoryNode


@dataclass(frozen=True, slots=True)
class RelationJudgment:
    """候选边的关系级判别结果。"""

    should_link: bool
    relation_type: str
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "should_link": self.should_link,
            "relation_type": self.relation_type,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


class RelationJudge(Protocol):
    """关系判别器接口，可由 GPT-5.4 或本地规则实现。"""

    def judge(
        self,
        seed: MemoryNode,
        other: MemoryNode,
        score_breakdown: dict[str, object],
    ) -> RelationJudgment:
        raise NotImplementedError


class ChatRelationJudge:
    """使用聊天模型判断两个记忆节点是否应建立语义边。"""

    def __init__(
        self,
        chat_client: ChatClient | None = None,
        *,
        min_confidence: float = 0.55,
        fail_open: bool = True,
    ) -> None:
        self.chat_client = chat_client or create_chat_client("azure_openai")
        self.min_confidence = min_confidence
        self.fail_open = fail_open

    def judge(
        self,
        seed: MemoryNode,
        other: MemoryNode,
        score_breakdown: dict[str, object],
    ) -> RelationJudgment:
        prompt = _relation_prompt(seed, other, score_breakdown)
        try:
            content = self.chat_client.complete(
                [
                    {
                        "role": "system",
                        "content": "你是知识图谱关系判别器，只输出 JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
            )
            judgment = _parse_relation_judgment(content)
        except Exception as exc:
            if self.fail_open:
                return RelationJudgment(
                    should_link=True,
                    relation_type=str(score_breakdown.get("relation_type_hint", "model_unavailable")),
                    confidence=0.0,
                    reason=f"关系判别失败，按 fail_open 保留候选边：{exc}",
                )
            return RelationJudgment(
                should_link=False,
                relation_type="model_error",
                confidence=0.0,
                reason=f"关系判别失败：{exc}",
            )
        if judgment.confidence < self.min_confidence:
            return RelationJudgment(
                should_link=False,
                relation_type=judgment.relation_type,
                confidence=judgment.confidence,
                reason=f"关系置信度低于阈值：{judgment.reason}",
            )
        return judgment


def create_relation_judge(name: str | None = None) -> RelationJudge | None:
    provider = name or "disabled"
    if provider in {"disabled", "none", ""}:
        return None
    if provider in {"gpt", "gpt54", "azure_openai", "chat"}:
        return ChatRelationJudge()
    raise ValueError(f"未知关系判别器：{provider}")


def _relation_prompt(
    seed: MemoryNode,
    other: MemoryNode,
    score_breakdown: dict[str, object],
) -> str:
    return f"""请判断两个记忆节点之间是否应该建立知识图谱语义边。

要求：
1. 只有当两个节点存在可解释的实体关系、事件关系、因果/从属/同一主题关系时，should_link 才能为 true。
2. 只因为泛化词、模板词或偶然关键词重叠，不应该建边。
3. relation_type 使用英文短标签，例如 shared_entity、same_topic、temporal_relation、causal_relation、unrelated。
4. 只输出 JSON，不要输出额外解释。

输出格式：
{{"should_link": true, "relation_type": "same_topic", "confidence": 0.82, "reason": "简短原因"}}

候选边打分：
{json.dumps(score_breakdown, ensure_ascii=False)}

节点 A：
标题：{seed.metadata.get("title", "")}
关键词：{", ".join(seed.keywords)}
摘要：{seed.summary}
正文：{seed.text[:900]}

节点 B：
标题：{other.metadata.get("title", "")}
关键词：{", ".join(other.keywords)}
摘要：{other.summary}
正文：{other.text[:900]}
"""


def _parse_relation_judgment(content: str) -> RelationJudgment:
    payload = _extract_json_object(content)
    data = json.loads(payload)
    return RelationJudgment(
        should_link=bool(data.get("should_link", False)),
        relation_type=str(data.get("relation_type", "unrelated")),
        confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
        reason=str(data.get("reason", "")),
    )


def _extract_json_object(content: str) -> str:
    clean = content.strip()
    if clean.startswith("{") and clean.endswith("}"):
        return clean
    match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("模型输出中没有 JSON 对象")
