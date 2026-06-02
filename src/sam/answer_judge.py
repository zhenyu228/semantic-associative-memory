from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from sam.llm import ChatClient, create_chat_client
from sam.text import tokenize


@dataclass(frozen=True, slots=True)
class AnswerJudgment:
    """生成答案是否等价于标准答案的判别结果。"""

    answer_hit: bool
    status: str
    score: float
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_hit": self.answer_hit,
            "status": self.status,
            "score": self.score,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class AnswerJudge(Protocol):
    """答案判别接口。"""

    def judge(self, question: str, gold_answer: str, generated_answer: str) -> AnswerJudgment:
        raise NotImplementedError


class RuleBasedAnswerJudge:
    """本地规则答案判别器，用于无 API 场景和 smoke test。"""

    def judge(self, question: str, gold_answer: str, generated_answer: str) -> AnswerJudgment:
        normalized_gold = _normalize(gold_answer)
        normalized_generated = _normalize(generated_answer)
        if not normalized_gold:
            return AnswerJudgment(
                answer_hit=False,
                status="empty_gold_answer",
                score=0.0,
                reason="标准答案为空，无法自动判别。",
                metadata={"judge": "rule"},
            )
        if normalized_gold in normalized_generated or normalized_generated in normalized_gold:
            return AnswerJudgment(
                answer_hit=True,
                status="exact_or_substring_match",
                score=1.0,
                reason="标准答案和生成答案存在规范化后的包含关系。",
                metadata={"judge": "rule"},
            )
        coverage = _key_term_coverage(gold_answer, generated_answer)
        if coverage >= 0.5:
            return AnswerJudgment(
                answer_hit=True,
                status="key_terms_covered",
                score=coverage,
                reason="生成答案覆盖了标准答案中的关键内容词。",
                metadata={"judge": "rule", "term_coverage": round(coverage, 4)},
            )
        return AnswerJudgment(
            answer_hit=False,
            status="not_matched",
            score=coverage,
            reason="生成答案未覆盖标准答案或关键内容词不足。",
            metadata={"judge": "rule", "term_coverage": round(coverage, 4)},
        )


class ChatAnswerJudge:
    """基于聊天模型的答案语义等价判别器。"""

    def __init__(self, client: ChatClient | None = None) -> None:
        self.client = client or create_chat_client("azure_openai")
        self.fallback = RuleBasedAnswerJudge()

    def judge(self, question: str, gold_answer: str, generated_answer: str) -> AnswerJudgment:
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": (
                    "你是问答评测器。判断 generated_answer 是否在语义上回答了问题，"
                    "并与 gold_answer 等价或覆盖主要信息。只输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "gold_answer": gold_answer,
                        "generated_answer": generated_answer,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            payload = json.loads(_strip_json_fence(self.client.complete(messages, max_tokens=300)))
            answer_hit = bool(payload.get("answer_hit", False))
            score = float(payload.get("score", 1.0 if answer_hit else 0.0))
            return AnswerJudgment(
                answer_hit=answer_hit,
                status=str(payload.get("status", "llm_equivalent" if answer_hit else "llm_not_equivalent")),
                score=max(0.0, min(1.0, score)),
                reason=str(payload.get("reason", "")),
                metadata={"judge": "chat"},
            )
        except Exception as exc:
            fallback = self.fallback.judge(question, gold_answer, generated_answer)
            return AnswerJudgment(
                answer_hit=fallback.answer_hit,
                status=f"chat_fallback_{fallback.status}",
                score=fallback.score,
                reason=f"聊天模型判别失败，回退到规则判别：{exc}",
                metadata={**fallback.metadata, "judge": "chat_fallback"},
            )


def create_answer_judge(name: str | None = None) -> AnswerJudge:
    judge_name = (name or "rule").strip().lower()
    if judge_name in {"rule", "heuristic", "local"}:
        return RuleBasedAnswerJudge()
    if judge_name in {"gpt54", "azure_openai", "chat"}:
        return ChatAnswerJudge()
    raise ValueError(f"未知 answer judge: {name}")


def _key_term_coverage(gold_answer: str, generated_answer: str) -> float:
    terms = [term for term in tokenize(gold_answer) if len(term) > 2]
    if len(terms) < 2:
        return 0.0
    generated_terms = set(tokenize(generated_answer))
    matched = sum(1 for term in terms if term in generated_terms)
    return matched / len(terms)


def _normalize(text: str) -> str:
    return " ".join(tokenize(text))


def _strip_json_fence(content: str) -> str:
    clean = content.strip()
    if clean.startswith("```"):
        clean = clean.removeprefix("```json").removeprefix("```").strip()
        clean = clean.removesuffix("```").strip()
    return clean
