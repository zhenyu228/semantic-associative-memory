from __future__ import annotations

import json
import re
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
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
        chat_provider: str | None = None,
    ) -> None:
        provider = (
            chat_provider
            or os.environ.get("SAM_RELATION_JUDGE_CHAT_PROVIDER")
            or os.environ.get("SAM_CHAT_PROVIDER")
            or "azure_openai_sdk"
        )
        self.chat_client = chat_client or create_chat_client(provider)
        self.min_confidence = min_confidence
        self.fail_open = fail_open
        self.chat_provider = provider

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


class CachedRelationJudge:
    """为关系判别器增加本地缓存，减少重复模型调用。"""

    def __init__(
        self,
        base_judge: RelationJudge,
        *,
        cache_path: str | Path | None = None,
    ) -> None:
        self.base_judge = base_judge
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict[str, object]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        if self.cache_path and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def judge(
        self,
        seed: MemoryNode,
        other: MemoryNode,
        score_breakdown: dict[str, object],
    ) -> RelationJudgment:
        key = _relation_cache_key(seed, other, score_breakdown)
        cached = self._cache.get(key)
        if cached:
            self.cache_hits += 1
            if bool(cached.get("should_link", False)) and str(cached.get("relation_type")) == "budget_exhausted":
                judgment = _budget_exhausted_skip_judgment(score_breakdown)
                self._cache[key] = judgment.to_dict()
                self._write_cache()
                return judgment
            return RelationJudgment(
                should_link=bool(cached.get("should_link", False)),
                relation_type=str(cached.get("relation_type", "unrelated")),
                confidence=max(0.0, min(1.0, float(cached.get("confidence", 0.0)))),
                reason=str(cached.get("reason", "")),
            )
        self.cache_misses += 1
        judgment = self.base_judge.judge(seed, other, score_breakdown)
        self._cache[key] = judgment.to_dict()
        self._write_cache()
        return judgment

    def _write_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class BudgetedRelationJudge:
    """为在线关系判别增加调用预算，避免低额度实验失控。"""

    def __init__(
        self,
        base_judge: RelationJudge,
        *,
        max_calls: int | None = None,
        on_exhausted: str = "skip",
    ) -> None:
        if max_calls is not None and max_calls < 0:
            raise ValueError("max_calls 不能为负数")
        if on_exhausted not in {"skip", "reject"}:
            raise ValueError("on_exhausted 只能是 skip 或 reject")
        self.base_judge = base_judge
        self.max_calls = max_calls
        self.on_exhausted = on_exhausted
        self.calls_made = 0
        self.skipped_count = 0

    def judge(
        self,
        seed: MemoryNode,
        other: MemoryNode,
        score_breakdown: dict[str, object],
    ) -> RelationJudgment:
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            self.skipped_count += 1
            should_link = self.on_exhausted == "skip"
            if should_link:
                return _budget_exhausted_skip_judgment(score_breakdown)
            return RelationJudgment(
                should_link=False,
                relation_type="budget_exhausted",
                confidence=0.0,
                reason=(
                    f"关系判别预算已耗尽：max_calls={self.max_calls}，"
                    f"策略={self.on_exhausted}"
                ),
            )
        self.calls_made += 1
        return self.base_judge.judge(seed, other, score_breakdown)


def create_relation_judge(name: str | None = None) -> RelationJudge | None:
    provider = name or "disabled"
    if provider in {"disabled", "none", ""}:
        return None
    if provider in {"gpt", "gpt54", "azure_openai", "chat", "gpt54_sdk", "azure_openai_sdk"}:
        return _with_relation_budget(
            ChatRelationJudge(
                min_confidence=_relation_min_confidence(),
                fail_open=_relation_fail_open(),
                chat_provider=_relation_chat_provider(provider),
            )
        )
    if provider in {
        "cached_gpt",
        "cached_gpt54",
        "cached_azure_openai",
        "cached_chat",
        "cached_gpt54_sdk",
        "cached_azure_openai_sdk",
    }:
        return CachedRelationJudge(
            _with_relation_budget(
                ChatRelationJudge(
                    min_confidence=_relation_min_confidence(),
                    fail_open=_relation_fail_open(),
                    chat_provider=_relation_chat_provider(provider),
                )
            ),
            cache_path=_relation_cache_path(),
        )
    raise ValueError(f"未知关系判别器：{provider}")


def _with_relation_budget(judge: RelationJudge) -> RelationJudge:
    max_calls = _relation_max_calls()
    if max_calls is None:
        return judge
    return BudgetedRelationJudge(
        judge,
        max_calls=max_calls,
        on_exhausted=_relation_budget_exhausted_policy(),
    )


def _relation_max_calls() -> int | None:
    raw = os.environ.get("SAM_RELATION_JUDGE_MAX_CALLS", "").strip()
    if not raw:
        return None
    return int(raw)


def _relation_budget_exhausted_policy() -> str:
    return os.environ.get("SAM_RELATION_JUDGE_BUDGET_EXHAUSTED", "skip").strip().lower()


def _budget_exhausted_skip_judgment(score_breakdown: dict[str, object]) -> RelationJudgment:
    relation_type = str(score_breakdown.get("relation_type_hint") or "unknown")
    return RelationJudgment(
        should_link=True,
        relation_type=relation_type,
        confidence=1.0,
        reason=(
            "关系判别预算已耗尽，按 skip 策略保留原始候选边："
            f"relation_type={relation_type}"
        ),
    )


def relation_judge_stats(judge: RelationJudge | None) -> dict[str, object]:
    if judge is None:
        return {"enabled": False, "type": None}
    if isinstance(judge, CachedRelationJudge):
        return {
            "enabled": True,
            "type": "CachedRelationJudge",
            "cache_path": str(judge.cache_path) if judge.cache_path else None,
            "cache_size": len(judge._cache),
            "cache_hits": judge.cache_hits,
            "cache_misses": judge.cache_misses,
            "base": relation_judge_stats(judge.base_judge),
        }
    if isinstance(judge, BudgetedRelationJudge):
        return {
            "enabled": True,
            "type": "BudgetedRelationJudge",
            "max_calls": judge.max_calls,
            "on_exhausted": judge.on_exhausted,
            "calls_made": judge.calls_made,
            "skipped_count": judge.skipped_count,
            "base": relation_judge_stats(judge.base_judge),
        }
    if isinstance(judge, ChatRelationJudge):
        return {
            "enabled": True,
            "type": "ChatRelationJudge",
            "chat_provider": judge.chat_provider,
            "min_confidence": judge.min_confidence,
            "fail_open": judge.fail_open,
        }
    return {"enabled": True, "type": type(judge).__name__}


def _relation_chat_provider(provider: str) -> str | None:
    configured = os.environ.get("SAM_RELATION_JUDGE_CHAT_PROVIDER")
    if configured:
        return configured
    if provider in {"azure_openai", "cached_azure_openai"}:
        return "azure_openai"
    return "azure_openai_sdk"


def _relation_min_confidence() -> float:
    return float(os.environ.get("SAM_RELATION_JUDGE_MIN_CONFIDENCE", "0.55"))


def _relation_fail_open() -> bool:
    raw = os.environ.get("SAM_RELATION_JUDGE_FAIL_OPEN", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _relation_cache_path() -> Path:
    return Path(os.environ.get("SAM_RELATION_JUDGE_CACHE_PATH", "outputs/cache/relation_judge_cache.json"))


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


def _relation_cache_key(
    seed: MemoryNode,
    other: MemoryNode,
    score_breakdown: dict[str, object],
) -> str:
    payload = {
        "seed_id": seed.id,
        "other_id": other.id,
        "seed_text_hash": _text_hash(seed.text),
        "other_text_hash": _text_hash(other.text),
        "relation_type_hint": score_breakdown.get("relation_type_hint"),
        "keyword_overlap": score_breakdown.get("keyword_overlap", []),
        "shared_entities": score_breakdown.get("shared_entities", []),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text[:1200].encode("utf-8")).hexdigest()[:16]
