from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sam.llm import ChatClient
from sam.models import RetrievalHit


@dataclass(slots=True)
class GeneratedAnswer:
    """基于检索上下文生成的答案。"""

    query_id: str
    method: str
    question: str
    gold_answer: str
    generated_answer: str
    answer_hit: bool
    context_titles: list[str]
    prompt_tokens_estimate: int
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "method": self.method,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "generated_answer": self.generated_answer,
            "answer_hit": self.answer_hit,
            "context_titles": self.context_titles,
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
            "metadata": self.metadata,
        }


class ContextAnswerGenerator:
    """把检索结果转成 LLM 可用上下文并生成最终答案。"""

    def __init__(self, chat_client: ChatClient, max_context_chars: int = 6000) -> None:
        self.chat_client = chat_client
        self.max_context_chars = max_context_chars

    def generate_from_hits(
        self,
        *,
        query_id: str,
        method: str,
        question: str,
        gold_answer: str,
        hits: list[RetrievalHit] | list[dict[str, object]],
        analogy_hints: list[str] | None = None,
        include_gold_in_prompt: bool = False,
    ) -> GeneratedAnswer:
        contexts = [_hit_to_context(hit, index + 1) for index, hit in enumerate(hits)]
        context_text = "\n\n".join(contexts)
        context_text = context_text[: self.max_context_chars]
        hints = "\n".join(f"- {hint}" for hint in (analogy_hints or []))
        gold_line = f"\n标准答案：{gold_answer}" if include_gold_in_prompt else ""
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": (
                    "你是一个严格基于检索证据回答问题的研究助手。"
                    "只能依据给定上下文作答；如果证据不足，回答“证据不足”。"
                    "答案要简洁，并给出使用了哪些上下文编号。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}{gold_line}\n\n"
                    f"类比提示：\n{hints or '无'}\n\n"
                    f"上下文：\n{context_text}\n\n"
                    "请输出最终答案和一句依据。"
                ),
            },
        ]
        answer = self.chat_client.complete(messages, max_tokens=500)
        return GeneratedAnswer(
            query_id=query_id,
            method=method,
            question=question,
            gold_answer=gold_answer,
            generated_answer=answer,
            answer_hit=_answer_matches(gold_answer, answer),
            context_titles=[_hit_title(hit) for hit in hits],
            prompt_tokens_estimate=max(1, len(json.dumps(messages, ensure_ascii=False)) // 4),
            metadata={"analogy_hints": analogy_hints or []},
        )

    def generate_for_case(
        self,
        case: dict[str, object],
        *,
        method: str,
        analogy_hints: list[str] | None = None,
    ) -> GeneratedAnswer:
        methods = case.get("methods", {})
        if not isinstance(methods, dict) or method not in methods:
            raise ValueError(f"case {case.get('query_id')} 不包含方法 {method}")
        hits = methods[method]
        if not isinstance(hits, list):
            raise ValueError(f"case {case.get('query_id')} 的 {method} 结果格式不正确")
        return self.generate_from_hits(
            query_id=str(case.get("query_id", "")),
            method=method,
            question=str(case.get("question", "")),
            gold_answer=str(case.get("answer", "")),
            hits=hits,
            analogy_hints=analogy_hints,
        )


def generate_answers_for_cases(
    cases: list[dict[str, object]],
    generator: ContextAnswerGenerator,
    *,
    method: str,
    limit: int | None = None,
) -> list[GeneratedAnswer]:
    selected_cases = cases[:limit] if limit is not None else cases
    return [
        generator.generate_for_case(case, method=method)
        for case in selected_cases
    ]


def write_generation_reports(
    answers: list[GeneratedAnswer],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "generated_answers.json"
    markdown_path = target / "generated_answers.md"
    json_path.write_text(
        json.dumps([answer.to_dict() for answer in answers], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    hit_count = sum(1 for answer in answers if answer.answer_hit)
    lines = [
        "# 生成式答案评测",
        "",
        f"- 样本数：{len(answers)}",
        f"- 答案命中数：{hit_count}",
        f"- 答案命中率：{hit_count / len(answers):.3f}" if answers else "- 答案命中率：N/A",
        "",
        "| Query | 方法 | 命中 | 标准答案 | 生成答案 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for answer in answers:
        lines.append(
            "| "
            + " | ".join(
                [
                    answer.query_id,
                    answer.method,
                    "是" if answer.answer_hit else "否",
                    answer.gold_answer.replace("|", "/"),
                    answer.generated_answer.replace("\n", " ").replace("|", "/")[:240],
                ]
            )
            + " |"
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def _hit_to_context(hit: RetrievalHit | dict[str, object], index: int) -> str:
    if isinstance(hit, RetrievalHit):
        title = str(hit.node.metadata.get("title") or hit.node.summary or hit.node.id)
        text = hit.node.text
        reason = hit.reason
    else:
        title = str(hit.get("title") or hit.get("node_id") or "")
        text = str(hit.get("text") or "")
        reason = str(hit.get("reason") or "")
    return f"[{index}] {title}\n{text}\n检索依据：{reason}"


def _hit_title(hit: RetrievalHit | dict[str, object]) -> str:
    if isinstance(hit, RetrievalHit):
        return str(hit.node.metadata.get("title") or hit.node.id)
    return str(hit.get("title") or hit.get("node_id") or "")


def _answer_matches(gold_answer: str, generated_answer: str) -> bool:
    gold = _normalize_answer(gold_answer)
    generated = _normalize_answer(generated_answer)
    if not gold:
        return False
    return gold in generated or generated in gold


def _normalize_answer(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", lowered)
    return " ".join(lowered.split())
