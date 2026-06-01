from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sam.llm import ChatClient
from sam.models import RetrievalHit
from sam.text import extract_keywords


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
    analogy_hint_builder: "CaseAnalogyHintBuilder | None" = None,
    analogy_top_k: int = 2,
) -> list[GeneratedAnswer]:
    selected_cases = cases[:limit] if limit is not None else cases
    return [
        generator.generate_for_case(
            case,
            method=method,
            analogy_hints=(
                analogy_hint_builder.hints_for(case, top_k=analogy_top_k)
                if analogy_hint_builder
                else None
            ),
        )
        for case in selected_cases
    ]


class CaseAnalogyHintBuilder:
    """从 cases.json 中检索相似历史案例，为生成阶段提供类比提示。"""

    def __init__(self, cases: list[dict[str, object]], method: str) -> None:
        self.cases = cases
        self.method = method

    def hints_for(self, case: dict[str, object], top_k: int = 2) -> list[str]:
        current_query_id = str(case.get("query_id", ""))
        current_keywords = _case_keywords(case)
        current_relations = _case_relation_types(case, self.method)
        scored: list[tuple[float, dict[str, object], list[str]]] = []
        for candidate in self.cases:
            if str(candidate.get("query_id", "")) == current_query_id:
                continue
            score, shared_relations = self._score_candidate(
                current_keywords=current_keywords,
                current_relations=current_relations,
                candidate=candidate,
            )
            if score > 0.0:
                scored.append((score, candidate, shared_relations))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            self._format_hint(candidate, score, shared_relations)
            for score, candidate, shared_relations in scored[:top_k]
        ]

    def _score_candidate(
        self,
        *,
        current_keywords: set[str],
        current_relations: list[str],
        candidate: dict[str, object],
    ) -> tuple[float, list[str]]:
        candidate_keywords = _case_keywords(candidate)
        keyword_score = _overlap_ratio(current_keywords, candidate_keywords)
        candidate_relations = _case_relation_types(candidate, self.method)
        shared_relations = [
            relation for relation in current_relations
            if relation in candidate_relations
        ]
        relation_score = len(shared_relations) / max(1, len(current_relations))
        support_hits = _support_hits(candidate, self.method)
        answer_status = _answer_status(candidate, self.method)
        success_score = 1.0 if support_hits > 0 or answer_status in {
            "found_in_retrieved_context",
            "matched_option",
        } else 0.0
        score = 0.54 * keyword_score + 0.32 * relation_score + 0.14 * success_score
        return score, shared_relations

    def _format_hint(
        self,
        candidate: dict[str, object],
        score: float,
        shared_relations: list[str],
    ) -> str:
        query_id = str(candidate.get("query_id", ""))
        question = str(candidate.get("question", ""))
        support_hits = _support_hits(candidate, self.method)
        relation_text = " -> ".join(shared_relations) if shared_relations else "无显式共享路径"
        return (
            f"历史案例 {query_id} 与当前问题相似，问题为：{question}。"
            f"该案例中 {self.method} 命中支持证据 {support_hits} 个，"
            f"共享关系路径：{relation_text}，类比分数={score:.3f}。"
        )


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


def _case_keywords(case: dict[str, object]) -> set[str]:
    text_parts = [
        str(case.get("question", "")),
        str(case.get("answer", "")),
    ]
    methods = case.get("methods", {})
    if isinstance(methods, dict):
        for hits in methods.values():
            if not isinstance(hits, list):
                continue
            for hit in hits:
                if isinstance(hit, dict):
                    text_parts.append(str(hit.get("title", "")))
                    text_parts.append(str(hit.get("reason", "")))
    return set(extract_keywords(" ".join(text_parts), limit=24))


def _case_relation_types(case: dict[str, object], method: str) -> list[str]:
    methods = case.get("methods", {})
    if not isinstance(methods, dict):
        return []
    hits = methods.get(method, [])
    if not isinstance(hits, list):
        return []
    relation_types: list[str] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        for candidate_path in hit.get("candidate_paths", []):
            if not isinstance(candidate_path, dict):
                continue
            relation_type = candidate_path.get("relation_type")
            if relation_type and relation_type not in relation_types:
                relation_types.append(str(relation_type))
        reason = str(hit.get("reason", ""))
        for relation_type in [
            "shared_entity",
            "keyword_overlap",
            "embedding_similarity",
            "context_cooccurrence",
            "summary_parent",
            "summary_child",
        ]:
            if relation_type in reason and relation_type not in relation_types:
                relation_types.append(relation_type)
    return relation_types


def _support_hits(case: dict[str, object], method: str) -> int:
    support_hits_by_method = case.get("support_hits_by_method", {})
    if isinstance(support_hits_by_method, dict):
        try:
            return int(support_hits_by_method.get(method, 0))
        except (TypeError, ValueError):
            return 0
    return 0


def _answer_status(case: dict[str, object], method: str) -> str:
    final_answers = case.get("final_answers", {})
    if not isinstance(final_answers, dict):
        return ""
    payload = final_answers.get(method, {})
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("status", ""))


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


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
