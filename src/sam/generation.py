from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sam.answer_judge import AnswerJudge, RuleBasedAnswerJudge
from sam.badcase import GenerationBadCaseAnalyzer, write_generation_bad_case_reports
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

    def __init__(
        self,
        chat_client: ChatClient,
        max_context_chars: int = 6000,
        answer_judge: AnswerJudge | None = None,
    ) -> None:
        self.chat_client = chat_client
        self.max_context_chars = max_context_chars
        self.answer_judge = answer_judge or RuleBasedAnswerJudge()

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
                    "只能依据给定上下文作答；先逐条检查上下文是否包含问题所问的实体、地点、时间、职位或数量。"
                    "如果上下文中存在答案线索，必须抽取最短、最直接的答案短语，不要因为证据分散就过早回答“证据不足”。"
                    "只有所有上下文都缺少答案线索时，才回答“证据不足”。"
                    "答案要简洁，并给出使用了哪些上下文编号。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}{gold_line}\n\n"
                    f"类比提示：\n{hints or '无'}\n\n"
                    f"上下文：\n{context_text}\n\n"
                    "请输出：最终答案：<最短答案短语>。依据：<一句话，包含上下文编号>。"
                ),
            },
        ]
        answer = self.chat_client.complete(messages, max_tokens=500)
        judgment = self.answer_judge.judge(question, gold_answer, answer)
        context_judgment = RuleBasedAnswerJudge().judge(question, gold_answer, context_text)
        grounded_answer_hit = judgment.answer_hit and context_judgment.answer_hit
        return GeneratedAnswer(
            query_id=query_id,
            method=method,
            question=question,
            gold_answer=gold_answer,
            generated_answer=answer,
            answer_hit=grounded_answer_hit,
            context_titles=[_hit_title(hit) for hit in hits],
            prompt_tokens_estimate=max(1, len(json.dumps(messages, ensure_ascii=False)) // 4),
            metadata={
                "analogy_hints": analogy_hints or [],
                "answer_judgment": judgment.to_dict(),
                "context_answer_judgment": context_judgment.to_dict(),
                "ungrounded_answer_hit": judgment.answer_hit and not context_judgment.answer_hit,
                "grounding_policy": "answer_hit requires generated answer match and retrieved context support",
            },
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


def compare_generation_variants(
    cases: list[dict[str, object]],
    *,
    all_cases: list[dict[str, object]] | None,
    generator: ContextAnswerGenerator,
    method: str,
    analogy_top_k: int = 2,
) -> dict[str, object]:
    """对比无类比提示与有类比提示的生成结果。"""

    candidate_cases = all_cases or cases
    hint_builder = CaseAnalogyHintBuilder(candidate_cases, method=method)
    baseline_answers = generate_answers_for_cases(
        cases,
        generator,
        method=method,
    )
    analogy_answers = generate_answers_for_cases(
        cases,
        generator,
        method=method,
        analogy_hint_builder=hint_builder,
        analogy_top_k=analogy_top_k,
    )
    case_deltas = [
        _generation_case_delta(baseline, with_analogy)
        for baseline, with_analogy in zip(baseline_answers, analogy_answers, strict=True)
    ]
    baseline_metrics = _generation_metrics(baseline_answers)
    analogy_metrics = _generation_metrics(analogy_answers)
    return {
        "method": method,
        "query_count": len(cases),
        "variants": {
            "baseline": baseline_metrics,
            "with_analogy": analogy_metrics,
        },
        "delta": {
            "answer_hit_count": (
                int(analogy_metrics["answer_hit_count"])
                - int(baseline_metrics["answer_hit_count"])
            ),
            "answer_hit_rate": (
                float(analogy_metrics["answer_hit_rate"])
                - float(baseline_metrics["answer_hit_rate"])
            ),
        },
        "case_deltas": case_deltas,
        "answers": {
            "baseline": [answer.to_dict() for answer in baseline_answers],
            "with_analogy": [answer.to_dict() for answer in analogy_answers],
        },
    }


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
    generation_bad_cases = GenerationBadCaseAnalyzer().analyze(
        [answer.to_dict() for answer in answers]
    )
    write_generation_bad_case_reports(generation_bad_cases, target)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def write_generation_comparison_reports(
    comparison: dict[str, object],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "generation_comparison.json"
    markdown_path = target / "generation_comparison.md"
    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    variants = comparison.get("variants", {})
    baseline = variants.get("baseline", {}) if isinstance(variants, dict) else {}
    with_analogy = variants.get("with_analogy", {}) if isinstance(variants, dict) else {}
    delta = comparison.get("delta", {})
    case_deltas = comparison.get("case_deltas", [])
    lines = [
        "# 类比提示生成对照实验",
        "",
        f"- 方法：{comparison.get('method', '')}",
        f"- 样本数：{comparison.get('query_count', 0)}",
        "",
        "| 变体 | 答案命中数 | 答案命中率 | 平均提示数 |",
        "| --- | ---: | ---: | ---: |",
        _comparison_metric_row("无类比提示", baseline),
        _comparison_metric_row("有类比提示", with_analogy),
        "",
        f"- 答案命中率变化：{float(delta.get('answer_hit_rate', 0.0)):.3f}" if isinstance(delta, dict) else "- 答案命中率变化：0.000",
        "",
        "| Query | 状态 | 无类比命中 | 有类比命中 | 类比提示数 |",
        "| --- | --- | --- | --- | ---: |",
    ]
    if isinstance(case_deltas, list):
        for item in case_deltas:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("query_id", "")),
                        str(item.get("status", "")),
                        "是" if item.get("baseline_hit") else "否",
                        "是" if item.get("with_analogy_hit") else "否",
                        str(item.get("analogy_hint_count", 0)),
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


def _generation_metrics(answers: list[GeneratedAnswer]) -> dict[str, object]:
    hit_count = sum(1 for answer in answers if answer.answer_hit)
    hint_counts = [
        len(answer.metadata.get("analogy_hints", []))
        for answer in answers
        if isinstance(answer.metadata.get("analogy_hints", []), list)
    ]
    return {
        "query_count": len(answers),
        "answer_hit_count": hit_count,
        "answer_hit_rate": hit_count / len(answers) if answers else 0.0,
        "average_analogy_hint_count": (
            sum(hint_counts) / len(hint_counts)
            if hint_counts
            else 0.0
        ),
    }


def _generation_case_delta(
    baseline: GeneratedAnswer,
    with_analogy: GeneratedAnswer,
) -> dict[str, object]:
    if not baseline.answer_hit and with_analogy.answer_hit:
        status = "improved"
    elif baseline.answer_hit and not with_analogy.answer_hit:
        status = "regressed"
    else:
        status = "unchanged"
    analogy_hints = with_analogy.metadata.get("analogy_hints", [])
    return {
        "query_id": baseline.query_id,
        "question": baseline.question,
        "gold_answer": baseline.gold_answer,
        "baseline_hit": baseline.answer_hit,
        "with_analogy_hit": with_analogy.answer_hit,
        "status": status,
        "analogy_hint_count": len(analogy_hints) if isinstance(analogy_hints, list) else 0,
    }


def _comparison_metric_row(label: str, metrics: object) -> str:
    if not isinstance(metrics, dict):
        metrics = {}
    return (
        f"| {label} | "
        f"{int(metrics.get('answer_hit_count', 0))} | "
        f"{float(metrics.get('answer_hit_rate', 0.0)):.3f} | "
        f"{float(metrics.get('average_analogy_hint_count', 0.0)):.2f} |"
    )


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
