from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class BadCase:
    """实验失败样本分析结果。"""

    query_id: str
    question: str
    gold_answer: str
    method: str
    categories: list[str]
    diagnosis: str
    recommendation: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "method": self.method,
            "categories": self.categories,
            "diagnosis": self.diagnosis,
            "recommendation": self.recommendation,
            "metadata": self.metadata,
        }


class BadCaseAnalyzer:
    """根据 cases.json 分析 SAM 当前失败原因。"""

    def analyze(
        self,
        cases: list[dict[str, object]],
        *,
        method: str = "sam_full",
    ) -> list[BadCase]:
        bad_cases: list[BadCase] = []
        for case in cases:
            resolved_method = _resolve_method(case, method)
            if not resolved_method:
                continue
            categories = _case_categories(case, resolved_method)
            if not categories:
                continue
            bad_cases.append(
                BadCase(
                    query_id=str(case.get("query_id", "")),
                    question=str(case.get("question", "")),
                    gold_answer=str(case.get("answer", "")),
                    method=resolved_method,
                    categories=categories,
                    diagnosis=_diagnosis(categories),
                    recommendation=_recommendation(categories),
                    metadata={
                        "supporting_doc_count": len(case.get("supporting_doc_ids", [])),
                        "method_support_hits": _support_hits(case, resolved_method),
                        "vector_support_hits": case.get("vector_support_hits", 0),
                    },
                )
            )
        bad_cases.sort(key=lambda item: (len(item.categories), item.query_id), reverse=True)
        return bad_cases


class GenerationBadCaseAnalyzer:
    """根据 generated_answers.json 分析生成阶段失败原因。"""

    def analyze(self, answers: list[dict[str, object]]) -> list[BadCase]:
        bad_cases: list[BadCase] = []
        for answer in answers:
            categories = _generation_categories(answer)
            if not categories:
                continue
            judgment = _answer_judgment(answer)
            bad_cases.append(
                BadCase(
                    query_id=str(answer.get("query_id", "")),
                    question=str(answer.get("question", "")),
                    gold_answer=str(answer.get("gold_answer", "")),
                    method=str(answer.get("method", "")),
                    categories=categories,
                    diagnosis=_generation_diagnosis(categories),
                    recommendation=_generation_recommendation(categories),
                    metadata={
                        "generated_answer": answer.get("generated_answer", ""),
                        "context_titles": answer.get("context_titles", []),
                        "answer_judgment": judgment,
                        "context_answer_judgment": _context_answer_judgment(answer),
                    },
                )
            )
        bad_cases.sort(key=lambda item: (len(item.categories), item.query_id), reverse=True)
        return bad_cases


def write_bad_case_reports(
    bad_cases: list[BadCase],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "bad_cases.json"
    markdown_path = target / "bad_cases.md"
    json_path.write_text(
        json.dumps([case.to_dict() for case in bad_cases], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Bad Case 分析",
        "",
        f"- 失败样本数：{len(bad_cases)}",
        "",
        "| Query | 类型 | 诊断 | 改进建议 |",
        "| --- | --- | --- | --- |",
    ]
    for case in bad_cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    case.query_id,
                    ", ".join(case.categories),
                    case.diagnosis.replace("|", "/"),
                    case.recommendation.replace("|", "/"),
                ]
            )
            + " |"
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def write_generation_bad_case_reports(
    bad_cases: list[BadCase],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "generation_bad_cases.json"
    markdown_path = target / "generation_bad_cases.md"
    json_path.write_text(
        json.dumps([case.to_dict() for case in bad_cases], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 生成 Bad Case 分析",
        "",
        f"- 失败样本数：{len(bad_cases)}",
        "",
        "| Query | 类型 | 判别状态 | 诊断 | 改进建议 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in bad_cases:
        judgment = case.metadata.get("answer_judgment", {})
        status = judgment.get("status", "") if isinstance(judgment, dict) else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    case.query_id,
                    ", ".join(case.categories),
                    str(status),
                    case.diagnosis.replace("|", "/"),
                    case.recommendation.replace("|", "/"),
                ]
            )
            + " |"
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def _resolve_method(case: dict[str, object], preferred: str) -> str | None:
    methods = case.get("methods", {})
    if not isinstance(methods, dict):
        return None
    if preferred in methods:
        return preferred
    if "sam" in methods:
        return "sam"
    for method in methods:
        if str(method).startswith("sam"):
            return str(method)
    return None


def _case_categories(case: dict[str, object], method: str) -> list[str]:
    categories: list[str] = []
    supporting_count = len(case.get("supporting_doc_ids", []))
    method_support_hits = _support_hits(case, method)
    vector_support_hits = int(case.get("vector_support_hits", 0) or 0)
    final_answers = case.get("final_answers", {})
    answer_status = ""
    if isinstance(final_answers, dict) and isinstance(final_answers.get(method), dict):
        answer_status = str(final_answers[method].get("status", ""))
    methods = case.get("methods", {})
    hits = methods.get(method, []) if isinstance(methods, dict) else []
    path_lengths = [
        len(hit.get("path", []))
        for hit in hits
        if isinstance(hit, dict)
    ]
    non_support_graph_hits = [
        hit for hit in hits
        if isinstance(hit, dict)
        and not hit.get("is_supporting")
        and len(hit.get("path", [])) > 1
    ]

    if supporting_count and method_support_hits < supporting_count:
        categories.append("missing_support_evidence")
    if answer_status and answer_status not in {"found_in_retrieved_context", "matched_option"}:
        categories.append("answer_not_covered")
    if path_lengths and max(path_lengths) <= 1 and supporting_count:
        categories.append("graph_not_used")
    if method_support_hits < vector_support_hits:
        categories.append("worse_than_vector")
    if non_support_graph_hits and method_support_hits < supporting_count:
        categories.append("graph_noise")
    return categories


def _support_hits(case: dict[str, object], method: str) -> int:
    support_hits_by_method = case.get("support_hits_by_method", {})
    if isinstance(support_hits_by_method, dict):
        return int(support_hits_by_method.get(method, 0) or 0)
    return 0


def _diagnosis(categories: list[str]) -> str:
    if "worse_than_vector" in categories:
        return "图扩展或重排把有效向量候选挤出 top-k。"
    if "graph_noise" in categories:
        return "图扩展引入了非支持证据，说明边质量或路径权重需要约束。"
    if "graph_not_used" in categories:
        return "结果主要停留在初始召回，联想扩展没有进入最终答案上下文。"
    if "missing_support_evidence" in categories:
        return "支持证据链没有被完整召回，当前检索上下文不足。"
    if "answer_not_covered" in categories:
        return "检索文本未覆盖标准答案，生成阶段缺少必要证据。"
    return "失败原因需要结合案例进一步人工检查。"


def _recommendation(categories: list[str]) -> str:
    if "worse_than_vector" in categories:
        return "增加保底向量候选比例，并在重排中惩罚低置信噪声路径。"
    if "graph_noise" in categories:
        return "引入 LLM 关系判断或更强实体链接，降低弱关键词边权重。"
    if "graph_not_used" in categories:
        return "提高有效边覆盖率，增加二跳路径候选和路径支持分权重。"
    if "missing_support_evidence" in categories:
        return "优化初始 embedding 召回和按需建边策略，补充跨文档桥接边。"
    if "answer_not_covered" in categories:
        return "接入生成式答案模块，并扩大可用证据上下文。"
    return "保留该样本进入人工误差分析集合。"


def _generation_categories(answer: dict[str, object]) -> list[str]:
    categories: list[str] = []
    generated_answer = str(answer.get("generated_answer", "")).strip()
    answer_hit = bool(answer.get("answer_hit", False))
    judgment = _answer_judgment(answer)
    context_judgment = _context_answer_judgment(answer)
    status = str(judgment.get("status", ""))
    score = _safe_float(judgment.get("score", 0.0))
    context_titles = answer.get("context_titles", [])
    has_context = isinstance(context_titles, list) and len(context_titles) > 0
    context_answer_hit = context_judgment.get("answer_hit")
    ungrounded_answer_hit = bool(
        isinstance(answer.get("metadata"), dict)
        and answer["metadata"].get("ungrounded_answer_hit")
    )

    if not generated_answer:
        categories.append("empty_generated_answer")
    if ungrounded_answer_hit:
        categories.append("ungrounded_generated_answer")
    elif not answer_hit:
        categories.append("generated_answer_not_equivalent")
    if judgment and score < 0.5:
        categories.append("judge_low_confidence")
    if has_context and context_answer_hit is False and not answer_hit:
        categories.append("retrieval_context_missing_answer")
    elif has_context and not answer_hit:
        categories.append("context_available_but_generation_failed")
    if status.startswith("chat_fallback"):
        categories.append("judge_fallback_used")
    return categories


def _answer_judgment(answer: dict[str, object]) -> dict[str, object]:
    metadata = answer.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    judgment = metadata.get("answer_judgment", {})
    return judgment if isinstance(judgment, dict) else {}


def _context_answer_judgment(answer: dict[str, object]) -> dict[str, object]:
    metadata = answer.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    judgment = metadata.get("context_answer_judgment", {})
    return judgment if isinstance(judgment, dict) else {}


def _generation_diagnosis(categories: list[str]) -> str:
    if "empty_generated_answer" in categories:
        return "生成阶段没有产出有效答案。"
    if "ungrounded_generated_answer" in categories:
        return "生成答案匹配标准答案，但检索上下文没有覆盖该答案，存在外部知识或幻觉风险。"
    if "retrieval_context_missing_answer" in categories:
        return "检索上下文没有覆盖标准答案，生成阶段缺少必要证据。"
    if "context_available_but_generation_failed" in categories:
        return "检索上下文已经进入生成阶段，但生成答案未通过语义等价判别。"
    if "judge_fallback_used" in categories:
        return "LLM 判别失败后回退到规则判别，当前命中结论可靠性较弱。"
    if "generated_answer_not_equivalent" in categories:
        return "生成答案与标准答案不等价或覆盖不足。"
    if "judge_low_confidence" in categories:
        return "答案判别分数较低，需要检查生成答案和 gold answer 的表述差异。"
    return "生成失败原因需要人工复查。"


def _generation_recommendation(categories: list[str]) -> str:
    if "empty_generated_answer" in categories:
        return "检查聊天模型调用和提示词，必要时降低上下文长度或增加重试。"
    if "ungrounded_generated_answer" in categories:
        return "将命中判定改为需要检索上下文支撑，并优先补足缺失证据，而不是采纳模型外部知识。"
    if "retrieval_context_missing_answer" in categories:
        return "优先改进检索召回、图扩展和路径重排，而不是只调整生成提示词。"
    if "context_available_but_generation_failed" in categories:
        return "优化生成提示词，要求模型显式引用上下文编号并覆盖标准答案关键实体。"
    if "judge_fallback_used" in categories:
        return "检查 GPT-5.4 判别器配置，并保留规则判别作为兜底。"
    if "generated_answer_not_equivalent" in categories:
        return "结合检索证据检查是上下文不足、答案抽取失败还是语义判别过严。"
    if "judge_low_confidence" in categories:
        return "使用 GPT-5.4 语义判别或人工复查边界样本。"
    return "保留该生成样本进入人工误差分析集合。"


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
