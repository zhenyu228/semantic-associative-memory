from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sam.badcase import BadCaseAnalyzer, GenerationBadCaseAnalyzer


def audit_run_directory(
    run_dir: str | Path,
    *,
    primary_method: str = "sam_full",
    baseline_method: str = "embedding_topk",
) -> dict[str, object]:
    """审计一次实验 run，输出瓶颈、bad case 类型和下一步动作。"""

    target = Path(run_dir)
    metrics = _read_json(target / "metrics.json", default={})
    cases = _read_json(target / "cases.json", default=[])
    generated_answers = _read_json(target / "generated_answers.json", default=[])
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(cases, list):
        cases = []
    if not isinstance(generated_answers, list):
        generated_answers = []

    method_metrics = metrics.get("method_metrics", {})
    if not isinstance(method_metrics, dict):
        method_metrics = {}
    baseline = _method_metric(method_metrics, baseline_method)
    primary = _method_metric(method_metrics, primary_method)
    retrieval_bad_cases = BadCaseAnalyzer().analyze(cases, method=primary_method)
    generation_bad_cases = GenerationBadCaseAnalyzer().analyze(generated_answers)
    retrieval_categories = _category_counts([case.categories for case in retrieval_bad_cases])
    generation_categories = _category_counts([case.categories for case in generation_bad_cases])
    generation_hit_rate = _generation_hit_rate(generated_answers)
    bottlenecks = _bottlenecks(
        baseline=baseline,
        primary=primary,
        retrieval_categories=retrieval_categories,
        generation_categories=generation_categories,
        generation_hit_rate=generation_hit_rate,
        generated_answer_count=len(generated_answers),
    )
    return {
        "run_dir": str(target),
        "primary_method": primary_method,
        "baseline_method": baseline_method,
        "key_metrics": {
            baseline_method: baseline,
            primary_method: primary,
            "generation_answer_hit_rate": generation_hit_rate,
        },
        "bad_case_summary": {
            "retrieval_bad_case_count": len(retrieval_bad_cases),
            "generation_bad_case_count": len(generation_bad_cases),
            "retrieval_categories": retrieval_categories,
            "generation_categories": generation_categories,
        },
        "bottlenecks": bottlenecks,
        "next_actions": _next_actions(bottlenecks),
    }


def write_experiment_audit(
    audit: dict[str, object],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """写出实验审计 JSON 和 Markdown。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "experiment_audit.json"
    markdown_path = target / "experiment_audit.md"
    json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(_audit_markdown(audit), encoding="utf-8")
    return json_path, markdown_path


def _read_json(path: Path, *, default: object) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _method_metric(method_metrics: dict[str, object], method: str) -> dict[str, float]:
    metric = method_metrics.get(method, {})
    if not isinstance(metric, dict):
        metric = {}
    return {
        "evidence_recall": float(metric.get("evidence_recall", 0.0) or 0.0),
        "answer_hit_rate": float(metric.get("answer_hit_rate", 0.0) or 0.0),
        "support_hits": float(metric.get("support_hits", 0.0) or 0.0),
    }


def _category_counts(category_lists: list[list[str]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for categories in category_lists:
        counter.update(categories)
    return dict(counter)


def _generation_hit_rate(generated_answers: list[object]) -> float:
    if not generated_answers:
        return 0.0
    hit_count = sum(
        1
        for answer in generated_answers
        if isinstance(answer, dict) and bool(answer.get("answer_hit"))
    )
    return hit_count / len(generated_answers)


def _bottlenecks(
    *,
    baseline: dict[str, float],
    primary: dict[str, float],
    retrieval_categories: dict[str, int],
    generation_categories: dict[str, int],
    generation_hit_rate: float,
    generated_answer_count: int,
) -> list[dict[str, object]]:
    bottlenecks: list[dict[str, object]] = []
    recall_gain = primary["evidence_recall"] - baseline["evidence_recall"]
    answer_gain = primary["answer_hit_rate"] - baseline["answer_hit_rate"]
    if recall_gain < -0.01:
        bottlenecks.append(
            _bottleneck(
                "graph_regression",
                "high",
                f"SAM 证据召回率比 baseline 低 {abs(recall_gain):.3f}。",
                "降低噪声图路径权重，增加向量锚点保底，并审计 worse_than_vector 样本。",
            )
        )
    elif recall_gain <= 0.01:
        bottlenecks.append(
            _bottleneck(
                "weak_graph_gain",
                "medium",
                f"SAM 证据召回率相对 baseline 仅提升 {recall_gain:.3f}。",
                "优先改进按需建边质量、二跳路径候选和关系判别，而不是继续调生成。",
            )
        )
    if answer_gain <= 0.01 and primary["answer_hit_rate"] < 0.6:
        bottlenecks.append(
            _bottleneck(
                "retrieval_answer_gap",
                "medium",
                f"SAM 检索答案命中率为 {primary['answer_hit_rate']:.3f}。",
                "检查缺失答案的支持证据是否未召回，必要时提高 top-k 或优化 QueryPlanner。",
            )
        )
    if retrieval_categories.get("graph_noise", 0) > 0:
        bottlenecks.append(
            _bottleneck(
                "graph_noise",
                "high",
                f"{retrieval_categories['graph_noise']} 个检索 bad case 包含图噪声。",
                "启用 GPT-5.4 RelationJudge 或降低弱关键词边、context_cooccurrence 边权重。",
            )
        )
    if retrieval_categories.get("missing_support_evidence", 0) > 0:
        bottlenecks.append(
            _bottleneck(
                "missing_support_evidence",
                "high",
                f"{retrieval_categories['missing_support_evidence']} 个检索 bad case 缺失支持证据。",
                "优先提高 embedding 质量、查询规划和跨文档桥接边覆盖率。",
            )
        )
    if generated_answer_count and generation_hit_rate < 0.2:
        bottlenecks.append(
            _bottleneck(
                "generation_failure",
                "high",
                f"生成答案命中率为 {generation_hit_rate:.3f}。",
                "使用 GPT-5.4 生成与 AnswerJudge 复核，并检查生成上下文是否包含完整证据链。",
            )
        )
    if generation_categories.get("insufficient_evidence_answer", 0) > 0:
        bottlenecks.append(
            _bottleneck(
                "insufficient_evidence_generation",
                "medium",
                f"{generation_categories['insufficient_evidence_answer']} 个生成 bad case 输出证据不足。",
                "区分检索证据缺失和生成器过度保守；若证据已在上下文中，优化生成 prompt。",
            )
        )
    return bottlenecks


def _bottleneck(
    bottleneck_type: str,
    severity: str,
    evidence: str,
    recommendation: str,
) -> dict[str, object]:
    return {
        "type": bottleneck_type,
        "severity": severity,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _next_actions(bottlenecks: list[dict[str, object]]) -> list[str]:
    priority = [
        "graph_regression",
        "missing_support_evidence",
        "graph_noise",
        "weak_graph_gain",
        "generation_failure",
        "retrieval_answer_gap",
        "insufficient_evidence_generation",
    ]
    actions: list[str] = []
    seen: set[str] = set()
    for bottleneck_type in priority:
        for item in bottlenecks:
            if item.get("type") != bottleneck_type or bottleneck_type in seen:
                continue
            actions.append(str(item["recommendation"]))
            seen.add(bottleneck_type)
    return actions[:5]


def _audit_markdown(audit: dict[str, object]) -> str:
    bad_case_summary = audit.get("bad_case_summary", {})
    key_metrics = audit.get("key_metrics", {})
    bottlenecks = audit.get("bottlenecks", [])
    next_actions = audit.get("next_actions", [])
    lines = [
        "# 实验审计报告",
        "",
        f"- Primary 方法：{audit.get('primary_method')}",
        f"- Baseline 方法：{audit.get('baseline_method')}",
        "",
        "## 关键指标",
        "",
    ]
    if isinstance(key_metrics, dict):
        for name, metric in key_metrics.items():
            lines.append(f"- {name}: {json.dumps(metric, ensure_ascii=False)}")
    lines.extend(["", "## Bad Case 摘要", ""])
    if isinstance(bad_case_summary, dict):
        lines.append(f"- 检索 bad case 数：{bad_case_summary.get('retrieval_bad_case_count', 0)}")
        lines.append(f"- 生成 bad case 数：{bad_case_summary.get('generation_bad_case_count', 0)}")
        lines.append(f"- 检索类型：{json.dumps(bad_case_summary.get('retrieval_categories', {}), ensure_ascii=False)}")
        lines.append(f"- 生成类型：{json.dumps(bad_case_summary.get('generation_categories', {}), ensure_ascii=False)}")
    lines.extend(["", "## 瓶颈判断", ""])
    if isinstance(bottlenecks, list) and bottlenecks:
        for item in bottlenecks:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- [{item.get('severity')}] {item.get('type')}: "
                f"{item.get('evidence')} 建议：{item.get('recommendation')}"
            )
    else:
        lines.append("- 暂未识别到明确瓶颈。")
    lines.extend(["", "## 下一步动作", ""])
    if isinstance(next_actions, list) and next_actions:
        for index, action in enumerate(next_actions, start=1):
            lines.append(f"{index}. {action}")
    else:
        lines.append("1. 保留当前配置，扩大样本继续验证。")
    return "\n".join(lines) + "\n"
