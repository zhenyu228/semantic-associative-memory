from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_STRATEGIES = [
    "no_graph",
    "semantic_only",
    "position_only",
    "cam_style",
    "context_path_only",
    "sam_context",
]

REQUIRED_METRICS = [
    "query_count",
    "supporting_evidence_count",
    "support_hits",
    "evidence_recall",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "graph_path_support_hits",
    "graph_path_evidence_recall",
    "graph_rescue_rate",
    "average_path_length",
    "average_expanded_node_count",
]

REQUIRED_COST_FIELDS = [
    "pair_scope",
    "candidate_pair_count",
    "theoretical_full_pair_count",
    "candidate_pair_coverage",
    "edge_count",
    "average_edges_per_node",
    "average_edge_score",
    "edge_keep_rate",
    "build_pairs_per_second",
    "build_time_seconds",
    "retrieval_time_seconds",
    "total_time_seconds",
    "average_retrieval_time_ms",
    "uses_llm",
]

REQUIRED_COST_EFFECTIVENESS_FIELDS = [
    "recall_per_100_edges",
    "recall_per_second",
    "normalized_edge_cost",
    "normalized_candidate_pair_cost",
    "normalized_build_time_cost",
    "cost_index",
    "cost_effectiveness_score",
    "balanced_score",
    "recall_gain_vs_no_graph",
    "gain_per_100_extra_edges",
    "gain_per_extra_second",
]


def audit_graph_strategy_report(
    report: dict[str, Any],
    *,
    expected_pair_scope: str | None = None,
    require_real_embedding: bool = False,
) -> dict[str, Any]:
    """审计建图策略实验报告是否具备答辩可检查的证据字段。"""

    checks: list[dict[str, Any]] = []
    _check_dataset(report, checks)
    _check_embedding(report, checks, require_real_embedding=require_real_embedding)
    _check_context_path(report, checks)
    _check_config(report, checks, expected_pair_scope=expected_pair_scope)
    _check_summary(report, checks)
    _check_strategies(report, checks, expected_pair_scope=expected_pair_scope)
    failed = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warn"]
    return {
        "passed": not failed,
        "summary": {
            "total_checks": len(checks),
            "failed_checks": len(failed),
            "warning_checks": len(warnings),
        },
        "checks": checks,
    }


def load_and_audit_graph_strategy_report(
    report_path: str | Path,
    *,
    expected_pair_scope: str | None = None,
    require_real_embedding: bool = False,
) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    audit = audit_graph_strategy_report(
        report,
        expected_pair_scope=expected_pair_scope,
        require_real_embedding=require_real_embedding,
    )
    audit["report_path"] = str(report_path)
    return audit


def write_graph_strategy_audit(audit: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "graph_strategy_audit.json"
    md_path = target / "graph_strategy_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_audit_markdown(audit), encoding="utf-8")
    return json_path, md_path


def _check_dataset(report: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    dataset = report.get("dataset")
    if not isinstance(dataset, dict):
        _add_check(checks, "dataset_present", "fail", "缺少 dataset 摘要")
        return
    required_positive = ["document_count", "query_count", "supporting_evidence_count"]
    missing = [field for field in required_positive if field not in dataset]
    non_positive = [
        field
        for field in required_positive
        if field in dataset and _as_float(dataset.get(field)) <= 0.0
    ]
    if missing or non_positive:
        _add_check(
            checks,
            "dataset_complete",
            "fail",
            f"dataset 摘要不完整，缺失={missing}，非正数={non_positive}",
        )
        return
    _add_check(checks, "dataset_complete", "pass", "dataset 摘要包含文档数、query 数和 gold evidence 数")


def _check_embedding(
    report: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    require_real_embedding: bool,
) -> None:
    embedding = report.get("embedding")
    if not isinstance(embedding, dict):
        _add_check(checks, "embedding_present", "fail", "缺少 embedding 摘要")
        return
    provider = str(embedding.get("provider") or "")
    if not provider:
        _add_check(checks, "embedding_provider_present", "fail", "缺少 embedding provider")
    elif require_real_embedding and provider == "local_hash":
        _add_check(checks, "embedding_provider_real", "fail", "正式实验不能使用 local_hash embedding")
    else:
        _add_check(checks, "embedding_provider_real", "pass", f"embedding provider={provider}")


def _check_context_path(report: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    context_path = report.get("context_path")
    if not isinstance(context_path, dict):
        _add_check(checks, "context_path_present", "fail", "缺少 context path 审计")
        return
    leaking_count = int(_as_float(context_path.get("context_paths_containing_query_ids")))
    if not context_path.get("is_leak_safe") or leaking_count != 0:
        _add_check(
            checks,
            "context_path_no_leakage",
            "fail",
            f"context_path 存在评测字段泄漏，泄漏路径数={leaking_count}",
        )
        return
    _add_check(checks, "context_path_no_leakage", "pass", "context_path 泄漏审计通过")


def _check_config(
    report: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    expected_pair_scope: str | None,
) -> None:
    config = report.get("config")
    if not isinstance(config, dict):
        _add_check(checks, "config_present", "fail", "缺少 config")
        return
    pair_scope = str(config.get("pair_scope") or "")
    if expected_pair_scope and pair_scope != expected_pair_scope:
        _add_check(
            checks,
            "config_pair_scope",
            "fail",
            f"pair_scope={pair_scope}，期望={expected_pair_scope}",
        )
    elif not pair_scope:
        _add_check(checks, "config_pair_scope", "fail", "缺少 pair_scope")
    else:
        _add_check(checks, "config_pair_scope", "pass", f"pair_scope={pair_scope}")


def _check_summary(report: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    summary = report.get("summary")
    strategies = report.get("strategies")
    if not isinstance(summary, dict):
        _add_check(checks, "summary_present", "fail", "缺少 summary")
        return
    recommended = str(summary.get("recommended_strategy") or "")
    if not recommended:
        _add_check(checks, "summary_recommendation", "fail", "缺少 recommended_strategy")
        return
    if (
        isinstance(strategies, dict)
        and recommended != "no_improving_graph_strategy"
        and recommended not in strategies
    ):
        _add_check(checks, "summary_recommendation", "fail", f"推荐策略不存在：{recommended}")
        return
    _add_check(checks, "summary_recommendation", "pass", f"推荐策略={recommended}")


def _check_strategies(
    report: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    expected_pair_scope: str | None,
) -> None:
    strategies = report.get("strategies")
    if not isinstance(strategies, dict) or not strategies:
        _add_check(checks, "strategies_present", "fail", "缺少 strategies")
        return
    missing_strategies = [strategy for strategy in REQUIRED_STRATEGIES if strategy not in strategies]
    if missing_strategies:
        _add_check(
            checks,
            "strategies_complete",
            "fail",
            f"缺少建图策略：{missing_strategies}",
        )
    else:
        _add_check(checks, "strategies_complete", "pass", "包含 no_graph 和全部建图策略")

    for strategy, payload in strategies.items():
        if not isinstance(payload, dict):
            _add_check(checks, f"strategy_{strategy}_payload", "fail", "策略 payload 不是对象")
            continue
        _check_strategy_metrics(strategy, payload, checks)
        _check_strategy_cost(strategy, payload, checks, expected_pair_scope=expected_pair_scope)
        _check_strategy_cost_effectiveness(strategy, payload, checks)


def _check_strategy_metrics(
    strategy: str,
    payload: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        _add_check(checks, f"strategy_{strategy}_metrics_complete", "fail", "缺少 metrics")
        return
    missing = [field for field in REQUIRED_METRICS if field not in metrics]
    out_of_range = [
        field
        for field in ["evidence_recall", "precision_at_k", "mrr", "ndcg_at_k"]
        if field in metrics and not 0.0 <= _as_float(metrics.get(field)) <= 1.0
    ]
    if missing or out_of_range:
        _add_check(
            checks,
            f"strategy_{strategy}_metrics_complete",
            "fail",
            f"metrics 不完整或越界，缺失={missing}，越界={out_of_range}",
        )
        return
    _add_check(checks, f"strategy_{strategy}_metrics_complete", "pass", "效果指标完整")


def _check_strategy_cost(
    strategy: str,
    payload: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    expected_pair_scope: str | None,
) -> None:
    cost = payload.get("cost")
    if not isinstance(cost, dict):
        _add_check(checks, f"strategy_{strategy}_cost_complete", "fail", "缺少 cost")
        return
    missing = [field for field in REQUIRED_COST_FIELDS if field not in cost]
    if missing:
        _add_check(checks, f"strategy_{strategy}_cost_complete", "fail", f"cost 缺失字段：{missing}")
        return
    if expected_pair_scope and str(cost.get("pair_scope")) != expected_pair_scope:
        _add_check(
            checks,
            f"strategy_{strategy}_cost_complete",
            "fail",
            f"pair_scope={cost.get('pair_scope')}，期望={expected_pair_scope}",
        )
        return
    if strategy != "no_graph" and _as_float(cost.get("candidate_pair_count")) <= 0.0:
        _add_check(checks, f"strategy_{strategy}_cost_complete", "fail", "图策略候选对数不能为 0")
        return
    if not 0.0 <= _as_float(cost.get("candidate_pair_coverage")) <= 1.0:
        _add_check(checks, f"strategy_{strategy}_cost_complete", "fail", "候选覆盖率不在 [0,1]")
        return
    _add_check(checks, f"strategy_{strategy}_cost_complete", "pass", "成本指标完整")


def _check_strategy_cost_effectiveness(
    strategy: str,
    payload: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    cost_effectiveness = payload.get("cost_effectiveness")
    if not isinstance(cost_effectiveness, dict):
        _add_check(checks, f"strategy_{strategy}_cost_effectiveness_complete", "fail", "缺少 cost_effectiveness")
        return
    missing = [field for field in REQUIRED_COST_EFFECTIVENESS_FIELDS if field not in cost_effectiveness]
    if missing:
        _add_check(
            checks,
            f"strategy_{strategy}_cost_effectiveness_complete",
            "fail",
            f"性价比字段缺失：{missing}",
        )
        return
    _add_check(checks, f"strategy_{strategy}_cost_effectiveness_complete", "pass", "性价比指标完整")


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# 建图策略实验审计",
        "",
        f"是否通过：{'是' if audit.get('passed') else '否'}",
        f"检查总数：{audit.get('summary', {}).get('total_checks', 0)}",
        f"失败检查：{audit.get('summary', {}).get('failed_checks', 0)}",
        f"警告检查：{audit.get('summary', {}).get('warning_checks', 0)}",
        "",
        "| 检查项 | 状态 | 说明 |",
        "| --- | --- | --- |",
    ]
    for check in audit.get("checks", []):
        if not isinstance(check, dict):
            continue
        lines.append(f"| {check.get('id', '')} | {check.get('status', '')} | {check.get('message', '')} |")
    return "\n".join(lines) + "\n"


def _add_check(checks: list[dict[str, Any]], check_id: str, status: str, message: str) -> None:
    checks.append({"id": check_id, "status": status, "message": message})


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
