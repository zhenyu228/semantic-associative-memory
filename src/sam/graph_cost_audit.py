from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def audit_graph_build_cost(
    *,
    edge_log: list[dict[str, object]],
    document_count: int,
    query_count: int,
) -> dict[str, object]:
    """统计按需建图相对全量建图的成本。

    这里的全量建图成本指对同一候选集合中所有文档两两比较的理论边数。
    """

    safe_document_count = max(0, document_count)
    safe_query_count = max(0, query_count)
    theoretical_full_edge_count = safe_document_count * (safe_document_count - 1) // 2
    created_edges = [entry for entry in edge_log if entry.get("action") == "created"]
    touched_edge_log_count = len(edge_log)
    created_edge_log_count = len(created_edges)
    unique_created_directed_edges = {
        (str(entry.get("source_id")), str(entry.get("target_id")))
        for entry in created_edges
        if entry.get("source_id") and entry.get("target_id")
    }
    unique_created_undirected_pairs = {
        tuple(sorted((str(entry.get("source_id")), str(entry.get("target_id")))))
        for entry in created_edges
        if entry.get("source_id") and entry.get("target_id")
    }
    relation_type_counts = Counter(
        str(entry.get("relation_type") or "unknown")
        for entry in edge_log
    )
    created_relation_type_counts = Counter(
        str(entry.get("relation_type") or "unknown")
        for entry in created_edges
    )
    unique_created_pair_to_full_ratio = (
        len(unique_created_undirected_pairs) / theoretical_full_edge_count
        if theoretical_full_edge_count
        else 0.0
    )
    return {
        "summary": {
            "document_count": safe_document_count,
            "query_count": safe_query_count,
            "touched_edge_log_count": touched_edge_log_count,
            "created_edge_log_count": created_edge_log_count,
            "unique_created_directed_edge_count": len(unique_created_directed_edges),
            "unique_created_undirected_pair_count": len(unique_created_undirected_pairs),
            "theoretical_full_edge_count": theoretical_full_edge_count,
            "unique_created_pair_to_full_ratio": unique_created_pair_to_full_ratio,
            "estimated_edge_saving_ratio": 1.0 - unique_created_pair_to_full_ratio if theoretical_full_edge_count else 0.0,
            "average_created_edge_logs_per_query": (
                created_edge_log_count / safe_query_count
                if safe_query_count
                else 0.0
            ),
            "average_created_undirected_pairs_per_query": (
                len(unique_created_undirected_pairs) / safe_query_count
                if safe_query_count
                else 0.0
            ),
            "average_touched_edge_logs_per_query": (
                touched_edge_log_count / safe_query_count
                if safe_query_count
                else 0.0
            ),
        },
        "relation_type_counts": dict(sorted(relation_type_counts.items())),
        "created_relation_type_counts": dict(sorted(created_relation_type_counts.items())),
    }


def write_graph_build_cost_audit(
    audit: dict[str, object],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """写出按需建图成本审计 JSON 和 Markdown。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "graph_build_cost_audit.json"
    markdown_path = target / "graph_build_cost_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_audit_to_markdown(audit), encoding="utf-8")
    return json_path, markdown_path


def _audit_to_markdown(audit: dict[str, object]) -> str:
    summary = audit.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    relation_counts = audit.get("relation_type_counts", {})
    if not isinstance(relation_counts, dict):
        relation_counts = {}
    lines = [
        "# 按需建图成本审计",
        "",
        f"- 文档节点数：{summary.get('document_count', 0)}",
        f"- 查询数量：{summary.get('query_count', 0)}",
        f"- 新建边日志数：{summary.get('created_edge_log_count', 0)}",
        f"- 触达边日志数：{summary.get('touched_edge_log_count', 0)}",
        f"- 唯一新建有向边数：{summary.get('unique_created_directed_edge_count', 0)}",
        f"- 唯一新建无向节点对数：{summary.get('unique_created_undirected_pair_count', 0)}",
        f"- 全量建图理论边数：{summary.get('theoretical_full_edge_count', 0)}",
        f"- 唯一新建节点对 / 全量边比例：{float(summary.get('unique_created_pair_to_full_ratio', 0.0)):.6f}",
        f"- 估算边比较节省比例：{float(summary.get('estimated_edge_saving_ratio', 0.0)):.6f}",
        f"- 平均每 query 新建无向节点对数：{float(summary.get('average_created_undirected_pairs_per_query', 0.0)):.3f}",
        "",
        "| 关系类型 | 触达次数 |",
        "| --- | ---: |",
    ]
    for relation_type, count in relation_counts.items():
        lines.append(f"| {relation_type} | {count} |")
    return "\n".join(lines)
