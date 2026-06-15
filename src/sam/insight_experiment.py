from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sam.models import EvaluationQuery, MemoryNode
from sam.store import MemoryStore


def summarize_insight_memory_reconstruction(
    *,
    store: MemoryStore,
    queries: list[EvaluationQuery],
    max_cases: int = 8,
) -> dict[str, Any]:
    """统计高层洞察记忆的生成规模、证据覆盖和回溯能力。"""

    nodes = store.get_nodes()
    node_by_id = {node.id: node for node in nodes}
    original_to_node = {
        str(node.metadata["original_doc_id"]): node.id
        for node in nodes
        if "original_doc_id" in node.metadata
    }
    consolidated_nodes = [
        node for node in nodes
        if node.metadata.get("node_type") == "consolidated_memory"
    ]
    insight_nodes = [
        node for node in nodes
        if node.metadata.get("node_type") == "insight_memory"
    ]
    consolidated_ids = {node.id for node in consolidated_nodes}
    insight_source_consolidated_ids = {
        str(node_id)
        for insight in insight_nodes
        for node_id in insight.metadata.get("source_consolidated_node_ids", [])
    }
    consolidated_evidence_ids = {
        str(node_id)
        for node in consolidated_nodes
        for node_id in node.metadata.get("evidence_node_ids", [])
    }
    insight_evidence_ids = {
        str(node_id)
        for node in insight_nodes
        for node_id in node.metadata.get("evidence_node_ids", [])
    }
    support_node_ids = {
        original_to_node[doc_id]
        for query in queries
        for doc_id in query.supporting_doc_ids
        if doc_id in original_to_node
    }
    support_nodes_traced_by_insight = support_node_ids & insight_evidence_ids
    insight_edges = [
        edge for edge in store.get_edges()
        if edge.relation_type.startswith("insight_")
        or edge.relation_type in {"memory_summarized_by_insight", "evidence_supports_insight"}
    ]
    trace_edges = [
        edge for edge in insight_edges
        if edge.relation_type == "insight_traces_evidence"
    ]
    summary: dict[str, Any] = {
        "query_count": len(queries),
        "document_count": len([
            node for node in nodes
            if node.metadata.get("node_type") not in {"query_summary", "consolidated_memory", "insight_memory"}
        ]),
        "consolidated_memory_count": len(consolidated_nodes),
        "insight_memory_count": len(insight_nodes),
        "insight_edge_count": len(insight_edges),
        "insight_trace_edge_count": len(trace_edges),
        "source_consolidated_total": len(consolidated_ids),
        "source_consolidated_covered_count": len(insight_source_consolidated_ids & consolidated_ids),
        "consolidated_coverage_rate": _safe_divide(
            len(insight_source_consolidated_ids & consolidated_ids),
            len(consolidated_ids),
        ),
        "unique_consolidated_evidence_count": len(consolidated_evidence_ids),
        "unique_insight_evidence_count": len(insight_evidence_ids),
        "insight_evidence_coverage_rate": _safe_divide(
            len(insight_evidence_ids & consolidated_evidence_ids),
            len(consolidated_evidence_ids),
        ),
        "support_node_count": len(support_node_ids),
        "support_nodes_traced_by_insight_count": len(support_nodes_traced_by_insight),
        "support_trace_rate": _safe_divide(
            len(support_nodes_traced_by_insight),
            len(support_node_ids),
        ),
        "average_consolidated_per_insight": _average(
            len(node.metadata.get("source_consolidated_node_ids", []))
            for node in insight_nodes
        ),
        "average_evidence_per_insight": _average(
            len(node.metadata.get("evidence_node_ids", []))
            for node in insight_nodes
        ),
        "average_source_queries_per_insight": _average(
            len(node.metadata.get("source_query_ids", []))
            for node in insight_nodes
        ),
    }
    summary["insight_cases"] = _insight_cases(insight_nodes, node_by_id, max_cases=max_cases)
    summary["query_trace_cases"] = _query_trace_cases(
        queries,
        original_to_node,
        node_by_id,
        insight_nodes,
        max_cases=max_cases,
    )
    return summary


def write_insight_memory_reports(
    *,
    output_dir: str | Path,
    summary: dict[str, Any],
    warmup_metrics: dict[str, Any],
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "warmup_metrics": warmup_metrics,
    }
    json_path = target / "insight_memory_results.json"
    markdown_path = target / "insight_memory_results.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_insight_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def _insight_cases(
    insight_nodes: list[MemoryNode],
    node_by_id: dict[str, MemoryNode],
    *,
    max_cases: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for node in insight_nodes[:max_cases]:
        evidence_ids = [str(item) for item in node.metadata.get("evidence_node_ids", [])]
        cases.append(
            {
                "insight_node_id": node.id,
                "summary": node.summary,
                "confidence": node.confidence,
                "shared_keywords": node.metadata.get("shared_keywords", []),
                "source_query_ids": node.metadata.get("source_query_ids", []),
                "source_consolidated_count": len(node.metadata.get("source_consolidated_node_ids", [])),
                "evidence_count": len(evidence_ids),
                "evidence_titles": [
                    _node_title(node_by_id[evidence_id])
                    for evidence_id in evidence_ids
                    if evidence_id in node_by_id
                ],
                "traceability": node.metadata.get("traceability"),
            }
        )
    return cases


def _query_trace_cases(
    queries: list[EvaluationQuery],
    original_to_node: dict[str, str],
    node_by_id: dict[str, MemoryNode],
    insight_nodes: list[MemoryNode],
    *,
    max_cases: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for query in queries[:max_cases]:
        support_ids = [
            original_to_node[doc_id]
            for doc_id in query.supporting_doc_ids
            if doc_id in original_to_node
        ]
        matched_insights = [
            insight for insight in insight_nodes
            if set(support_ids) & {
                str(node_id) for node_id in insight.metadata.get("evidence_node_ids", [])
            }
        ]
        traced_ids = sorted(
            set(support_ids)
            & {
                str(node_id)
                for insight in matched_insights
                for node_id in insight.metadata.get("evidence_node_ids", [])
            }
        )
        cases.append(
            {
                "query_id": query.id,
                "question": query.question,
                "answer": query.answer,
                "support_node_ids": support_ids,
                "traced_support_node_ids": traced_ids,
                "trace_hit_rate": _safe_divide(len(traced_ids), len(support_ids)),
                "matched_insight_node_ids": [insight.id for insight in matched_insights],
                "traced_support_titles": [
                    _node_title(node_by_id[node_id])
                    for node_id in traced_ids
                    if node_id in node_by_id
                ],
            }
        )
    return cases


def _insight_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, dict)
    lines = [
        "# SAM 高层洞察记忆重构实验",
        "",
        f"- 查询数量：{summary.get('query_count')}",
        f"- 原始文档节点数：{summary.get('document_count')}",
        f"- 单次巩固记忆数：{summary.get('consolidated_memory_count')}",
        f"- 高层洞察记忆数：{summary.get('insight_memory_count')}",
        f"- 洞察关联边数：{summary.get('insight_edge_count')}",
        f"- 洞察可回溯证据数：{summary.get('unique_insight_evidence_count')}",
        f"- 巩固证据覆盖率：{float(summary.get('insight_evidence_coverage_rate', 0.0)):.3f}",
        f"- 支持证据回溯率：{float(summary.get('support_trace_rate', 0.0)):.3f}",
        f"- 平均每个洞察覆盖巩固记忆数：{float(summary.get('average_consolidated_per_insight', 0.0)):.2f}",
        f"- 平均每个洞察覆盖底层证据数：{float(summary.get('average_evidence_per_insight', 0.0)):.2f}",
        "",
        "## 典型洞察节点",
        "",
    ]
    for case in summary.get("insight_cases", [])[:5]:
        if not isinstance(case, dict):
            continue
        lines.extend(
            [
                f"### {case.get('insight_node_id')}",
                "",
                f"- 摘要：{case.get('summary')}",
                f"- 来源巩固记忆数：{case.get('source_consolidated_count')}",
                f"- 可回溯证据数：{case.get('evidence_count')}",
                f"- 共享关键词：{', '.join(str(item) for item in case.get('shared_keywords', [])[:8])}",
                f"- 证据标题：{', '.join(str(item) for item in case.get('evidence_titles', [])[:8])}",
                "",
            ]
        )
    lines.extend(["## 查询级回溯案例", ""])
    for case in summary.get("query_trace_cases", [])[:5]:
        if not isinstance(case, dict):
            continue
        lines.extend(
            [
                f"### {case.get('query_id')}",
                "",
                f"- 问题：{case.get('question')}",
                f"- 标准答案：{case.get('answer')}",
                f"- 回溯命中率：{float(case.get('trace_hit_rate', 0.0)):.3f}",
                f"- 命中的洞察节点：{', '.join(str(item) for item in case.get('matched_insight_node_ids', []))}",
                f"- 回溯证据标题：{', '.join(str(item) for item in case.get('traced_support_titles', []))}",
                "",
            ]
        )
    return "\n".join(lines)


def _node_title(node: MemoryNode) -> str:
    return str(node.metadata.get("title") or node.summary or node.id)


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _average(values: Any) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0
