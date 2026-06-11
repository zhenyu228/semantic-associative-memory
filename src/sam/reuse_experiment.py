from __future__ import annotations

import json
from pathlib import Path

from sam.models import EvaluationQuery, MemoryEdge
from sam.store import MemoryStore


def build_masked_queries(queries: list[EvaluationQuery]) -> list[EvaluationQuery]:
    """构造连续复用 probe：保留问题和 gold，移除候选集中的 gold 支持文档。"""

    masked: list[EvaluationQuery] = []
    for query in queries:
        support_ids = set(query.supporting_doc_ids)
        masked_candidates = [
            doc_id for doc_id in query.candidate_doc_ids
            if doc_id not in support_ids
        ]
        masked.append(
            EvaluationQuery(
                id=f"{query.id}::reuse_probe",
                dataset=query.dataset,
                question=query.question,
                answer=query.answer,
                supporting_doc_ids=list(query.supporting_doc_ids),
                candidate_doc_ids=masked_candidates,
                metadata={
                    **query.metadata,
                    "reuse_probe": True,
                    "source_query_id": query.id,
                    "masked_support_doc_ids": list(query.supporting_doc_ids),
                },
            )
        )
    return masked


def memory_reuse_candidate_ids(
    *,
    store: MemoryStore,
    query: EvaluationQuery,
    method: str,
    base_candidate_ids: list[str],
) -> list[str]:
    """为连续记忆复用 probe 构造候选池。

    普通 baseline 保持被 mask 后的候选集；SAM 方法可以读取 warmup 阶段
    形成的巩固记忆节点，以及这些巩固记忆连接回的支持证据节点。
    """

    candidate_ids = list(base_candidate_ids)
    if not query.metadata.get("reuse_probe") or not method.startswith("sam"):
        return list(dict.fromkeys(candidate_ids))

    source_query_id = str(query.metadata.get("source_query_id") or query.id)
    for node in store.get_nodes():
        if node.metadata.get("node_type") != "consolidated_memory":
            continue
        if str(node.metadata.get("query_id")) != source_query_id:
            continue
        candidate_ids.append(node.id)
        candidate_ids.extend(
            str(support_id)
            for support_id in node.metadata.get("support_node_ids", [])
        )
        candidate_ids.extend(
            str(evidence_id)
            for evidence_id in node.metadata.get("evidence_node_ids", [])
        )
    return list(dict.fromkeys(candidate_ids))


def summarize_memory_reuse(
    *,
    warmup_consolidated_count: int,
    warmup_consolidation_edge_count: int,
    baseline_metric: dict[str, object],
    sam_metric: dict[str, object],
) -> dict[str, object]:
    baseline_support_hits = int(baseline_metric.get("support_hits", 0))
    sam_support_hits = int(sam_metric.get("support_hits", 0))
    baseline_recall = float(baseline_metric.get("evidence_recall", 0.0))
    sam_recall = float(sam_metric.get("evidence_recall", 0.0))
    return {
        "warmup_consolidated_count": warmup_consolidated_count,
        "warmup_consolidation_edge_count": warmup_consolidation_edge_count,
        "baseline_support_hits": baseline_support_hits,
        "sam_support_hits": sam_support_hits,
        "support_hit_gain": sam_support_hits - baseline_support_hits,
        "baseline_evidence_recall": baseline_recall,
        "sam_evidence_recall": sam_recall,
        "evidence_recall_gain": sam_recall - baseline_recall,
    }


def write_memory_reuse_reports(
    *,
    output_dir: str | Path,
    summary: dict[str, object],
    warmup_metrics: dict[str, object],
    probe_metrics: dict[str, object],
    probe_cases: list[dict[str, object]],
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "memory_reuse_results.json"
    markdown_path = target / "memory_reuse_results.md"
    payload = {
        "summary": summary,
        "warmup_metrics": warmup_metrics,
        "probe_metrics": probe_metrics,
        "probe_cases": probe_cases,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_memory_reuse_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def snapshot_edges(edges: list[MemoryEdge]) -> dict[str, dict[str, object]]:
    """记录 probe 前的边状态，用于对比反馈后的边权和激活次数。"""

    return {
        _edge_key(edge): {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relation_type": edge.relation_type,
            "weight": edge.weight,
            "activation_count": edge.activation_count,
            "updated_at": edge.updated_at,
        }
        for edge in edges
    }


def write_memory_reuse_event_reports(
    *,
    output_dir: str | Path,
    events: list[dict[str, object]],
    edges_after: list[MemoryEdge],
    edges_before: dict[str, dict[str, object]],
) -> tuple[Path, Path, Path, Path]:
    """输出连续记忆实验的事件流和反馈后边变化案例。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    events_json = target / "memory_events.json"
    events_md = target / "memory_events.md"
    changes_json = target / "feedback_edge_changes.json"
    changes_md = target / "feedback_edge_changes.md"
    changes = _edge_changes(edges_after, edges_before)
    events_json.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    events_md.write_text(_memory_events_markdown(events), encoding="utf-8")
    changes_json.write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    changes_md.write_text(_edge_changes_markdown(changes), encoding="utf-8")
    return events_json, events_md, changes_json, changes_md


def _memory_reuse_markdown(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, dict)
    probe_metrics = payload.get("probe_metrics", {})
    method_metrics = {}
    if isinstance(probe_metrics, dict):
        method_metrics = probe_metrics.get("method_metrics", {})
    assert isinstance(method_metrics, dict)
    lines = [
        "# SAM 连续记忆复用实验",
        "",
        f"- Warmup 巩固记忆节点数：{summary.get('warmup_consolidated_count')}",
        f"- Warmup 巩固边数量：{summary.get('warmup_consolidation_edge_count')}",
        f"- Baseline 支持证据命中数：{summary.get('baseline_support_hits')}",
        f"- SAM 支持证据命中数：{summary.get('sam_support_hits')}",
        f"- 支持证据命中增益：{summary.get('support_hit_gain')}",
        f"- Baseline 证据召回率：{float(summary.get('baseline_evidence_recall', 0.0)):.3f}",
        f"- SAM 证据召回率：{float(summary.get('sam_evidence_recall', 0.0)):.3f}",
        f"- 证据召回增益：{float(summary.get('evidence_recall_gain', 0.0)):.3f}",
    ]
    if method_metrics:
        lines.extend(
            [
                "",
                "## Probe 方法对比",
                "",
                "| 方法 | 支持证据命中数 | 证据召回率 | 答案命中率 | 平均路径长度 | 平均边记忆分 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method, metric in method_metrics.items():
            if not isinstance(metric, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(metric.get("display_name", method)),
                        str(metric.get("support_hits", 0)),
                        f"{float(metric.get('evidence_recall', 0.0)):.3f}",
                        f"{float(metric.get('answer_hit_rate', 0.0)):.3f}",
                        f"{float(metric.get('average_path_length', 0.0)):.2f}",
                        f"{float(metric.get('average_edge_memory_score', 0.0)):.3f}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def _memory_events_markdown(events: list[dict[str, object]]) -> str:
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type", "unknown"))
        counts[event_type] = counts.get(event_type, 0) + 1
    lines = [
        "# 连续记忆复用事件流",
        "",
        "| 事件类型 | 数量 |",
        "| --- | ---: |",
    ]
    for event_type, count in sorted(counts.items()):
        lines.append(f"| {event_type} | {count} |")
    lines.extend(
        [
            "",
            "## 最近事件",
            "",
            "| 时间 | 类型 | 方法 | 节点 | 分数 |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for event in events[:30]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(event.get("created_at", "")),
                    str(event.get("event_type", "")),
                    str(event.get("mode", "")),
                    str(event.get("node_id") or ""),
                    f"{float(event.get('score', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _edge_changes(
    edges_after: list[MemoryEdge],
    edges_before: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    for edge in edges_after:
        key = _edge_key(edge)
        before = edges_before.get(key)
        if before is None:
            continue
        before_weight = float(before["weight"])
        before_activation_count = int(before["activation_count"])
        if edge.weight == before_weight and edge.activation_count == before_activation_count:
            continue
        changes.append(
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation_type": edge.relation_type,
                "weight_before": round(before_weight, 4),
                "weight_after": round(edge.weight, 4),
                "weight_delta": round(edge.weight - before_weight, 4),
                "activation_count_before": before_activation_count,
                "activation_count_after": edge.activation_count,
                "activation_delta": edge.activation_count - before_activation_count,
                "reason": edge.reason,
                "updated_at": edge.updated_at,
            }
        )
    changes.sort(
        key=lambda item: (
            abs(float(item["weight_delta"])),
            int(item["activation_delta"]),
        ),
        reverse=True,
    )
    return changes


def _edge_changes_markdown(changes: list[dict[str, object]]) -> str:
    lines = [
        "# 连续记忆复用反馈边变化",
        "",
        "| 关系类型 | 边权变化 | 激活次数变化 | 原因 |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in changes[:30]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["relation_type"]),
                    f"{float(item['weight_before']):.4f} -> {float(item['weight_after']):.4f}",
                    f"{int(item['activation_count_before'])} -> {int(item['activation_count_after'])}",
                    str(item.get("reason", ""))[:120].replace("|", "/"),
                ]
            )
            + " |"
        )
    if not changes:
        lines.append("| 无变化 | 0.0000 -> 0.0000 | 0 -> 0 | 本次 probe 未改变边状态 |")
    return "\n".join(lines)


def _edge_key(edge: MemoryEdge) -> str:
    return f"{edge.source_id}\t{edge.target_id}\t{edge.relation_type}"
