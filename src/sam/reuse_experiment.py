from __future__ import annotations

import json
from pathlib import Path

from sam.models import EvaluationQuery


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


def _memory_reuse_markdown(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, dict)
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
    return "\n".join(lines)
