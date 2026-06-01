from __future__ import annotations

import json
from pathlib import Path

from sam.analogy import AnalogyEngine
from sam.models import EvaluationQuery


def run_analogy_reuse_probe(
    engine: AnalogyEngine,
    queries: list[EvaluationQuery],
    *,
    top_k: int = 3,
) -> dict[str, object]:
    """评估类比检索是否能命中 warmup 阶段巩固出的历史案例。"""

    cases: list[dict[str, object]] = []
    consolidated_case_hits = 0
    support_overlap_hits = 0
    for query in queries:
        source_query_id = str(query.metadata.get("source_query_id", query.id))
        matches = engine.retrieve_cases(
            query.question,
            top_k=top_k,
            exclude_case_id=query.id,
        )
        serialized = [_serialize_match(match) for match in matches]
        top_match = serialized[0] if serialized else {}
        is_consolidated_hit = bool(top_match.get("is_consolidated_case")) and top_match.get("case_id") == source_query_id
        support_overlap = bool(
            set(query.supporting_doc_ids)
            & set(str(item) for item in top_match.get("support_original_doc_ids", []))
        )
        if is_consolidated_hit:
            consolidated_case_hits += 1
        if support_overlap:
            support_overlap_hits += 1
        cases.append(
            {
                "query_id": query.id,
                "source_query_id": source_query_id,
                "question": query.question,
                "answer": query.answer,
                "top_match": top_match,
                "matches": serialized,
                "consolidated_case_hit": is_consolidated_hit,
                "support_overlap_hit": support_overlap,
            }
        )
    query_count = len(queries)
    return {
        "query_count": query_count,
        "consolidated_case_hit_count": consolidated_case_hits,
        "support_overlap_hit_count": support_overlap_hits,
        "consolidated_case_hit_rate": consolidated_case_hits / query_count if query_count else 0.0,
        "support_overlap_hit_rate": support_overlap_hits / query_count if query_count else 0.0,
        "cases": cases,
    }


def write_analogy_reuse_reports(
    *,
    output_dir: str | Path,
    result: dict[str, object],
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "analogy_reuse_results.json"
    markdown_path = target / "analogy_reuse_results.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_analogy_reuse_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def _serialize_match(match) -> dict[str, object]:
    support_node_ids = [
        str(node_id)
        for node_id in match.metadata.get("support_node_ids", [])
    ]
    support_original_doc_ids = list(
        match.metadata.get("support_original_doc_ids", [])
    ) or [
        str(node.metadata.get("original_doc_id"))
        for node in match.matched_nodes
        if node.metadata.get("original_doc_id")
    ]
    return {
        "case_id": match.case_id,
        "score": match.score,
        "is_consolidated_case": match.metadata.get("is_consolidated_case", False),
        "case_answer": match.metadata.get("case_answer"),
        "support_node_ids": support_node_ids,
        "support_original_doc_ids": support_original_doc_ids,
        "support_titles": match.metadata.get("support_titles", []),
        "matched_relation_path": match.metadata.get("matched_relation_path", []),
        "prompt_hint": match.prompt_hint,
    }


def _analogy_reuse_markdown(result: dict[str, object]) -> str:
    return "\n".join(
        [
            "# SAM 类比复用实验",
            "",
            f"- 查询数量：{result.get('query_count')}",
            f"- 巩固案例命中数：{result.get('consolidated_case_hit_count')}",
            f"- 巩固案例命中率：{float(result.get('consolidated_case_hit_rate', 0.0)):.3f}",
            f"- 支持证据重叠命中数：{result.get('support_overlap_hit_count')}",
            f"- 支持证据重叠命中率：{float(result.get('support_overlap_hit_rate', 0.0)):.3f}",
        ]
    )
