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
    use_source_relation_pattern: bool = True,
) -> dict[str, object]:
    """评估类比检索是否能命中 warmup 阶段巩固出的历史案例。"""

    cases: list[dict[str, object]] = []
    source_case_hits = 0
    consolidated_case_hits = 0
    support_overlap_hits = 0
    structure_pattern_available = 0
    structure_match_hits = 0
    total_top_match_score = 0.0
    bad_case_counts: dict[str, int] = {}
    for query in queries:
        source_query_id = str(query.metadata.get("source_query_id", query.id))
        relation_pattern = (
            engine.relation_pattern_for_case(source_query_id)
            if use_source_relation_pattern
            else []
        )
        if relation_pattern:
            structure_pattern_available += 1
        matches = engine.retrieve_cases(
            query.question,
            top_k=top_k,
            exclude_case_id=query.id,
            relation_pattern=relation_pattern,
        )
        serialized = [_serialize_match(match) for match in matches]
        top_match = serialized[0] if serialized else {}
        is_source_case_hit = top_match.get("case_id") == source_query_id
        is_consolidated_hit = bool(top_match.get("is_consolidated_case")) and top_match.get("case_id") == source_query_id
        structure_match = bool(top_match.get("matched_relation_path"))
        support_overlap = bool(
            set(query.supporting_doc_ids)
            & set(str(item) for item in top_match.get("support_original_doc_ids", []))
        )
        if is_source_case_hit:
            source_case_hits += 1
        if is_consolidated_hit:
            consolidated_case_hits += 1
        if structure_match:
            structure_match_hits += 1
        if support_overlap:
            support_overlap_hits += 1
        if top_match:
            total_top_match_score += float(top_match.get("score", 0.0))
        bad_case_type = _analogy_bad_case_type(
            top_match=top_match,
            is_consolidated_hit=is_consolidated_hit,
            is_source_case_hit=is_source_case_hit,
            support_overlap=support_overlap,
            relation_pattern=relation_pattern,
            structure_match=structure_match,
        )
        bad_case_counts[bad_case_type] = bad_case_counts.get(bad_case_type, 0) + 1
        cases.append(
            {
                "query_id": query.id,
                "source_query_id": source_query_id,
                "question": query.question,
                "answer": query.answer,
                "source_relation_pattern": relation_pattern,
                "top_match": top_match,
                "matches": serialized,
                "source_case_hit": is_source_case_hit,
                "consolidated_case_hit": is_consolidated_hit,
                "structure_match_hit": structure_match,
                "support_overlap_hit": support_overlap,
                "bad_case_type": bad_case_type,
            }
        )
    query_count = len(queries)
    return {
        "query_count": query_count,
        "source_case_hit_count": source_case_hits,
        "consolidated_case_hit_count": consolidated_case_hits,
        "support_overlap_hit_count": support_overlap_hits,
        "structure_pattern_available_count": structure_pattern_available,
        "structure_match_hit_count": structure_match_hits,
        "source_case_hit_rate": source_case_hits / query_count if query_count else 0.0,
        "consolidated_case_hit_rate": consolidated_case_hits / query_count if query_count else 0.0,
        "support_overlap_hit_rate": support_overlap_hits / query_count if query_count else 0.0,
        "structure_pattern_available_rate": structure_pattern_available / query_count if query_count else 0.0,
        "structure_match_hit_rate": structure_match_hits / query_count if query_count else 0.0,
        "average_top_match_score": total_top_match_score / query_count if query_count else 0.0,
        "bad_case_counts": bad_case_counts,
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
    evidence_node_ids = [
        str(node_id)
        for node_id in match.metadata.get("evidence_node_ids", [])
    ]
    return {
        "case_id": match.case_id,
        "score": match.score,
        "is_consolidated_case": match.metadata.get("is_consolidated_case", False),
        "case_answer": match.metadata.get("case_answer"),
        "support_node_ids": support_node_ids,
        "support_original_doc_ids": support_original_doc_ids,
        "support_titles": match.metadata.get("support_titles", []),
        "evidence_node_ids": evidence_node_ids,
        "evidence_original_doc_ids": match.metadata.get("evidence_original_doc_ids", []),
        "evidence_titles": match.metadata.get("evidence_titles", []),
        "consolidation_source": match.metadata.get("consolidation_source"),
        "matched_relation_path": match.metadata.get("matched_relation_path", []),
        "path_pattern_score": match.metadata.get("path_pattern_score", 0.0),
        "relation_path_count": match.metadata.get("relation_path_count", 0),
        "longest_relation_path": match.metadata.get("longest_relation_path", []),
        "prompt_hint": match.prompt_hint,
    }


def _analogy_reuse_markdown(result: dict[str, object]) -> str:
    return "\n".join(
        [
            "# SAM 类比复用实验",
            "",
            f"- 查询数量：{result.get('query_count')}",
            f"- 来源案例命中数：{result.get('source_case_hit_count')}",
            f"- 来源案例命中率：{float(result.get('source_case_hit_rate', 0.0)):.3f}",
            f"- 巩固案例命中数：{result.get('consolidated_case_hit_count')}",
            f"- 巩固案例命中率：{float(result.get('consolidated_case_hit_rate', 0.0)):.3f}",
            f"- 支持证据重叠命中数：{result.get('support_overlap_hit_count')}",
            f"- 支持证据重叠命中率：{float(result.get('support_overlap_hit_rate', 0.0)):.3f}",
            f"- 来源结构模式可用数：{result.get('structure_pattern_available_count')}",
            f"- 结构路径匹配数：{result.get('structure_match_hit_count')}",
            f"- 结构路径匹配率：{float(result.get('structure_match_hit_rate', 0.0)):.3f}",
            f"- 平均 Top-1 类比分数：{float(result.get('average_top_match_score', 0.0)):.3f}",
            f"- Bad case 分布：{json.dumps(result.get('bad_case_counts', {}), ensure_ascii=False)}",
        ]
    )


def _analogy_bad_case_type(
    *,
    top_match: dict[str, object],
    is_consolidated_hit: bool,
    is_source_case_hit: bool,
    support_overlap: bool,
    relation_pattern: list[str],
    structure_match: bool,
) -> str:
    if not top_match:
        return "no_match"
    if is_consolidated_hit and support_overlap and (not relation_pattern or structure_match):
        return "success"
    if relation_pattern and not structure_match:
        return "structure_mismatch"
    if is_source_case_hit and not is_consolidated_hit:
        return "source_case_without_consolidation"
    if not is_source_case_hit:
        return "wrong_case"
    if not support_overlap:
        return "no_support_overlap"
    return "partial_success"
