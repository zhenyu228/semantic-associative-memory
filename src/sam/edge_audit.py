from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def audit_edge_quality(
    cases: list[dict[str, object]],
    *,
    method: str = "sam_full",
) -> dict[str, object]:
    """根据 cases.json 审计图边关系类型的支持与噪声分布。"""

    relation_support: Counter[str] = Counter()
    relation_noise: Counter[str] = Counter()
    relation_graph_scores: dict[str, list[float]] = {}
    graph_noise_case_ids: set[str] = set()
    graph_hit_count = 0
    support_graph_hit_count = 0
    noise_graph_hit_count = 0

    for case in cases:
        hits = _method_hits(case, method)
        supporting_count = len(case.get("supporting_doc_ids", []))
        support_hits = _support_hits(case, method)
        case_has_noise = False
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            if len(hit.get("path", [])) <= 1:
                continue
            relations = _relations_for_hit(hit)
            if not relations:
                continue
            graph_hit_count += 1
            is_supporting = bool(hit.get("is_supporting"))
            if is_supporting:
                support_graph_hit_count += 1
            else:
                noise_graph_hit_count += 1
                if supporting_count and support_hits < supporting_count:
                    case_has_noise = True
            for relation_type, graph_score in relations:
                if is_supporting:
                    relation_support[relation_type] += 1
                else:
                    relation_noise[relation_type] += 1
                relation_graph_scores.setdefault(relation_type, []).append(graph_score)
        if case_has_noise:
            graph_noise_case_ids.add(str(case.get("query_id", "")))

    relation_stats = _relation_stats(
        support_counts=relation_support,
        noise_counts=relation_noise,
        graph_scores=relation_graph_scores,
    )
    return {
        "method": method,
        "summary": {
            "case_count": len(cases),
            "graph_hit_count": graph_hit_count,
            "support_graph_hit_count": support_graph_hit_count,
            "noise_graph_hit_count": noise_graph_hit_count,
            "graph_noise_case_count": len(graph_noise_case_ids),
            "graph_noise_case_ids": sorted(graph_noise_case_ids),
        },
        "relation_stats": relation_stats,
        "recommendations": _recommendations(relation_stats),
    }


def write_edge_quality_audit(
    audit: dict[str, object],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """写出图边质量审计 JSON 和 Markdown。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "edge_quality_audit.json"
    markdown_path = target / "edge_quality_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_audit_markdown(audit), encoding="utf-8")
    return json_path, markdown_path


def _method_hits(case: dict[str, object], method: str) -> list[object]:
    methods = case.get("methods", {})
    if not isinstance(methods, dict):
        return []
    if method in methods and isinstance(methods[method], list):
        return methods[method]
    for name, hits in methods.items():
        if str(name).startswith("sam") and isinstance(hits, list):
            return hits
    return []


def _support_hits(case: dict[str, object], method: str) -> int:
    support_hits_by_method = case.get("support_hits_by_method", {})
    if isinstance(support_hits_by_method, dict):
        return int(support_hits_by_method.get(method, 0) or 0)
    return 0


def _relations_for_hit(hit: dict[str, object]) -> list[tuple[str, float]]:
    relations: list[tuple[str, float]] = []
    candidate_paths = hit.get("candidate_paths", [])
    if isinstance(candidate_paths, list):
        for path in candidate_paths:
            if not isinstance(path, dict):
                continue
            relation_type = path.get("relation_type")
            if not relation_type:
                continue
            relations.append(
                (
                    str(relation_type),
                    float(path.get("graph_score", 0.0) or 0.0),
                )
            )
    if relations:
        return relations
    reason = str(hit.get("reason", ""))
    for relation_type in [
        "shared_entity",
        "keyword_overlap",
        "embedding_similarity",
        "context_cooccurrence",
        "summary_parent",
        "summary_child",
        "consolidates_support",
        "analogy_case_reuse",
    ]:
        if relation_type in reason:
            relations.append((relation_type, float(hit.get("graph_score", 0.0) or 0.0)))
    return relations


def _relation_stats(
    *,
    support_counts: Counter[str],
    noise_counts: Counter[str],
    graph_scores: dict[str, list[float]],
) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for relation_type in sorted(set(support_counts) | set(noise_counts) | set(graph_scores)):
        support_count = int(support_counts.get(relation_type, 0))
        noise_count = int(noise_counts.get(relation_type, 0))
        total_count = support_count + noise_count
        scores = graph_scores.get(relation_type, [])
        stats[relation_type] = {
            "support_count": support_count,
            "noise_count": noise_count,
            "total_count": total_count,
            "noise_rate": round(noise_count / total_count, 4) if total_count else 0.0,
            "average_graph_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        }
    return stats


def _recommendations(relation_stats: dict[str, dict[str, object]]) -> list[str]:
    recommendations: list[str] = []
    weak_relations = {"embedding_similarity", "context_cooccurrence", "keyword_overlap"}
    for relation_type, stats in relation_stats.items():
        noise_rate = float(stats.get("noise_rate", 0.0))
        total_count = int(stats.get("total_count", 0) or 0)
        if relation_type in weak_relations and total_count and noise_rate >= 0.6:
            recommendations.append(
                f"降低 {relation_type} 在二跳路径中的权重，或要求更高的实体/LLM 关系判别置信度。"
            )
        elif total_count and noise_rate >= 0.8:
            recommendations.append(
                f"审查 {relation_type} 的建边规则，该关系在当前 run 中主要出现在非支持证据路径。"
            )
    if not recommendations:
        recommendations.append("当前 run 未发现单一关系类型的高噪声集中问题，建议继续扩大样本。")
    return recommendations


def _audit_markdown(audit: dict[str, object]) -> str:
    summary = audit.get("summary", {})
    relation_stats = audit.get("relation_stats", {})
    recommendations = audit.get("recommendations", [])
    lines = [
        "# 图边质量审计",
        "",
        f"- 方法：{audit.get('method')}",
        f"- 样本数：{summary.get('case_count', 0) if isinstance(summary, dict) else 0}",
        f"- 图路径命中数：{summary.get('graph_hit_count', 0) if isinstance(summary, dict) else 0}",
        f"- 噪声图路径命中数：{summary.get('noise_graph_hit_count', 0) if isinstance(summary, dict) else 0}",
        f"- 图噪声 bad case 数：{summary.get('graph_noise_case_count', 0) if isinstance(summary, dict) else 0}",
        "",
        "| 关系类型 | 支持次数 | 噪声次数 | 噪声率 | 平均图分 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if isinstance(relation_stats, dict):
        for relation_type, stats in relation_stats.items():
            if not isinstance(stats, dict):
                continue
            lines.append(
                f"| {relation_type} | {stats.get('support_count', 0)} | "
                f"{stats.get('noise_count', 0)} | {float(stats.get('noise_rate', 0.0)):.3f} | "
                f"{float(stats.get('average_graph_score', 0.0)):.3f} |"
            )
    lines.extend(["", "## 建议", ""])
    if isinstance(recommendations, list):
        for item in recommendations:
            lines.append(f"- {item}")
    return "\n".join(lines)
