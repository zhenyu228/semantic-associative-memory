from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sam.evidence_rescue_experiment import evaluate_evidence_rescue
from sam.graph_strategy_experiment import (
    GraphStrategyConfig,
    _allowed_pair_keys_for_scope,
    _edge_for_score,
    score_pair,
)
from sam.models import EvaluationQuery, MemoryNode
from sam.progress import progress_iter


def run_graph_density_sweep(
    *,
    nodes: list[MemoryNode],
    queries: list[EvaluationQuery],
    query_embeddings: dict[str, list[float]],
    strategy: str = "sam_context",
    top_k_edges_values: list[int] | None = None,
    threshold_values: list[float] | None = None,
    alpha: float = 0.55,
    top_k: int = 5,
    seed_k: int = 2,
    hops: int = 1,
    pair_scope: str = "query_candidates",
    max_rescue_per_seed: int = 2,
    min_expansion_similarity: float = -1.0,
) -> dict[str, Any]:
    """扫描图密度配置，观察补证据收益和噪声扩展变化。"""

    top_k_edges_values = top_k_edges_values or [1, 2, 4, 8, 16]
    threshold_values = threshold_values or [0.10, 0.18, 0.25]
    rows: list[dict[str, Any]] = []
    scored_graph = _precompute_density_scores(
        nodes=nodes,
        queries=queries,
        strategy=strategy,
        alpha=alpha,
        pair_scope=pair_scope,
    )
    for threshold in threshold_values:
        for top_k_edges in top_k_edges_values:
            build_start = time.perf_counter()
            edges = _edges_from_precomputed_scores(
                scored_by_source=scored_graph["scored_by_source"],
                strategy=strategy,
                top_k_edges=top_k_edges,
                threshold=threshold,
            )
            selection_time_seconds = time.perf_counter() - build_start
            result = evaluate_evidence_rescue(
                nodes=nodes,
                edges=edges,
                queries=queries,
                query_embeddings=query_embeddings,
                top_k=top_k,
                seed_k=seed_k,
                hops=hops,
                max_rescue_per_seed=max_rescue_per_seed,
                min_expansion_similarity=min_expansion_similarity,
            )
            cost = _density_cost_payload(
                nodes=nodes,
                edges=edges,
                candidate_pair_count=int(scored_graph["candidate_pair_count"]),
                theoretical_full_pair_count=int(scored_graph["theoretical_full_pair_count"]),
                score_precompute_time_seconds=float(scored_graph["score_precompute_time_seconds"]),
                selection_time_seconds=selection_time_seconds,
                pair_scope=pair_scope,
            )
            row = _density_row(
                strategy=strategy,
                alpha=alpha,
                top_k_edges=top_k_edges,
                threshold=threshold,
                metrics=result["metrics"],
                cost=cost,
            )
            rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row["threshold"]),
            int(row["top_k_edges"]),
        )
    )
    return {
        "config": {
            "strategy": strategy,
            "alpha": alpha,
            "top_k_edges_values": top_k_edges_values,
            "threshold_values": threshold_values,
            "top_k": top_k,
            "seed_k": seed_k,
            "hops": hops,
            "pair_scope": pair_scope,
            "max_rescue_per_seed": max_rescue_per_seed,
            "min_expansion_similarity": min_expansion_similarity,
        },
        "dataset": {
            "document_count": len(nodes),
            "query_count": len(queries),
            "supporting_evidence_count": sum(len(query.supporting_doc_ids) for query in queries),
        },
        "summary": _density_summary(rows),
        "density_rows": rows,
        "score_precompute": {
            "candidate_pair_count": scored_graph["candidate_pair_count"],
            "theoretical_full_pair_count": scored_graph["theoretical_full_pair_count"],
            "score_precompute_time_seconds": scored_graph["score_precompute_time_seconds"],
        },
    }


def write_graph_density_report(
    report: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """写出图密度实验 JSON 和 Markdown 报告。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "graph_density_results.json"
    markdown_path = target / "graph_density_results.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_density_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _density_row(
    *,
    strategy: str,
    alpha: float,
    top_k_edges: int,
    threshold: float,
    metrics: dict[str, Any],
    cost: dict[str, Any],
) -> dict[str, Any]:
    expanded_node_count = int(metrics.get("expanded_node_count", 0))
    rescued_support_count = int(metrics.get("rescued_support_count", 0))
    edge_count = int(cost.get("edge_count", 0))
    recall_gain = float(metrics.get("recall_gain", 0.0))
    rescue_precision = float(metrics.get("rescue_precision", 0.0))
    noise_expansion_count = max(0, expanded_node_count - rescued_support_count)
    noise_expansion_rate = (
        noise_expansion_count / expanded_node_count
        if expanded_node_count
        else 0.0
    )
    return {
        "strategy": strategy,
        "alpha": alpha,
        "top_k_edges": top_k_edges,
        "threshold": threshold,
        "edge_count": edge_count,
        "candidate_pair_count": int(cost.get("candidate_pair_count", 0)),
        "candidate_pair_coverage": float(cost.get("candidate_pair_coverage", 0.0)),
        "build_time_seconds": float(cost.get("build_time_seconds", 0.0)),
        "score_precompute_time_seconds": float(cost.get("score_precompute_time_seconds", 0.0)),
        "edge_selection_time_seconds": float(cost.get("edge_selection_time_seconds", 0.0)),
        "average_edges_per_node": float(cost.get("average_edges_per_node", 0.0)),
        "baseline_evidence_recall": float(metrics.get("baseline_evidence_recall", 0.0)),
        "evidence_recall_with_rescue": float(metrics.get("evidence_recall_with_rescue", 0.0)),
        "recall_gain": recall_gain,
        "rescued_support_count": rescued_support_count,
        "rescue_success_query_count": int(metrics.get("rescue_success_query_count", 0)),
        "eligible_query_count": int(metrics.get("eligible_query_count", 0)),
        "rescue_precision": rescue_precision,
        "expanded_node_count": expanded_node_count,
        "average_new_expansions_per_query": float(metrics.get("average_new_expansions_per_query", 0.0)),
        "noise_expansion_count": noise_expansion_count,
        "noise_expansion_rate": noise_expansion_rate,
        "recall_gain_per_100_edges": (
            100.0 * recall_gain / edge_count if edge_count else 0.0
        ),
        "rescued_support_per_100_edges": (
            100.0 * rescued_support_count / edge_count if edge_count else 0.0
        ),
        "density_score": _density_score(
            recall_gain=recall_gain,
            rescue_precision=rescue_precision,
            edge_count=edge_count,
            build_time_seconds=float(cost.get("build_time_seconds", 0.0)),
            noise_expansion_rate=noise_expansion_rate,
        ),
    }


def _precompute_density_scores(
    *,
    nodes: list[MemoryNode],
    queries: list[EvaluationQuery],
    strategy: str,
    alpha: float,
    pair_scope: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    config = GraphStrategyConfig(strategy=strategy, alpha=alpha)
    allowed_pair_keys = _allowed_pair_keys_for_scope(
        nodes=nodes,
        queries=queries,
        pair_scope=pair_scope,
    )
    scored_by_source: dict[str, list[tuple[float, MemoryNode, dict[str, object]]]] = {}
    candidate_pair_count = 0
    for source in progress_iter(nodes, total=len(nodes), desc=f"预计算边分数:{strategy}"):
        scored: list[tuple[float, MemoryNode, dict[str, object]]] = []
        for target in nodes:
            if source.id == target.id:
                continue
            if allowed_pair_keys is not None and (source.id, target.id) not in allowed_pair_keys:
                continue
            candidate_pair_count += 1
            score, breakdown = score_pair(source, target, config)
            scored.append((score, target, breakdown))
        scored.sort(key=lambda item: item[0], reverse=True)
        scored_by_source[source.id] = scored
    return {
        "scored_by_source": scored_by_source,
        "candidate_pair_count": candidate_pair_count,
        "theoretical_full_pair_count": len(nodes) * max(0, len(nodes) - 1),
        "score_precompute_time_seconds": time.perf_counter() - start,
    }


def _edges_from_precomputed_scores(
    *,
    scored_by_source: dict[str, list[tuple[float, MemoryNode, dict[str, object]]]],
    strategy: str,
    top_k_edges: int,
    threshold: float,
) -> list[Any]:
    if top_k_edges <= 0:
        return []
    edges = []
    for source_id, scored in scored_by_source.items():
        kept = 0
        source_node = None
        for score, target, breakdown in scored:
            if score < threshold:
                break
            if source_node is None:
                source_node = _source_proxy(source_id)
            edges.append(_edge_for_score(source_node, target, strategy, score, breakdown))
            kept += 1
            if kept >= top_k_edges:
                break
    return edges


def _source_proxy(source_id: str) -> MemoryNode:
    return MemoryNode(
        id=source_id,
        text="",
        summary="",
        keywords=[],
        tags=[],
        source="density_precompute",
        created_at="",
        last_accessed_at=None,
        usage_count=0,
        confidence=0.0,
        embedding=[],
        metadata={},
    )


def _density_cost_payload(
    *,
    nodes: list[MemoryNode],
    edges: list[Any],
    candidate_pair_count: int,
    theoretical_full_pair_count: int,
    score_precompute_time_seconds: float,
    selection_time_seconds: float,
    pair_scope: str,
) -> dict[str, Any]:
    edge_count = len(edges)
    node_ids = {
        node_id
        for edge in edges
        for node_id in [edge.source_id, edge.target_id]
    }
    return {
        "pair_scope": pair_scope,
        "candidate_pair_count": candidate_pair_count,
        "theoretical_full_pair_count": theoretical_full_pair_count,
        "candidate_pair_coverage": (
            candidate_pair_count / theoretical_full_pair_count
            if theoretical_full_pair_count
            else 0.0
        ),
        "edge_count": edge_count,
        "average_edges_per_node": edge_count / len(node_ids) if node_ids else 0.0,
        "build_time_seconds": score_precompute_time_seconds + selection_time_seconds,
        "score_precompute_time_seconds": score_precompute_time_seconds,
        "edge_selection_time_seconds": selection_time_seconds,
    }


def _density_score(
    *,
    recall_gain: float,
    rescue_precision: float,
    edge_count: int,
    build_time_seconds: float,
    noise_expansion_rate: float,
) -> float:
    quality = max(0.0, recall_gain) + 0.5 * rescue_precision
    cost_penalty = 1.0 + edge_count / 1000.0 + build_time_seconds
    noise_penalty = 1.0 + noise_expansion_rate
    return quality / (cost_penalty * noise_penalty)


def _density_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "configuration_count": 0,
            "best_balanced_configuration": None,
            "best_recall_gain_configuration": None,
            "lowest_noise_configuration": None,
        }
    improving_rows = [row for row in rows if float(row["recall_gain"]) > 0.0]
    best_balanced = max(rows, key=lambda row: (float(row["density_score"]), float(row["recall_gain"])))
    best_recall = max(rows, key=lambda row: (float(row["recall_gain"]), float(row["rescue_precision"])))
    noise_pool = improving_rows or rows
    lowest_noise = min(
        noise_pool,
        key=lambda row: (
            float(row["noise_expansion_rate"]),
            int(row["edge_count"]),
        ),
    )
    return {
        "configuration_count": len(rows),
        "best_balanced_configuration": _compact_row(best_balanced),
        "best_recall_gain_configuration": _compact_row(best_recall),
        "lowest_noise_configuration": _compact_row(lowest_noise),
        "max_recall_gain": max(float(row["recall_gain"]) for row in rows),
        "min_noise_expansion_rate": min(float(row["noise_expansion_rate"]) for row in rows),
        "max_edge_count": max(int(row["edge_count"]) for row in rows),
        "min_edge_count": min(int(row["edge_count"]) for row in rows),
    }


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "top_k_edges": row["top_k_edges"],
        "threshold": row["threshold"],
        "edge_count": row["edge_count"],
        "recall_gain": row["recall_gain"],
        "rescue_precision": row["rescue_precision"],
        "noise_expansion_rate": row["noise_expansion_rate"],
        "density_score": row["density_score"],
    }


def _density_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    config = report.get("config", {})
    dataset = report.get("dataset", {})
    rows = report.get("density_rows", [])
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(config, dict):
        config = {}
    if not isinstance(dataset, dict):
        dataset = {}
    if not isinstance(rows, list):
        rows = []
    lines = [
        "# 图密度与噪声实验",
        "",
        "本实验固定 embedding top-k，只改变图建边密度，观察边数增加后补证据收益和噪声扩展如何变化。",
        "",
        "## 实验配置",
        "",
        f"- 建图策略：{config.get('strategy')}",
        f"- alpha：{config.get('alpha')}",
        f"- 数据文档数：{dataset.get('document_count')}",
        f"- Query 数：{dataset.get('query_count')}",
        f"- Gold evidence 数：{dataset.get('supporting_evidence_count')}",
        f"- pair scope：{config.get('pair_scope')}",
        "",
        "## 关键结论",
        "",
        f"- 配置数量：{summary.get('configuration_count')}",
        f"- 最佳综合配置：`{summary.get('best_balanced_configuration')}`",
        f"- 最大 recall 增益配置：`{summary.get('best_recall_gain_configuration')}`",
        f"- 最低噪声配置：`{summary.get('lowest_noise_configuration')}`",
        "",
        "## 密度扫描结果",
        "",
        "| threshold | top_k_edges | 边数 | Baseline Recall | Rescue Recall | Recall 增益 | 补回证据数 | 扩展节点数 | 图扩展 Precision | 噪声扩展率 | Gain/100 edges | 密度综合分 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {threshold:.3f} | {top_k_edges} | {edge_count} | {baseline:.4f} | {rescue:.4f} | "
            "{gain:.4f} | {rescued} | {expanded} | {precision:.4f} | {noise:.4f} | "
            "{gain_edges:.6f} | {score:.6f} |".format(
                threshold=float(row.get("threshold", 0.0)),
                top_k_edges=int(row.get("top_k_edges", 0)),
                edge_count=int(row.get("edge_count", 0)),
                baseline=float(row.get("baseline_evidence_recall", 0.0)),
                rescue=float(row.get("evidence_recall_with_rescue", 0.0)),
                gain=float(row.get("recall_gain", 0.0)),
                rescued=int(row.get("rescued_support_count", 0)),
                expanded=int(row.get("expanded_node_count", 0)),
                precision=float(row.get("rescue_precision", 0.0)),
                noise=float(row.get("noise_expansion_rate", 0.0)),
                gain_edges=float(row.get("recall_gain_per_100_edges", 0.0)),
                score=float(row.get("density_score", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## 指标解释",
            "",
            "- 图扩展 Precision = 图补回 gold evidence 数 / 图额外扩展节点数。",
            "- 噪声扩展率 = 1 - 图扩展 Precision，表示扩展出来但不是 gold evidence 的节点比例。",
            "- Gain/100 edges = 每 100 条图边带来的 evidence recall 增益。",
            "- 密度综合分只用于选择阶段性配置，综合 recall 增益、precision、边数、耗时和噪声惩罚。",
        ]
    )
    return "\n".join(lines) + "\n"
