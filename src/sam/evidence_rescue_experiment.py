from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from sam.graph_strategy_experiment import (
    GraphStrategyConfig,
    _allowed_pair_keys_for_scope,
    build_graph_for_strategy,
)
from sam.models import EvaluationQuery, MemoryEdge, MemoryNode
from sam.text import cosine_similarity


def run_evidence_rescue_strategies(
    *,
    nodes: list[MemoryNode],
    queries: list[EvaluationQuery],
    query_embeddings: dict[str, list[float]],
    strategies: list[str],
    top_k: int = 5,
    seed_k: int = 1,
    hops: int = 1,
    alpha: float = 0.55,
    top_k_edges: int = 4,
    threshold: float = 0.18,
    pair_scope: str = "query_candidates",
    max_rescue_per_seed: int = 2,
    min_expansion_similarity: float = -1.0,
) -> dict[str, object]:
    """运行多种建图策略的补证据实验。"""

    allowed_pair_keys = _allowed_pair_keys_for_scope(
        nodes=nodes,
        queries=queries,
        pair_scope=pair_scope,
    )
    payload: dict[str, object] = {}
    for strategy in strategies:
        build_result = build_graph_for_strategy(
            nodes,
            GraphStrategyConfig(
                strategy=strategy,
                alpha=alpha,
                top_k_edges=top_k_edges,
                threshold=threshold,
            ),
            allowed_pair_keys=allowed_pair_keys,
            pair_scope=pair_scope,
        )
        result = evaluate_evidence_rescue(
            nodes=nodes,
            edges=build_result.edges,
            queries=queries,
            query_embeddings=query_embeddings,
            top_k=top_k,
            seed_k=seed_k,
            hops=hops,
            max_rescue_per_seed=max_rescue_per_seed,
            min_expansion_similarity=min_expansion_similarity,
        )
        payload[strategy] = {
            "strategy": strategy,
            "metrics": result["metrics"],
            "cost": build_result.cost_payload(),
            "cases": result["cases"],
        }
    return {
        "config": {
            "top_k": top_k,
            "seed_k": seed_k,
            "hops": hops,
            "alpha": alpha,
            "top_k_edges": top_k_edges,
            "threshold": threshold,
            "pair_scope": pair_scope,
            "max_rescue_per_seed": max_rescue_per_seed,
            "min_expansion_similarity": min_expansion_similarity,
        },
        "summary": _strategy_summary(payload),
        "strategies": payload,
    }


def evaluate_evidence_rescue(
    *,
    nodes: list[MemoryNode],
    edges: list[MemoryEdge],
    queries: list[EvaluationQuery],
    query_embeddings: dict[str, list[float]],
    top_k: int = 5,
    seed_k: int = 1,
    hops: int = 1,
    max_rescue_per_seed: int = 2,
    min_expansion_similarity: float = -1.0,
) -> dict[str, object]:
    """评估图是否能在不替换 embedding top-k 的前提下补回遗漏证据。"""

    node_by_original = {
        str(node.metadata.get("original_doc_id")): node.id
        for node in nodes
        if node.metadata.get("original_doc_id")
    }
    node_by_id = {node.id: node for node in nodes}
    adjacency = _adjacency(edges)

    support_total = 0
    baseline_support_hits = 0
    rescued_support_count = 0
    missing_support_total_for_eligible = 0
    expanded_node_count = 0
    eligible_query_count = 0
    rescue_success_query_count = 0
    cases: list[dict[str, object]] = []

    for query in queries:
        query_embedding = query_embeddings.get(query.id)
        if query_embedding is None:
            raise ValueError(f"缺少 query embedding：{query.id}")
        candidate_ids = [
            node_by_original[doc_id]
            for doc_id in query.candidate_doc_ids
            if doc_id in node_by_original
        ]
        support_node_ids = {
            node_by_original[doc_id]
            for doc_id in query.supporting_doc_ids
            if doc_id in node_by_original
        }
        support_total += len(support_node_ids)
        candidate_nodes = [node_by_id[node_id] for node_id in candidate_ids]
        baseline_hits = _embedding_hits(query_embedding, candidate_nodes, top_k=top_k)
        baseline_hit_ids = [node.id for _score, node in baseline_hits]
        baseline_hit_set = set(baseline_hit_ids)
        query_baseline_support = len(baseline_hit_set & support_node_ids)
        baseline_support_hits += query_baseline_support

        missing_support_ids = sorted(support_node_ids - baseline_hit_set)
        expanded = _expand_from_seeds(
            query_embedding=query_embedding,
            seed_ids=baseline_hit_ids[: max(seed_k, 1)],
            candidate_ids=set(candidate_ids),
            node_by_id=node_by_id,
            adjacency=adjacency,
            hops=hops,
            max_rescue_per_seed=max_rescue_per_seed,
            min_expansion_similarity=min_expansion_similarity,
        )
        expanded_new_ids = [item["node_id"] for item in expanded if item["node_id"] not in baseline_hit_set]
        rescued_ids = sorted(set(expanded_new_ids) & set(missing_support_ids))
        is_eligible = bool(query_baseline_support > 0 and missing_support_ids)
        if is_eligible:
            eligible_query_count += 1
            missing_support_total_for_eligible += len(missing_support_ids)
            if rescued_ids:
                rescue_success_query_count += 1
        rescued_support_count += len(rescued_ids)
        expanded_node_count += len(expanded_new_ids)
        union_support_hits = len((baseline_hit_set | set(rescued_ids)) & support_node_ids)
        cases.append(
            {
                "query_id": query.id,
                "question": query.question,
                "support_node_ids": sorted(support_node_ids),
                "baseline_hit_node_ids": baseline_hit_ids,
                "baseline_support_hits": query_baseline_support,
                "missing_support_node_ids": missing_support_ids,
                "expanded_node_ids": expanded_new_ids,
                "rescued_support_node_ids": rescued_ids,
                "union_support_hits": union_support_hits,
                "eligible_for_rescue": is_eligible,
                "rescue_success": bool(rescued_ids),
                "expanded_paths": expanded,
            }
        )

    union_support_hits = baseline_support_hits + rescued_support_count
    metrics = {
        "query_count": len(queries),
        "supporting_evidence_count": support_total,
        "baseline_support_hits": baseline_support_hits,
        "baseline_evidence_recall": baseline_support_hits / support_total if support_total else 0.0,
        "eligible_query_count": eligible_query_count,
        "missing_support_total_for_eligible": missing_support_total_for_eligible,
        "rescued_support_count": rescued_support_count,
        "rescue_success_query_count": rescue_success_query_count,
        "rescue_success_rate": rescue_success_query_count / eligible_query_count if eligible_query_count else 0.0,
        "rescue_recall_over_missing": (
            rescued_support_count / missing_support_total_for_eligible
            if missing_support_total_for_eligible
            else 0.0
        ),
        "expanded_node_count": expanded_node_count,
        "rescue_precision": rescued_support_count / expanded_node_count if expanded_node_count else 0.0,
        "evidence_recall_with_rescue": union_support_hits / support_total if support_total else 0.0,
        "recall_gain": (
            (union_support_hits / support_total) - (baseline_support_hits / support_total)
            if support_total
            else 0.0
        ),
        "average_new_expansions_per_query": expanded_node_count / len(queries) if queries else 0.0,
        "average_new_expansions_per_eligible_query": (
            expanded_node_count / eligible_query_count if eligible_query_count else 0.0
        ),
    }
    return {"metrics": metrics, "cases": cases}


def write_evidence_rescue_report(report: dict[str, object], output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "evidence_rescue_results.json"
    markdown_path = target / "evidence_rescue_results.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def _embedding_hits(
    query_embedding: list[float],
    candidate_nodes: list[MemoryNode],
    *,
    top_k: int,
) -> list[tuple[float, MemoryNode]]:
    hits = [
        (cosine_similarity(query_embedding, node.embedding), node)
        for node in candidate_nodes
    ]
    hits.sort(key=lambda item: item[0], reverse=True)
    return hits[:top_k]


def _expand_from_seeds(
    *,
    query_embedding: list[float],
    seed_ids: list[str],
    candidate_ids: set[str],
    node_by_id: dict[str, MemoryNode],
    adjacency: dict[str, list[MemoryEdge]],
    hops: int,
    max_rescue_per_seed: int,
    min_expansion_similarity: float,
) -> list[dict[str, object]]:
    expanded_by_id: dict[str, dict[str, object]] = {}
    for seed_id in seed_ids:
        queue: deque[tuple[str, list[str], int]] = deque([(seed_id, [seed_id], 0)])
        rescued_from_seed = 0
        while queue:
            current_id, path, depth = queue.popleft()
            if depth >= hops:
                continue
            edges = sorted(adjacency.get(current_id, []), key=lambda edge: edge.weight, reverse=True)
            for edge in edges:
                if rescued_from_seed >= max_rescue_per_seed:
                    break
                if edge.target_id not in candidate_ids or edge.target_id in path:
                    continue
                target = node_by_id.get(edge.target_id)
                if target is None:
                    continue
                similarity = cosine_similarity(query_embedding, target.embedding)
                if similarity < min_expansion_similarity:
                    continue
                next_path = [*path, edge.target_id]
                previous = expanded_by_id.get(edge.target_id)
                payload = {
                    "node_id": edge.target_id,
                    "seed_id": seed_id,
                    "path": next_path,
                    "edge_weight": round(edge.weight, 6),
                    "query_similarity": round(similarity, 6),
                    "relation_type": edge.relation_type,
                    "reason": edge.reason,
                }
                if previous is None or float(payload["edge_weight"]) > float(previous["edge_weight"]):
                    expanded_by_id[edge.target_id] = payload
                rescued_from_seed += 1
                queue.append((edge.target_id, next_path, depth + 1))
    return sorted(
        expanded_by_id.values(),
        key=lambda item: (float(item["edge_weight"]), float(item["query_similarity"])),
        reverse=True,
    )


def _adjacency(edges: list[MemoryEdge]) -> dict[str, list[MemoryEdge]]:
    adjacency: dict[str, list[MemoryEdge]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source_id, []).append(edge)
    return adjacency


def _strategy_summary(strategies: dict[str, object]) -> dict[str, object]:
    best_rescue_strategy = ""
    best_recall_gain_strategy = ""
    best_rescue_count = -1
    best_recall_gain = -1.0
    rows: list[dict[str, object]] = []
    for strategy, payload in strategies.items():
        if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
            continue
        metrics = payload["metrics"]
        rescued = int(metrics.get("rescued_support_count", 0))
        recall_gain = float(metrics.get("recall_gain", 0.0))
        row = {
            "strategy": strategy,
            "baseline_evidence_recall": round(float(metrics.get("baseline_evidence_recall", 0.0)), 6),
            "evidence_recall_with_rescue": round(float(metrics.get("evidence_recall_with_rescue", 0.0)), 6),
            "recall_gain": round(recall_gain, 6),
            "rescued_support_count": rescued,
            "rescue_success_query_count": int(metrics.get("rescue_success_query_count", 0)),
            "rescue_precision": round(float(metrics.get("rescue_precision", 0.0)), 6),
        }
        rows.append(row)
        if strategy != "no_graph" and rescued > best_rescue_count:
            best_rescue_count = rescued
            best_rescue_strategy = strategy
        if strategy != "no_graph" and recall_gain > best_recall_gain:
            best_recall_gain = recall_gain
            best_recall_gain_strategy = strategy
    rows.sort(
        key=lambda row: (
            int(row["rescued_support_count"]),
            float(row["recall_gain"]),
            float(row["rescue_precision"]),
        ),
        reverse=True,
    )
    return {
        "best_rescue_strategy": best_rescue_strategy or "no_rescue_strategy",
        "best_recall_gain_strategy": best_recall_gain_strategy or "no_rescue_strategy",
        "selection_rule": "优先选择能在不替换 embedding top-k 的前提下补回更多 gold evidence 的图策略。",
        "ranking": rows,
    }


def _markdown_report(report: dict[str, object]) -> str:
    strategies = report.get("strategies")
    if isinstance(strategies, dict):
        return _strategy_markdown_report(report)
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    lines = [
        "# 图联想补证据实验",
        "",
        "本实验固定保留 embedding top-k，不让图扩展结果替换原始检索结果，只统计图从已激活 seed 出发额外补回的 gold evidence。",
        "",
        "## 总体指标",
        "",
        f"- Query 数：{metrics.get('query_count', 0)}",
        f"- Gold evidence 数：{metrics.get('supporting_evidence_count', 0)}",
        f"- Embedding baseline 命中证据数：{metrics.get('baseline_support_hits', 0)}",
        f"- Baseline evidence recall：{float(metrics.get('baseline_evidence_recall', 0.0)):.4f}",
        f"- 可补证据 query 数：{metrics.get('eligible_query_count', 0)}",
        f"- 图补回证据数：{metrics.get('rescued_support_count', 0)}",
        f"- 图补证据成功 query 数：{metrics.get('rescue_success_query_count', 0)}",
        f"- 图补证据成功率：{float(metrics.get('rescue_success_rate', 0.0)):.4f}",
        f"- 对遗漏证据的补回率：{float(metrics.get('rescue_recall_over_missing', 0.0)):.4f}",
        f"- 图扩展 precision：{float(metrics.get('rescue_precision', 0.0)):.4f}",
        f"- Baseline + graph rescue evidence recall：{float(metrics.get('evidence_recall_with_rescue', 0.0)):.4f}",
        f"- Recall 增益：{float(metrics.get('recall_gain', 0.0)):.4f}",
        "",
        "## 成功案例",
        "",
    ]
    cases = report.get("cases", [])
    success_cases = [
        case for case in cases
        if isinstance(case, dict) and case.get("rescue_success")
    ]
    if not success_cases:
        lines.append("当前配置下没有找到图补回 gold evidence 的成功案例。")
    for index, case in enumerate(success_cases[:10], start=1):
        lines.extend(
            [
                f"### Case {index}: {case.get('query_id', '')}",
                "",
                f"- 问题：{case.get('question', '')}",
                f"- Baseline 命中节点：`{case.get('baseline_hit_node_ids', [])}`",
                f"- Baseline 漏掉证据：`{case.get('missing_support_node_ids', [])}`",
                f"- 图补回证据：`{case.get('rescued_support_node_ids', [])}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _strategy_markdown_report(report: dict[str, object]) -> str:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    lines = [
        "# 图联想补证据策略实验",
        "",
        "本实验固定保留 embedding top-k，不让图扩展结果替换原始检索结果，只统计图从已激活 seed 出发额外补回的 gold evidence。",
        "",
        f"最佳补证据策略：{summary.get('best_rescue_strategy', '')}",
        f"最高 recall 增益策略：{summary.get('best_recall_gain_strategy', '')}",
        "",
        "## 策略对比",
        "",
        "| 策略 | Baseline Recall | Rescue 后 Recall | Recall 增益 | 可补证据 Query | 补回证据数 | 成功 Query | 补证据成功率 | 补回遗漏证据率 | 图扩展 Precision | 扩展节点数 | 边数 | 候选对数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    strategies = report.get("strategies", {})
    if isinstance(strategies, dict):
        for strategy, payload in strategies.items():
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics", {})
            cost = payload.get("cost", {})
            if not isinstance(metrics, dict) or not isinstance(cost, dict):
                continue
            lines.append(
                "| "
                f"{strategy} | "
                f"{float(metrics.get('baseline_evidence_recall', 0.0)):.4f} | "
                f"{float(metrics.get('evidence_recall_with_rescue', 0.0)):.4f} | "
                f"{float(metrics.get('recall_gain', 0.0)):.4f} | "
                f"{int(metrics.get('eligible_query_count', 0))} | "
                f"{int(metrics.get('rescued_support_count', 0))} | "
                f"{int(metrics.get('rescue_success_query_count', 0))} | "
                f"{float(metrics.get('rescue_success_rate', 0.0)):.4f} | "
                f"{float(metrics.get('rescue_recall_over_missing', 0.0)):.4f} | "
                f"{float(metrics.get('rescue_precision', 0.0)):.4f} | "
                f"{int(metrics.get('expanded_node_count', 0))} | "
                f"{int(cost.get('edge_count', 0))} | "
                f"{int(cost.get('candidate_pair_count', 0))} |"
            )
    lines.extend(["", "## 成功案例", ""])
    if isinstance(strategies, dict):
        for strategy, payload in strategies.items():
            if strategy == "no_graph" or not isinstance(payload, dict):
                continue
            success_cases = [
                case for case in payload.get("cases", [])
                if isinstance(case, dict) and case.get("rescue_success")
            ]
            if not success_cases:
                continue
            lines.extend([f"### {strategy}", ""])
            for index, case in enumerate(success_cases[:5], start=1):
                lines.extend(
                    [
                        f"{index}. Query `{case.get('query_id', '')}`",
                        f"   - 问题：{case.get('question', '')}",
                        f"   - Baseline 漏掉证据：`{case.get('missing_support_node_ids', [])}`",
                        f"   - 图补回证据：`{case.get('rescued_support_node_ids', [])}`",
                    ]
                )
    return "\n".join(lines) + "\n"
