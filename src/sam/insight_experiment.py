from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

from sam.models import EvaluationQuery, MemoryNode
from sam.store import MemoryStore
from sam.text import stable_id


@dataclass(frozen=True, slots=True)
class ReconstructionGroup:
    """一个重构策略形成的高层记忆分组。"""

    strategy: str
    group_id: str
    label: str
    source_consolidated_node_ids: list[str]
    evidence_node_ids: list[str]
    source_query_ids: list[str]
    source_answers: list[str]
    shared_keywords: list[str]
    score: float


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


def compare_insight_reconstruction_strategies(
    *,
    store: MemoryStore,
    queries: list[EvaluationQuery],
    dataset: str | None = None,
    max_cases: int = 8,
    embedding_threshold: float = 0.82,
    hybrid_threshold: float = 0.34,
) -> dict[str, Any]:
    """在同一批巩固记忆上比较不同高层记忆重构策略。

    这个实验固定前置检索和巩固记忆，只改变“如何从巩固记忆重构高层洞察”。
    因此它适合回答：SAM 的动态重构是否比不重构、逐条保存或简单聚类更有价值。
    """

    nodes = store.get_nodes()
    node_by_id = {node.id: node for node in nodes}
    original_to_node = {
        str(node.metadata["original_doc_id"]): node.id
        for node in nodes
        if "original_doc_id" in node.metadata
    }
    consolidated_nodes = [
        node
        for node in nodes
        if node.metadata.get("node_type") == "consolidated_memory"
        and (dataset is None or node.metadata.get("dataset") == dataset)
    ]
    consolidated_nodes = sorted(consolidated_nodes, key=lambda node: (node.created_at, node.id))
    support_node_ids = {
        original_to_node[doc_id]
        for query in queries
        for doc_id in query.supporting_doc_ids
        if doc_id in original_to_node
    }
    query_support_node_ids = {
        query.id: [
            original_to_node[doc_id]
            for doc_id in query.supporting_doc_ids
            if doc_id in original_to_node
        ]
        for query in queries
    }
    context = {
        "node_by_id": node_by_id,
        "support_node_ids": support_node_ids,
        "query_support_node_ids": query_support_node_ids,
    }

    strategy_outputs: dict[str, tuple[list[ReconstructionGroup], dict[str, Any]]] = {}
    builders = [
        ("no_reconstruction", _build_no_reconstruction_groups),
        ("flat_consolidated", _build_flat_consolidated_groups),
        ("keyword_cluster", _build_keyword_cluster_groups),
        ("embedding_cluster", lambda nodes_: _build_embedding_cluster_groups(nodes_, threshold=embedding_threshold)),
        (
            "sam_hybrid_reconstruction",
            lambda nodes_: _build_sam_hybrid_groups(nodes_, threshold=hybrid_threshold),
        ),
    ]

    for strategy, builder in builders:
        started = time.perf_counter()
        groups, cost = builder(consolidated_nodes)
        build_time_ms = (time.perf_counter() - started) * 1000.0
        strategy_outputs[strategy] = (
            groups,
            {
                **cost,
                "build_time_ms": build_time_ms,
                "embedding_threshold": embedding_threshold,
                "hybrid_threshold": hybrid_threshold,
            },
        )

    strategy_metrics = {
        strategy: _reconstruction_strategy_metrics(
            strategy=strategy,
            groups=groups,
            consolidated_nodes=consolidated_nodes,
            context=context,
            cost=cost,
        )
        for strategy, (groups, cost) in strategy_outputs.items()
    }
    return {
        "dataset": dataset or _infer_dataset(consolidated_nodes, queries),
        "query_count": len(queries),
        "consolidated_memory_count": len(consolidated_nodes),
        "support_node_count": len(support_node_ids),
        "strategies": strategy_metrics,
        "strategy_rankings": _strategy_rankings(strategy_metrics),
        "cases": _reconstruction_comparison_cases(
            queries=queries,
            groups_by_strategy={strategy: groups for strategy, (groups, _) in strategy_outputs.items()},
            context=context,
            max_cases=max_cases,
        ),
    }


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


def write_insight_reconstruction_comparison_reports(
    *,
    output_dir: str | Path,
    comparison: dict[str, Any],
    warmup_metrics: dict[str, Any],
) -> tuple[Path, Path]:
    """写出高层记忆重构对照实验报告。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "comparison": comparison,
        "warmup_metrics": warmup_metrics,
    }
    json_path = target / "insight_reconstruction_comparison.json"
    markdown_path = target / "insight_reconstruction_comparison.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_comparison_markdown(payload), encoding="utf-8")
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


def _comparison_markdown(payload: dict[str, Any]) -> str:
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    strategies = comparison.get("strategies", {})
    assert isinstance(strategies, dict)
    lines = [
        "# SAM 高层记忆重构对照实验",
        "",
        "## 实验目的",
        "",
        "本实验固定底层检索和单次巩固记忆，只比较不同高层记忆重构策略。",
        "重点观察重构是否同时带来压缩、证据保真、查询级回溯和合理构建成本。",
        "",
        "## 对照策略",
        "",
        "- `no_reconstruction`：不做高层重构，只保留底层巩固记忆。",
        "- `flat_consolidated`：每条巩固记忆单独作为一个高层单元，不压缩。",
        "- `keyword_cluster`：按共享关键词聚合巩固记忆。",
        "- `embedding_cluster`：按巩固记忆向量相似度聚合。",
        "- `sam_hybrid_reconstruction`：综合语义相似、关键词重叠、证据重叠和答案一致性形成高层洞察。",
        "",
        "## 指标说明",
        "",
        "- 压缩率：巩固记忆数量 / 重构后记忆单元数量，越高表示高层重构越能减少记忆冗余。",
        "- 支持证据回溯率：高层记忆覆盖标准支持证据节点的比例。",
        "- 查询完整回溯率：一个问题的全部支持证据都能被高层记忆追溯到的比例。",
        "- 答案一致性：同一高层记忆中来源问题答案是否集中，越高表示聚合更稳。",
        "- 冗余率：同一证据被多个高层记忆重复覆盖的程度，越低越好。",
        "- 检索单元减少率：高层重构后需要优先检索的记忆单元减少比例。",
        "- Trace边减少率：高层记忆到证据的回溯边相对原始巩固记忆证据边的减少比例。",
        "- Trace噪声率：高层记忆回溯到非标准支持证据的比例，越低越好。",
        "- Query级Trace噪声率：单个 query 命中高层记忆后暴露的额外证据比例，越低越好。",
        "- 有效Trace精度：高层记忆回溯证据中属于标准支持证据的比例，越高越好。",
        "- 质量成本综合分：综合证据保真、查询完整回溯、答案一致性、压缩率和构建耗时后的阶段性评分。",
        "",
        "## 结果表",
        "",
        "| 策略 | 高层单元数 | 压缩率 | 检索单元减少率 | 支持证据回溯率 | 查询完整回溯率 | 平均暴露证据数 | Query级Trace噪声率 | Trace边数 | Trace噪声率 | 有效Trace精度 | 构建耗时ms | 质量成本综合分 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, metrics in strategies.items():
        if not isinstance(metrics, dict):
            continue
        lines.append(
            "| {strategy} | {units} | {compression:.3f} | {unit_reduction:.3f} | {support:.3f} | {query_full:.3f} | "
            "{exposed:.2f} | {query_noise:.3f} | {trace_edges} | {trace_noise:.3f} | {trace_precision:.3f} | "
            "{time_ms:.3f} | {score:.3f} |".format(
                strategy=strategy,
                units=metrics.get("reconstructed_unit_count", 0),
                compression=float(metrics.get("compression_ratio", 0.0)),
                unit_reduction=float(metrics.get("retrieval_unit_reduction_rate", 0.0)),
                support=float(metrics.get("support_trace_rate", 0.0)),
                query_full=float(metrics.get("query_full_trace_rate", 0.0)),
                exposed=float(metrics.get("average_exposed_evidence_per_query", 0.0)),
                query_noise=float(metrics.get("query_trace_noise_rate", 0.0)),
                trace_edges=int(metrics.get("trace_edge_count", 0)),
                trace_noise=float(metrics.get("trace_noise_rate", 0.0)),
                trace_precision=float(metrics.get("effective_trace_precision", 0.0)),
                time_ms=float(metrics.get("build_time_ms", 0.0)),
                score=float(metrics.get("quality_cost_score", 0.0)),
            )
        )

    lines.extend(["", "## 排名", ""])
    rankings = comparison.get("strategy_rankings", {})
    if isinstance(rankings, dict):
        for metric, ranking in rankings.items():
            if not isinstance(ranking, list):
                continue
            readable = " > ".join(
                f"{item.get('strategy')}({float(item.get('value', 0.0)):.3f})"
                for item in ranking
                if isinstance(item, dict)
            )
            lines.append(f"- {metric}：{readable}")

    lines.extend(["", "## 典型高层记忆分组", ""])
    cases = comparison.get("cases", {})
    if isinstance(cases, dict):
        for case in cases.get("insight_groups", [])[:5]:
            if not isinstance(case, dict):
                continue
            lines.extend(
                [
                    f"### {case.get('strategy')} / {case.get('group_id')}",
                    "",
                    f"- 标签：{case.get('label')}",
                    f"- 来源巩固记忆数：{case.get('source_consolidated_count')}",
                    f"- 证据数量：{case.get('evidence_count')}",
                    f"- 共享关键词：{', '.join(str(item) for item in case.get('shared_keywords', [])[:8])}",
                    f"- 证据标题：{', '.join(str(item) for item in case.get('evidence_titles', [])[:8])}",
                    "",
                ]
            )

        lines.extend(["## 查询级回溯案例", ""])
        for case in cases.get("query_traces", [])[:5]:
            if not isinstance(case, dict):
                continue
            lines.extend(
                [
                    f"### {case.get('query_id')}",
                    "",
                    f"- 问题：{case.get('question')}",
                    f"- 标准答案：{case.get('answer')}",
                    f"- 各策略回溯情况：{json.dumps(case.get('strategy_trace_rates', {}), ensure_ascii=False)}",
                    "",
                ]
            )
    return "\n".join(lines)


def _build_no_reconstruction_groups(
    consolidated_nodes: list[MemoryNode],
) -> tuple[list[ReconstructionGroup], dict[str, Any]]:
    return [], {
        "candidate_pair_count": 0,
        "accepted_pair_count": 0,
        "strategy_cost_note": "no_high_level_reconstruction",
    }


def _build_flat_consolidated_groups(
    consolidated_nodes: list[MemoryNode],
) -> tuple[list[ReconstructionGroup], dict[str, Any]]:
    groups = [
        _make_reconstruction_group(
            strategy="flat_consolidated",
            label=str(node.metadata.get("query_id") or node.id),
            nodes=[node],
            score=node.confidence,
        )
        for node in consolidated_nodes
    ]
    return groups, {
        "candidate_pair_count": 0,
        "accepted_pair_count": len(groups),
        "strategy_cost_note": "one_group_per_consolidated_memory",
    }


def _build_keyword_cluster_groups(
    consolidated_nodes: list[MemoryNode],
) -> tuple[list[ReconstructionGroup], dict[str, Any]]:
    keyword_counts: dict[str, int] = {}
    for node in consolidated_nodes:
        for keyword in _content_keywords(node, limit=20):
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

    buckets: dict[str, list[MemoryNode]] = {}
    for node in consolidated_nodes:
        keywords = _content_keywords(node, limit=20)
        label = next(
            (keyword for keyword in keywords if keyword_counts.get(keyword, 0) >= 2),
            keywords[0] if keywords else "general",
        )
        buckets.setdefault(label, []).append(node)
    groups = [
        _make_reconstruction_group(
            strategy="keyword_cluster",
            label=label,
            nodes=nodes,
            score=float(len(nodes)),
        )
        for label, nodes in sorted(buckets.items())
        if nodes
    ]
    return groups, {
        "candidate_pair_count": len(consolidated_nodes),
        "accepted_pair_count": len(groups),
        "strategy_cost_note": "primary_keyword_bucket",
    }


def _build_embedding_cluster_groups(
    consolidated_nodes: list[MemoryNode],
    *,
    threshold: float,
) -> tuple[list[ReconstructionGroup], dict[str, Any]]:
    clusters: list[list[MemoryNode]] = []
    candidate_pairs = 0
    accepted_pairs = 0
    for node in consolidated_nodes:
        best_index = -1
        best_score = -1.0
        for index, cluster in enumerate(clusters):
            candidate_pairs += 1
            score = _cosine(node.embedding, _cluster_centroid(cluster))
            if score > best_score:
                best_index = index
                best_score = score
        if best_index >= 0 and best_score >= threshold:
            clusters[best_index].append(node)
            accepted_pairs += 1
        else:
            clusters.append([node])
    groups = [
        _make_reconstruction_group(
            strategy="embedding_cluster",
            label=f"embedding_cluster_{index}",
            nodes=cluster,
            score=_average_pairwise_similarity(cluster),
        )
        for index, cluster in enumerate(clusters)
    ]
    return groups, {
        "candidate_pair_count": candidate_pairs,
        "accepted_pair_count": accepted_pairs,
        "strategy_cost_note": "greedy_embedding_centroid_cluster",
    }


def _build_sam_hybrid_groups(
    consolidated_nodes: list[MemoryNode],
    *,
    threshold: float,
) -> tuple[list[ReconstructionGroup], dict[str, Any]]:
    pair_scores: dict[tuple[str, str], float] = {}
    candidate_pair_ids = _hybrid_candidate_pair_ids(consolidated_nodes)
    candidate_pairs = len(candidate_pair_ids)
    accepted_pairs = 0
    adjacency: dict[str, set[str]] = {node.id: set() for node in consolidated_nodes}
    node_by_id = {node.id: node for node in consolidated_nodes}
    for left_id, right_id in candidate_pair_ids:
        left = node_by_id[left_id]
        right = node_by_id[right_id]
        score = _hybrid_reconstruction_score(left, right)
        pair_scores[(left.id, right.id)] = score
        if score >= threshold:
            adjacency[left.id].add(right.id)
            adjacency[right.id].add(left.id)
            accepted_pairs += 1

    components = _connected_components(adjacency)
    groups: list[ReconstructionGroup] = []
    for index, component in enumerate(components):
        component_nodes = [node_by_id[node_id] for node_id in component]
        label = _group_label(component_nodes) or f"hybrid_component_{index}"
        groups.append(
            _make_reconstruction_group(
                strategy="sam_hybrid_reconstruction",
                label=label,
                nodes=component_nodes,
                score=_average_connected_score(component, pair_scores),
            )
        )
    return groups, {
        "candidate_pair_count": candidate_pairs,
        "accepted_pair_count": accepted_pairs,
        "strategy_cost_note": "keyword_blocked_semantic_evidence_answer_components",
    }


def _hybrid_candidate_pair_ids(consolidated_nodes: list[MemoryNode]) -> list[tuple[str, str]]:
    """用关键词倒排生成候选对，避免高层重构阶段全量两两比较。"""

    keyword_to_node_ids: dict[str, list[str]] = {}
    for node in consolidated_nodes:
        for keyword in _content_keywords(node, limit=8):
            keyword_to_node_ids.setdefault(keyword, []).append(node.id)

    candidate_pairs: set[tuple[str, str]] = set()
    for node_ids in keyword_to_node_ids.values():
        unique_ids = sorted(set(node_ids))
        if len(unique_ids) < 2:
            continue
        for left_index, left_id in enumerate(unique_ids):
            for right_id in unique_ids[left_index + 1:]:
                candidate_pairs.add((left_id, right_id))
    return sorted(candidate_pairs)


def _make_reconstruction_group(
    *,
    strategy: str,
    label: str,
    nodes: list[MemoryNode],
    score: float,
) -> ReconstructionGroup:
    return ReconstructionGroup(
        strategy=strategy,
        group_id=f"{strategy}:{stable_group_id(label, nodes)}",
        label=label,
        source_consolidated_node_ids=[node.id for node in nodes],
        evidence_node_ids=_unique_strings(
            evidence_id
            for node in nodes
            for evidence_id in node.metadata.get("evidence_node_ids", [])
        ),
        source_query_ids=_unique_strings(
            node.metadata.get("query_id")
            for node in nodes
            if node.metadata.get("query_id")
        ),
        source_answers=_unique_strings(
            node.metadata.get("answer")
            for node in nodes
            if node.metadata.get("answer")
        ),
        shared_keywords=_shared_group_keywords(nodes, limit=10),
        score=float(score),
    )


def _reconstruction_strategy_metrics(
    *,
    strategy: str,
    groups: list[ReconstructionGroup],
    consolidated_nodes: list[MemoryNode],
    context: dict[str, Any],
    cost: dict[str, Any],
) -> dict[str, Any]:
    consolidated_count = len(consolidated_nodes)
    reconstructed_units = consolidated_count if strategy == "no_reconstruction" else len(groups)
    reconstructed_units = reconstructed_units or 0
    consolidated_ids = {node.id for node in consolidated_nodes}
    consolidated_evidence_ids = {
        str(evidence_id)
        for node in consolidated_nodes
        for evidence_id in node.metadata.get("evidence_node_ids", [])
    }
    grouped_consolidated_ids = {
        node_id
        for group in groups
        for node_id in group.source_consolidated_node_ids
    }
    grouped_evidence_ids = {
        node_id
        for group in groups
        for node_id in group.evidence_node_ids
    }
    support_node_ids = set(context["support_node_ids"])
    query_support_node_ids = context["query_support_node_ids"]
    assert isinstance(query_support_node_ids, dict)
    query_trace_rates = [
        _safe_divide(len(set(support_ids) & grouped_evidence_ids), len(support_ids))
        for support_ids in query_support_node_ids.values()
        if support_ids
    ]
    query_trace_budget = _query_trace_budget(groups, query_support_node_ids)
    query_full_trace_rate = _safe_divide(
        sum(1 for value in query_trace_rates if value >= 1.0),
        len(query_trace_rates),
    )
    answer_consistency = _average(_group_answer_consistency(group) for group in groups)
    pairwise_similarity = _average(
        _average_pairwise_similarity(
            [
                node
                for node in consolidated_nodes
                if node.id in group.source_consolidated_node_ids
            ]
        )
        for group in groups
    )
    total_group_evidence_mentions = sum(len(group.evidence_node_ids) for group in groups)
    duplicate_evidence_mentions = max(0, total_group_evidence_mentions - len(grouped_evidence_ids))
    redundancy_rate = _safe_divide(duplicate_evidence_mentions, total_group_evidence_mentions)
    raw_trace_edge_count = sum(
        len(node.metadata.get("evidence_node_ids", []))
        for node in consolidated_nodes
    )
    trace_edge_count = total_group_evidence_mentions
    trace_noise_count = len(grouped_evidence_ids - support_node_ids)
    trace_noise_rate = _safe_divide(trace_noise_count, len(grouped_evidence_ids))
    effective_trace_precision = _safe_divide(
        len(grouped_evidence_ids & support_node_ids),
        len(grouped_evidence_ids),
    )
    compression_ratio = _safe_divide(consolidated_count, reconstructed_units)
    retrieval_unit_reduction_rate = 1.0 - _safe_divide(reconstructed_units, consolidated_count)
    trace_edge_reduction_rate = 1.0 - _safe_divide(trace_edge_count, raw_trace_edge_count)
    if strategy == "no_reconstruction":
        retrieval_unit_reduction_rate = 0.0
        trace_edge_reduction_rate = 0.0
        trace_noise_rate = 0.0
        effective_trace_precision = 0.0
        query_trace_budget = {
            **query_trace_budget,
            "query_trace_noise_rate": 0.0,
            "query_effective_trace_precision": 0.0,
            "average_exposed_evidence_per_query": 0.0,
            "average_noise_evidence_per_query": 0.0,
        }
    support_trace_rate = _safe_divide(len(grouped_evidence_ids & support_node_ids), len(support_node_ids))
    evidence_coverage_rate = _safe_divide(
        len(grouped_evidence_ids & consolidated_evidence_ids),
        len(consolidated_evidence_ids),
    )
    quality = (
        0.34 * support_trace_rate
        + 0.24 * query_full_trace_rate
        + 0.18 * answer_consistency
        + 0.14 * evidence_coverage_rate
        + 0.10 * min(1.0, math.log2(1.0 + max(0.0, compression_ratio)) / 3.0)
    )
    time_penalty = 1.0 + float(cost.get("build_time_ms", 0.0)) / 1000.0
    redundancy_penalty = 1.0 + redundancy_rate
    trace_noise_penalty = 1.0 + trace_noise_rate + float(query_trace_budget["query_trace_noise_rate"])
    quality_cost_score = quality / (time_penalty * redundancy_penalty * trace_noise_penalty)
    if strategy == "no_reconstruction":
        quality_cost_score = 0.0
    return {
        "strategy": strategy,
        "consolidated_memory_count": consolidated_count,
        "reconstructed_unit_count": reconstructed_units,
        "insight_count": len(groups) if strategy != "no_reconstruction" else 0,
        "compression_ratio": compression_ratio,
        "source_consolidated_covered_count": len(grouped_consolidated_ids & consolidated_ids),
        "source_consolidated_coverage_rate": _safe_divide(
            len(grouped_consolidated_ids & consolidated_ids),
            len(consolidated_ids),
        ),
        "unique_consolidated_evidence_count": len(consolidated_evidence_ids),
        "unique_reconstructed_evidence_count": len(grouped_evidence_ids),
        "evidence_coverage_rate": evidence_coverage_rate,
        "support_node_count": len(support_node_ids),
        "support_nodes_traced_count": len(grouped_evidence_ids & support_node_ids),
        "support_trace_rate": support_trace_rate,
        "query_partial_trace_rate": _average(query_trace_rates),
        "query_full_trace_rate": query_full_trace_rate,
        "answer_consistency": answer_consistency,
        "average_pairwise_similarity": pairwise_similarity,
        "average_group_size": _average(len(group.source_consolidated_node_ids) for group in groups),
        "average_evidence_per_group": _average(len(group.evidence_node_ids) for group in groups),
        "evidence_redundancy_rate": redundancy_rate,
        "raw_trace_edge_count": raw_trace_edge_count,
        "trace_edge_count": trace_edge_count,
        "trace_edge_reduction_rate": trace_edge_reduction_rate,
        "retrieval_unit_reduction_rate": retrieval_unit_reduction_rate,
        "trace_noise_count": trace_noise_count,
        "trace_noise_rate": trace_noise_rate,
        "effective_trace_precision": effective_trace_precision,
        **query_trace_budget,
        "quality_cost_score": quality_cost_score,
        **cost,
    }


def _query_trace_budget(
    groups: list[ReconstructionGroup],
    query_support_node_ids: dict[str, list[str]],
) -> dict[str, float]:
    """按 query 统计高层记忆命中后会暴露多少额外证据。

    全局 trace 噪声只能说明高层记忆是否最终仍能回到 gold evidence。
    但如果一个高层记忆把大量 query 压在一起，单个 query 命中该高层记忆时会暴露很多额外证据。
    这个函数用 source_query_ids 模拟压缩单元被对应 query 访问后的证据暴露规模。
    """

    groups_by_query: dict[str, list[ReconstructionGroup]] = {}
    for group in groups:
        for query_id in group.source_query_ids:
            groups_by_query.setdefault(query_id, []).append(group)

    exposed_counts: list[float] = []
    noise_counts: list[float] = []
    trace_precisions: list[float] = []
    trace_noise_rates: list[float] = []
    for query_id, support_ids in query_support_node_ids.items():
        support_set = {str(item) for item in support_ids}
        if not support_set:
            continue
        exposed_ids = {
            evidence_id
            for group in groups_by_query.get(query_id, [])
            for evidence_id in group.evidence_node_ids
        }
        if not exposed_ids:
            exposed_counts.append(0.0)
            noise_counts.append(0.0)
            trace_precisions.append(0.0)
            trace_noise_rates.append(0.0)
            continue
        noise_count = len(exposed_ids - support_set)
        exposed_counts.append(float(len(exposed_ids)))
        noise_counts.append(float(noise_count))
        trace_precisions.append(_safe_divide(len(exposed_ids & support_set), len(exposed_ids)))
        trace_noise_rates.append(_safe_divide(noise_count, len(exposed_ids)))

    return {
        "average_exposed_evidence_per_query": _average(exposed_counts),
        "average_noise_evidence_per_query": _average(noise_counts),
        "query_effective_trace_precision": _average(trace_precisions),
        "query_trace_noise_rate": _average(trace_noise_rates),
    }


def _strategy_rankings(strategy_metrics: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ranking_metrics = [
        "quality_cost_score",
        "support_trace_rate",
        "query_full_trace_rate",
        "compression_ratio",
        "answer_consistency",
    ]
    rankings: dict[str, list[dict[str, Any]]] = {}
    for metric in ranking_metrics:
        ranked = sorted(
            (
                {"strategy": strategy, "value": float(metrics.get(metric, 0.0))}
                for strategy, metrics in strategy_metrics.items()
            ),
            key=lambda item: (-item["value"], item["strategy"]),
        )
        rankings["balanced_score" if metric == "quality_cost_score" else metric] = ranked
    return rankings


def _reconstruction_comparison_cases(
    *,
    queries: list[EvaluationQuery],
    groups_by_strategy: dict[str, list[ReconstructionGroup]],
    context: dict[str, Any],
    max_cases: int,
) -> dict[str, Any]:
    node_by_id = context["node_by_id"]
    assert isinstance(node_by_id, dict)
    query_support_node_ids = context["query_support_node_ids"]
    assert isinstance(query_support_node_ids, dict)
    selected_groups: list[dict[str, Any]] = []
    for strategy in [
        "sam_hybrid_reconstruction",
        "embedding_cluster",
        "keyword_cluster",
        "flat_consolidated",
    ]:
        for group in groups_by_strategy.get(strategy, [])[:max_cases]:
            selected_groups.append(
                {
                    "strategy": group.strategy,
                    "group_id": group.group_id,
                    "label": group.label,
                    "score": group.score,
                    "source_consolidated_count": len(group.source_consolidated_node_ids),
                    "source_query_ids": group.source_query_ids,
                    "source_answers": group.source_answers,
                    "evidence_count": len(group.evidence_node_ids),
                    "evidence_titles": [
                        _node_title(node_by_id[evidence_id])
                        for evidence_id in group.evidence_node_ids
                        if evidence_id in node_by_id
                    ],
                    "shared_keywords": group.shared_keywords,
                }
            )
            if len(selected_groups) >= max_cases:
                break
        if len(selected_groups) >= max_cases:
            break

    query_cases: list[dict[str, Any]] = []
    for query in queries[:max_cases]:
        support_ids = set(query_support_node_ids.get(query.id, []))
        strategy_trace_rates: dict[str, float] = {}
        strategy_traced_titles: dict[str, list[str]] = {}
        for strategy, groups in groups_by_strategy.items():
            traced_ids = support_ids & {
                evidence_id
                for group in groups
                for evidence_id in group.evidence_node_ids
            }
            strategy_trace_rates[strategy] = _safe_divide(len(traced_ids), len(support_ids))
            strategy_traced_titles[strategy] = [
                _node_title(node_by_id[evidence_id])
                for evidence_id in sorted(traced_ids)
                if evidence_id in node_by_id
            ]
        query_cases.append(
            {
                "query_id": query.id,
                "question": query.question,
                "answer": query.answer,
                "support_node_ids": sorted(support_ids),
                "strategy_trace_rates": strategy_trace_rates,
                "strategy_traced_titles": strategy_traced_titles,
            }
        )
    return {
        "insight_groups": selected_groups,
        "query_traces": query_cases,
    }


def _node_title(node: MemoryNode) -> str:
    return str(node.metadata.get("title") or node.summary or node.id)


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _average(values: Any) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0


def stable_group_id(label: str, nodes: list[MemoryNode]) -> str:
    raw = "|".join([label, *sorted(node.id for node in nodes)])
    return stable_id("group", raw).removeprefix("group_")


def _unique_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _infer_dataset(consolidated_nodes: list[MemoryNode], queries: list[EvaluationQuery]) -> str:
    if consolidated_nodes:
        return str(consolidated_nodes[0].metadata.get("dataset") or "unknown")
    if queries:
        return queries[0].dataset
    return "unknown"


def _content_keywords(node: MemoryNode, *, limit: int = 10) -> list[str]:
    keywords: list[str] = []
    for keyword in node.keywords:
        text = str(keyword).strip()
        if not text:
            continue
        if text.lower() in _RECONSTRUCTION_STOP_KEYWORDS:
            continue
        if text not in keywords:
            keywords.append(text)
        if len(keywords) >= limit:
            break
    return keywords


def _shared_group_keywords(nodes: list[MemoryNode], *, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for node in nodes:
        for keyword in _content_keywords(node, limit=20):
            counts[keyword] = counts.get(keyword, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [keyword for keyword, _ in ranked[:limit]]


def _group_label(nodes: list[MemoryNode]) -> str:
    keywords = _shared_group_keywords(nodes, limit=1)
    if keywords:
        return keywords[0]
    answers = _unique_strings(
        node.metadata.get("answer")
        for node in nodes
        if node.metadata.get("answer")
    )
    return answers[0] if answers else "general"


def _cluster_centroid(nodes: list[MemoryNode]) -> list[float]:
    valid_nodes = [node for node in nodes if node.embedding]
    if not valid_nodes:
        return []
    dimensions = min(len(node.embedding) for node in valid_nodes)
    centroid = [0.0] * dimensions
    for node in valid_nodes:
        for index, value in enumerate(node.embedding[:dimensions]):
            centroid[index] += float(value)
    centroid = [value / len(valid_nodes) for value in centroid]
    norm = math.sqrt(sum(value * value for value in centroid))
    if norm == 0.0:
        return centroid
    return [value / norm for value in centroid]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dimensions = min(len(left), len(right))
    numerator = sum(float(left[index]) * float(right[index]) for index in range(dimensions))
    left_norm = math.sqrt(sum(float(left[index]) ** 2 for index in range(dimensions)))
    right_norm = math.sqrt(sum(float(right[index]) ** 2 for index in range(dimensions)))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def _hybrid_reconstruction_score(left: MemoryNode, right: MemoryNode) -> float:
    semantic_score = max(0.0, _cosine(left.embedding, right.embedding))
    left_keywords = set(_content_keywords(left, limit=16))
    right_keywords = set(_content_keywords(right, limit=16))
    keyword_score = _jaccard(left_keywords, right_keywords)
    evidence_score = _jaccard(
        {str(item) for item in left.metadata.get("evidence_node_ids", [])},
        {str(item) for item in right.metadata.get("evidence_node_ids", [])},
    )
    answer_score = 1.0 if left.metadata.get("answer") and left.metadata.get("answer") == right.metadata.get("answer") else 0.0
    return (
        0.35 * semantic_score
        + 0.35 * keyword_score
        + 0.20 * evidence_score
        + 0.10 * answer_score
    )


def _connected_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    remaining = set(adjacency)
    components: list[list[str]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[str] = set()
        remaining.remove(start)
        while stack:
            node_id = stack.pop()
            component.add(node_id)
            for neighbor in adjacency.get(node_id, set()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda item: (-len(item), item[0] if item else ""))


def _average_connected_score(
    component: list[str],
    pair_scores: dict[tuple[str, str], float],
) -> float:
    if len(component) <= 1:
        return 1.0
    scores: list[float] = []
    for left_index, left in enumerate(component):
        for right in component[left_index + 1:]:
            scores.append(pair_scores.get((left, right), pair_scores.get((right, left), 0.0)))
    return _average(scores)


def _average_pairwise_similarity(nodes: list[MemoryNode]) -> float:
    if len(nodes) <= 1:
        return 1.0 if nodes else 0.0
    scores: list[float] = []
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1:]:
            scores.append(max(0.0, _cosine(left.embedding, right.embedding)))
    return _average(scores)


def _group_answer_consistency(group: ReconstructionGroup) -> float:
    answers = group.source_answers
    if len(group.source_consolidated_node_ids) <= 1:
        return 1.0 if group.source_consolidated_node_ids else 0.0
    if not answers:
        return 0.0
    return 1.0 if len(answers) == 1 else 1.0 / len(answers)


_RECONSTRUCTION_STOP_KEYWORDS = {
    "问题",
    "答案",
    "检索方法",
    "答案状态",
    "支持证据",
    "证据",
    "evidence",
    "support",
    "claim",
    "sam",
    "sam_full",
    "sam_no_graph",
    "embedding_topk",
    "found",
    "context",
    "question",
    "answer",
    "method",
    "able",
    "about",
    "after",
    "also",
    "another",
    "became",
    "been",
    "being",
    "between",
    "called",
    "came",
    "could",
    "during",
    "first",
    "from",
    "have",
    "into",
    "made",
    "many",
    "more",
    "most",
    "other",
    "over",
    "same",
    "some",
    "than",
    "that",
    "their",
    "then",
    "there",
    "these",
    "this",
    "through",
    "under",
    "used",
    "were",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}
