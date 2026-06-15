from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import deque

from sam.embedding import EmbeddingProvider
from sam.models import EvaluationQuery, MemoryEdge, MemoryNode, RetrievalHit, utc_now_iso
from sam.text import cosine_similarity


SUPPORTED_GRAPH_STRATEGIES = {
    "no_graph",
    "semantic_only",
    "position_only",
    "cam_style",
    "context_path_only",
    "sam_context",
}


@dataclass(frozen=True, slots=True)
class GraphStrategyConfig:
    """非 LLM 建图策略配置。"""

    strategy: str
    alpha: float = 0.55
    top_k_edges: int = 4
    threshold: float = 0.18

    def __post_init__(self) -> None:
        if self.strategy not in SUPPORTED_GRAPH_STRATEGIES:
            raise ValueError(f"未知建图策略：{self.strategy}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha 必须位于 [0, 1]")
        if self.top_k_edges < 0:
            raise ValueError("top_k_edges 不能小于 0")


@dataclass(slots=True)
class GraphBuildResult:
    """一次建图策略的边与成本统计。"""

    strategy: str
    edges: list[MemoryEdge]
    candidate_pair_count: int
    build_time_seconds: float
    average_edge_score: float
    config: GraphStrategyConfig

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def average_edges_per_node(self) -> float:
        node_ids = {
            node_id
            for edge in self.edges
            for node_id in [edge.source_id, edge.target_id]
        }
        return len(self.edges) / len(node_ids) if node_ids else 0.0

    def cost_payload(self) -> dict[str, object]:
        return {
            "candidate_pair_count": self.candidate_pair_count,
            "edge_count": self.edge_count,
            "average_edges_per_node": round(self.average_edges_per_node, 4),
            "average_edge_score": round(self.average_edge_score, 4),
            "build_time_seconds": round(self.build_time_seconds, 6),
            "uses_llm": False,
        }


class GraphStrategyExperiment:
    """比较多种非 LLM 建图公式的效果和成本。"""

    def __init__(
        self,
        nodes: list[MemoryNode],
        queries: list[EvaluationQuery],
        alpha: float = 0.55,
        top_k_edges: int = 4,
        threshold: float = 0.18,
    ) -> None:
        self.nodes = nodes
        self.queries = queries
        self.alpha = alpha
        self.top_k_edges = top_k_edges
        self.threshold = threshold

    def compare_build_strategies(
        self,
        strategies: list[str],
    ) -> dict[str, GraphBuildResult]:
        return {
            strategy: build_graph_for_strategy(
                self.nodes,
                GraphStrategyConfig(
                    strategy=strategy,
                    alpha=self.alpha,
                    top_k_edges=self.top_k_edges,
                    threshold=self.threshold,
                ),
            )
            for strategy in strategies
        }

    def run(
        self,
        strategies: list[str],
        top_k: int = 4,
        seed_k: int = 1,
        hops: int = 1,
    ) -> dict[str, object]:
        build_results = self.compare_build_strategies(strategies)
        strategy_payload: dict[str, object] = {}
        for strategy, build_result in build_results.items():
            metrics, cases = evaluate_strategy(
                nodes=self.nodes,
                edges=build_result.edges,
                queries=self.queries,
                top_k=top_k,
                seed_k=seed_k,
                hops=hops,
            )
            strategy_payload[strategy] = {
                "strategy": strategy,
                "metrics": metrics,
                "cost": build_result.cost_payload(),
                "cost_effectiveness": _cost_effectiveness(metrics, build_result),
                "cases": cases,
            }
        return {
            "config": {
                "alpha": self.alpha,
                "top_k_edges": self.top_k_edges,
                "threshold": self.threshold,
                "top_k": top_k,
                "seed_k": seed_k,
                "hops": hops,
            },
            "summary": _summary(strategy_payload),
            "strategies": strategy_payload,
        }


def build_graph_for_strategy(
    nodes: list[MemoryNode],
    config: GraphStrategyConfig,
) -> GraphBuildResult:
    start = time.perf_counter()
    if config.strategy == "no_graph" or config.top_k_edges == 0:
        return GraphBuildResult(
            strategy=config.strategy,
            edges=[],
            candidate_pair_count=0,
            build_time_seconds=time.perf_counter() - start,
            average_edge_score=0.0,
            config=config,
        )

    edge_by_key: dict[tuple[str, str, str], MemoryEdge] = {}
    candidate_pair_count = 0
    for source in nodes:
        scored: list[tuple[float, MemoryNode, dict[str, object]]] = []
        for target in nodes:
            if source.id == target.id:
                continue
            candidate_pair_count += 1
            score, breakdown = score_pair(source, target, config)
            if score >= config.threshold:
                scored.append((score, target, breakdown))
        scored.sort(key=lambda item: item[0], reverse=True)
        for score, target, breakdown in scored[: config.top_k_edges]:
            edge = _edge_for_score(source, target, config.strategy, score, breakdown)
            edge_by_key[edge.key] = edge
    edges = list(edge_by_key.values())
    average_edge_score = sum(edge.weight for edge in edges) / len(edges) if edges else 0.0
    return GraphBuildResult(
        strategy=config.strategy,
        edges=edges,
        candidate_pair_count=candidate_pair_count,
        build_time_seconds=time.perf_counter() - start,
        average_edge_score=average_edge_score,
        config=config,
    )


def score_pair(
    source: MemoryNode,
    target: MemoryNode,
    config: GraphStrategyConfig,
) -> tuple[float, dict[str, object]]:
    semantic = cosine_similarity(source.embedding, target.embedding)
    position = position_proximity(_position(source), _position(target))
    context = context_path_proximity(_context_path(source), _context_path(target))
    if config.strategy == "semantic_only":
        score = semantic
    elif config.strategy == "position_only":
        score = position
    elif config.strategy == "cam_style":
        score = config.alpha * semantic + (1.0 - config.alpha) * position
    elif config.strategy == "context_path_only":
        score = context
    elif config.strategy == "sam_context":
        score = config.alpha * semantic + (1.0 - config.alpha) * context
    else:
        score = 0.0
    return score, {
        "semantic_similarity": round(semantic, 4),
        "position_proximity": round(position, 4),
        "context_path_proximity": round(context, 4),
        "alpha": config.alpha,
        "formula": config.strategy,
    }


def position_proximity(left_position: int | None, right_position: int | None) -> float:
    if left_position is None or right_position is None:
        return 0.0
    distance = abs(left_position - right_position)
    return 1.0 / (1.0 + distance)


def context_path_proximity(left_path: list[str], right_path: list[str]) -> float:
    if not left_path or not right_path:
        return 0.0
    common_prefix = 0
    for left, right in zip(left_path, right_path, strict=False):
        if left != right:
            break
        common_prefix += 1
    if common_prefix == 0:
        return 0.0
    max_depth = max(len(left_path), len(right_path))
    depth_score = common_prefix / max_depth
    tail_distance = abs(len(left_path) - len(right_path))
    return depth_score / (1.0 + 0.25 * tail_distance)


def evaluate_strategy(
    *,
    nodes: list[MemoryNode],
    edges: list[MemoryEdge],
    queries: list[EvaluationQuery],
    top_k: int,
    seed_k: int,
    hops: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not queries:
        return (
            {
                "query_count": 0,
                "supporting_evidence_count": 0,
                "support_hits": 0,
                "evidence_recall": 0.0,
                "average_path_length": 0.0,
                "average_expanded_node_count": 0.0,
            },
            [],
        )
    node_by_original = {
        str(node.metadata.get("original_doc_id")): node.id
        for node in nodes
        if node.metadata.get("original_doc_id")
    }
    node_by_id = {node.id: node for node in nodes}
    adjacency = _adjacency(edges)
    support_hits = 0
    support_total = 0
    path_lengths: list[int] = []
    expanded_counts: list[int] = []
    cases: list[dict[str, object]] = []
    for query in queries:
        query_embedding = _query_embedding_from_nodes(query.question, nodes)
        candidate_ids = [
            node_by_original[doc_id]
            for doc_id in query.candidate_doc_ids
            if doc_id in node_by_original
        ]
        candidate_nodes = [node_by_id[node_id] for node_id in candidate_ids]
        support_node_ids = {
            node_by_original[doc_id]
            for doc_id in query.supporting_doc_ids
            if doc_id in node_by_original
        }
        support_total += len(support_node_ids)
        hits = _retrieve_with_edges(
            query_embedding=query_embedding,
            candidate_nodes=candidate_nodes,
            adjacency=adjacency,
            top_k=top_k,
            seed_k=seed_k,
            hops=hops,
        )
        hit_ids = {hit.node.id for hit in hits}
        query_support_hits = len(hit_ids & support_node_ids)
        support_hits += query_support_hits
        path_lengths.extend(max(0, len(hit.path) - 1) for hit in hits)
        expanded_counts.append(len({node_id for hit in hits for node_id in hit.path}))
        cases.append(
            {
                "query_id": query.id,
                "question": query.question,
                "support_node_ids": sorted(support_node_ids),
                "hit_node_ids": [hit.node.id for hit in hits],
                "support_hits": query_support_hits,
                "hits": [
                    {
                        "node_id": hit.node.id,
                        "score": round(hit.score, 4),
                        "path": hit.path,
                        "reason": hit.reason,
                    }
                    for hit in hits
                ],
            }
        )
    metrics = {
        "query_count": len(queries),
        "supporting_evidence_count": support_total,
        "support_hits": support_hits,
        "evidence_recall": support_hits / support_total if support_total else 0.0,
        "average_path_length": sum(path_lengths) / len(path_lengths) if path_lengths else 0.0,
        "average_expanded_node_count": sum(expanded_counts) / len(expanded_counts) if expanded_counts else 0.0,
    }
    return metrics, cases


def write_graph_strategy_report(report: dict[str, object], output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "graph_strategy_results.json"
    md_path = target / "graph_strategy_results.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def run_alpha_sweep(
    *,
    nodes: list[MemoryNode],
    queries: list[EvaluationQuery],
    alphas: list[float],
    top_k_edges: int,
    threshold: float,
    top_k: int,
    seed_k: int,
    hops: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_score = -1.0
    for alpha in alphas:
        report = GraphStrategyExperiment(
            nodes=nodes,
            queries=queries,
            alpha=alpha,
            top_k_edges=top_k_edges,
            threshold=threshold,
        ).run(
            strategies=["sam_context"],
            top_k=top_k,
            seed_k=seed_k,
            hops=hops,
        )
        payload = report["strategies"]["sam_context"]
        metrics = payload["metrics"]
        cost = payload["cost"]
        row = {
            "alpha": alpha,
            "evidence_recall": metrics["evidence_recall"],
            "edge_count": cost["edge_count"],
            "candidate_pair_count": cost["candidate_pair_count"],
            "build_time_seconds": cost["build_time_seconds"],
            "recall_per_100_edges": payload["cost_effectiveness"]["recall_per_100_edges"],
        }
        rows.append(row)
        score = float(row["evidence_recall"]) - 0.0005 * float(row["edge_count"])
        if score > best_score:
            best_row = row
            best_score = score
    return {
        "strategy": "sam_context",
        "best_alpha": best_row["alpha"] if best_row else None,
        "selection_rule": "优先 evidence_recall，同时轻微惩罚边规模。",
        "rows": rows,
    }


def write_alpha_sweep_report(sweep: dict[str, object], output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "graph_strategy_alpha_sweep.json"
    md_path = target / "graph_strategy_alpha_sweep.md"
    json_path.write_text(json.dumps(sweep, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_alpha_sweep_markdown(sweep), encoding="utf-8")
    return json_path, md_path


def _edge_for_score(
    source: MemoryNode,
    target: MemoryNode,
    strategy: str,
    score: float,
    breakdown: dict[str, object],
) -> MemoryEdge:
    now = utc_now_iso()
    return MemoryEdge(
        source_id=source.id,
        target_id=target.id,
        relation_type=f"strategy_{strategy}",
        weight=score,
        reason=f"非 LLM 建图策略 {strategy} 产生的候选边",
        created_at=now,
        updated_at=now,
        activation_count=0,
        last_activated_at=None,
        metadata={
            "strategy": strategy,
            "uses_llm": False,
            "score_breakdown": breakdown,
        },
    )


def _position(node: MemoryNode) -> int | None:
    value = node.metadata.get("position")
    if value is None:
        value = node.metadata.get("paragraph_index")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _context_path(node: MemoryNode) -> list[str]:
    value = node.metadata.get("context_path")
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [part for part in value.split("/") if part]
    fallback = [
        node.metadata.get("dataset"),
        node.metadata.get("query_id"),
        node.metadata.get("title"),
    ]
    return [str(item) for item in fallback if item]


def _adjacency(edges: list[MemoryEdge]) -> dict[str, list[MemoryEdge]]:
    adjacency: dict[str, list[MemoryEdge]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source_id, []).append(edge)
    return adjacency


def _query_embedding_from_nodes(query: str, nodes: list[MemoryNode]) -> list[float]:
    if not nodes:
        return []
    # 避免在该实验模块中绑定具体 embedding provider；用节点向量的关键词近似加权作为轻量查询向量。
    query_terms = set(query.lower().replace("-", " ").split())
    weighted = [0.0] * len(nodes[0].embedding)
    total_weight = 0.0
    for node in nodes:
        node_terms = set(" ".join([node.text, node.summary, " ".join(node.keywords)]).lower().split())
        weight = len(query_terms & node_terms)
        if weight <= 0:
            continue
        total_weight += weight
        for index, value in enumerate(node.embedding):
            weighted[index] += weight * value
    if total_weight == 0.0:
        return [0.0] * len(nodes[0].embedding)
    return [value / total_weight for value in weighted]


def _retrieve_with_edges(
    *,
    query_embedding: list[float],
    candidate_nodes: list[MemoryNode],
    adjacency: dict[str, list[MemoryEdge]],
    top_k: int,
    seed_k: int,
    hops: int,
) -> list[RetrievalHit]:
    vector_hits = sorted(
        [
            (cosine_similarity(query_embedding, node.embedding), node)
            for node in candidate_nodes
        ],
        key=lambda item: item[0],
        reverse=True,
    )
    seed_hits = vector_hits[: max(seed_k, 1)]
    candidate_ids = {node.id for node in candidate_nodes}
    node_by_id = {node.id: node for node in candidate_nodes}
    best: dict[str, tuple[list[str], float, str]] = {}
    queue: deque[tuple[str, list[str], float, int]] = deque()
    for similarity, node in vector_hits[:top_k]:
        best[node.id] = ([node.id], 0.0, f"embedding top-k 基础候选，相似度={similarity:.3f}")
    for similarity, node in seed_hits:
        best[node.id] = ([node.id], 0.0, f"图扩展种子节点，相似度={similarity:.3f}")
        queue.append((node.id, [node.id], 0.0, 0))
    while queue:
        current_id, path, graph_score, depth = queue.popleft()
        if depth >= hops:
            continue
        for edge in adjacency.get(current_id, []):
            if edge.target_id not in candidate_ids or edge.target_id in path:
                continue
            next_path = [*path, edge.target_id]
            next_graph_score = graph_score + edge.weight / max(1, depth + 1)
            previous = best.get(edge.target_id)
            if previous is None or next_graph_score > previous[1]:
                best[edge.target_id] = (
                    next_path,
                    next_graph_score,
                    f"沿 {edge.relation_type} 扩展，边权={edge.weight:.3f}",
                )
                queue.append((edge.target_id, next_path, next_graph_score, depth + 1))
    hits: list[RetrievalHit] = []
    for node_id, (path, graph_score, reason) in best.items():
        node = node_by_id[node_id]
        similarity = cosine_similarity(query_embedding, node.embedding)
        score = 0.7 * similarity + 0.3 * graph_score
        hits.append(
            RetrievalHit(
                node=node,
                score=score,
                similarity_score=similarity,
                graph_score=graph_score,
                usage_score=0.0,
                confidence_score=node.confidence * 0.03,
                path=path,
                reason=reason,
                metadata={"score_breakdown": {"similarity": similarity, "graph_score": graph_score}},
            )
        )
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:top_k]


def _cost_effectiveness(metrics: dict[str, object], build_result: GraphBuildResult) -> dict[str, object]:
    recall = float(metrics.get("evidence_recall", 0.0))
    edge_count = max(1, build_result.edge_count)
    build_time = max(0.000001, build_result.build_time_seconds)
    return {
        "recall_per_100_edges": round(recall * 100.0 / edge_count, 6),
        "recall_per_second": round(recall / build_time, 6),
        "llm_call_count": 0,
        "uses_llm": False,
    }


def _summary(strategy_payload: dict[str, object]) -> dict[str, object]:
    best_strategy = ""
    best_score = -1.0
    for strategy, payload in strategy_payload.items():
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics", {})
        cost = payload.get("cost", {})
        if not isinstance(metrics, dict) or not isinstance(cost, dict):
            continue
        recall = float(metrics.get("evidence_recall", 0.0))
        edge_count = float(cost.get("edge_count", 0.0))
        build_time = float(cost.get("build_time_seconds", 0.0))
        score = recall - 0.0005 * edge_count - 0.01 * build_time
        if score > best_score:
            best_strategy = strategy
            best_score = score
    return {
        "recommended_strategy": best_strategy,
        "selection_rule": "优先 evidence_recall，同时轻微惩罚边规模和建图耗时；所有策略均不使用 LLM 建图。",
    }


def _markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# 非 LLM 建图策略性价比实验",
        "",
        f"推荐策略：{report.get('summary', {}).get('recommended_strategy', '')}",
        "",
        "| 策略 | Evidence Recall | 边数 | 候选对数 | 建图耗时(s) | 是否用 LLM |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
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
                f"{float(metrics.get('evidence_recall', 0.0)):.3f} | "
                f"{int(cost.get('edge_count', 0))} | "
                f"{int(cost.get('candidate_pair_count', 0))} | "
                f"{float(cost.get('build_time_seconds', 0.0)):.6f} | "
                f"{'是' if cost.get('uses_llm') else '否'} |"
            )
    lines.extend(
        [
            "",
            "本实验只比较非生成式建图公式。主线假设是：简洁的 `semantic similarity + context path proximity` "
            "比单独语义、单独位置或线性位置版本更适合作为通用 SAM 建图策略。",
        ]
    )
    return "\n".join(lines) + "\n"


def _alpha_sweep_markdown(sweep: dict[str, object]) -> str:
    lines = [
        "# SAM-style alpha 扫描",
        "",
        f"最佳 alpha：{sweep.get('best_alpha')}",
        "",
        "| alpha | Evidence Recall | 边数 | 候选对数 | 建图耗时(s) | Recall / 100 edges |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sweep.get("rows", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            f"{float(row.get('alpha', 0.0)):.2f} | "
            f"{float(row.get('evidence_recall', 0.0)):.3f} | "
            f"{int(row.get('edge_count', 0))} | "
            f"{int(row.get('candidate_pair_count', 0))} | "
            f"{float(row.get('build_time_seconds', 0.0)):.6f} | "
            f"{float(row.get('recall_per_100_edges', 0.0)):.6f} |"
        )
    lines.append("")
    lines.append("alpha 越接近 1.0，越偏向语义相似；越接近 0.0，越偏向上下文路径邻近。")
    return "\n".join(lines) + "\n"
