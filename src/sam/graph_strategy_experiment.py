from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import deque

from sam.embedding import EmbeddingProvider
from sam.models import EvaluationQuery, MemoryEdge, MemoryNode, RetrievalHit, utc_now_iso
from sam.progress import progress_iter
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
        query_embeddings: dict[str, list[float]] | None = None,
        alpha: float = 0.55,
        top_k_edges: int = 4,
        threshold: float = 0.18,
    ) -> None:
        self.nodes = nodes
        self.queries = queries
        self.query_embeddings = query_embeddings or {}
        self.alpha = alpha
        self.top_k_edges = top_k_edges
        self.threshold = threshold

    def compare_build_strategies(
        self,
        strategies: list[str],
    ) -> dict[str, GraphBuildResult]:
        results: dict[str, GraphBuildResult] = {}
        for strategy in progress_iter(strategies, total=len(strategies), desc="建图策略"):
            results[strategy] = build_graph_for_strategy(
                self.nodes,
                GraphStrategyConfig(
                    strategy=strategy,
                    alpha=self.alpha,
                    top_k_edges=self.top_k_edges,
                    threshold=self.threshold,
                ),
            )
        return results

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
                query_embeddings=self.query_embeddings,
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
        _attach_comparative_cost_effectiveness(strategy_payload)
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
    for source in progress_iter(nodes, total=len(nodes), desc=f"建边:{config.strategy}"):
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
    query_embeddings: dict[str, list[float]] | None = None,
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
    for query in progress_iter(queries, total=len(queries), desc="评估查询"):
        supplied_query_embedding = query_embeddings.get(query.id) if query_embeddings else None
        query_embedding = supplied_query_embedding or _query_embedding_from_nodes(query.question, nodes)
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
                "query_embedding_source": "provided" if supplied_query_embedding is not None else "fallback",
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
    query_embeddings: dict[str, list[float]] | None = None,
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
    for alpha in progress_iter(alphas, total=len(alphas), desc="alpha扫描"):
        report = GraphStrategyExperiment(
            nodes=nodes,
            queries=queries,
            query_embeddings=query_embeddings,
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
            "recall_per_second": payload["cost_effectiveness"]["recall_per_second"],
        }
        rows.append(row)
    _attach_alpha_cost_effectiveness(rows)
    for row in rows:
        score = float(row["cost_effectiveness_score"])
        if score > best_score:
            best_row = row
            best_score = score
    return {
        "strategy": "sam_context",
        "best_alpha": best_row["alpha"] if best_row else None,
        "selection_rule": "按综合性价比分选择：Evidence Recall / (1 + 成本指数)。",
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
    edge_count = build_result.edge_count
    build_time = max(0.000001, build_result.build_time_seconds)
    return {
        "recall_per_100_edges": round(recall * 100.0 / edge_count, 6) if edge_count else 0.0,
        "recall_per_second": round(recall / build_time, 6),
        "llm_call_count": 0,
        "uses_llm": False,
    }


def _attach_comparative_cost_effectiveness(strategy_payload: dict[str, object]) -> None:
    payloads = [
        payload
        for payload in strategy_payload.values()
        if isinstance(payload, dict)
        and isinstance(payload.get("metrics"), dict)
        and isinstance(payload.get("cost"), dict)
        and isinstance(payload.get("cost_effectiveness"), dict)
    ]
    if not payloads:
        return
    max_edges = max(float(payload["cost"].get("edge_count", 0.0)) for payload in payloads) or 1.0
    max_pairs = max(float(payload["cost"].get("candidate_pair_count", 0.0)) for payload in payloads) or 1.0
    max_time = max(float(payload["cost"].get("build_time_seconds", 0.0)) for payload in payloads) or 0.000001
    no_graph_payload = strategy_payload.get("no_graph")
    if isinstance(no_graph_payload, dict) and isinstance(no_graph_payload.get("metrics"), dict) and isinstance(no_graph_payload.get("cost"), dict):
        baseline_recall = float(no_graph_payload["metrics"].get("evidence_recall", 0.0))
        baseline_edges = float(no_graph_payload["cost"].get("edge_count", 0.0))
        baseline_time = float(no_graph_payload["cost"].get("build_time_seconds", 0.0))
    else:
        baseline_recall = 0.0
        baseline_edges = 0.0
        baseline_time = 0.0
    for payload in payloads:
        metrics = payload["metrics"]
        cost = payload["cost"]
        cost_effectiveness = payload["cost_effectiveness"]
        recall = float(metrics.get("evidence_recall", 0.0))
        edge_count = float(cost.get("edge_count", 0.0))
        candidate_pairs = float(cost.get("candidate_pair_count", 0.0))
        build_time = float(cost.get("build_time_seconds", 0.0))
        normalized_edge_cost = edge_count / max_edges if max_edges else 0.0
        normalized_candidate_pair_cost = candidate_pairs / max_pairs if max_pairs else 0.0
        normalized_build_time_cost = build_time / max_time if max_time else 0.0
        cost_index = (
            0.40 * normalized_edge_cost
            + 0.30 * normalized_candidate_pair_cost
            + 0.30 * normalized_build_time_cost
        )
        recall_gain = recall - baseline_recall
        extra_edges = max(0.0, edge_count - baseline_edges)
        extra_time = max(0.0, build_time - baseline_time)
        cost_effectiveness.update(
            {
                "normalized_edge_cost": round(normalized_edge_cost, 6),
                "normalized_candidate_pair_cost": round(normalized_candidate_pair_cost, 6),
                "normalized_build_time_cost": round(normalized_build_time_cost, 6),
                "cost_index": round(cost_index, 6),
                "cost_effectiveness_score": round(recall / (1.0 + cost_index), 6),
                "balanced_score": round(recall - 0.15 * cost_index, 6),
                "recall_gain_vs_no_graph": round(recall_gain, 6),
                "gain_per_100_extra_edges": round(recall_gain * 100.0 / extra_edges, 6) if extra_edges else 0.0,
                "gain_per_extra_second": round(recall_gain / extra_time, 6) if extra_time > 0.0 else 0.0,
            }
        )


def _attach_alpha_cost_effectiveness(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    max_edges = max(float(row.get("edge_count", 0.0)) for row in rows) or 1.0
    max_pairs = max(float(row.get("candidate_pair_count", 0.0)) for row in rows) or 1.0
    max_time = max(float(row.get("build_time_seconds", 0.0)) for row in rows) or 0.000001
    for row in rows:
        recall = float(row.get("evidence_recall", 0.0))
        normalized_edge_cost = float(row.get("edge_count", 0.0)) / max_edges
        normalized_candidate_pair_cost = float(row.get("candidate_pair_count", 0.0)) / max_pairs
        normalized_build_time_cost = float(row.get("build_time_seconds", 0.0)) / max_time
        cost_index = (
            0.40 * normalized_edge_cost
            + 0.30 * normalized_candidate_pair_cost
            + 0.30 * normalized_build_time_cost
        )
        row["cost_index"] = round(cost_index, 6)
        row["cost_effectiveness_score"] = round(recall / (1.0 + cost_index), 6)


def _summary(strategy_payload: dict[str, object]) -> dict[str, object]:
    best_balanced_strategy = ""
    best_cost_effectiveness_strategy = ""
    best_recall_strategy = ""
    best_balanced_score = -1.0
    best_cost_effectiveness_score = -1.0
    best_recall = -1.0
    ranking: list[dict[str, object]] = []
    for strategy, payload in strategy_payload.items():
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics", {})
        cost_effectiveness = payload.get("cost_effectiveness", {})
        if not isinstance(metrics, dict) or not isinstance(cost_effectiveness, dict):
            continue
        recall = float(metrics.get("evidence_recall", 0.0))
        cost_effectiveness_score = float(cost_effectiveness.get("cost_effectiveness_score", 0.0))
        balanced_score = float(cost_effectiveness.get("balanced_score", 0.0))
        ranking.append(
            {
                "strategy": strategy,
                "evidence_recall": round(recall, 6),
                "cost_effectiveness_score": round(cost_effectiveness_score, 6),
                "balanced_score": round(balanced_score, 6),
                "cost_index": cost_effectiveness.get("cost_index", 0.0),
            }
        )
        if recall > best_recall:
            best_recall = recall
            best_recall_strategy = strategy
        can_recommend = strategy != "no_graph" or len(strategy_payload) == 1
        if can_recommend and cost_effectiveness_score > best_cost_effectiveness_score:
            best_cost_effectiveness_score = cost_effectiveness_score
            best_cost_effectiveness_strategy = strategy
        if can_recommend and balanced_score > best_balanced_score:
            best_balanced_score = balanced_score
            best_balanced_strategy = strategy
    ranking.sort(key=lambda row: (float(row["balanced_score"]), float(row["evidence_recall"])), reverse=True)
    return {
        "recommended_strategy": best_balanced_strategy,
        "best_recall_strategy": best_recall_strategy,
        "best_cost_effectiveness_strategy": best_cost_effectiveness_strategy,
        "best_balanced_strategy": best_balanced_strategy,
        "selection_rule": "推荐策略按 balanced_score 选择：优先证据召回，同时惩罚归一化边规模、候选比较次数和建图耗时；所有策略均不使用 LLM 建图。",
        "ranking": ranking,
    }


def _markdown_report(report: dict[str, object]) -> str:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    lines = [
        "# 非 LLM 建图策略性价比实验",
        "",
        f"推荐策略：{summary.get('recommended_strategy', '')}",
        f"最高召回策略：{summary.get('best_recall_strategy', '')}",
        f"最高性价比策略：{summary.get('best_cost_effectiveness_strategy', '')}",
        "",
        str(summary.get("selection_rule", "")),
        "",
        "| 策略 | Evidence Recall | 相对 no_graph 召回增益 | 边数 | 候选对数 | 建图耗时(s) | 平均路径长度 | 平均扩展节点数 | Recall / 100 edges | Recall/s | 成本指数 | 综合性价比分 | 是否用 LLM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    strategies = report.get("strategies", {})
    if isinstance(strategies, dict):
        for strategy, payload in strategies.items():
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics", {})
            cost = payload.get("cost", {})
            cost_effectiveness = payload.get("cost_effectiveness", {})
            if not isinstance(metrics, dict) or not isinstance(cost, dict) or not isinstance(cost_effectiveness, dict):
                continue
            lines.append(
                "| "
                f"{strategy} | "
                f"{float(metrics.get('evidence_recall', 0.0)):.3f} | "
                f"{float(cost_effectiveness.get('recall_gain_vs_no_graph', 0.0)):.3f} | "
                f"{int(cost.get('edge_count', 0))} | "
                f"{int(cost.get('candidate_pair_count', 0))} | "
                f"{float(cost.get('build_time_seconds', 0.0)):.6f} | "
                f"{float(metrics.get('average_path_length', 0.0)):.3f} | "
                f"{float(metrics.get('average_expanded_node_count', 0.0)):.3f} | "
                f"{float(cost_effectiveness.get('recall_per_100_edges', 0.0)):.6f} | "
                f"{float(cost_effectiveness.get('recall_per_second', 0.0)):.6f} | "
                f"{float(cost_effectiveness.get('cost_index', 0.0)):.6f} | "
                f"{float(cost_effectiveness.get('cost_effectiveness_score', 0.0)):.6f} | "
                f"{'是' if cost.get('uses_llm') else '否'} |"
            )
    lines.extend(
        [
            "",
            "成本指数由归一化边规模、归一化候选比较次数和归一化建图耗时加权得到。综合性价比分使用 `Evidence Recall / (1 + 成本指数)`，数值越高表示单位建图代价下的召回表现越好。",
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
        f"选择规则：{sweep.get('selection_rule')}",
        "",
        "| alpha | Evidence Recall | 边数 | 候选对数 | 建图耗时(s) | Recall / 100 edges | Recall/s | 成本指数 | 综合性价比分 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
            f"{float(row.get('recall_per_100_edges', 0.0)):.6f} | "
            f"{float(row.get('recall_per_second', 0.0)):.6f} | "
            f"{float(row.get('cost_index', 0.0)):.6f} | "
            f"{float(row.get('cost_effectiveness_score', 0.0)):.6f} |"
        )
    lines.append("")
    lines.append("alpha 越接近 1.0，越偏向语义相似；越接近 0.0，越偏向上下文路径邻近。")
    return "\n".join(lines) + "\n"
