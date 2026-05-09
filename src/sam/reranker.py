from __future__ import annotations

import math
from dataclasses import dataclass

from sam.models import MemoryNode


@dataclass(frozen=True, slots=True)
class PathScore:
    """SAM 路径重排分数。"""

    total: float
    path_support_score: float
    edge_memory_score: float
    usage_score: float
    recency_score: float
    confidence_score: float
    breakdown: dict[str, float]


class PathReranker:
    """将联想检索结果的路径质量和记忆状态合成为排序分。"""

    def score(
        self,
        *,
        similarity: float,
        graph_score: float,
        signals: list[dict[str, object]],
        node: MemoryNode,
        use_multipath: bool,
        use_memory_state: bool,
    ) -> PathScore:
        path_support_score = _path_support_score(signals) if use_multipath else 0.0
        edge_memory_score = _edge_memory_score(signals) if use_memory_state else 0.0
        usage_score = min(0.14, math.log1p(node.usage_count) * 0.045) if use_memory_state else 0.0
        recency_score = _recency_score(node.last_accessed_at) if use_memory_state else 0.0
        confidence_score = node.confidence * 0.04
        breakdown = {
            "similarity_component": round(0.56 * similarity, 4),
            "graph_component": round(0.21 * graph_score, 4),
            "confidence_component": round(confidence_score, 4),
        }
        if use_multipath:
            breakdown["path_support_component"] = round(0.09 * path_support_score, 4)
        if use_memory_state:
            breakdown["edge_memory_component"] = round(0.05 * edge_memory_score, 4)
            breakdown["usage_component"] = round(usage_score, 4)
            breakdown["recency_component"] = round(recency_score, 4)
        total = (
            0.56 * similarity
            + 0.21 * graph_score
            + 0.09 * path_support_score
            + 0.05 * edge_memory_score
            + usage_score
            + recency_score
            + confidence_score
        )
        return PathScore(
            total=total,
            path_support_score=path_support_score,
            edge_memory_score=edge_memory_score,
            usage_score=usage_score,
            recency_score=recency_score,
            confidence_score=confidence_score,
            breakdown=breakdown,
        )


def _path_support_score(signals: list[dict[str, object]]) -> float:
    non_seed_paths = [
        signal for signal in signals
        if len(signal.get("path", [])) > 1
    ]
    if not non_seed_paths:
        return 0.0
    weighted = 0.0
    for signal in non_seed_paths:
        depth = max(1, int(signal.get("depth", 1)))
        weighted += float(signal.get("graph_score", 0.0)) / depth
    return min(1.0, math.log1p(weighted + len(non_seed_paths) * 0.2))


def _edge_memory_score(signals: list[dict[str, object]]) -> float:
    activations = [
        int(signal.get("edge_activation_count", 0))
        for signal in signals
        if int(signal.get("edge_activation_count", 0)) > 0
    ]
    if not activations:
        return 0.0
    return min(1.0, math.log1p(sum(activations)) / 3.0)


def _recency_score(last_accessed_at: str | None) -> float:
    if not last_accessed_at:
        return 0.0
    return 0.015
