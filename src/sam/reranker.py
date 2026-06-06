from __future__ import annotations

import math
import os
import json
from dataclasses import dataclass
from pathlib import Path

from sam.models import MemoryNode


@dataclass(frozen=True, slots=True)
class RerankerWeights:
    """路径重排权重配置，用于 bad case 后的可控消融。"""

    similarity: float
    graph: float
    path_support: float
    edge_memory: float
    confidence: float
    usage_scale: float
    usage_cap: float
    recency: float
    path_noise_cap: float = 0.12
    path_noise_threshold: int = 12
    weak_relation_cap: float = 0.10


RERANKER_PROFILES = {
    "balanced": RerankerWeights(
        similarity=0.56,
        graph=0.21,
        path_support=0.09,
        edge_memory=0.05,
        confidence=0.04,
        usage_scale=0.045,
        usage_cap=0.14,
        recency=0.015,
    ),
    "semantic_heavy": RerankerWeights(
        similarity=0.68,
        graph=0.14,
        path_support=0.05,
        edge_memory=0.03,
        confidence=0.04,
        usage_scale=0.03,
        usage_cap=0.10,
        recency=0.01,
    ),
    "graph_heavy": RerankerWeights(
        similarity=0.42,
        graph=0.34,
        path_support=0.13,
        edge_memory=0.05,
        confidence=0.04,
        usage_scale=0.035,
        usage_cap=0.12,
        recency=0.012,
    ),
    "memory_heavy": RerankerWeights(
        similarity=0.48,
        graph=0.18,
        path_support=0.08,
        edge_memory=0.09,
        confidence=0.04,
        usage_scale=0.07,
        usage_cap=0.20,
        recency=0.03,
    ),
}

DEFAULT_RERANKER_PROFILE = "semantic_heavy"


@dataclass(frozen=True, slots=True)
class PathScore:
    """SAM 路径重排分数。"""

    profile: str
    total: float
    path_support_score: float
    edge_memory_score: float
    path_noise_penalty: float
    weak_relation_penalty: float
    relation_noise_penalty: float
    usage_score: float
    recency_score: float
    confidence_score: float
    breakdown: dict[str, float]


class PathReranker:
    """将联想检索结果的路径质量和记忆状态合成为排序分。"""

    def __init__(
        self,
        profile: str = DEFAULT_RERANKER_PROFILE,
        relation_noise_rates: dict[str, float] | None = None,
    ) -> None:
        if profile not in RERANKER_PROFILES:
            raise ValueError(f"未知 reranker profile: {profile}")
        self.profile = profile
        self.weights = RERANKER_PROFILES[profile]
        self.relation_noise_rates = relation_noise_rates or {}

    @classmethod
    def from_env(cls) -> "PathReranker":
        return cls(
            os.environ.get("SAM_RERANKER_PROFILE", DEFAULT_RERANKER_PROFILE),
            relation_noise_rates=_load_relation_noise_rates_from_env(),
        )

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
        weights = self.weights
        path_support_score = _path_support_score(signals) if use_multipath else 0.0
        path_noise_penalty = _path_noise_penalty(signals, weights) if use_multipath else 0.0
        weak_relation_penalty = _weak_relation_penalty(signals, weights) if use_multipath else 0.0
        relation_noise_penalty = (
            _relation_noise_penalty(signals, self.relation_noise_rates)
            if use_multipath
            else 0.0
        )
        effective_similarity = _effective_similarity(similarity, graph_score, path_support_score)
        edge_memory_score = _edge_memory_score(signals) if use_memory_state else 0.0
        usage_score = (
            min(weights.usage_cap, math.log1p(node.usage_count) * weights.usage_scale)
            if use_memory_state
            else 0.0
        )
        recency_score = _recency_score(node.last_accessed_at, weights.recency) if use_memory_state else 0.0
        confidence_score = node.confidence * weights.confidence
        breakdown = {
            "similarity_component": round(weights.similarity * effective_similarity, 4),
            "graph_component": round(weights.graph * graph_score, 4),
            "confidence_component": round(confidence_score, 4),
        }
        if effective_similarity != similarity:
            breakdown["similarity_floor"] = round(effective_similarity, 4)
        if use_multipath:
            breakdown["path_support_component"] = round(weights.path_support * path_support_score, 4)
            if path_noise_penalty > 0.0:
                breakdown["path_noise_penalty"] = round(path_noise_penalty, 4)
            if weak_relation_penalty > 0.0:
                breakdown["weak_relation_penalty"] = round(weak_relation_penalty, 4)
            if relation_noise_penalty > 0.0:
                breakdown["relation_noise_penalty"] = round(relation_noise_penalty, 4)
        if use_memory_state:
            breakdown["edge_memory_component"] = round(weights.edge_memory * edge_memory_score, 4)
            breakdown["usage_component"] = round(usage_score, 4)
            breakdown["recency_component"] = round(recency_score, 4)
        total = (
            weights.similarity * effective_similarity
            + weights.graph * graph_score
            + weights.path_support * path_support_score
            + weights.edge_memory * edge_memory_score
            + usage_score
            + recency_score
            + confidence_score
            - path_noise_penalty
            - weak_relation_penalty
            - relation_noise_penalty
        )
        return PathScore(
            profile=self.profile,
            total=total,
            path_support_score=path_support_score,
            edge_memory_score=edge_memory_score,
            path_noise_penalty=path_noise_penalty,
            weak_relation_penalty=weak_relation_penalty,
            relation_noise_penalty=relation_noise_penalty,
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


def _effective_similarity(
    similarity: float,
    graph_score: float,
    path_support_score: float,
) -> float:
    """强图路径命中时，避免负向量相似度压制桥接证据。"""

    if graph_score >= 0.8 and path_support_score >= 0.45:
        return max(0.0, similarity)
    return similarity


def _edge_memory_score(signals: list[dict[str, object]]) -> float:
    activations = [
        int(signal.get("edge_activation_count", 0))
        for signal in signals
        if int(signal.get("edge_activation_count", 0)) > 0
    ]
    if not activations:
        return 0.0
    return min(1.0, math.log1p(sum(activations)) / 3.0)


def _path_noise_penalty(signals: list[dict[str, object]], weights: RerankerWeights) -> float:
    non_seed_path_count = sum(
        1 for signal in signals
        if len(signal.get("path", [])) > 1
    )
    if non_seed_path_count <= weights.path_noise_threshold:
        return 0.0
    overflow = non_seed_path_count - weights.path_noise_threshold
    return min(weights.path_noise_cap, math.log1p(overflow) * 0.035)


def _weak_relation_penalty(signals: list[dict[str, object]], weights: RerankerWeights) -> float:
    weak_relation_types = {"embedding_similarity", "context_cooccurrence"}
    penalty = 0.0
    for signal in signals:
        relation_type = str(signal.get("relation_type", ""))
        path = signal.get("path", [])
        fallback_depth = max(0, len(path) - 1) if isinstance(path, list) else 0
        depth = int(signal.get("depth", fallback_depth))
        if depth < 2:
            continue
        if relation_type in weak_relation_types:
            penalty += 0.045
        elif relation_type == "keyword_overlap":
            penalty += 0.02
    return min(weights.weak_relation_cap, penalty)


def _relation_noise_penalty(
    signals: list[dict[str, object]],
    relation_noise_rates: dict[str, float],
) -> float:
    if not relation_noise_rates:
        return 0.0
    penalty = 0.0
    for signal in signals:
        relation_type = str(signal.get("relation_type", ""))
        if not relation_type:
            continue
        noise_rate = float(relation_noise_rates.get(relation_type, 0.0))
        if noise_rate < 0.6:
            continue
        path = signal.get("path", [])
        fallback_depth = max(0, len(path) - 1) if isinstance(path, list) else 0
        depth = max(1, int(signal.get("depth", fallback_depth) or 1))
        penalty += (noise_rate - 0.55) * 0.08 / depth
    return min(0.14, penalty)


def _load_relation_noise_rates_from_env() -> dict[str, float]:
    path = os.environ.get("SAM_EDGE_QUALITY_AUDIT_PATH")
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    relation_stats = data.get("relation_stats", {})
    if not isinstance(relation_stats, dict):
        return {}
    rates: dict[str, float] = {}
    for relation_type, stats in relation_stats.items():
        if not isinstance(stats, dict):
            continue
        rates[str(relation_type)] = float(stats.get("noise_rate", 0.0) or 0.0)
    return rates


def _recency_score(last_accessed_at: str | None, weight: float) -> float:
    if not last_accessed_at:
        return 0.0
    return weight
