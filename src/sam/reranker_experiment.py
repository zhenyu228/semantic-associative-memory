from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from sam.badcase import BadCaseAnalyzer
from sam.embedding import EmbeddingProvider
from sam.evaluator import Evaluator
from sam.graph import GraphBuilder
from sam.models import DatasetDocument, EvaluationQuery
from sam.reranker import RERANKER_PROFILES
from sam.store import MemoryStore


DEFAULT_RERANKER_PROFILES = ["balanced", "semantic_heavy", "graph_heavy", "memory_heavy"]


def run_reranker_profile_comparison(
    *,
    documents: list[DatasetDocument],
    queries: list[EvaluationQuery],
    embedding_provider: EmbeddingProvider,
    profiles: list[str] | None = None,
    top_k: int = 4,
    seed_k: int = 1,
    hops: int = 2,
    method: str = "sam_full",
) -> dict[str, object]:
    """在同一批样本上隔离比较多种 PathReranker profile。"""

    active_profiles = profiles or DEFAULT_RERANKER_PROFILES
    _validate_profiles(active_profiles)
    original_profile = os.environ.get("SAM_RERANKER_PROFILE")
    profile_results: dict[str, dict[str, object]] = {}
    try:
        for profile in active_profiles:
            os.environ["SAM_RERANKER_PROFILE"] = profile
            with tempfile.TemporaryDirectory() as temp_dir:
                store = MemoryStore(Path(temp_dir) / f"{profile}.sqlite")
                try:
                    graph_builder = GraphBuilder(store)
                    evaluator = Evaluator(store, embedding_provider, graph_builder)
                    evaluator.ingest(documents)
                    result = evaluator.evaluate(
                        queries,
                        top_k=top_k,
                        seed_k=seed_k,
                        hops=hops,
                        methods=[method],
                    )
                    bad_cases = BadCaseAnalyzer().analyze(result.cases, method=method)
                    profile_results[profile] = {
                        "metrics": result.method_metrics[method],
                        "bad_case_summary": _bad_case_summary(bad_cases),
                        "case_count": len(result.cases),
                    }
                finally:
                    store.close()
    finally:
        if original_profile is None:
            os.environ.pop("SAM_RERANKER_PROFILE", None)
        else:
            os.environ["SAM_RERANKER_PROFILE"] = original_profile

    best_profile = _best_profile(profile_results)
    return {
        "profiles": active_profiles,
        "method": method,
        "query_count": len(queries),
        "document_count": len(documents),
        "top_k": top_k,
        "seed_k": seed_k,
        "hops": hops,
        "best_profile": best_profile,
        "profile_results": profile_results,
        "analysis": _comparison_analysis(profile_results, best_profile),
    }


def write_reranker_profile_reports(
    comparison: dict[str, object],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "reranker_profile_comparison.json"
    markdown_path = target / "reranker_profile_comparison.md"
    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(_comparison_to_markdown(comparison), encoding="utf-8")
    return json_path, markdown_path


def _validate_profiles(profiles: list[str]) -> None:
    unknown = [profile for profile in profiles if profile not in RERANKER_PROFILES]
    if unknown:
        raise ValueError(f"未知 reranker profile: {', '.join(unknown)}")


def _bad_case_summary(bad_cases) -> dict[str, object]:
    category_counts: Counter[str] = Counter()
    for case in bad_cases:
        category_counts.update(case.categories)
    return {
        "bad_case_count": len(bad_cases),
        "category_counts": dict(sorted(category_counts.items())),
    }


def _best_profile(profile_results: dict[str, dict[str, object]]) -> str:
    def key(item: tuple[str, dict[str, object]]) -> tuple[float, float, int]:
        metrics = item[1]["metrics"]
        bad_case_summary = item[1]["bad_case_summary"]
        assert isinstance(metrics, dict)
        assert isinstance(bad_case_summary, dict)
        return (
            float(metrics.get("evidence_recall", 0.0)),
            float(metrics.get("answer_hit_rate", 0.0)),
            -int(bad_case_summary.get("bad_case_count", 0)),
        )

    return max(profile_results.items(), key=key)[0]


def _comparison_analysis(
    profile_results: dict[str, dict[str, object]],
    best_profile: str,
) -> dict[str, object]:
    best = profile_results[best_profile]
    best_metrics = best["metrics"]
    assert isinstance(best_metrics, dict)
    baseline = profile_results.get("balanced", best)
    baseline_metrics = baseline["metrics"]
    assert isinstance(baseline_metrics, dict)
    return {
        "best_profile": best_profile,
        "best_evidence_recall": best_metrics.get("evidence_recall", 0.0),
        "best_answer_hit_rate": best_metrics.get("answer_hit_rate", 0.0),
        "gain_over_balanced": {
            "evidence_recall": float(best_metrics.get("evidence_recall", 0.0))
            - float(baseline_metrics.get("evidence_recall", 0.0)),
            "answer_hit_rate": float(best_metrics.get("answer_hit_rate", 0.0))
            - float(baseline_metrics.get("answer_hit_rate", 0.0)),
        },
    }


def _comparison_to_markdown(comparison: dict[str, object]) -> str:
    profile_results = comparison["profile_results"]
    assert isinstance(profile_results, dict)
    lines = [
        "# PathReranker Profile 对比实验",
        "",
        f"- 查询数量：{comparison['query_count']}",
        f"- 候选文档数量：{comparison['document_count']}",
        f"- 方法：{comparison['method']}",
        f"- 最优 profile：{comparison['best_profile']}",
        "",
        "| Profile | 证据召回率 | 答案命中率 | 平均路径长度 | Bad case 数量 | 主要失败类型 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for profile in comparison["profiles"]:
        result = profile_results[str(profile)]
        metrics = result["metrics"]
        bad_summary = result["bad_case_summary"]
        assert isinstance(metrics, dict)
        assert isinstance(bad_summary, dict)
        category_counts = bad_summary.get("category_counts", {})
        assert isinstance(category_counts, dict)
        category_text = ", ".join(
            f"{category}:{count}"
            for category, count in category_counts.items()
        ) or "无"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(profile),
                    f"{float(metrics.get('evidence_recall', 0.0)):.3f}",
                    f"{float(metrics.get('answer_hit_rate', 0.0)):.3f}",
                    f"{float(metrics.get('average_path_length', 0.0)):.2f}",
                    str(bad_summary.get("bad_case_count", 0)),
                    category_text,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"当前最优 profile 为 `{comparison['best_profile']}`。该结论只对本次数据规模、embedding provider、top-k 和 hops 设置成立，后续正式实验需要在 HotpotQA 300 条和 NovelQA 上分别复跑。",
            "",
        ]
    )
    return "\n".join(lines)
