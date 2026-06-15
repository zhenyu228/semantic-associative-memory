"""在线批量插入场景下的建图时间成本实验。"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from sam.cost_effect_figure import strip_trailing_whitespace
from sam.graph_strategy_experiment import GraphStrategyConfig, build_graph_for_strategy
from sam.models import EvaluationQuery, MemoryNode


METHOD_LABELS = {
    "full_rebuild": "全量重建图",
    "query_activated_local": "SAM 查询激活局部建图",
    "sam_lazy_insert": "SAM 懒插入",
}


def run_insertion_time_benchmark(
    *,
    nodes: list[MemoryNode],
    queries: list[EvaluationQuery],
    batch_sizes: list[int],
    strategy: str = "sam_context",
    alpha: float = 0.55,
    top_k_edges: int = 4,
    threshold: float = 0.18,
    repeats: int = 3,
) -> dict[str, Any]:
    """比较全量重建、查询触发局部建图和 SAM lazy insert 的时间成本。"""

    if not nodes:
        raise ValueError("nodes 不能为空")
    if repeats <= 0:
        raise ValueError("repeats 必须大于 0")
    normalized_batch_sizes = _normalize_batch_sizes(batch_sizes, len(nodes))
    config = GraphStrategyConfig(
        strategy=strategy,
        alpha=alpha,
        top_k_edges=top_k_edges,
        threshold=threshold,
    )
    rows: list[dict[str, Any]] = []
    for batch_size in normalized_batch_sizes:
        batch_nodes = nodes[:batch_size]
        rows.append(
            _measure_full_rebuild(
                batch_nodes=batch_nodes,
                config=config,
                repeats=repeats,
            )
        )
        rows.append(
            _measure_query_activated_local(
                batch_nodes=batch_nodes,
                queries=queries,
                config=config,
                repeats=repeats,
            )
        )
        rows.append(
            _measure_lazy_insert(
                batch_nodes=batch_nodes,
                repeats=repeats,
            )
        )
    return {
        "config": {
            "strategy": strategy,
            "alpha": alpha,
            "top_k_edges": top_k_edges,
            "threshold": threshold,
            "repeats": repeats,
            "batch_sizes": normalized_batch_sizes,
            "timing_scope": (
                "只统计节点插入和图边构建时间，不统计 embedding 生成时间；"
                "embedding 对所有方法是共同前置成本。"
            ),
        },
        "summary": _summary(rows),
        "rows": rows,
    }


def write_insertion_time_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """写出在线插入时间实验的 JSON 和 Markdown 报告。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "insertion_time_benchmark.json"
    markdown_path = target / "insertion_time_benchmark.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def plot_insertion_time_figure(
    report: dict[str, Any],
    output_png: str | Path,
    *,
    output_svg: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """绘制 CAM Figure 3(a) 风格的 batch size 与插入时间关系图。"""

    import matplotlib.pyplot as plt
    import seaborn as sns

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg_path = Path(output_svg) if output_svg else None
    if output_svg_path:
        output_svg_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": ["Arial Unicode MS", "DejaVu Sans", "Arial", "sans-serif"],
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7DBE7",
            "axes.labelcolor": "#1F2430",
            "xtick.color": "#1F2430",
            "ytick.color": "#1F2430",
        }
    )
    rows = list(report["rows"])
    fig, ax = plt.subplots(figsize=(7.6, 5.3), dpi=180)
    palette = {
        "full_rebuild": "#7A828F",
        "query_activated_local": "#5477C4",
        "sam_lazy_insert": "#CC6F47",
    }
    styles = {
        "full_rebuild": (0, (2, 2)),
        "query_activated_local": "solid",
        "sam_lazy_insert": "solid",
    }
    markers = {
        "full_rebuild": "s",
        "query_activated_local": "o",
        "sam_lazy_insert": "D",
    }
    for method in ["full_rebuild", "query_activated_local", "sam_lazy_insert"]:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: row["batch_size"])
        ax.plot(
            [row["batch_size"] for row in method_rows],
            [row["time_seconds"] for row in method_rows],
            label=METHOD_LABELS[method],
            color=palette[method],
            linestyle=styles[method],
            marker=markers[method],
            linewidth=2.2,
            markersize=5,
        )
    ax.set_title("在线批量插入时间成本", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("在线批大小（memory items）")
    ax.set_ylabel("批量插入 / 图更新时间（秒）")
    ax.grid(axis="y", color="#E6E8F0", linestyle="--", linewidth=0.8)
    ax.grid(axis="x", color="#F4F5F7", linestyle=":", linewidth=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DBE7")
    ax.spines["bottom"].set_color("#D7DBE7")
    ax.text(
        0.01,
        -0.2,
        "注：为隔离建图成本，图中不统计 embedding 生成时间；三条线使用同一批已嵌入 memory items。",
        transform=ax.transAxes,
        fontsize=9,
        color="#6F768A",
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.22, top=0.9)
    fig.savefig(output_png, bbox_inches="tight")
    if output_svg_path:
        fig.savefig(output_svg_path, bbox_inches="tight")
        strip_trailing_whitespace(output_svg_path)
    plt.close(fig)
    return output_png, output_svg_path


def _normalize_batch_sizes(batch_sizes: list[int], node_count: int) -> list[int]:
    normalized = sorted({size for size in batch_sizes if 0 < size <= node_count})
    if not normalized:
        raise ValueError("batch_sizes 至少需要包含一个有效正整数")
    return normalized


def _measure_full_rebuild(
    *,
    batch_nodes: list[MemoryNode],
    config: GraphStrategyConfig,
    repeats: int,
) -> dict[str, Any]:
    results = [
        build_graph_for_strategy(batch_nodes, config, pair_scope="global")
        for _ in range(repeats)
    ]
    result = results[-1]
    return {
        "method": "full_rebuild",
        "method_label": METHOD_LABELS["full_rebuild"],
        "batch_size": len(batch_nodes),
        "time_seconds": round(statistics.median(item.build_time_seconds for item in results), 6),
        "candidate_pair_count": result.candidate_pair_count,
        "edge_count": result.edge_count,
        "pair_scope": "global",
        "repeat_times_seconds": [round(item.build_time_seconds, 6) for item in results],
    }


def _measure_query_activated_local(
    *,
    batch_nodes: list[MemoryNode],
    queries: list[EvaluationQuery],
    config: GraphStrategyConfig,
    repeats: int,
) -> dict[str, Any]:
    allowed_pair_keys = _allowed_local_pair_keys(batch_nodes, queries)
    results = [
        build_graph_for_strategy(
            batch_nodes,
            config,
            allowed_pair_keys=allowed_pair_keys,
            pair_scope="query_candidates",
        )
        for _ in range(repeats)
    ]
    result = results[-1]
    return {
        "method": "query_activated_local",
        "method_label": METHOD_LABELS["query_activated_local"],
        "batch_size": len(batch_nodes),
        "time_seconds": round(statistics.median(item.build_time_seconds for item in results), 6),
        "candidate_pair_count": result.candidate_pair_count,
        "edge_count": result.edge_count,
        "pair_scope": "query_candidates",
        "repeat_times_seconds": [round(item.build_time_seconds, 6) for item in results],
    }


def _measure_lazy_insert(
    *,
    batch_nodes: list[MemoryNode],
    repeats: int,
) -> dict[str, Any]:
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _inserted_nodes = list(batch_nodes)
        times.append(time.perf_counter() - start)
    return {
        "method": "sam_lazy_insert",
        "method_label": METHOD_LABELS["sam_lazy_insert"],
        "batch_size": len(batch_nodes),
        "time_seconds": round(statistics.median(times), 6),
        "candidate_pair_count": 0,
        "edge_count": 0,
        "pair_scope": "none",
        "repeat_times_seconds": [round(item, 6) for item in times],
    }


def _allowed_local_pair_keys(
    batch_nodes: list[MemoryNode],
    queries: list[EvaluationQuery],
) -> set[tuple[str, str]]:
    node_by_doc_id = {
        str(node.metadata.get("original_doc_id")): node
        for node in batch_nodes
        if node.metadata.get("original_doc_id")
    }
    allowed: set[tuple[str, str]] = set()
    for query in queries:
        candidate_node_ids = [
            node_by_doc_id[doc_id].id
            for doc_id in query.candidate_doc_ids
            if doc_id in node_by_doc_id
        ]
        for source_id in candidate_node_ids:
            for target_id in candidate_node_ids:
                if source_id != target_id:
                    allowed.add((source_id, target_id))
    return allowed


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_batch_size = max(int(row["batch_size"]) for row in rows)
    max_batch_rows = [row for row in rows if int(row["batch_size"]) == max_batch_size]
    time_by_method = {
        row["method"]: float(row["time_seconds"])
        for row in max_batch_rows
    }
    full_time = time_by_method.get("full_rebuild", 0.0)
    local_time = time_by_method.get("query_activated_local", 0.0)
    lazy_time = time_by_method.get("sam_lazy_insert", 0.0)
    return {
        "max_batch_size": max_batch_size,
        "full_rebuild_time_seconds_at_max_batch": round(full_time, 6),
        "query_activated_time_seconds_at_max_batch": round(local_time, 6),
        "lazy_insert_time_seconds_at_max_batch": round(lazy_time, 6),
        "local_speedup_vs_full_at_max_batch": round(full_time / local_time, 4) if local_time > 0 else None,
        "lazy_speedup_vs_full_at_max_batch": round(full_time / lazy_time, 4) if lazy_time > 0 else None,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# 在线插入时间成本实验",
        "",
        "本实验只统计节点插入和图边构建时间，不统计 embedding 生成时间。",
        "",
        "## 摘要",
        "",
    ]
    for key, value in report["summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## 结果",
            "",
            "| 批大小 | 方法 | 时间（秒） | 候选边对数 | 实际边数 |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["rows"]:
        lines.append(
            "| {batch_size} | {method_label} | {time_seconds:.6f} | {candidate_pair_count} | {edge_count} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)
