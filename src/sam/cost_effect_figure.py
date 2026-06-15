"""生成 SAM 按需建图成本-效果图所需的数据和静态图。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_evidence_rescue_reports(paths: dict[str, str | Path]) -> dict[str, dict[str, Any]]:
    """读取多组图联想补证据实验结果。"""

    reports: dict[str, dict[str, Any]] = {}
    for dataset_name, path in paths.items():
        reports[dataset_name] = json.loads(Path(path).read_text(encoding="utf-8"))
    return reports


def strip_trailing_whitespace(path: str | Path) -> None:
    """清理文本文件行尾空格，避免生成的 SVG 触发 diff 检查。"""

    target = Path(path)
    lines = target.read_text(encoding="utf-8").splitlines()
    target.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def build_cost_effect_rows(
    reports: dict[str, dict[str, Any]],
    *,
    method: str = "sam_context",
) -> list[dict[str, Any]]:
    """从实验结果中抽取 Figure 需要的成本和召回指标。"""

    rows: list[dict[str, Any]] = []
    for dataset_name, report in reports.items():
        strategies = report.get("strategies", {})
        if method not in strategies:
            available = ", ".join(sorted(strategies))
            raise KeyError(f"实验结果缺少方法 {method!r}，可用方法：{available}")
        strategy = strategies[method]
        cost = strategy.get("cost", {})
        metrics = strategy.get("metrics", {})
        dataset = report.get("dataset", {})
        candidate_pair_coverage = float(cost.get("candidate_pair_coverage", 0.0))
        baseline_recall = float(metrics.get("baseline_evidence_recall", 0.0))
        rescue_recall = float(metrics.get("evidence_recall_with_rescue", 0.0))
        recall_gain = float(metrics.get("recall_gain", rescue_recall - baseline_recall))
        rows.append(
            {
                "dataset": dataset_name,
                "document_count": int(dataset.get("document_count", 0)),
                "query_count": int(dataset.get("query_count", 0)),
                "full_pair_count": int(cost.get("theoretical_full_pair_count", 0)),
                "candidate_pair_count": int(cost.get("candidate_pair_count", 0)),
                "edge_count": int(cost.get("edge_count", 0)),
                "candidate_pair_coverage_percent": round(candidate_pair_coverage * 100, 4),
                "build_time_seconds": float(cost.get("build_time_seconds", 0.0)),
                "uses_llm": bool(cost.get("uses_llm", False)),
                "baseline_recall_percent": round(baseline_recall * 100, 4),
                "rescue_recall_percent": round(rescue_recall * 100, 4),
                "recall_gain_pp": round(recall_gain * 100, 4),
            }
        )
    return rows


def plot_cost_effect_figure(
    rows: list[dict[str, Any]],
    output_png: str | Path,
    *,
    output_svg: str | Path | None = None,
    method_label: str = "SAM-context",
) -> tuple[Path, Path | None]:
    """绘制类似 CAM Figure 3 的双面板成本-效果图。"""

    import matplotlib.pyplot as plt
    import numpy as np
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
            "axes.edgecolor": "#D7DBE7",
            "axes.labelcolor": "#1F2430",
            "xtick.color": "#1F2430",
            "ytick.color": "#1F2430",
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
        }
    )

    datasets = [str(row["dataset"]) for row in rows]
    x = np.arange(len(datasets))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), dpi=180)
    fig.suptitle(
        "Figure: SAM 按需建图的成本-效果分析",
        fontsize=15,
        fontweight="bold",
        x=0.03,
        y=0.965,
        ha="left",
        color="#1F2430",
    )
    fig.text(
        0.03,
        0.91,
        "成本口径为候选边比较次数和保留边数量；效果口径为 evidence recall。标注为实际比较边 / 全量理论候选边；建边不调用大模型。",
        fontsize=10,
        color="#6F768A",
        ha="left",
    )

    full_pairs = [row["full_pair_count"] for row in rows]
    candidate_pairs = [row["candidate_pair_count"] for row in rows]
    kept_edges = [row["edge_count"] for row in rows]
    coverage = [row["candidate_pair_coverage_percent"] for row in rows]

    ax = axes[0]
    ax.bar(x - width, full_pairs, width, label="全量图理论候选边", color="#C5CAD3", edgecolor="#7A828F")
    ax.bar(x, candidate_pairs, width, label=f"{method_label} 实际比较边", color="#A3BEFA", edgecolor="#2E4780")
    ax.bar(x + width, kept_edges, width, label=f"{method_label} 保留边", color="#A3D576", edgecolor="#386411")
    ax.set_yscale("log")
    ax.set_ylabel("边数量（log scale）")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_title("(a) 建图成本：只比较局部候选边", loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#E6E8F0", linestyle="--", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    for i, pct in enumerate(coverage):
        ax.text(
            x[i],
            candidate_pairs[i] * 1.18,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2E4780",
            fontweight="bold",
        )
    baseline = [row["baseline_recall_percent"] for row in rows]
    rescue = [row["rescue_recall_percent"] for row in rows]
    gain = [row["recall_gain_pp"] for row in rows]

    ax = axes[1]
    ax.bar(x - width / 2, baseline, width, label="Embedding Top-k", color="#C5CAD3", edgecolor="#7A828F")
    ax.bar(x + width / 2, rescue, width, label=f"{method_label} 补证据后", color="#FFE15B", edgecolor="#736422")
    ax.set_ylim(45, 100)
    ax.set_ylabel("Evidence recall (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_title("(b) 检索效果：局部图补回遗漏证据", loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#E6E8F0", linestyle="--", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    for i, delta in enumerate(gain):
        ax.plot([x[i] - width / 2, x[i] + width / 2], [baseline[i], rescue[i]], color="#804126", linewidth=1.2)
        ax.text(
            x[i] + width / 2,
            rescue[i] + 1.2,
            f"+{delta:.1f} pp",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#804126",
            fontweight="bold",
        )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#D7DBE7")
        axis.spines["bottom"].set_color("#D7DBE7")

    fig.subplots_adjust(top=0.78, left=0.07, right=0.985, bottom=0.13, wspace=0.24)
    fig.savefig(output_png, bbox_inches="tight")
    if output_svg_path:
        fig.savefig(output_svg_path, bbox_inches="tight")
        strip_trailing_whitespace(output_svg_path)
    plt.close(fig)
    return output_png, output_svg_path
