#!/usr/bin/env python
"""根据图联想补证据实验结果生成成本-效果图。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.cost_effect_figure import build_cost_effect_rows, load_evidence_rescue_reports, plot_cost_effect_figure


DEFAULT_REPORTS = {
    "HotpotQA": "outputs/evidence_rescue_hotpotqa30/evidence_rescue_results.json",
    "QASPER": "outputs/evidence_rescue_qasper30/evidence_rescue_results.json",
    "LitSearch": "outputs/evidence_rescue_litsearch30/evidence_rescue_results.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 SAM 按需建图成本-效果双面板图。")
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="实验结果，格式为 数据集名=路径。可重复传入；不传则使用三组默认正式实验结果。",
    )
    parser.add_argument("--method", default="sam_context", help="用于作图的策略名，默认 sam_context。")
    parser.add_argument("--method-label", default="SAM-context", help="图中展示的方法名称。")
    parser.add_argument(
        "--output-png",
        default="docs/figures/sam_cost_effect_figure.png",
        help="输出 PNG 路径。",
    )
    parser.add_argument(
        "--output-svg",
        default="docs/figures/sam_cost_effect_figure.svg",
        help="输出 SVG 路径；传空字符串则不生成 SVG。",
    )
    parser.add_argument(
        "--output-data",
        default="docs/figures/sam_cost_effect_figure_data.json",
        help="输出作图数据 JSON 路径。",
    )
    return parser.parse_args()


def _parse_report_args(items: list[str]) -> dict[str, Path]:
    if not items:
        return {name: ROOT / path for name, path in DEFAULT_REPORTS.items()}
    reports: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--report 需要使用 数据集名=路径 格式：{item}")
        name, path = item.split("=", 1)
        reports[name.strip()] = Path(path).expanduser()
    return reports


def main() -> None:
    args = parse_args()
    report_paths = _parse_report_args(args.report)
    reports = load_evidence_rescue_reports(report_paths)
    rows = build_cost_effect_rows(reports, method=args.method)

    output_data = Path(args.output_data)
    output_data.parent.mkdir(parents=True, exist_ok=True)
    output_data.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    output_svg = args.output_svg or None
    png_path, svg_path = plot_cost_effect_figure(
        rows,
        args.output_png,
        output_svg=output_svg,
        method_label=args.method_label,
    )
    print(f"成本-效果图 PNG：{png_path}")
    if svg_path:
        print(f"成本-效果图 SVG：{svg_path}")
    print(f"作图数据：{output_data}")


if __name__ == "__main__":
    main()
