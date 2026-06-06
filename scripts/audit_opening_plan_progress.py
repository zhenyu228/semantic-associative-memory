from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.opening_audit import build_opening_plan_audit, write_opening_plan_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 SAM 当前实现与开题计划的对齐进度")
    parser.add_argument("--output-dir", default="docs", help="审计报告输出目录，默认 docs")
    parser.add_argument("--json", action="store_true", help="同时在终端输出 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_opening_plan_audit(ROOT)
    output_dir = ROOT / args.output_dir
    json_path, markdown_path = write_opening_plan_audit(audit, output_dir)
    if args.json:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    else:
        print("开题计划进度审计完成")
        print(f"估算总体进度：{audit['overall_progress']}%")
        print(f"JSON：{json_path}")
        print(f"Markdown：{markdown_path}")


if __name__ == "__main__":
    main()
