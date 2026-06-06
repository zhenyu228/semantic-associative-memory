from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.edge_audit import audit_edge_quality, write_edge_quality_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 cases.json 审计 SAM 图边质量")
    parser.add_argument("--cases-file", required=True, help="实验 run 输出的 cases.json")
    parser.add_argument("--method", default="sam_full", help="需要审计的检索方法")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认写到 cases.json 所在目录")
    parser.add_argument("--json", action="store_true", help="同时在终端打印 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_path = ROOT / args.cases_file if not Path(args.cases_file).is_absolute() else Path(args.cases_file)
    output_dir = (
        ROOT / args.output_dir
        if args.output_dir and not Path(args.output_dir).is_absolute()
        else Path(args.output_dir) if args.output_dir else cases_path.parent
    )
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("cases 文件必须是列表")
    audit = audit_edge_quality(cases, method=args.method)
    json_path, markdown_path = write_edge_quality_audit(audit, output_dir)
    if args.json:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    summary = audit.get("summary", {})
    print(f"图边质量审计完成：{json_path}")
    print(f"Markdown：{markdown_path}")
    if isinstance(summary, dict):
        print(f"图路径命中数：{summary.get('graph_hit_count', 0)}")
        print(f"噪声图路径命中数：{summary.get('noise_graph_hit_count', 0)}")
        print(f"图噪声 bad case 数：{summary.get('graph_noise_case_count', 0)}")


if __name__ == "__main__":
    main()
