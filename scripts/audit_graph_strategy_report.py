from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.graph_strategy_audit import load_and_audit_graph_strategy_report, write_graph_strategy_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计建图策略实验报告是否包含完整成本、效果和无泄漏证据")
    parser.add_argument("--report", required=True, help="graph_strategy_results.json 路径")
    parser.add_argument("--output-dir", default="", help="审计结果输出目录；默认写到 report 所在目录")
    parser.add_argument(
        "--expected-pair-scope",
        choices=["global", "query_candidates"],
        default=None,
        help="期望的建图候选节点对范围",
    )
    parser.add_argument(
        "--require-real-embedding",
        action="store_true",
        help="要求正式实验不能使用 local_hash embedding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report)
    output_dir = Path(args.output_dir) if args.output_dir else report_path.parent
    audit = load_and_audit_graph_strategy_report(
        report_path,
        expected_pair_scope=args.expected_pair_scope,
        require_real_embedding=args.require_real_embedding,
    )
    json_path, md_path = write_graph_strategy_audit(audit, output_dir)
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))
    print(f"审计 JSON：{json_path}")
    print(f"审计 Markdown：{md_path}")
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
