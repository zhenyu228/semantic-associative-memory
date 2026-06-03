from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.experiment_audit import audit_run_directory, write_experiment_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 SAM 实验 run，输出瓶颈和下一步改进建议")
    parser.add_argument("run_dir", help="实验 run 目录，例如 outputs/runs/provider_smoke_local_check")
    parser.add_argument("--primary-method", default="sam_full", help="主方法，默认 sam_full")
    parser.add_argument("--baseline-method", default="embedding_topk", help="baseline 方法，默认 embedding_topk")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_run_directory(
        ROOT / args.run_dir,
        primary_method=args.primary_method,
        baseline_method=args.baseline_method,
    )
    json_path, markdown_path = write_experiment_audit(audit, ROOT / args.run_dir)
    print("SAM 实验审计完成")
    print(f"JSON：{json_path}")
    print(f"Markdown：{markdown_path}")
    print(f"识别瓶颈数：{len(audit['bottlenecks'])}")


if __name__ == "__main__":
    main()
