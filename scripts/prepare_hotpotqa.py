from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.dataset_format import save_sam_dataset, summarize_sam_dataset  # noqa: E402
from sam.datasets import DATASET_REFERENCES, download_hotpotqa_dev, load_hotpotqa_real_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 HotpotQA 转换成 SAM 项目统一数据格式")
    parser.add_argument("--raw-path", default="data/raw/hotpot_dev_distractor_v1.json", help="HotpotQA 原始 JSON 路径")
    parser.add_argument("--output", default="data/processed/hotpotqa_sam_sample.json", help="SAM 格式输出路径")
    parser.add_argument("--sample-size", type=int, default=8, help="抽取问题数量")
    parser.add_argument("--max-scan", type=int, default=800, help="最多扫描 HotpotQA 原始样本数量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = download_hotpotqa_dev(ROOT / args.raw_path)
    documents, queries, manifest = load_hotpotqa_real_sample(
        raw_path=raw_path,
        sample_size=args.sample_size,
        max_scan=args.max_scan,
    )
    output_path = save_sam_dataset(
        path=ROOT / args.output,
        documents=documents,
        queries=queries,
        dataset_info=DATASET_REFERENCES["hotpotqa_real"],
        processing={
            "source_script": "scripts/prepare_hotpotqa.py",
            "raw_path": str(raw_path),
            "sample_size": args.sample_size,
            "max_scan": args.max_scan,
            "selection_policy": "选择 supporting paragraph 之间存在标题提及的 bridge-style 样本",
            "manifest": manifest,
        },
    )
    summary = summarize_sam_dataset(output_path)
    print("HotpotQA 已转换为 SAM 数据格式")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

