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
from sam.datasets import DATASET_REFERENCES, load_qasper_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 QASPER 转换成 SAM 项目统一数据格式")
    parser.add_argument("--source", default="", help="QASPER 本地目录/JSON；为空时从 Hugging Face 读取")
    parser.add_argument("--output", default="data/processed/qasper_sam_sample.json", help="SAM 格式输出路径")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation", help="QASPER split")
    parser.add_argument("--sample-size", type=int, default=30, help="抽取 QA 数量")
    parser.add_argument("--max-papers", type=int, default=20, help="最多扫描多少篇论文")
    parser.add_argument("--max-paragraphs-per-paper", type=int, default=120, help="每篇论文最多保留多少段落")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = ROOT / args.source if args.source else None
    documents, queries, manifest = load_qasper_sample(
        source_path=source_path,
        split=args.split,
        sample_size=args.sample_size,
        max_papers=args.max_papers,
        max_paragraphs_per_paper=args.max_paragraphs_per_paper,
    )
    output_path = save_sam_dataset(
        path=ROOT / args.output,
        documents=documents,
        queries=queries,
        dataset_info=DATASET_REFERENCES["qasper"],
        processing={
            "source_script": "scripts/prepare_qasper.py",
            "source_path": str(source_path or "huggingface"),
            "split": args.split,
            "sample_size": args.sample_size,
            "max_papers": args.max_papers,
            "max_paragraphs_per_paper": args.max_paragraphs_per_paper,
            "selection_policy": "论文段落作为 MemoryItem；QA evidence 文本匹配到段落作为 gold supporting docs",
            "manifest": manifest,
        },
    )
    print("QASPER 已转换为 SAM 数据格式")
    print(json.dumps(summarize_sam_dataset(output_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
