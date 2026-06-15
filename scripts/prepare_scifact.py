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
from sam.datasets import DATASET_REFERENCES, download_scifact_data, load_scifact_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 SciFact 转换成 SAM 项目统一数据格式")
    parser.add_argument("--source", default="data/raw/scifact", help="SciFact 原始目录；缺失时可配合 --download 自动下载")
    parser.add_argument("--output", default="data/processed/scifact_sam_sample.json", help="SAM 格式输出路径")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev", help="SciFact claim split")
    parser.add_argument("--sample-size", type=int, default=50, help="抽取 claim 数量")
    parser.add_argument("--negative-docs-per-query", type=int, default=20, help="每个 claim 额外加入的 hard negative 文档数")
    parser.add_argument("--max-corpus-docs", type=int, default=0, help="最多读取多少篇 corpus 文档；0 表示读取全部")
    parser.add_argument("--download", action="store_true", help="若 source 不完整，则从 SciFact 官方地址下载并解压")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = ROOT / args.source
    if args.download:
        source_path = download_scifact_data(source_path)
    documents, queries, manifest = load_scifact_sample(
        source_path=source_path,
        split=args.split,
        sample_size=args.sample_size,
        negative_docs_per_query=args.negative_docs_per_query,
        max_corpus_docs=args.max_corpus_docs,
    )
    output_path = save_sam_dataset(
        path=ROOT / args.output,
        documents=documents,
        queries=queries,
        dataset_info=DATASET_REFERENCES["scifact"],
        processing={
            "source_script": "scripts/prepare_scifact.py",
            "source_path": str(source_path),
            "split": args.split,
            "sample_size": args.sample_size,
            "negative_docs_per_query": args.negative_docs_per_query,
            "max_corpus_docs": args.max_corpus_docs,
            "selection_policy": "选择带 gold evidence 的 claim；候选集包含 evidence docs、cited docs 和词项重叠 hard negatives",
            "manifest": manifest,
        },
    )
    print("SciFact 已转换为 SAM 数据格式")
    print(json.dumps(summarize_sam_dataset(output_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
