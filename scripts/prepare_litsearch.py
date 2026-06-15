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
from sam.datasets import DATASET_REFERENCES, load_litsearch_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 LitSearch 转换成 SAM 项目统一数据格式")
    parser.add_argument("--source", default="", help="LitSearch 本地目录/JSON；为空时从 Hugging Face 读取")
    parser.add_argument("--output", default="data/processed/litsearch_sam_sample.json", help="SAM 格式输出路径")
    parser.add_argument("--sample-size", type=int, default=30, help="抽取 query 数量")
    parser.add_argument("--negative-docs-per-query", type=int, default=20, help="每个 query 额外加入的 hard negative 文档数")
    parser.add_argument("--max-corpus-docs", type=int, default=0, help="最多扫描多少篇 corpus 文档；0 表示尽量扫描完整 corpus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = ROOT / args.source if args.source else None
    documents, queries, manifest = load_litsearch_sample(
        source_path=source_path,
        sample_size=args.sample_size,
        negative_docs_per_query=args.negative_docs_per_query,
        max_corpus_docs=args.max_corpus_docs,
    )
    output_path = save_sam_dataset(
        path=ROOT / args.output,
        documents=documents,
        queries=queries,
        dataset_info=DATASET_REFERENCES["litsearch"],
        processing={
            "source_script": "scripts/prepare_litsearch.py",
            "source_path": str(source_path or "huggingface"),
            "sample_size": args.sample_size,
            "negative_docs_per_query": args.negative_docs_per_query,
            "max_corpus_docs": args.max_corpus_docs,
            "selection_policy": "选择带 gold corpusids 且 gold paper 已加载的 query；候选集包含 gold papers 和词项重叠 hard negatives",
            "manifest": manifest,
        },
    )
    print("LitSearch 已转换为 SAM 数据格式")
    print(json.dumps(summarize_sam_dataset(output_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
