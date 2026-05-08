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
from sam.datasets import DATASET_REFERENCES, load_novelqa_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将本地 NovelQA zip/目录转换成 SAM 项目统一数据格式")
    parser.add_argument("--source", default="data/raw/NovelQA", help="NovelQA 本地 zip 或解压目录")
    parser.add_argument("--output", default="data/processed/novelqa_sam_sample.json", help="SAM 格式输出路径")
    parser.add_argument("--sample-size", type=int, default=8, help="抽取问题数量")
    parser.add_argument("--max-books", type=int, default=1, help="最多读取小说数量")
    parser.add_argument("--chunk-chars", type=int, default=1800, help="小说切块字符数")
    parser.add_argument("--chunk-overlap", type=int, default=180, help="相邻 chunk 的重叠字符数")
    parser.add_argument("--max-chunks-per-book", type=int, default=80, help="每本小说最多保留 chunk 数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents, queries, manifest = load_novelqa_sample(
        source_path=ROOT / args.source,
        sample_size=args.sample_size,
        max_books=args.max_books,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        max_chunks_per_book=args.max_chunks_per_book,
    )
    output_path = save_sam_dataset(
        path=ROOT / args.output,
        documents=documents,
        queries=queries,
        dataset_info=DATASET_REFERENCES["novelqa"],
        processing={
            "source_script": "scripts/prepare_novelqa.py",
            "source_path": str(ROOT / args.source),
            "sample_size": args.sample_size,
            "max_books": args.max_books,
            "chunk_chars": args.chunk_chars,
            "chunk_overlap": args.chunk_overlap,
            "max_chunks_per_book": args.max_chunks_per_book,
            "selection_policy": "按 NovelQA 本地文件顺序读取小说与 QA，小说正文切分为固定窗口 chunk",
            "manifest": manifest,
        },
    )
    print("NovelQA 已转换为 SAM 数据格式")
    print(json.dumps(summarize_sam_dataset(output_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
