from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.datasets import (  # noqa: E402
    load_builtin_benchmark_sample,
    load_multihop_rag_from_huggingface,
    write_dataset_manifest,
)
from sam.embedding import create_embedding_provider  # noqa: E402
from sam.evaluator import Evaluator  # noqa: E402
from sam.graph import GraphBuilder  # noqa: E402
from sam.store import MemoryStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 SAM 两阶段检索 demo")
    parser.add_argument("--db", default="data/sam_demo.sqlite", help="SQLite 数据库路径")
    parser.add_argument("--report-dir", default="reports", help="实验报告输出目录")
    parser.add_argument("--embedding-provider", default=None, help="local 或 openai")
    parser.add_argument("--top-k", type=int, default=2, help="最终返回文档数")
    parser.add_argument("--seed-k", type=int, default=1, help="联想检索种子节点数")
    parser.add_argument("--hops", type=int, default=2, help="图扩展跳数")
    parser.add_argument("--reset", action="store_true", help="运行前清空本地记忆库")
    parser.add_argument(
        "--try-download",
        action="store_true",
        help="尝试下载公开数据集元信息；失败后自动使用内置小样本",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = MemoryStore(ROOT / args.db)
    if args.reset:
        store.reset()

    report_dir = ROOT / args.report_dir
    write_dataset_manifest(report_dir / "dataset_references.json")
    if args.try_download:
        try:
            info = load_multihop_rag_from_huggingface(report_dir / "multihop_rag_readme.md")
            print(f"已下载公开数据集元信息：{info}")
        except Exception as exc:
            print(f"公开数据集元信息下载失败，使用内置小样本兜底：{exc}")

    embedding_provider = create_embedding_provider(args.embedding_provider)
    graph_builder = GraphBuilder(store)
    evaluator = Evaluator(store, embedding_provider, graph_builder)

    documents, queries = load_builtin_benchmark_sample()
    evaluator.ingest(documents)
    result = evaluator.evaluate(
        queries=queries,
        top_k=args.top_k,
        seed_k=args.seed_k,
        hops=args.hops,
    )
    json_path, markdown_path = evaluator.write_reports(result, report_dir)

    print("SAM demo 已完成")
    print(f"查询数量：{result.query_count}")
    print(f"纯向量证据召回率：{result.vector_recall:.3f}")
    print(f"联想检索证据召回率：{result.associative_recall:.3f}")
    print(f"联想检索新增有效证据数：{result.associative_gain}")
    print(f"JSON 结果：{json_path}")
    print(f"Markdown 报告：{markdown_path}")


if __name__ == "__main__":
    main()

