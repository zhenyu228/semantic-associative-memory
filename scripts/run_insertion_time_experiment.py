#!/usr/bin/env python
"""运行 SAM online batch insertion time 实验并生成时间成本图。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam.dataset_format import load_sam_dataset
from sam.datasets import documents_to_nodes
from sam.insertion_time_experiment import (
    plot_insertion_time_figure,
    run_insertion_time_benchmark,
    write_insertion_time_report,
)

from scripts.run_graph_strategy_experiment import _attach_context_metadata, _create_embedding_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较 online batch size 下的插入/建图时间成本。")
    parser.add_argument("--dataset-file", default="data/processed/litsearch_query30_sam_sample.json", help="SAM 统一数据格式文件")
    parser.add_argument("--output-dir", default="outputs/insertion_time_litsearch", help="实验输出目录")
    parser.add_argument("--batch-sizes", default="25,50,100,200,300,500,630", help="逗号分隔的 online batch size")
    parser.add_argument("--strategy", default="sam_context", help="全量图和局部图使用的同一建边评分策略")
    parser.add_argument("--alpha", type=float, default=0.55, help="sam_context/cam_style 中语义相似度权重")
    parser.add_argument("--top-k-edges", type=int, default=4, help="每个节点最多保留多少条边")
    parser.add_argument("--threshold", type=float, default=0.18, help="建边阈值")
    parser.add_argument("--repeats", type=int, default=3, help="每个点重复计时次数，取中位数")
    parser.add_argument("--limit-docs", type=int, default=0, help="最多使用多少个文档；0 表示不限制")
    parser.add_argument(
        "--context-path-policy",
        choices=["intrinsic", "metadata", "query_grouped_legacy"],
        default="intrinsic",
        help="context_path 构造策略",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["local_hash", "openai", "azure_openai", "azure_openai_sdk", "sentence_transformers"],
        default="local_hash",
        help="embedding provider。时间图默认使用 local_hash，因为本实验隔离建图时间，不统计 embedding API 时间。",
    )
    parser.add_argument("--embedding-concurrency", type=int, default=None, help="在线 embedding 最大并发数")
    parser.add_argument("--embedding-batch-size", type=int, default=None, help="在线 embedding 批大小")
    parser.add_argument("--embedding-cache", action="store_true", help="启用 embedding 缓存")
    parser.add_argument("--embedding-cache-path", default=None, help="embedding 缓存 SQLite 路径")
    parser.add_argument(
        "--embedding-input-mode",
        choices=["single", "batch"],
        default=None,
        help="azure_openai_sdk 输入模式",
    )
    parser.add_argument("--figure-png", default="docs/figures/sam_insertion_time_figure.png", help="输出 PNG 图路径")
    parser.add_argument("--figure-svg", default="docs/figures/sam_insertion_time_figure.svg", help="输出 SVG 图路径")
    parser.add_argument("--figure-data", default="docs/figures/sam_insertion_time_figure_data.json", help="输出作图数据路径")
    return parser.parse_args()


def _parse_batch_sizes(raw: str) -> list[int]:
    sizes = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not sizes:
        raise ValueError("--batch-sizes 至少需要一个正整数")
    return sizes


def main() -> None:
    args = parse_args()
    documents, queries, _payload = load_sam_dataset(args.dataset_file)
    if args.limit_docs > 0:
        documents = documents[: args.limit_docs]
    embedding = _create_embedding_provider(
        args.embedding_provider,
        embedding_concurrency=args.embedding_concurrency,
        embedding_batch_size=args.embedding_batch_size,
        embedding_input_mode=args.embedding_input_mode,
        embedding_cache=args.embedding_cache,
        embedding_cache_path=args.embedding_cache_path,
    )
    nodes = documents_to_nodes(documents, embedding)
    context_audit = _attach_context_metadata(nodes, policy=args.context_path_policy)
    report = run_insertion_time_benchmark(
        nodes=nodes,
        queries=queries,
        batch_sizes=_parse_batch_sizes(args.batch_sizes),
        strategy=args.strategy,
        alpha=args.alpha,
        top_k_edges=args.top_k_edges,
        threshold=args.threshold,
        repeats=args.repeats,
    )
    report["dataset"] = {
        "dataset_file": args.dataset_file,
        "document_count": len(documents),
        "query_count": len(queries),
    }
    report["context_path"] = context_audit
    json_path, markdown_path = write_insertion_time_report(report, args.output_dir)
    png_path, svg_path = plot_insertion_time_figure(
        report,
        args.figure_png,
        output_svg=args.figure_svg or None,
    )
    figure_data_path = Path(args.figure_data)
    figure_data_path.parent.mkdir(parents=True, exist_ok=True)
    figure_data_path.write_text(json.dumps(report["rows"], ensure_ascii=False, indent=2), encoding="utf-8")
    print("Online insertion time 实验完成")
    print(f"JSON：{json_path}")
    print(f"Markdown：{markdown_path}")
    print(f"PNG：{png_path}")
    if svg_path:
        print(f"SVG：{svg_path}")
    print(f"作图数据：{figure_data_path}")


if __name__ == "__main__":
    main()
