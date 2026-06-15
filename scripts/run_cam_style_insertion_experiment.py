#!/usr/bin/env python
"""运行 CAM Figure 3(a) 同口径的 SAM 在线插入时间实验。"""

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
from sam.env import load_env_file
from sam.insertion_time_experiment import (
    plot_cam_style_insertion_time_figure,
    run_cam_style_insertion_benchmark,
    write_insertion_time_report,
)
from sam.llm import create_chat_client
from sam.semantic_compressor import ExtractiveSemanticCompressor, LLMSemanticCompressor

from scripts.run_graph_strategy_experiment import _attach_context_metadata, _create_embedding_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复用 CAM Figure 3(a) 口径测试 SAM 在线插入时间。")
    parser.add_argument("--dataset-file", default="data/processed/litsearch_query30_sam_sample.json", help="SAM 统一数据格式文件")
    parser.add_argument("--output-dir", default="outputs/cam_style_insertion_litsearch", help="实验输出目录")
    parser.add_argument("--batch-sizes", default="1,100,200,300,400,500", help="逗号分隔的 online batch size")
    parser.add_argument("--chunk-token-size", type=int, default=512, help="CAM 口径中的 chunk token size")
    parser.add_argument("--compression-group-size", type=int, default=20, help="每次高层语义压缩聚合多少个 memory items")
    parser.add_argument("--compressor-mode", choices=["extractive", "llm"], default="extractive", help="高层语义压缩实现")
    parser.add_argument("--env-file", default=None, help="加载本地环境变量文件，例如 .env.local")
    parser.add_argument("--chat-provider", default=None, help="compressor-mode=llm 时使用的 chat provider")
    parser.add_argument("--llm-max-tokens", type=int, default=300, help="LLM 高层压缩输出 token 上限")
    parser.add_argument("--strategy", default="sam_context", help="低层非 LLM 建边评分策略")
    parser.add_argument("--alpha", type=float, default=0.55, help="sam_context/cam_style 中语义相似度权重")
    parser.add_argument("--top-k-edges", type=int, default=4, help="每个节点最多保留多少条低层边")
    parser.add_argument("--threshold", type=float, default=0.18, help="低层建边阈值")
    parser.add_argument("--repeats", type=int, default=3, help="每个点重复计时次数，取中位数；LLM 模式建议设为 1")
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
        help="embedding provider。默认 local_hash，用于隔离插入和建图时间。",
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
    parser.add_argument("--figure-png", default="docs/figures/sam_cam_style_insertion_time.png", help="输出 PNG 图路径")
    parser.add_argument("--figure-svg", default="docs/figures/sam_cam_style_insertion_time.svg", help="输出 SVG 图路径")
    parser.add_argument("--figure-data", default="docs/figures/sam_cam_style_insertion_time_data.json", help="输出作图数据路径")
    return parser.parse_args()


def _parse_batch_sizes(raw: str) -> list[int]:
    sizes = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not sizes:
        raise ValueError("--batch-sizes 至少需要一个正整数")
    return sizes


def _create_compressor(args: argparse.Namespace):
    if args.compressor_mode == "extractive":
        return ExtractiveSemanticCompressor(max_sentences=4)
    return LLMSemanticCompressor(
        create_chat_client(args.chat_provider),
        max_tokens=args.llm_max_tokens,
    )


def _token_stats(texts: list[str]) -> dict[str, float | int]:
    lengths = [len(text.split()) for text in texts]
    if not lengths:
        return {"count": 0, "average_tokens": 0.0, "max_tokens": 0, "min_tokens": 0}
    return {
        "count": len(lengths),
        "average_tokens": round(sum(lengths) / len(lengths), 2),
        "max_tokens": max(lengths),
        "min_tokens": min(lengths),
    }


def main() -> None:
    args = parse_args()
    if args.env_file:
        load_env_file(args.env_file)
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
    compressor = _create_compressor(args)
    report = run_cam_style_insertion_benchmark(
        nodes=nodes,
        queries=queries,
        batch_sizes=_parse_batch_sizes(args.batch_sizes),
        compressor=compressor,
        compression_group_size=args.compression_group_size,
        chunk_token_size=args.chunk_token_size,
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
        "token_stats": _token_stats([document.text for document in documents]),
        "note": "CAM 原实验使用 512-token chunks；本字段记录当前 SAM 数据文件的实际 token 长度。",
    }
    report["context_path"] = context_audit
    report["compressor"] = {
        "mode": args.compressor_mode,
        "compression_group_size": args.compression_group_size,
        "llm_max_tokens": args.llm_max_tokens if args.compressor_mode == "llm" else None,
    }
    json_path, markdown_path = write_insertion_time_report(report, args.output_dir)
    png_path, svg_path = plot_cam_style_insertion_time_figure(
        report,
        args.figure_png,
        output_svg=args.figure_svg or None,
    )
    figure_data_path = Path(args.figure_data)
    figure_data_path.parent.mkdir(parents=True, exist_ok=True)
    figure_data_path.write_text(json.dumps(report["rows"], ensure_ascii=False, indent=2), encoding="utf-8")
    print("CAM-style online insertion time 实验完成")
    print(f"JSON：{json_path}")
    print(f"Markdown：{markdown_path}")
    print(f"PNG：{png_path}")
    if svg_path:
        print(f"SVG：{svg_path}")
    print(f"作图数据：{figure_data_path}")


if __name__ == "__main__":
    main()
