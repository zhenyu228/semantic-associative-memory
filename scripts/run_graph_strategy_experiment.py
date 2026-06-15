from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.dataset_format import load_sam_dataset
from sam.datasets import documents_to_nodes
from sam.embedding import LocalHashEmbeddingProvider, create_embedding_provider
from sam.env import load_default_env_file
from sam.graph_strategy_experiment import (
    GraphStrategyExperiment,
    run_alpha_sweep,
    write_alpha_sweep_report,
    write_graph_strategy_report,
)


DEFAULT_STRATEGIES = [
    "no_graph",
    "semantic_only",
    "position_only",
    "cam_style",
    "context_path_only",
    "sam_context",
]


def _create_embedding_provider(
    provider_name: str,
    *,
    embedding_concurrency: int | None = None,
    embedding_batch_size: int | None = None,
    embedding_input_mode: str | None = None,
):
    """创建 embedding provider；远端 provider 先加载本地环境变量。"""

    if provider_name == "local_hash":
        return LocalHashEmbeddingProvider()
    load_default_env_file()
    _apply_embedding_runtime_options(
        embedding_concurrency=embedding_concurrency,
        embedding_batch_size=embedding_batch_size,
        embedding_input_mode=embedding_input_mode,
    )
    return create_embedding_provider(provider_name)


def _apply_embedding_runtime_options(
    *,
    embedding_concurrency: int | None,
    embedding_batch_size: int | None,
    embedding_input_mode: str | None,
) -> None:
    """把脚本参数转为 provider 使用的环境变量。

    provider 构造函数统一从环境变量读取配置；这里让命令行参数覆盖
    `.env.local`，避免真实实验时还要手动 export 多个变量。
    """

    if embedding_concurrency is not None and embedding_concurrency > 0:
        os.environ["SAM_AZURE_EMBEDDING_CONCURRENCY"] = str(embedding_concurrency)
        os.environ["SAM_OPENAI_EMBEDDING_CONCURRENCY"] = str(embedding_concurrency)
    if embedding_batch_size is not None and embedding_batch_size > 0:
        os.environ["SAM_AZURE_EMBEDDING_BATCH_SIZE"] = str(embedding_batch_size)
        os.environ["SAM_OPENAI_EMBEDDING_BATCH_SIZE"] = str(embedding_batch_size)
    if embedding_input_mode:
        os.environ["SAM_AZURE_EMBEDDING_INPUT_MODE"] = embedding_input_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="比较多种非 LLM 建图策略的效果、耗时和边规模")
    parser.add_argument("--dataset-file", default="data/processed/hotpotqa_sam_sample.json", help="SAM 统一数据格式文件")
    parser.add_argument("--output-dir", default="outputs/graph_strategy_experiment", help="实验输出目录")
    parser.add_argument("--limit-queries", type=int, default=30, help="最多评测多少条 query")
    parser.add_argument("--alpha", type=float, default=0.55, help="语义相似度权重")
    parser.add_argument("--top-k-edges", type=int, default=4, help="每个节点最多保留多少条出边")
    parser.add_argument("--threshold", type=float, default=0.18, help="建边得分阈值")
    parser.add_argument("--top-k", type=int, default=4, help="检索返回证据数")
    parser.add_argument("--seed-k", type=int, default=1, help="初始召回种子数")
    parser.add_argument("--hops", type=int, default=1, help="图扩展跳数")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES), help="逗号分隔的策略列表")
    parser.add_argument("--alpha-sweep", default="", help="可选，逗号分隔 alpha 列表，例如 0,0.25,0.5,0.75,1")
    parser.add_argument(
        "--embedding-provider",
        choices=["local_hash", "openai", "azure_openai", "azure_openai_sdk", "sentence_transformers"],
        default="local_hash",
        help="embedding provider；正式实验可换成 azure_openai_sdk",
    )
    parser.add_argument("--embedding-concurrency", type=int, default=None, help="在线 embedding 最大并发数")
    parser.add_argument("--embedding-batch-size", type=int, default=None, help="在线 embedding 批大小；batch 模式下更重要")
    parser.add_argument(
        "--embedding-input-mode",
        choices=["single", "batch"],
        default=None,
        help="azure_openai_sdk 输入模式；single 表示每条文本一个异步请求，batch 表示按批发送列表",
    )
    args = parser.parse_args()

    documents, queries, _payload = load_sam_dataset(args.dataset_file)
    if args.limit_queries > 0:
        queries = queries[: args.limit_queries]
        candidate_doc_ids = {
            doc_id
            for query in queries
            for doc_id in query.candidate_doc_ids
        }
        documents = [document for document in documents if document.id in candidate_doc_ids]

    embedding = _create_embedding_provider(
        args.embedding_provider,
        embedding_concurrency=args.embedding_concurrency,
        embedding_batch_size=args.embedding_batch_size,
        embedding_input_mode=args.embedding_input_mode,
    )
    nodes = documents_to_nodes(documents, embedding)
    _attach_context_metadata(nodes)
    experiment = GraphStrategyExperiment(
        nodes=nodes,
        queries=queries,
        alpha=args.alpha,
        top_k_edges=args.top_k_edges,
        threshold=args.threshold,
    )
    report = experiment.run(
        strategies=[strategy.strip() for strategy in args.strategies.split(",") if strategy.strip()],
        top_k=args.top_k,
        seed_k=args.seed_k,
        hops=args.hops,
    )
    json_path, markdown_path = write_graph_strategy_report(report, args.output_dir)
    if args.alpha_sweep:
        sweep = run_alpha_sweep(
            nodes=nodes,
            queries=queries,
            alphas=[float(value.strip()) for value in args.alpha_sweep.split(",") if value.strip()],
            top_k_edges=args.top_k_edges,
            threshold=args.threshold,
            top_k=args.top_k,
            seed_k=args.seed_k,
            hops=args.hops,
        )
        write_alpha_sweep_report(sweep, args.output_dir)
    print("非 LLM 建图策略实验完成")
    print(f"JSON：{json_path}")
    print(f"Markdown：{markdown_path}")
    print(f"推荐策略：{report['summary']['recommended_strategy']}")


def _attach_context_metadata(nodes: list[object]) -> None:
    """为旧数据集节点补充通用 context_path 和 position。

    该函数只使用已有 metadata，不引入领域专用语义边。
    """

    for index, node in enumerate(nodes):
        metadata = node.metadata
        dataset = str(metadata.get("dataset") or "dataset")
        query_id = str(metadata.get("query_id") or metadata.get("book_id") or metadata.get("source_id") or "global")
        title = str(metadata.get("title") or metadata.get("original_doc_id") or node.id)
        metadata.setdefault("context_path", [dataset, query_id, title])
        metadata.setdefault("position", index)


if __name__ == "__main__":
    main()
