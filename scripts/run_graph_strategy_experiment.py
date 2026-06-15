from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
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
    embedding_cache: bool = False,
    embedding_cache_path: str | None = None,
):
    """创建 embedding provider；远端 provider 先加载本地环境变量。"""

    if provider_name == "local_hash":
        return LocalHashEmbeddingProvider()
    load_default_env_file()
    _apply_embedding_runtime_options(
        embedding_concurrency=embedding_concurrency,
        embedding_batch_size=embedding_batch_size,
        embedding_input_mode=embedding_input_mode,
        embedding_cache=embedding_cache,
        embedding_cache_path=embedding_cache_path,
    )
    return create_embedding_provider(provider_name)


def _apply_embedding_runtime_options(
    *,
    embedding_concurrency: int | None,
    embedding_batch_size: int | None,
    embedding_input_mode: str | None,
    embedding_cache: bool,
    embedding_cache_path: str | None,
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
    if embedding_cache:
        os.environ["SAM_EMBEDDING_CACHE"] = "1"
    if embedding_cache_path:
        cache_path = Path(embedding_cache_path)
        if not cache_path.is_absolute():
            cache_path = ROOT / cache_path
        os.environ["SAM_EMBEDDING_CACHE_PATH"] = str(cache_path)


def _embed_queries(queries: list[object], embedding_provider) -> tuple[dict[str, list[float]], float]:
    """为 query 生成真实 embedding，保证检索评测和文档向量在同一空间。"""

    start = time.perf_counter()
    texts = [str(query.question) for query in queries]
    embeddings = embedding_provider.embed_many(texts)
    return {
        str(query.id): embedding
        for query, embedding in zip(queries, embeddings, strict=True)
    }, time.perf_counter() - start


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
        "--context-path-policy",
        choices=["intrinsic", "metadata", "query_grouped_legacy"],
        default="intrinsic",
        help="context_path 构造策略；默认 intrinsic 禁止使用 query_id/hotpotqa_id/original_doc_id",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["local_hash", "openai", "azure_openai", "azure_openai_sdk", "sentence_transformers"],
        default="local_hash",
        help="embedding provider；正式实验可换成 azure_openai_sdk",
    )
    parser.add_argument("--embedding-concurrency", type=int, default=None, help="在线 embedding 最大并发数")
    parser.add_argument("--embedding-batch-size", type=int, default=None, help="在线 embedding 批大小；batch 模式下更重要")
    parser.add_argument("--embedding-cache", action="store_true", help="启用 SQLite embedding 缓存")
    parser.add_argument("--embedding-cache-path", default=None, help="自定义 embedding 缓存 SQLite 路径")
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
        embedding_cache=args.embedding_cache,
        embedding_cache_path=args.embedding_cache_path,
    )
    document_embedding_start = time.perf_counter()
    nodes = documents_to_nodes(documents, embedding)
    document_embedding_time = time.perf_counter() - document_embedding_start
    query_embeddings, query_embedding_time = _embed_queries(queries, embedding)
    context_path_audit = _attach_context_metadata(nodes, policy=args.context_path_policy)
    experiment = GraphStrategyExperiment(
        nodes=nodes,
        queries=queries,
        query_embeddings=query_embeddings,
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
    report["embedding"] = {
        "provider": args.embedding_provider,
        "document_embedding_count": len(documents),
        "query_embedding_count": len(queries),
        "document_embedding_time_seconds": round(document_embedding_time, 6),
        "query_embedding_time_seconds": round(query_embedding_time, 6),
        "embedding_concurrency": args.embedding_concurrency,
        "embedding_batch_size": args.embedding_batch_size,
        "embedding_input_mode": args.embedding_input_mode,
        "embedding_cache": args.embedding_cache,
        "embedding_cache_path": args.embedding_cache_path,
    }
    report["context_path"] = context_path_audit
    report["dataset"] = {
        "dataset_file": args.dataset_file,
        "document_count": len(documents),
        "query_count": len(queries),
        "limit_queries": args.limit_queries,
    }
    json_path, markdown_path = write_graph_strategy_report(report, args.output_dir)
    if args.alpha_sweep:
        sweep = run_alpha_sweep(
            nodes=nodes,
            queries=queries,
            query_embeddings=query_embeddings,
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


def _attach_context_metadata(nodes: list[object], *, policy: str = "intrinsic") -> dict[str, object]:
    """为旧数据集节点补充 context_path 和 position，并返回泄漏审计。

    默认 `intrinsic` 只使用文档自身结构字段，显式排除 query_id、
    hotpotqa_id 和 original_doc_id，避免在 HotpotQA 中把候选集分组信息
    写进 context_path。
    """

    if policy not in {"intrinsic", "metadata", "query_grouped_legacy"}:
        raise ValueError(f"未知 context_path policy: {policy}")
    query_ids = {
        str(node.metadata.get("query_id"))
        for node in nodes
        if node.metadata.get("query_id")
    }
    hotpotqa_ids = {
        str(node.metadata.get("hotpotqa_id"))
        for node in nodes
        if node.metadata.get("hotpotqa_id")
    }
    original_doc_ids = {
        str(node.metadata.get("original_doc_id"))
        for node in nodes
        if node.metadata.get("original_doc_id")
    }
    leaking_paths = 0
    position_sources: dict[str, int] = {}
    path_examples: list[list[str]] = []
    for index, node in enumerate(nodes):
        metadata = node.metadata
        path = _context_path_for_node(node, policy=policy)
        metadata["context_path"] = path
        position, position_source = _position_for_node(metadata, fallback=index)
        metadata.setdefault("position", position)
        metadata["position_source"] = position_source
        position_sources[position_source] = position_sources.get(position_source, 0) + 1
        if len(path_examples) < 5:
            path_examples.append(path)
        if _path_contains_any(path, query_ids | hotpotqa_ids | original_doc_ids):
            leaking_paths += 1
    return {
        "policy": policy,
        "is_leak_safe": leaking_paths == 0,
        "node_count": len(nodes),
        "nodes_with_query_id_metadata": len(query_ids),
        "nodes_with_hotpotqa_id_metadata": len(hotpotqa_ids),
        "context_paths_containing_query_ids": leaking_paths,
        "excluded_fields": ["query_id", "hotpotqa_id", "original_doc_id"],
        "position_sources": position_sources,
        "examples": path_examples,
        "notes": (
            "intrinsic 策略只使用文档自身结构字段；query_grouped_legacy 会复现旧逻辑，"
            "仅用于对照，不建议作为正式实验结论。"
        ),
    }


def _context_path_for_node(node: object, *, policy: str) -> list[str]:
    metadata = node.metadata
    if policy == "metadata" and metadata.get("context_path"):
        return _coerce_path(metadata["context_path"])
    if policy == "query_grouped_legacy":
        dataset = _safe_segment(metadata.get("dataset") or "dataset")
        group = _safe_segment(metadata.get("query_id") or metadata.get("book_id") or metadata.get("source_id") or "global")
        title = _safe_segment(metadata.get("title") or "untitled")
        return [f"dataset:{dataset}", f"group:{group}", f"title:{title}"]
    return _intrinsic_context_path(node)


def _intrinsic_context_path(node: object) -> list[str]:
    metadata = node.metadata
    if metadata.get("book_id"):
        path = [f"book:{_safe_segment(metadata.get('book_id'))}"]
        if metadata.get("chapter") is not None:
            path.append(f"chapter:{_safe_segment(metadata.get('chapter'))}")
        if metadata.get("chunk_index") is not None:
            chunk_index = _safe_int(metadata.get("chunk_index"))
            path.append(f"chunk_block:{chunk_index // 10}")
            path.append(f"chunk:{chunk_index}")
        return path
    if metadata.get("source_id"):
        path = [f"source:{_safe_segment(metadata.get('source_id'))}"]
        if metadata.get("section") is not None:
            path.append(f"section:{_safe_segment(metadata.get('section'))}")
        if metadata.get("title"):
            path.append(f"title:{_safe_segment(metadata.get('title'))}")
        return path
    if metadata.get("section") and metadata.get("title"):
        return [f"section:{_safe_segment(metadata.get('section'))}", f"title:{_safe_segment(metadata.get('title'))}"]
    if metadata.get("title"):
        return [f"title:{_safe_segment(metadata.get('title'))}"]
    stable_fallback = hashlib.sha1(str(getattr(node, "id", "node")).encode("utf-8")).hexdigest()[:10]
    return [f"node:{stable_fallback}"]


def _position_for_node(metadata: dict[str, object], *, fallback: int) -> tuple[int, str]:
    for key in ["chunk_index", "paragraph_index", "position"]:
        if metadata.get(key) is not None:
            return _safe_int(metadata.get(key)), key
    return fallback, "script_order_fallback"


def _coerce_path(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [part for part in value.split("/") if part]
    return []


def _safe_segment(value: object) -> str:
    text = str(value or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _path_contains_any(path: list[str], values: set[str]) -> bool:
    path_text = "/".join(path)
    return any(value and value in path_text for value in values)


if __name__ == "__main__":
    main()
