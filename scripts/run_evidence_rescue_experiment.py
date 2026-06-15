from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.dataset_format import load_sam_dataset  # noqa: E402
from sam.datasets import documents_to_nodes  # noqa: E402
from sam.evidence_rescue_experiment import run_evidence_rescue_strategies, write_evidence_rescue_report  # noqa: E402
from scripts.run_graph_strategy_experiment import (  # noqa: E402
    DEFAULT_STRATEGIES,
    _attach_context_metadata,
    _create_embedding_provider,
    _embed_queries,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估图联想是否能在保留 embedding top-k 的前提下补回遗漏证据")
    parser.add_argument("--dataset-file", default="data/processed/hotpotqa_sam_sample.json", help="SAM 统一数据格式文件")
    parser.add_argument("--output-dir", default="outputs/evidence_rescue_experiment", help="实验输出目录")
    parser.add_argument("--limit-queries", type=int, default=30, help="最多评测多少条 query")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES), help="逗号分隔的策略列表")
    parser.add_argument("--alpha", type=float, default=0.55, help="SAM/CAM 混合公式中的语义权重")
    parser.add_argument("--top-k-edges", type=int, default=4, help="每个节点最多保留多少条出边")
    parser.add_argument("--threshold", type=float, default=0.18, help="建边得分阈值")
    parser.add_argument("--top-k", type=int, default=5, help="embedding baseline 返回证据数")
    parser.add_argument("--seed-k", type=int, default=1, help="从 embedding top-k 中选多少个 seed 做图扩展")
    parser.add_argument("--hops", type=int, default=1, help="图扩展跳数")
    parser.add_argument("--max-rescue-per-seed", type=int, default=2, help="每个 seed 最多保留多少个图扩展候选")
    parser.add_argument("--min-expansion-similarity", type=float, default=-1.0, help="扩展节点和 query 的最低相似度")
    parser.add_argument(
        "--pair-scope",
        choices=["global", "query_candidates"],
        default="query_candidates",
        help="建图候选节点对范围",
    )
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
        help="embedding provider",
    )
    parser.add_argument("--embedding-concurrency", type=int, default=None, help="在线 embedding 最大并发数")
    parser.add_argument("--embedding-batch-size", type=int, default=None, help="在线 embedding 批大小")
    parser.add_argument("--embedding-cache", action="store_true", help="启用 SQLite embedding 缓存")
    parser.add_argument("--embedding-cache-path", default=None, help="自定义 embedding 缓存 SQLite 路径")
    parser.add_argument(
        "--embedding-input-mode",
        choices=["single", "batch"],
        default=None,
        help="azure_openai_sdk 输入模式",
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

    report = run_evidence_rescue_strategies(
        nodes=nodes,
        queries=queries,
        query_embeddings=query_embeddings,
        strategies=[strategy.strip() for strategy in args.strategies.split(",") if strategy.strip()],
        top_k=args.top_k,
        seed_k=args.seed_k,
        hops=args.hops,
        alpha=args.alpha,
        top_k_edges=args.top_k_edges,
        threshold=args.threshold,
        pair_scope=args.pair_scope,
        max_rescue_per_seed=args.max_rescue_per_seed,
        min_expansion_similarity=args.min_expansion_similarity,
    )
    report["dataset"] = {
        "dataset_file": args.dataset_file,
        "document_count": len(documents),
        "query_count": len(queries),
        "limit_queries": args.limit_queries,
        "supporting_evidence_count": sum(len(query.supporting_doc_ids) for query in queries),
        "average_candidate_docs_per_query": (
            round(sum(len(query.candidate_doc_ids) for query in queries) / len(queries), 4)
            if queries
            else 0.0
        ),
    }
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
    json_path, markdown_path = write_evidence_rescue_report(report, args.output_dir)
    print("图联想补证据实验完成")
    print(f"JSON：{json_path}")
    print(f"Markdown：{markdown_path}")
    print(f"最佳补证据策略：{report['summary']['best_rescue_strategy']}")


if __name__ == "__main__":
    main()
