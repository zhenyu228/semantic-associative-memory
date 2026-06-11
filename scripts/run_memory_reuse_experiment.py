from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.dataset_format import load_sam_dataset, summarize_sam_dataset  # noqa: E402
from sam.embedding import create_embedding_provider  # noqa: E402
from sam.env import load_env_file  # noqa: E402
from sam.evaluator import Evaluator  # noqa: E402
from sam.graph import GraphBuilder  # noqa: E402
from sam.reuse_experiment import (  # noqa: E402
    build_masked_queries,
    snapshot_edges,
    summarize_memory_reuse,
    write_memory_reuse_event_reports,
    write_memory_reuse_reports,
)
from sam.store import MemoryStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 SAM 连续记忆复用实验")
    parser.add_argument("--dataset-file", default="data/processed/hotpotqa_sam_sample.json", help="SAM 统一数据格式文件")
    parser.add_argument("--output-root", default="outputs/runs", help="运行产物根目录")
    parser.add_argument("--run-name", default=None, help="本次运行名称")
    parser.add_argument("--db", default=None, help="SQLite 数据库路径，默认写入 run 目录")
    parser.add_argument("--limit", type=int, default=30, help="参与 warmup/probe 的查询数量")
    parser.add_argument("--env-file", default=None, help="可选：加载本地 .env.local；文件已被 gitignore 忽略")
    parser.add_argument("--embedding-provider", default=None, help="local、openai、azure_openai 或 azure_openai_sdk")
    parser.add_argument("--embedding-cache", action="store_true", help="启用 SQLite embedding 缓存，默认写入 data/embedding_cache.sqlite")
    parser.add_argument("--embedding-cache-path", default=None, help="自定义 embedding 缓存 SQLite 路径")
    parser.add_argument("--embedding-concurrency", type=int, default=None, help="在线 embedding 最大并发数")
    parser.add_argument("--top-k", type=int, default=4, help="最终返回文档数")
    parser.add_argument("--seed-k", type=int, default=1, help="SAM 种子节点数")
    parser.add_argument("--hops", type=int, default=1, help="图扩展跳数；默认采用一跳局部联想")
    parser.add_argument(
        "--probe-methods",
        default="embedding_topk,sam_full,sam_no_memory_state,sam_no_feedback,sam_static_graph",
        help="probe 阶段方法列表；默认比较 baseline 与连续记忆消融方法",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env_file:
        load_env_file(ROOT / args.env_file)
    if args.embedding_cache:
        os.environ["SAM_EMBEDDING_CACHE"] = "1"
    if args.embedding_cache_path:
        os.environ["SAM_EMBEDDING_CACHE_PATH"] = str(ROOT / args.embedding_cache_path)
    if args.embedding_concurrency is not None:
        os.environ["SAM_AZURE_EMBEDDING_CONCURRENCY"] = str(args.embedding_concurrency)
        os.environ["SAM_OPENAI_EMBEDDING_CONCURRENCY"] = str(args.embedding_concurrency)
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_memory_reuse"
    run_dir = ROOT / args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = ROOT / args.db if args.db else run_dir / "memory_reuse.sqlite"
    methods = [method.strip() for method in args.probe_methods.split(",") if method.strip()]
    (run_dir / "config.json").write_text(
        json.dumps(vars(args) | {"probe_methods": methods, "run_dir": str(run_dir)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataset_path = ROOT / args.dataset_file
    documents, queries, _ = load_sam_dataset(dataset_path)
    selected_queries = queries[: args.limit]
    masked_queries = build_masked_queries(selected_queries)
    (run_dir / "dataset_summary.json").write_text(
        json.dumps(
            summarize_sam_dataset(dataset_path) | {"selected_query_count": len(selected_queries)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    store = MemoryStore(db_path)
    try:
        embedding_provider = create_embedding_provider(args.embedding_provider)
        graph_builder = GraphBuilder(store)
        evaluator = Evaluator(store, embedding_provider, graph_builder)
        evaluator.ingest(documents)

        warmup_result = evaluator.evaluate(
            selected_queries,
            top_k=args.top_k,
            seed_k=args.seed_k,
            hops=args.hops,
            methods=["sam_full"],
        )
        warmup_consolidated_count = len(
            [
                node for node in store.get_nodes()
                if node.metadata.get("node_type") == "consolidated_memory"
            ]
        )
        warmup_consolidation_edge_count = len(
            [
                edge for edge in store.get_edges()
                if "consolidat" in edge.relation_type
            ]
        )
        edges_before_probe = snapshot_edges(store.get_edges())

        probe_result = evaluator.evaluate(
            masked_queries,
            top_k=args.top_k,
            seed_k=args.seed_k,
            hops=args.hops,
            methods=methods,
        )
        baseline_method = methods[0]
        sam_method = next((method for method in methods if method.startswith("sam")), methods[-1])
        summary = summarize_memory_reuse(
            warmup_consolidated_count=warmup_consolidated_count,
            warmup_consolidation_edge_count=warmup_consolidation_edge_count,
            baseline_metric=probe_result.method_metrics[baseline_method],
            sam_metric=probe_result.method_metrics[sam_method],
        )
        json_path, markdown_path = write_memory_reuse_reports(
            output_dir=run_dir,
            summary=summary,
            warmup_metrics=warmup_result.to_dict(),
            probe_metrics=probe_result.to_dict(),
            probe_cases=probe_result.cases,
        )
        write_memory_reuse_event_reports(
            output_dir=run_dir,
            events=store.get_memory_events(limit=10000),
            edges_after=store.get_edges(),
            edges_before=edges_before_probe,
        )
    finally:
        store.close()

    print("SAM 连续记忆复用实验完成")
    print(f"运行目录：{run_dir}")
    print(f"Warmup 巩固记忆节点数：{summary['warmup_consolidated_count']}")
    print(f"Baseline 证据召回率：{float(summary['baseline_evidence_recall']):.3f}")
    print(f"SAM 证据召回率：{float(summary['sam_evidence_recall']):.3f}")
    print(f"证据召回增益：{float(summary['evidence_recall_gain']):.3f}")
    print(f"JSON：{json_path}")
    print(f"Markdown：{markdown_path}")


if __name__ == "__main__":
    main()
