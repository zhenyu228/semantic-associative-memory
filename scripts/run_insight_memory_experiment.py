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
from sam.insight_experiment import (  # noqa: E402
    summarize_insight_memory_reconstruction,
    write_insight_memory_reports,
)
from sam.store import MemoryStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 SAM 高层洞察记忆重构实验")
    parser.add_argument("--dataset-file", default="data/processed/hotpotqa_midterm30_sam_sample.json", help="SAM 统一数据格式文件")
    parser.add_argument("--output-root", default="outputs/runs", help="运行产物根目录")
    parser.add_argument("--run-name", default=None, help="本次运行名称")
    parser.add_argument("--db", default=None, help="SQLite 数据库路径，默认写入 run 目录")
    parser.add_argument("--limit", type=int, default=30, help="参与 warmup 的查询数量")
    parser.add_argument("--env-file", default=None, help="可选：加载本地 .env.local；文件已被 gitignore 忽略")
    parser.add_argument("--embedding-provider", default=None, help="local_hash、azure_openai_sdk、openai 等")
    parser.add_argument("--embedding-cache", action="store_true", help="启用 SQLite embedding 缓存")
    parser.add_argument("--embedding-cache-path", default=None, help="自定义 embedding 缓存 SQLite 路径")
    parser.add_argument("--embedding-concurrency", type=int, default=None, help="在线 embedding 最大并发数")
    parser.add_argument("--top-k", type=int, default=4, help="最终返回文档数")
    parser.add_argument("--seed-k", type=int, default=1, help="SAM 种子节点数")
    parser.add_argument("--hops", type=int, default=1, help="图扩展跳数")
    parser.add_argument("--method", default="sam_full", help="用于生成巩固记忆和洞察记忆的 SAM 方法")
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

    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_insight_memory"
    run_dir = ROOT / args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = ROOT / args.db if args.db else run_dir / "insight_memory.sqlite"
    dataset_path = ROOT / args.dataset_file

    documents, queries, _ = load_sam_dataset(dataset_path)
    selected_queries = queries[: args.limit]
    if not selected_queries:
        raise ValueError("没有可评测 query，请检查 --dataset-file 和 --limit")

    config = vars(args) | {"run_dir": str(run_dir), "selected_query_count": len(selected_queries)}
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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
            methods=[args.method],
        )
        summary = summarize_insight_memory_reconstruction(
            store=store,
            queries=selected_queries,
        )
        warmup_payload = warmup_result.to_dict()
        (run_dir / "warmup_metrics.json").write_text(
            json.dumps(warmup_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        json_path, markdown_path = write_insight_memory_reports(
            output_dir=run_dir,
            summary=summary,
            warmup_metrics=warmup_payload,
        )
    finally:
        store.close()

    print("SAM 高层洞察记忆重构实验完成")
    print(f"运行目录：{run_dir}")
    print(f"查询数量：{summary['query_count']}")
    print(f"单次巩固记忆数：{summary['consolidated_memory_count']}")
    print(f"高层洞察记忆数：{summary['insight_memory_count']}")
    print(f"支持证据回溯率：{float(summary['support_trace_rate']):.3f}")
    print(f"巩固证据覆盖率：{float(summary['insight_evidence_coverage_rate']):.3f}")
    print(f"JSON：{json_path}")
    print(f"Markdown：{markdown_path}")


if __name__ == "__main__":
    main()
