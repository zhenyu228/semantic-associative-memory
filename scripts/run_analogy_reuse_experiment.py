from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.analogy import AnalogyEngine  # noqa: E402
from sam.analogy_experiment import run_analogy_reuse_probe, write_analogy_reuse_reports  # noqa: E402
from sam.dataset_format import load_sam_dataset, summarize_sam_dataset  # noqa: E402
from sam.embedding import create_embedding_provider  # noqa: E402
from sam.evaluator import Evaluator  # noqa: E402
from sam.graph import GraphBuilder  # noqa: E402
from sam.reuse_experiment import build_masked_queries  # noqa: E402
from sam.store import MemoryStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 SAM 类比复用实验")
    parser.add_argument("--dataset-file", default="data/processed/hotpotqa_sam_sample.json", help="SAM 统一数据格式文件")
    parser.add_argument("--output-root", default="outputs/runs", help="运行产物根目录")
    parser.add_argument("--run-name", default=None, help="本次运行名称")
    parser.add_argument("--db", default=None, help="SQLite 数据库路径，默认写入 run 目录")
    parser.add_argument("--limit", type=int, default=30, help="参与 warmup/probe 的查询数量")
    parser.add_argument("--embedding-provider", default=None, help="local、openai、azure_openai 或 azure_openai_sdk")
    parser.add_argument("--top-k", type=int, default=3, help="类比候选案例数量")
    parser.add_argument("--retrieval-top-k", type=int, default=4, help="warmup 检索返回文档数")
    parser.add_argument("--seed-k", type=int, default=1, help="SAM warmup 种子节点数")
    parser.add_argument("--hops", type=int, default=2, help="SAM warmup 图扩展跳数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_analogy_reuse"
    run_dir = ROOT / args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = ROOT / args.db if args.db else run_dir / "analogy_reuse.sqlite"
    (run_dir / "config.json").write_text(
        json.dumps(vars(args) | {"run_dir": str(run_dir)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataset_path = ROOT / args.dataset_file
    documents, queries, _ = load_sam_dataset(dataset_path)
    selected_queries = queries[: args.limit]
    probe_queries = build_masked_queries(selected_queries)
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
            top_k=args.retrieval_top_k,
            seed_k=args.seed_k,
            hops=args.hops,
            methods=["sam_full"],
        )
        engine = AnalogyEngine(store, embedding_provider, graph_builder)
        result = run_analogy_reuse_probe(engine, probe_queries, top_k=args.top_k)
        result["warmup"] = {
            "query_count": warmup_result.query_count,
            "consolidated_memory_count": len(
                [
                    node for node in store.get_nodes()
                    if node.metadata.get("node_type") == "consolidated_memory"
                ]
            ),
        }
        json_path, markdown_path = write_analogy_reuse_reports(output_dir=run_dir, result=result)
    finally:
        store.close()

    print("SAM 类比复用实验完成")
    print(f"运行目录：{run_dir}")
    print(f"巩固案例命中率：{float(result['consolidated_case_hit_rate']):.3f}")
    print(f"支持证据重叠命中率：{float(result['support_overlap_hit_rate']):.3f}")
    print(f"JSON：{json_path}")
    print(f"Markdown：{markdown_path}")


if __name__ == "__main__":
    main()
