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

from sam.answer_judge import create_answer_judge  # noqa: E402
from sam.dataset_format import load_sam_dataset, summarize_sam_dataset  # noqa: E402
from sam.embedding import create_embedding_provider  # noqa: E402
from sam.llm import create_chat_client  # noqa: E402
from sam.pipeline_experiment import run_retrieval_generation_pipeline  # noqa: E402
from sam.query_planner import create_query_planner  # noqa: E402
from sam.relation_judge import create_relation_judge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 SAM 检索-生成-判别端到端实验")
    parser.add_argument("--dataset-file", default="data/processed/hotpotqa_sam_sample.json", help="SAM 统一数据格式文件")
    parser.add_argument("--output-root", default="outputs/runs", help="运行产物根目录")
    parser.add_argument("--run-name", default=None, help="本次运行名称")
    parser.add_argument("--limit", type=int, default=8, help="参与实验的查询数量")
    parser.add_argument("--embedding-provider", default=None, help="local、openai 或 azure_openai")
    parser.add_argument("--chat-provider", default=None, help="heuristic 或 azure_openai")
    parser.add_argument("--answer-judge", default="rule", choices=["rule", "gpt54"], help="答案判别器")
    parser.add_argument("--query-planner", default="disabled", choices=["disabled", "heuristic", "gpt54"], help="查询规划器")
    parser.add_argument("--relation-judge", default="disabled", help="关系级建边判别器：disabled、gpt54 或 cached_gpt54")
    parser.add_argument(
        "--retrieval-methods",
        default="embedding_topk,sam_full",
        help="逗号分隔的检索方法列表",
    )
    parser.add_argument("--generation-method", default="sam_full", help="用于生成答案的检索方法")
    parser.add_argument(
        "--reranker-profile",
        default="semantic_heavy",
        choices=["balanced", "semantic_heavy", "graph_heavy", "memory_heavy"],
        help="SAM 路径重排权重配置",
    )
    parser.add_argument("--top-k", type=int, default=4, help="最终返回文档数")
    parser.add_argument("--seed-k", type=int, default=1, help="SAM 种子节点数")
    parser.add_argument("--hops", type=int, default=2, help="图扩展跳数")
    parser.add_argument("--max-context-chars", type=int, default=6000, help="生成阶段每题最多上下文字符数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_end_to_end"
    run_dir = ROOT / args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    retrieval_methods = [method.strip() for method in args.retrieval_methods.split(",") if method.strip()]
    (run_dir / "config.json").write_text(
        json.dumps(vars(args) | {"retrieval_methods": retrieval_methods, "run_dir": str(run_dir)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataset_path = ROOT / args.dataset_file
    documents, queries, _ = load_sam_dataset(dataset_path)
    selected_queries = queries[: args.limit]
    (run_dir / "dataset_summary.json").write_text(
        json.dumps(
            summarize_sam_dataset(dataset_path) | {"selected_query_count": len(selected_queries)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = run_retrieval_generation_pipeline(
        documents=documents,
        queries=selected_queries,
        output_dir=run_dir,
        embedding_provider=create_embedding_provider(args.embedding_provider),
        chat_client=create_chat_client(args.chat_provider),
        answer_judge=create_answer_judge(args.answer_judge),
        retrieval_methods=retrieval_methods,
        generation_method=args.generation_method,
        query_planner=create_query_planner(args.query_planner),
        relation_judge=create_relation_judge(args.relation_judge),
        reranker_profile=args.reranker_profile,
        top_k=args.top_k,
        seed_k=args.seed_k,
        hops=args.hops,
        max_context_chars=args.max_context_chars,
    )

    print("SAM 端到端实验完成")
    print(f"运行目录：{run_dir}")
    print(f"查询数量：{summary['query_count']}")
    generation = summary["generation"]
    print(f"生成答案命中率：{float(generation['answer_hit_rate']):.3f}")
    print(f"Pipeline Summary：{run_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()
