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

from sam.dataset_format import load_sam_dataset, summarize_sam_dataset  # noqa: E402
from sam.embedding import create_embedding_provider  # noqa: E402
from sam.reranker_experiment import (  # noqa: E402
    DEFAULT_RERANKER_PROFILES,
    run_reranker_profile_comparison,
    write_reranker_profile_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 PathReranker profile 对比实验")
    parser.add_argument("--dataset-file", default="data/processed/hotpotqa_sam_sample.json", help="SAM 统一数据格式文件")
    parser.add_argument("--output-root", default="outputs/runs", help="运行产物根目录")
    parser.add_argument("--run-name", default=None, help="本次运行名称")
    parser.add_argument("--limit", type=int, default=30, help="参与对比的查询数量")
    parser.add_argument("--embedding-provider", default=None, help="local、openai、azure_openai 或 azure_openai_sdk")
    parser.add_argument("--top-k", type=int, default=4, help="最终返回文档数")
    parser.add_argument("--seed-k", type=int, default=1, help="SAM 种子节点数")
    parser.add_argument("--hops", type=int, default=2, help="图扩展跳数")
    parser.add_argument("--method", default="sam_full", help="用于对比的 SAM 方法")
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_RERANKER_PROFILES),
        help="逗号分隔的 reranker profile 列表",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_reranker_profiles"
    run_dir = ROOT / args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    profiles = [profile.strip() for profile in args.profiles.split(",") if profile.strip()]
    (run_dir / "config.json").write_text(
        json.dumps(vars(args) | {"profiles": profiles, "run_dir": str(run_dir)}, ensure_ascii=False, indent=2),
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

    comparison = run_reranker_profile_comparison(
        documents=documents,
        queries=selected_queries,
        embedding_provider=create_embedding_provider(args.embedding_provider),
        profiles=profiles,
        top_k=args.top_k,
        seed_k=args.seed_k,
        hops=args.hops,
        method=args.method,
    )
    json_path, markdown_path = write_reranker_profile_reports(comparison, run_dir)

    print("PathReranker profile 对比实验完成")
    print(f"运行目录：{run_dir}")
    print(f"查询数量：{comparison['query_count']}")
    print(f"最优 profile：{comparison['best_profile']}")
    for profile in profiles:
        profile_result = comparison["profile_results"][profile]
        metrics = profile_result["metrics"]
        print(
            f"{profile}: 证据召回率={float(metrics['evidence_recall']):.3f}, "
            f"答案命中率={float(metrics['answer_hit_rate']):.3f}"
        )
    print(f"JSON：{json_path}")
    print(f"Markdown：{markdown_path}")


if __name__ == "__main__":
    main()
