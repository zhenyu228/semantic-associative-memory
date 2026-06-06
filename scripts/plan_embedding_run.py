from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.embedding_plan import build_embedding_run_plan, write_embedding_run_plan  # noqa: E402
from sam.env import load_env_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="规划一次 embedding 实验的请求量，不调用在线 API")
    parser.add_argument("--dataset-file", required=True, help="SAM 统一数据格式文件")
    parser.add_argument("--provider", default=None, help="local、openai、azure_openai 或 azure_openai_sdk")
    parser.add_argument("--env-file", default=None, help="可选：加载本地 env 文件")
    parser.add_argument("--cache-path", default=None, help="embedding cache SQLite 路径")
    parser.add_argument("--cache-namespace", default=None, help="可选：指定精确 cache namespace")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 provider 默认 batch size")
    parser.add_argument("--no-query-summaries", action="store_true", help="不统计 query summary 节点 embedding")
    parser.add_argument("--output-dir", default="outputs/plans", help="计划文件输出目录")
    parser.add_argument("--json", action="store_true", help="同时在终端打印 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env_file:
        load_env_file(ROOT / args.env_file)
    dataset_path = ROOT / args.dataset_file if not Path(args.dataset_file).is_absolute() else Path(args.dataset_file)
    cache_path = (
        ROOT / args.cache_path
        if args.cache_path and not Path(args.cache_path).is_absolute()
        else Path(args.cache_path) if args.cache_path else None
    )
    output_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    plan = build_embedding_run_plan(
        dataset_path=dataset_path,
        provider_name=args.provider,
        cache_path=cache_path,
        cache_namespace=args.cache_namespace,
        batch_size=args.batch_size,
        include_query_summaries=not args.no_query_summaries,
    )
    json_path, markdown_path = write_embedding_run_plan(plan, output_dir)
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    print(f"Embedding 运行计划完成：{json_path}")
    print(f"Markdown：{markdown_path}")
    print(f"预计需要请求文本数：{plan['cache_miss_count']}")
    print(f"预计 batch 数：{plan['estimated_batch_count']}")


if __name__ == "__main__":
    main()
