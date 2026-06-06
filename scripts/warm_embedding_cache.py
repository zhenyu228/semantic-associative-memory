from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.embedding_plan import warm_embedding_cache, write_embedding_warmup_result  # noqa: E402
from sam.env import load_env_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 SAM 数据集文本构造方式预热 embedding cache")
    parser.add_argument("--dataset-file", required=True, help="SAM 统一数据格式文件")
    parser.add_argument("--provider", default=None, help="local、openai、azure_openai 或 azure_openai_sdk")
    parser.add_argument("--env-file", default=None, help="可选：加载本地 env 文件")
    parser.add_argument("--cache-path", required=True, help="embedding cache SQLite 路径")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 provider 默认 batch size")
    parser.add_argument("--no-query-summaries", action="store_true", help="不预热 query summary 节点 embedding")
    parser.add_argument("--output-dir", default="outputs/plans", help="预热结果输出目录")
    parser.add_argument("--json", action="store_true", help="同时在终端打印 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env_file:
        load_env_file(ROOT / args.env_file)
    dataset_path = ROOT / args.dataset_file if not Path(args.dataset_file).is_absolute() else Path(args.dataset_file)
    cache_path = ROOT / args.cache_path if not Path(args.cache_path).is_absolute() else Path(args.cache_path)
    output_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    result = warm_embedding_cache(
        dataset_path=dataset_path,
        provider_name=args.provider,
        cache_path=cache_path,
        batch_size=args.batch_size,
        include_query_summaries=not args.no_query_summaries,
    )
    json_path, markdown_path = write_embedding_warmup_result(result, output_dir)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Embedding cache 预热完成：{json_path}")
    print(f"Markdown：{markdown_path}")
    print(f"本次写入文本数：{result['warmed_text_count']}")
    after = result.get("after", {})
    if isinstance(after, dict):
        print(f"预热后缺失文本数：{after.get('cache_miss_count')}")


if __name__ == "__main__":
    main()
