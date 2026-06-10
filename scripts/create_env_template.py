from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


ENV_TEMPLATE = """# SAM 本地模型配置模板
# 这个文件会被 .gitignore 忽略。请把占位符替换成真实值，不要提交真实 key。

SAM_EMBEDDING_PROVIDER=azure_openai_sdk
SAM_AZURE_EMBEDDING_ENDPOINT=https://aidp-i18ntt-sg.tiktok-row.net/gpt/openapi/online/v2/crawl
SAM_AZURE_EMBEDDING_API_VERSION=2023-07-01-preview
SAM_AZURE_EMBEDDING_MODEL=text-embedding-3-large
SAM_AZURE_EMBEDDING_DIMENSIONS=1024
SAM_AZURE_EMBEDDING_CONCURRENCY=1
SAM_AZURE_EMBEDDING_BATCH_SIZE=16
SAM_AZURE_EMBEDDING_INPUT_MODE=single
SAM_AZURE_EMBEDDING_TIMEOUT=120
SAM_AZURE_EMBEDDING_MAX_RETRIES=5
SAM_AZURE_EMBEDDING_RATE_LIMIT_RETRIES=30
SAM_AZURE_EMBEDDING_RATE_LIMIT_SLEEP_SECONDS=5
SAM_AZURE_EMBEDDING_RETRY_BASE_SECONDS=1
SAM_AZURE_EMBEDDING_API_KEY=replace-with-embedding-api-key
SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE=1

# 如果在线 embedding endpoint 不可用，可以安装可选依赖后切换到本地模型：
# conda run -n sam python -m pip install -e ".[local-embedding]"
# SAM_EMBEDDING_PROVIDER=sentence_transformers
# SAM_SENTENCE_TRANSFORMER_MODEL=/Users/bytedance/models/Qwen3-Embedding-0.6B
# SAM_SENTENCE_TRANSFORMER_DEVICE=cpu
# SAM_SENTENCE_TRANSFORMER_BATCH_SIZE=8
# SAM_SENTENCE_TRANSFORMER_NORMALIZE=1

SAM_CHAT_PROVIDER=azure_openai_sdk
SAM_AZURE_CHAT_ENDPOINT=https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl
SAM_AZURE_CHAT_API_VERSION=2024-02-01
SAM_AZURE_CHAT_MODEL=gpt-5.4-2026-03-05
SAM_AZURE_CHAT_API_KEY=replace-with-chat-api-key
"""


def write_env_template(path: str | Path, *, force: bool = False) -> Path:
    target = Path(path)
    if target.exists() and not force:
        raise FileExistsError(f"{target} 已存在；如需覆盖请使用 --force")
    target.write_text(ENV_TEMPLATE, encoding="utf-8")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 SAM 本地 .env.local 模板")
    parser.add_argument("--output", default=".env.local", help="输出路径，默认 .env.local")
    parser.add_argument("--force", action="store_true", help="允许覆盖已有文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    path = write_env_template(output, force=args.force)
    print(f"已生成本地配置模板：{path}")
    print("请替换 replace-with-* 占位符后再运行 provider 诊断。")


if __name__ == "__main__":
    main()
