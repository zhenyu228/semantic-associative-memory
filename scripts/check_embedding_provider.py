from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.embedding import create_embedding_provider, inspect_embedding_provider_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 SAM embedding provider 配置")
    parser.add_argument("--provider", default=None, help="local、openai、azure_openai 或 azure_openai_sdk；默认读取 SAM_EMBEDDING_PROVIDER")
    parser.add_argument("--probe", default=None, help="可选：发送一条测试文本并返回维度和范数，不打印向量内容")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出诊断结果")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = inspect_embedding_provider_config(args.provider)
    if args.probe and status.get("ready"):
        provider = create_embedding_provider(args.provider)
        embedding = provider.embed(args.probe)
        status["probe"] = {
            "dimension": len(embedding),
            "l2_norm": round(math.sqrt(sum(value * value for value in embedding)), 6),
        }
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print(f"provider: {status.get('provider')}")
        print(f"ready: {status.get('ready')}")
        if status.get("missing"):
            print("missing: " + ", ".join(str(item) for item in status["missing"]))
        if status.get("missing_packages"):
            print("missing packages: " + ", ".join(str(item) for item in status["missing_packages"]))
        if status.get("required_any_missing"):
            groups = [" 或 ".join(group) for group in status["required_any_missing"]]
            print("missing one of: " + "; ".join(groups))
        if status.get("configured_optional"):
            print("configured optional: " + ", ".join(str(item) for item in status["configured_optional"]))
        print(f"cache enabled: {status.get('cache_enabled')}")
        if status.get("probe"):
            probe = status["probe"]
            print(f"probe dimension: {probe['dimension']}")
            print(f"probe l2_norm: {probe['l2_norm']}")
        if status.get("error"):
            print(f"error: {status.get('error')}")

    if not status.get("ready"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
