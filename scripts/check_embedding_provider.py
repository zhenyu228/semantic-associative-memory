from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.embedding import create_embedding_provider, inspect_embedding_provider_config  # noqa: E402
from sam.env import load_env_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 SAM embedding provider 配置")
    parser.add_argument("--provider", default=None, help="local、openai、azure_openai 或 azure_openai_sdk；默认读取 SAM_EMBEDDING_PROVIDER")
    parser.add_argument("--env-file", default=None, help="可选：加载本地 .env.local；文件已被 gitignore 忽略")
    parser.add_argument("--probe", default=None, help="可选：发送一条测试文本并返回维度和范数，不打印向量内容")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出诊断结果")
    return parser.parse_args()


def build_embedding_status(
    *,
    provider_name: str | None = None,
    probe: str | None = None,
) -> dict[str, object]:
    """构建 embedding provider 诊断结果，不暴露 key、endpoint 或向量内容。"""

    status = inspect_embedding_provider_config(provider_name)
    if probe and status.get("ready"):
        provider = create_embedding_provider(provider_name)
        try:
            embedding = provider.embed(probe)
            status["probe"] = {
                "dimension": len(embedding),
                "l2_norm": round(math.sqrt(sum(value * value for value in embedding)), 6),
            }
        except Exception as exc:
            status["ready"] = False
            status["probe_error"] = _safe_probe_error(exc)
        finally:
            close = getattr(provider, "close", None)
            if callable(close):
                close()
    return status


def main() -> None:
    args = parse_args()
    if args.env_file:
        load_env_file(ROOT / args.env_file)
    status = build_embedding_status(provider_name=args.provider, probe=args.probe)

    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print(f"provider: {status.get('provider')}")
        print(f"ready: {status.get('ready')}")
        if status.get("missing"):
            print("missing: " + ", ".join(str(item) for item in status["missing"]))
        if status.get("missing_packages"):
            print("missing packages: " + ", ".join(str(item) for item in status["missing_packages"]))
        if status.get("install_hint"):
            print(f"install hint: {status.get('install_hint')}")
        if status.get("required_any_missing"):
            groups = [" 或 ".join(group) for group in status["required_any_missing"]]
            print("missing one of: " + "; ".join(groups))
        if status.get("configured_optional"):
            print("configured optional: " + ", ".join(str(item) for item in status["configured_optional"]))
        if status.get("alias_sources"):
            aliases = [
                f"{target}<-{source}"
                for target, source in dict(status["alias_sources"]).items()
            ]
            print("aliases: " + ", ".join(aliases))
        print(f"cache enabled: {status.get('cache_enabled')}")
        if status.get("probe"):
            probe = status["probe"]
            print(f"probe dimension: {probe['dimension']}")
            print(f"probe l2_norm: {probe['l2_norm']}")
        if status.get("probe_error"):
            print(f"probe error: {json.dumps(status['probe_error'], ensure_ascii=False)}")
        if status.get("error"):
            print(f"error: {status.get('error')}")

    if not status.get("ready"):
        raise SystemExit(2)


def _safe_probe_error(exc: Exception) -> dict[str, str]:
    message = str(exc)
    message = re.sub(r"https?://[^\s)]+", "<redacted-url>", message)
    message = re.sub(r"api[_-]?key[=:]\s*[^,\s]+", "api_key=<redacted>", message, flags=re.IGNORECASE)
    return {
        "type": type(exc).__name__,
        "message": message[:300],
    }


if __name__ == "__main__":
    main()
