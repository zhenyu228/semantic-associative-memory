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
from sam.llm import create_chat_client, inspect_chat_provider_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一检查 SAM embedding 与 GPT-5.4 provider 配置")
    parser.add_argument("--embedding-provider", default=None, help="local、openai、azure_openai 或 azure_openai_sdk；默认读取 SAM_EMBEDDING_PROVIDER")
    parser.add_argument("--chat-provider", default=None, help="heuristic 或 azure_openai；默认读取 SAM_CHAT_PROVIDER")
    parser.add_argument("--embedding-probe", default=None, help="可选：发送一条 embedding 测试文本")
    parser.add_argument("--chat-probe", default=None, help="可选：发送一条聊天模型测试消息")
    parser.add_argument("--chat-max-tokens", type=int, default=64, help="chat probe 最大输出 token")
    parser.add_argument(
        "--require",
        default="both",
        choices=["both", "embedding", "chat"],
        help="本次检查要求哪些 provider ready；默认两者都要求",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出诊断结果")
    return parser.parse_args()


def build_provider_status(
    *,
    embedding_provider: str | None = None,
    chat_provider: str | None = None,
    embedding_probe: str | None = None,
    chat_probe: str | None = None,
    chat_max_tokens: int = 64,
    required_providers: str = "both",
) -> dict[str, object]:
    """构建模型 provider 诊断结果。

    不返回 API key、endpoint 或完整向量。只有传入 probe 时才会发起真实请求。
    """

    embedding_status = inspect_embedding_provider_config(embedding_provider)
    chat_status = inspect_chat_provider_config(chat_provider)
    if embedding_probe and embedding_status.get("ready"):
        provider = create_embedding_provider(embedding_provider)
        embedding = provider.embed(embedding_probe)
        embedding_status["probe"] = {
            "dimension": len(embedding),
            "l2_norm": round(math.sqrt(sum(value * value for value in embedding)), 6),
        }
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    if chat_probe and chat_status.get("ready"):
        client = create_chat_client(chat_provider)
        answer = client.complete(
            [{"role": "user", "content": chat_probe}],
            max_tokens=chat_max_tokens,
        )
        chat_status["probe"] = {
            "answer_preview": answer[:200],
            "answer_chars": len(answer),
        }
    required_ready = {
        "both": bool(embedding_status.get("ready") and chat_status.get("ready")),
        "embedding": bool(embedding_status.get("ready")),
        "chat": bool(chat_status.get("ready")),
    }
    return {
        "ready": required_ready[required_providers],
        "required_providers": required_providers,
        "embedding": embedding_status,
        "chat": chat_status,
    }


def main() -> None:
    args = parse_args()
    status = build_provider_status(
        embedding_provider=args.embedding_provider,
        chat_provider=args.chat_provider,
        embedding_probe=args.embedding_probe,
        chat_probe=args.chat_probe,
        chat_max_tokens=args.chat_max_tokens,
        required_providers=args.require,
    )
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        _print_section("embedding", status["embedding"])
        _print_section("chat", status["chat"])
        print(f"overall ready: {status['ready']}")
    if not status["ready"]:
        raise SystemExit(2)


def _print_section(name: str, status: object) -> None:
    assert isinstance(status, dict)
    print(f"[{name}] provider: {status.get('provider')}")
    print(f"[{name}] ready: {status.get('ready')}")
    if status.get("missing"):
        print(f"[{name}] missing: " + ", ".join(str(item) for item in status["missing"]))
    if status.get("required_any_missing"):
        groups = [" 或 ".join(group) for group in status["required_any_missing"]]
        print(f"[{name}] missing one of: " + "; ".join(groups))
    if status.get("configured_optional"):
        print(f"[{name}] configured optional: " + ", ".join(str(item) for item in status["configured_optional"]))
    if status.get("cache_enabled") is not None:
        print(f"[{name}] cache enabled: {status.get('cache_enabled')}")
    if status.get("probe"):
        print(f"[{name}] probe: {json.dumps(status['probe'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
