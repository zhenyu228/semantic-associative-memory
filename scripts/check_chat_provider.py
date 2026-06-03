from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.llm import create_chat_client, inspect_chat_provider_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 SAM GPT-5.4 聊天模型配置")
    parser.add_argument("--provider", default=None, help="heuristic 或 azure_openai；默认读取 SAM_CHAT_PROVIDER")
    parser.add_argument("--probe", default=None, help="可选：发送一条测试消息并返回截断后的输出")
    parser.add_argument("--max-tokens", type=int, default=64, help="probe 最大输出 token")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出诊断结果")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = inspect_chat_provider_config(args.provider)
    if args.probe and status.get("ready"):
        client = create_chat_client(args.provider)
        answer = client.complete(
            [{"role": "user", "content": args.probe}],
            max_tokens=args.max_tokens,
        )
        status["probe"] = {
            "answer_preview": answer[:200],
            "answer_chars": len(answer),
        }

    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print(f"provider: {status.get('provider')}")
        print(f"ready: {status.get('ready')}")
        if status.get("missing"):
            print("missing: " + ", ".join(str(item) for item in status["missing"]))
        if status.get("required_any_missing"):
            groups = [" 或 ".join(group) for group in status["required_any_missing"]]
            print("missing one of: " + "; ".join(groups))
        if status.get("configured_optional"):
            print("configured optional: " + ", ".join(str(item) for item in status["configured_optional"]))
        if status.get("probe"):
            probe = status["probe"]
            print(f"probe answer chars: {probe['answer_chars']}")
            print(f"probe answer preview: {probe['answer_preview']}")
        if status.get("error"):
            print(f"error: {status.get('error')}")

    if not status.get("ready"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
