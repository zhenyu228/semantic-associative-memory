from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试公司 Azure-style OpenAI 网关是否可用")
    parser.add_argument("--api-key", default=None, help="默认读取 GPT54_API_KEY 或 OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None, help="默认读取 GPT54_BASE_URL 或 RAPTOR_AZURE_ENDPOINT")
    parser.add_argument("--api-version", default=None, help="默认读取 GPT54_API_VERSION 或 2024-02-01")
    parser.add_argument("--chat-model", default=None, help="默认读取 GPT54_MODEL 或 RAPTOR_QA_MODEL")
    parser.add_argument("--embedding-model", default=None, help="可选：测试 embedding deployment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv("GPT54_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("GPT54_BASE_URL") or os.getenv("RAPTOR_AZURE_ENDPOINT")
    api_version = args.api_version or os.getenv("GPT54_API_VERSION") or "2024-02-01"
    chat_model = args.chat_model or os.getenv("GPT54_MODEL") or os.getenv("RAPTOR_QA_MODEL")

    if not api_key:
        raise RuntimeError("缺少 API key：请设置 GPT54_API_KEY 或 OPENAI_API_KEY。")
    if not base_url:
        raise RuntimeError("缺少 base url：请设置 GPT54_BASE_URL 或 RAPTOR_AZURE_ENDPOINT。")
    if not chat_model:
        raise RuntimeError("缺少 chat model/deployment：请设置 GPT54_MODEL 或 RAPTOR_QA_MODEL。")

    chat_payload = {
        "messages": [{"role": "user", "content": "只回复 OK"}],
        "temperature": 0,
        "max_completion_tokens": 8,
    }
    chat_response = _post_azure(
        base_url=base_url,
        deployment=chat_model,
        endpoint="chat/completions",
        api_version=api_version,
        api_key=api_key,
        payload=chat_payload,
    )
    chat_text = chat_response["choices"][0]["message"]["content"].strip()
    print(f"chat_ok=true model={chat_model} response={chat_text}")

    embedding_model = args.embedding_model or os.getenv("RAPTOR_EMBEDDING_MODEL") or os.getenv("GRAPHRAG_EMBEDDING_MODEL")
    if not embedding_model:
        print("embedding_ok=skipped reason=未设置 embedding deployment")
        return

    embedding_payload = {"input": ["SAM API probe"]}
    embedding_response = _post_azure(
        base_url=base_url,
        deployment=embedding_model,
        endpoint="embeddings",
        api_version=api_version,
        api_key=api_key,
        payload=embedding_payload,
    )
    vector = embedding_response["data"][0]["embedding"]
    print(f"embedding_ok=true model={embedding_model} dimension={len(vector)}")


def _post_azure(
    base_url: str,
    deployment: str,
    endpoint: str,
    api_version: str,
    api_key: str,
    payload: dict,
) -> dict:
    url = f"{base_url.rstrip('/')}/openai/deployments/{deployment}/{endpoint}?api-version={api_version}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"请求失败：HTTP {exc.code} {body[:500]}") from exc


if __name__ == "__main__":
    main()
