from __future__ import annotations

import json
import os
import re
import urllib.request
from abc import ABC, abstractmethod


class ChatClient(ABC):
    """聊天模型抽象层。"""

    @abstractmethod
    def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
        raise NotImplementedError


class HeuristicChatClient(ChatClient):
    """无 API key 时的兜底生成器。

    该实现只用于本地测试和流程验证，不作为最终实验结论。
    """

    def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
        user_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "user"
        )
        for marker in ["标准答案：", "Gold answer:"]:
            if marker in user_text:
                return user_text.split(marker, 1)[1].splitlines()[0].strip()
        for marker in ["answer is ", "答案是"]:
            lowered = user_text.lower()
            if marker in lowered:
                start = lowered.index(marker) + len(marker)
                return user_text[start:].splitlines()[0].strip().strip("。.")[:max_tokens]
        context_text = _context_section(user_text)
        for pattern in [
            r"\bthe city is ([A-Z][A-Za-z0-9_\- ]{1,60})[\.。\n]",
            r"\blocated in ([A-Z][A-Za-z0-9_\- ]{1,60})[\.。\n]",
            r"\bis ([A-Z][A-Za-z0-9_\- ]{1,40})[\.。\n]",
        ]:
            match = re.search(pattern, context_text)
            if match:
                return match.group(1).strip()[:max_tokens]
        for line in user_text.splitlines():
            clean = line.strip()
            if clean in {"证据不足", "insufficient evidence"}:
                return clean[:max_tokens]
        return "证据不足"


def _context_section(text: str) -> str:
    for marker in ["上下文：", "Context:"]:
        if marker in text:
            return text.split(marker, 1)[1]
    return text


class AzureOpenAIChatClient(ChatClient):
    """Azure OpenAI 兼容 GPT 聊天接口。

    配置全部来自环境变量：
    - SAM_AZURE_CHAT_API_KEY
    - SAM_AZURE_CHAT_ENDPOINT
    - SAM_AZURE_CHAT_API_VERSION，默认 2024-02-01
    - SAM_AZURE_CHAT_MODEL，默认 gpt-5.4-2026-03-05
    - SAM_AZURE_CHAT_URL，可选；如果公司网关不是标准 Azure 路径，可直接传完整 URL
    - SAM_AZURE_CHAT_AUTH_HEADER，可选，默认 api-key
    """

    def __init__(self) -> None:
        self.api_key = _require_env("SAM_AZURE_CHAT_API_KEY")
        self.api_version = os.environ.get("SAM_AZURE_CHAT_API_VERSION", "2024-02-01")
        self.model = os.environ.get("SAM_AZURE_CHAT_MODEL", "gpt-5.4-2026-03-05")
        self.full_url = os.environ.get("SAM_AZURE_CHAT_URL")
        endpoint = os.environ.get("SAM_AZURE_CHAT_ENDPOINT")
        if not self.full_url and not endpoint:
            raise ValueError("缺少 SAM_AZURE_CHAT_ENDPOINT 或 SAM_AZURE_CHAT_URL")
        self.endpoint = endpoint.rstrip("/") if endpoint else ""
        self.auth_header = os.environ.get("SAM_AZURE_CHAT_AUTH_HEADER", "api-key")

    @property
    def request_url(self) -> str:
        if self.full_url:
            return self.full_url
        return (
            f"{self.endpoint}/openai/deployments/{self.model}/chat/completions"
            f"?api-version={self.api_version}"
        )

    def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            self.request_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                self.auth_header: self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"]).strip()


def create_chat_client(name: str | None = None) -> ChatClient:
    provider_name = name or os.environ.get("SAM_CHAT_PROVIDER", "heuristic")
    if provider_name in {"heuristic", "local"}:
        return HeuristicChatClient()
    if provider_name in {"azure_openai", "azure"}:
        return AzureOpenAIChatClient()
    raise ValueError(f"未知 chat provider: {provider_name}")


def inspect_chat_provider_config(name: str | None = None) -> dict[str, object]:
    """检查聊天模型配置是否完整。

    返回结果只包含环境变量名称和开关状态，不暴露 API key 或 endpoint 明文。
    """

    provider_name = name or os.environ.get("SAM_CHAT_PROVIDER", "heuristic")
    aliases = {"azure": "azure_openai", "local": "heuristic"}
    provider_name = aliases.get(provider_name, provider_name)
    if provider_name == "heuristic":
        missing: list[str] = []
        required_any_missing: list[list[str]] = []
        optional: list[str] = []
    elif provider_name == "azure_openai":
        missing = _missing_env(["SAM_AZURE_CHAT_API_KEY"])
        required_any_missing = [
            group
            for group in [["SAM_AZURE_CHAT_ENDPOINT", "SAM_AZURE_CHAT_URL"]]
            if not any(os.environ.get(item) for item in group)
        ]
        optional = [
            "SAM_AZURE_CHAT_API_VERSION",
            "SAM_AZURE_CHAT_MODEL",
            "SAM_AZURE_CHAT_AUTH_HEADER",
            "SAM_AZURE_CHAT_URL",
            "SAM_AZURE_CHAT_ENDPOINT",
        ]
    else:
        return {
            "provider": provider_name,
            "ready": False,
            "error": f"未知 chat provider: {provider_name}",
            "missing": [],
            "required_any_missing": [],
            "configured_optional": [],
        }
    return {
        "provider": provider_name,
        "ready": not missing and not required_any_missing,
        "missing": missing,
        "required_any_missing": required_any_missing,
        "configured_optional": [key for key in optional if os.environ.get(key)],
    }


def _missing_env(keys: list[str]) -> list[str]:
    return [key for key in keys if _is_missing_env_value(os.environ.get(key))]


def _require_env(key: str) -> str:
    value = os.environ.get(key)
    if _is_missing_env_value(value):
        raise ValueError(f"缺少环境变量 {key}")
    return value


def _is_missing_env_value(value: str | None) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    return not stripped or stripped.startswith("replace-with-")
