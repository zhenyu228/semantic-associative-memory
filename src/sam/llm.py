from __future__ import annotations

import json
import os
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
        for line in user_text.splitlines():
            clean = line.strip()
            if clean in {"证据不足", "insufficient evidence"}:
                return clean[:max_tokens]
        return "证据不足"


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
        self.api_key = os.environ["SAM_AZURE_CHAT_API_KEY"]
        self.endpoint = os.environ["SAM_AZURE_CHAT_ENDPOINT"].rstrip("/")
        self.api_version = os.environ.get("SAM_AZURE_CHAT_API_VERSION", "2024-02-01")
        self.model = os.environ.get("SAM_AZURE_CHAT_MODEL", "gpt-5.4-2026-03-05")
        self.full_url = os.environ.get("SAM_AZURE_CHAT_URL")
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
