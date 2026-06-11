from __future__ import annotations

import json
import importlib.util
import os
import re
import time
import urllib.request
from abc import ABC, abstractmethod

from sam.env import apply_provider_env_aliases, load_default_env_file


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
    - SAM_AZURE_CHAT_RATE_LIMIT_RETRIES，可选，限流错误重试次数
    - SAM_AZURE_CHAT_RATE_LIMIT_SLEEP_SECONDS，可选，限流错误最小等待秒数
    - SAM_AZURE_CHAT_MIN_INTERVAL_SECONDS，可选，相邻请求最小间隔
    """

    def __init__(self) -> None:
        apply_provider_env_aliases(target_prefix="SAM_AZURE_CHAT_")
        self.api_key = _require_env("SAM_AZURE_CHAT_API_KEY")
        self.api_version = os.environ.get("SAM_AZURE_CHAT_API_VERSION", "2024-02-01")
        self.model = os.environ.get("SAM_AZURE_CHAT_MODEL", "gpt-5.4-2026-03-05")
        self.full_url = os.environ.get("SAM_AZURE_CHAT_URL")
        endpoint = os.environ.get("SAM_AZURE_CHAT_ENDPOINT")
        if not self.full_url and not endpoint:
            raise ValueError("缺少 SAM_AZURE_CHAT_ENDPOINT 或 SAM_AZURE_CHAT_URL")
        self.endpoint = endpoint.rstrip("/") if endpoint else ""
        self.auth_header = os.environ.get("SAM_AZURE_CHAT_AUTH_HEADER", "api-key")
        self.request_timeout = float(os.environ.get("SAM_AZURE_CHAT_TIMEOUT", "120"))
        self.retry_base_seconds = float(os.environ.get("SAM_AZURE_CHAT_RETRY_BASE_SECONDS", "2"))
        self.rate_limit_retries = int(os.environ.get("SAM_AZURE_CHAT_RATE_LIMIT_RETRIES", "3"))
        self.rate_limit_sleep_seconds = float(
            os.environ.get("SAM_AZURE_CHAT_RATE_LIMIT_SLEEP_SECONDS", str(self.retry_base_seconds))
        )
        self.min_interval_seconds = float(os.environ.get("SAM_AZURE_CHAT_MIN_INTERVAL_SECONDS", "0"))
        self._last_request_at = 0.0

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
        for attempt in range(max(1, self.rate_limit_retries)):
            try:
                self._last_request_at = _throttle_request(
                    self._last_request_at,
                    self.min_interval_seconds,
                )
                request = urllib.request.Request(
                    self.request_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        self.auth_header: self.api_key,
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt == max(1, self.rate_limit_retries) - 1:
                    raise
                time.sleep(
                    _rate_limit_retry_delay(
                        exc,
                        attempt,
                        self.retry_base_seconds,
                        self.rate_limit_sleep_seconds,
                    )
                )
        return str(data["choices"][0]["message"]["content"]).strip()


class AzureOpenAISDKChatClient(ChatClient):
    """基于 OpenAI SDK AzureOpenAI 的 GPT 聊天接口。"""

    def __init__(self) -> None:
        apply_provider_env_aliases(target_prefix="SAM_AZURE_CHAT_")
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("使用 azure_openai_sdk 需要安装 openai 包") from exc
        self.api_key = _require_env("SAM_AZURE_CHAT_API_KEY")
        self.azure_endpoint = _require_env("SAM_AZURE_CHAT_ENDPOINT").rstrip("/")
        self.api_version = os.environ.get("SAM_AZURE_CHAT_API_VERSION", "2024-02-01")
        self.model = os.environ.get("SAM_AZURE_CHAT_MODEL", "gpt-5.4-2026-03-05")
        self.request_timeout = float(os.environ.get("SAM_AZURE_CHAT_TIMEOUT", "60"))
        self.max_retries = int(os.environ.get("SAM_AZURE_CHAT_MAX_RETRIES", "3"))
        self.retry_base_seconds = float(os.environ.get("SAM_AZURE_CHAT_RETRY_BASE_SECONDS", "2"))
        self.rate_limit_retries = int(
            os.environ.get("SAM_AZURE_CHAT_RATE_LIMIT_RETRIES", str(self.max_retries))
        )
        self.rate_limit_sleep_seconds = float(
            os.environ.get("SAM_AZURE_CHAT_RATE_LIMIT_SLEEP_SECONDS", str(self.retry_base_seconds))
        )
        self.min_interval_seconds = float(os.environ.get("SAM_AZURE_CHAT_MIN_INTERVAL_SECONDS", "0"))
        self._last_request_at = 0.0
        self.client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            timeout=self.request_timeout,
        )

    def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
        for attempt in range(max(1, self.rate_limit_retries)):
            try:
                self._last_request_at = _throttle_request(
                    self._last_request_at,
                    self.min_interval_seconds,
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    stream=False,
                )
                break
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt == max(1, self.rate_limit_retries) - 1:
                    raise
                time.sleep(
                    _rate_limit_retry_delay(
                        exc,
                        attempt,
                        self.retry_base_seconds,
                        self.rate_limit_sleep_seconds,
                    )
                )
        content = response.choices[0].message.content
        if isinstance(content, list):
            return "".join(str(item.get("text", item)) for item in content).strip()
        return str(content or "").strip()


def create_chat_client(name: str | None = None) -> ChatClient:
    if name is None:
        load_default_env_file()
    provider_name = name or os.environ.get("SAM_CHAT_PROVIDER", "heuristic")
    if provider_name in {"heuristic", "local"}:
        return HeuristicChatClient()
    if provider_name in {"azure_openai", "azure"}:
        apply_provider_env_aliases(target_prefix="SAM_AZURE_CHAT_")
        return AzureOpenAIChatClient()
    if provider_name in {"azure_openai_sdk", "azure_sdk"}:
        apply_provider_env_aliases(target_prefix="SAM_AZURE_CHAT_")
        return AzureOpenAISDKChatClient()
    raise ValueError(f"未知 chat provider: {provider_name}")


def inspect_chat_provider_config(name: str | None = None) -> dict[str, object]:
    """检查聊天模型配置是否完整。

    返回结果只包含环境变量名称和开关状态，不暴露 API key 或 endpoint 明文。
    """

    if name is None:
        load_default_env_file()
    provider_name = name or os.environ.get("SAM_CHAT_PROVIDER", "heuristic")
    aliases = {"azure": "azure_openai", "local": "heuristic", "azure_sdk": "azure_openai_sdk"}
    provider_name = aliases.get(provider_name, provider_name)
    alias_sources: dict[str, str] = {}
    if provider_name == "heuristic":
        missing: list[str] = []
        missing_packages: list[str] = []
        required_any_missing: list[list[str]] = []
        optional: list[str] = []
    elif provider_name in {"azure_openai", "azure_openai_sdk"}:
        alias_sources = apply_provider_env_aliases(target_prefix="SAM_AZURE_CHAT_")
        missing = _missing_env(["SAM_AZURE_CHAT_API_KEY"])
        missing_packages = []
        if provider_name == "azure_openai_sdk" and importlib.util.find_spec("openai") is None:
            missing_packages.append("openai")
        required_any_missing = [
            group
            for group in [[
                "SAM_AZURE_CHAT_ENDPOINT",
                *([] if provider_name == "azure_openai_sdk" else ["SAM_AZURE_CHAT_URL"]),
            ]]
            if not any(os.environ.get(item) for item in group)
        ]
        optional = [
            "SAM_AZURE_CHAT_API_VERSION",
            "SAM_AZURE_CHAT_MODEL",
            "SAM_AZURE_CHAT_AUTH_HEADER",
            "SAM_AZURE_CHAT_URL",
            "SAM_AZURE_CHAT_ENDPOINT",
            "SAM_AZURE_CHAT_TIMEOUT",
            "SAM_AZURE_CHAT_MAX_RETRIES",
            "SAM_AZURE_CHAT_RETRY_BASE_SECONDS",
            "SAM_AZURE_CHAT_RATE_LIMIT_RETRIES",
            "SAM_AZURE_CHAT_RATE_LIMIT_SLEEP_SECONDS",
            "SAM_AZURE_CHAT_MIN_INTERVAL_SECONDS",
        ]
    else:
        return {
            "provider": provider_name,
            "ready": False,
            "error": f"未知 chat provider: {provider_name}",
            "missing": [],
            "missing_packages": [],
            "install_hint": "",
            "required_any_missing": [],
            "configured_optional": [],
        }
    return {
        "provider": provider_name,
        "ready": not missing and not required_any_missing and not missing_packages,
        "missing": missing,
        "missing_packages": missing_packages,
        "install_hint": _install_hint(missing_packages),
        "required_any_missing": required_any_missing,
        "configured_optional": [key for key in optional if os.environ.get(key)],
        "alias_sources": {
            key: value
            for key, value in alias_sources.items()
            if key.startswith("SAM_AZURE_CHAT_")
        },
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


def _install_hint(missing_packages: list[str]) -> str:
    if "openai" in missing_packages:
        return "python -m pip install 'openai>=1.0.0'"
    return ""


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return "ratelimit" in text or "rate limit" in text or "qpm limit" in text or "429" in text


def _retry_delay_seconds(exc: Exception, attempt: int, base_seconds: float) -> float:
    retry_after = getattr(exc, "response", None)
    headers = getattr(retry_after, "headers", {}) if retry_after is not None else {}
    if headers:
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value:
            try:
                return max(0.0, min(60.0, float(value)))
            except ValueError:
                pass
    return max(0.0, min(60.0, base_seconds * (2 ** attempt)))


def _rate_limit_retry_delay(
    exc: Exception,
    attempt: int,
    base_seconds: float,
    minimum_seconds: float,
) -> float:
    return max(
        _retry_delay_seconds(exc, attempt, base_seconds),
        max(0.0, minimum_seconds),
    )


def _throttle_request(last_request_at: float, min_interval_seconds: float) -> float:
    min_interval_seconds = max(0.0, min_interval_seconds)
    if last_request_at > 0.0 and min_interval_seconds > 0.0:
        elapsed = time.monotonic() - last_request_at
        if elapsed < min_interval_seconds:
            time.sleep(min_interval_seconds - elapsed)
    return time.monotonic()
