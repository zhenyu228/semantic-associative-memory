from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from abc import ABC, abstractmethod

from sam.text import tokenize


class EmbeddingProvider(ABC):
    """Embedding 抽象层，后续可替换成本地模型或在线 API。"""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """无需依赖的本地哈希 embedding。

    它不是为了追求最终效果，而是确保原型在没有网络、没有 API key 时仍能复现。
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI 兼容 embedding provider。

    只从环境变量读取配置，不把 key 写入仓库：
    - OPENAI_API_KEY
    - OPENAI_BASE_URL，可选，默认 https://api.openai.com/v1
    - SAM_OPENAI_EMBEDDING_MODEL，可选，默认 text-embedding-3-small
    """

    def __init__(self) -> None:
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.environ.get("SAM_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        return [float(value) for value in data["data"][0]["embedding"]]


class AzureOpenAIEmbeddingProvider(EmbeddingProvider):
    """Azure OpenAI 兼容 embedding provider。

    配置全部来自环境变量，避免把 API key 写入仓库：
    - SAM_AZURE_EMBEDDING_API_KEY
    - SAM_AZURE_EMBEDDING_ENDPOINT
    - SAM_AZURE_EMBEDDING_API_VERSION，默认 2023-07-01-preview
    - SAM_AZURE_EMBEDDING_MODEL，默认 text-embedding-3-large
    - SAM_AZURE_EMBEDDING_DIMENSIONS，可选，例如 1024
    """

    def __init__(self) -> None:
        self.api_key = os.environ["SAM_AZURE_EMBEDDING_API_KEY"]
        self.endpoint = os.environ["SAM_AZURE_EMBEDDING_ENDPOINT"].rstrip("/")
        self.api_version = os.environ.get("SAM_AZURE_EMBEDDING_API_VERSION", "2023-07-01-preview")
        self.model = os.environ.get("SAM_AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
        dimensions = os.environ.get("SAM_AZURE_EMBEDDING_DIMENSIONS")
        self.dimensions = int(dimensions) if dimensions else None

    @property
    def request_url(self) -> str:
        return (
            f"{self.endpoint}/openai/deployments/{self.model}/embeddings"
            f"?api-version={self.api_version}"
        )

    def embed(self, text: str) -> list[float]:
        payload: dict[str, object] = {"input": text}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        request = urllib.request.Request(
            self.request_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "api-key": self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        return [float(value) for value in data["data"][0]["embedding"]]


def create_embedding_provider(name: str | None = None) -> EmbeddingProvider:
    provider_name = name or os.environ.get("SAM_EMBEDDING_PROVIDER", "local")
    if provider_name == "openai":
        return OpenAIEmbeddingProvider()
    if provider_name in {"azure_openai", "azure"}:
        return AzureOpenAIEmbeddingProvider()
    if provider_name == "local":
        return LocalHashEmbeddingProvider()
    raise ValueError(f"未知 embedding provider: {provider_name}")
