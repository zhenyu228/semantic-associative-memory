from __future__ import annotations

import hashlib
import asyncio
import inspect
import importlib.util
import json
import math
import os
import socket
import sqlite3
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sam.env import apply_provider_env_aliases, load_default_env_file
from sam.progress import progress_iter
from sam.text import tokenize


class EmbeddingProvider(ABC):
    """Embedding 抽象层，后续可替换成本地模型或在线 API。"""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """批量 embedding。默认串行，在线 provider 可覆盖为并发实现。"""

        return [self.embed(text) for text in texts]

    @property
    def cache_namespace(self) -> str:
        return self.__class__.__name__


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """无需依赖的本地哈希 embedding。

    它不是为了追求最终效果，而是确保原型在没有网络、没有 API key 时仍能复现。
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    @property
    def cache_namespace(self) -> str:
        return f"local_hash:{self.dimensions}"

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


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """本地 sentence-transformers embedding provider。

    该 provider 用于在线 embedding endpoint 不可用时接入本地真实语义模型，
    例如 Qwen/Qwen3-Embedding-0.6B、BGE 或 E5。模型路径和设备均从环境变量读取。
    """

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("使用 sentence_transformers 需要安装 sentence-transformers 包") from exc
        self.model_name = os.environ.get(
            "SAM_SENTENCE_TRANSFORMER_MODEL",
            "Qwen/Qwen3-Embedding-0.6B",
        )
        self.device = os.environ.get("SAM_SENTENCE_TRANSFORMER_DEVICE")
        self.batch_size = int(os.environ.get("SAM_SENTENCE_TRANSFORMER_BATCH_SIZE", "16"))
        self.normalize = os.environ.get("SAM_SENTENCE_TRANSFORMER_NORMALIZE", "1") != "0"
        trust_remote_code = os.environ.get("SAM_SENTENCE_TRANSFORMER_TRUST_REMOTE_CODE", "1") != "0"
        model_kwargs: dict[str, object] = {}
        init_signature = inspect.signature(SentenceTransformer)
        if self.device and "device" in init_signature.parameters:
            model_kwargs["device"] = self.device
        if "trust_remote_code" in init_signature.parameters:
            model_kwargs["trust_remote_code"] = trust_remote_code
        self.model = SentenceTransformer(self.model_name, **model_kwargs)

    @property
    def cache_namespace(self) -> str:
        device = self.device or "auto"
        return f"sentence_transformers:{self.model_name}:{device}:normalize={int(self.normalize)}"

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self.model.encode(
            texts,
            batch_size=max(1, self.batch_size),
            normalize_embeddings=self.normalize,
            convert_to_numpy=False,
            show_progress_bar=len(texts) > 1,
        )
        return [_to_float_list(vector) for vector in encoded]


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
        self.max_concurrency = int(os.environ.get("SAM_OPENAI_EMBEDDING_CONCURRENCY", "4"))
        self.batch_size = int(os.environ.get("SAM_OPENAI_EMBEDDING_BATCH_SIZE", "16"))
        dimensions = os.environ.get("SAM_OPENAI_EMBEDDING_DIMENSIONS")
        self.dimensions = int(dimensions) if dimensions else None

    @property
    def cache_namespace(self) -> str:
        return f"openai:{self.base_url}:{self.model}:{self.dimensions or 'default'}"

    def embed(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return _parallel_embed_batches(
            self._embed_batch,
            texts,
            batch_size=self.batch_size,
            max_concurrency=self.max_concurrency,
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, object] = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        return [
            [float(value) for value in item["embedding"]]
            for item in data["data"]
        ]


class AzureOpenAIEmbeddingProvider(EmbeddingProvider):
    """Azure OpenAI 兼容 embedding provider。

    配置全部来自环境变量，避免把 API key 写入仓库：
    - SAM_AZURE_EMBEDDING_API_KEY
    - SAM_AZURE_EMBEDDING_ENDPOINT
    - SAM_AZURE_EMBEDDING_URL，可选，填写后直接作为完整 embeddings 请求地址
    - SAM_AZURE_EMBEDDING_API_VERSION，默认 2023-07-01-preview
    - SAM_AZURE_EMBEDDING_MODEL，默认 text-embedding-3-large
    - SAM_AZURE_EMBEDDING_DIMENSIONS，可选，例如 1024
    """

    def __init__(self) -> None:
        apply_provider_env_aliases(target_prefix="SAM_AZURE_EMBEDDING_")
        self.api_key = _require_env("SAM_AZURE_EMBEDDING_API_KEY")
        self.api_version = os.environ.get("SAM_AZURE_EMBEDDING_API_VERSION", "2023-07-01-preview")
        self.model = os.environ.get("SAM_AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
        self.full_url = os.environ.get("SAM_AZURE_EMBEDDING_URL")
        endpoint = os.environ.get("SAM_AZURE_EMBEDDING_ENDPOINT")
        if not self.full_url and not endpoint:
            raise ValueError("缺少 SAM_AZURE_EMBEDDING_ENDPOINT 或 SAM_AZURE_EMBEDDING_URL")
        self.endpoint = endpoint.rstrip("/") if endpoint else ""
        self.auth_header = os.environ.get("SAM_AZURE_EMBEDDING_AUTH_HEADER", "api-key")
        self.max_concurrency = int(os.environ.get("SAM_AZURE_EMBEDDING_CONCURRENCY", "4"))
        self.batch_size = int(os.environ.get("SAM_AZURE_EMBEDDING_BATCH_SIZE", "16"))
        self.max_retries = int(os.environ.get("SAM_AZURE_EMBEDDING_MAX_RETRIES", "5"))
        self.request_timeout = int(os.environ.get("SAM_AZURE_EMBEDDING_TIMEOUT", "60"))
        self.send_model = os.environ.get("SAM_AZURE_EMBEDDING_SEND_MODEL", "1") != "0"
        dimensions = os.environ.get("SAM_AZURE_EMBEDDING_DIMENSIONS")
        self.dimensions = int(dimensions) if dimensions else None

    @property
    def cache_namespace(self) -> str:
        return f"azure:{self.request_url}:{self.dimensions or 'default'}"

    @property
    def request_url(self) -> str:
        if self.full_url:
            return self.full_url
        return (
            f"{self.endpoint}/openai/deployments/{self.model}/embeddings"
            f"?api-version={self.api_version}"
        )

    def embed(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return _parallel_embed_batches(
            self._embed_batch,
            texts,
            batch_size=self.batch_size,
            max_concurrency=self.max_concurrency,
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, object] = {"input": texts}
        if self.send_model:
            payload["model"] = self.model
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        for attempt in range(self.max_retries):
            try:
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
                return [
                    [float(value) for value in item["embedding"]]
                    for item in data["data"]
                ]
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(min(8.0, 1.0 + attempt))
        raise RuntimeError("embedding 请求失败")


class AzureOpenAISDKEmbeddingProvider(EmbeddingProvider):
    """基于 OpenAI SDK AsyncAzureOpenAI 的 Azure embedding provider。"""

    def __init__(self) -> None:
        apply_provider_env_aliases(target_prefix="SAM_AZURE_EMBEDDING_")
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("使用 azure_openai_sdk 需要安装 openai 包") from exc
        self.api_key = _require_env("SAM_AZURE_EMBEDDING_API_KEY")
        self.azure_endpoint = _require_env("SAM_AZURE_EMBEDDING_ENDPOINT").rstrip("/")
        self.api_version = os.environ.get("SAM_AZURE_EMBEDDING_API_VERSION", "2023-07-01-preview")
        self.model = os.environ.get("SAM_AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
        self.max_concurrency = int(os.environ.get("SAM_AZURE_EMBEDDING_CONCURRENCY", "4"))
        self.batch_size = int(os.environ.get("SAM_AZURE_EMBEDDING_BATCH_SIZE", "16"))
        self.max_retries = int(os.environ.get("SAM_AZURE_EMBEDDING_MAX_RETRIES", "5"))
        self.request_timeout = float(os.environ.get("SAM_AZURE_EMBEDDING_TIMEOUT", "120"))
        self.retry_base_seconds = float(os.environ.get("SAM_AZURE_EMBEDDING_RETRY_BASE_SECONDS", "1"))
        self.rate_limit_retries = int(os.environ.get("SAM_AZURE_EMBEDDING_RATE_LIMIT_RETRIES", "30"))
        self.rate_limit_sleep_seconds = float(os.environ.get("SAM_AZURE_EMBEDDING_RATE_LIMIT_SLEEP_SECONDS", "5"))
        self.input_mode = os.environ.get("SAM_AZURE_EMBEDDING_INPUT_MODE", "single").strip().lower()
        if self.input_mode not in {"single", "batch"}:
            raise ValueError("SAM_AZURE_EMBEDDING_INPUT_MODE 只能是 single 或 batch")
        dimensions = os.environ.get("SAM_AZURE_EMBEDDING_DIMENSIONS")
        self.dimensions = int(dimensions) if dimensions else None
        self.client = openai.AsyncAzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            api_key=self.api_key,
            timeout=self.request_timeout,
        )

    @property
    def cache_namespace(self) -> str:
        return f"azure_sdk:{self.azure_endpoint}:{self.api_version}:{self.model}:{self.dimensions or 'default'}:{self.input_mode}"

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self._embed_many_async(texts))

    async def _embed_many_async(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.input_mode == "single":
            return await self._embed_many_single_async(texts)
        safe_batch_size = max(1, self.batch_size)
        batches = [
            texts[index : index + safe_batch_size]
            for index in range(0, len(texts), safe_batch_size)
        ]
        semaphore = asyncio.Semaphore(max(1, self.max_concurrency))

        async def embed_batch(index: int, batch: list[str]) -> tuple[int, list[list[float]]]:
            async with semaphore:
                return index, await self._embed_batch_async(batch)

        tasks = [
            asyncio.create_task(embed_batch(index, batch))
            for index, batch in enumerate(batches)
        ]
        batch_results_by_index: list[list[list[float]] | None] = [None] * len(batches)
        for completed in progress_iter(asyncio.as_completed(tasks), total=len(tasks), desc="请求embedding批次"):
            index, embeddings = await completed
            batch_results_by_index[index] = embeddings
        if any(batch_result is None for batch_result in batch_results_by_index):
            raise RuntimeError("embedding 批次结果不完整")
        return [
            embedding
            for batch in batch_results_by_index
            if batch is not None
            for embedding in batch
        ]

    async def _embed_many_single_async(self, texts: list[str]) -> list[list[float]]:
        semaphore = asyncio.Semaphore(max(1, self.max_concurrency))

        async def embed_one(index: int, text: str) -> tuple[int, list[float]]:
            async with semaphore:
                return index, await self._embed_one_async(text)

        tasks = [
            asyncio.create_task(embed_one(index, text))
            for index, text in enumerate(texts)
        ]
        results: list[list[float] | None] = [None] * len(texts)
        for completed in progress_iter(asyncio.as_completed(tasks), total=len(tasks), desc="请求embedding"):
            index, embedding = await completed
            results[index] = embedding
        if any(embedding is None for embedding in results):
            raise RuntimeError("embedding 结果不完整")
        return [embedding for embedding in results if embedding is not None]

    async def _embed_one_async(self, text: str) -> list[float]:
        payload: dict[str, object] = {
            "input": text,
            "model": self.model,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        for attempt in range(self.max_retries + self.rate_limit_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.embeddings.create(**payload),
                    timeout=self.request_timeout,
                )
                return [float(value) for value in response.data[0].embedding]
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < self.max_retries + self.rate_limit_retries - 1:
                    await asyncio.sleep(self.rate_limit_sleep_seconds)
                    continue
                if attempt >= self.max_retries - 1:
                    raise
                await asyncio.sleep(min(8.0, self.retry_base_seconds + attempt))
        raise RuntimeError("embedding SDK 请求失败")

    async def _embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, object] = {
            "input": texts,
            "model": self.model,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        for attempt in range(self.max_retries + self.rate_limit_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.embeddings.create(**payload),
                    timeout=self.request_timeout,
                )
                return [
                    [float(value) for value in item.embedding]
                    for item in response.data
                ]
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < self.max_retries + self.rate_limit_retries - 1:
                    await asyncio.sleep(self.rate_limit_sleep_seconds)
                    continue
                if attempt >= self.max_retries - 1:
                    raise
                await asyncio.sleep(min(8.0, self.retry_base_seconds + attempt))
        raise RuntimeError("embedding SDK 请求失败")


class CachedEmbeddingProvider(EmbeddingProvider):
    """SQLite embedding 缓存，避免重复请求在线模型。"""

    def __init__(self, inner: EmbeddingProvider, cache_path: str | Path) -> None:
        self.inner = inner
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.cache_path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                text_sha1 TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    @property
    def cache_namespace(self) -> str:
        return f"cached:{self.inner.cache_namespace}"

    def embed(self, text: str) -> list[float]:
        cached = self._get(text)
        if cached is not None:
            return cached
        embedding = self.inner.embed(text)
        self._put(text, embedding)
        return embedding

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        missing_positions_by_text: dict[str, list[int]] = {}
        for index, text in progress_iter(enumerate(texts), total=len(texts), desc="检查embedding缓存"):
            cached = self._get(text)
            if cached is None:
                missing_positions_by_text.setdefault(text, []).append(index)
            else:
                results[index] = cached
        missing_texts = list(missing_positions_by_text)
        if missing_texts:
            write_batch_size = max(1, int(os.environ.get("SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE", "1")))
            missing_batches = [
                missing_texts[start : start + write_batch_size]
                for start in range(0, len(missing_texts), write_batch_size)
            ]
            for batch_texts in progress_iter(missing_batches, total=len(missing_batches), desc="生成缺失embedding"):
                embeddings = self.inner.embed_many(batch_texts)
                for text, embedding in zip(batch_texts, embeddings, strict=True):
                    self._put(text, embedding)
                    for position in missing_positions_by_text[text]:
                        results[position] = embedding
        if any(embedding is None for embedding in results):
            raise RuntimeError("embedding 缓存结果不完整")
        return [embedding for embedding in results if embedding is not None]

    def close(self) -> None:
        self.connection.close()

    def _get(self, text: str) -> list[float] | None:
        row = self.connection.execute(
            "SELECT embedding FROM embedding_cache WHERE cache_key = ?",
            (_cache_key(self.inner.cache_namespace, text),),
        ).fetchone()
        if not row:
            return None
        return [float(value) for value in json.loads(str(row[0]))]

    def _put(self, text: str, embedding: list[float]) -> None:
        text_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()
        self.connection.execute(
            """
            INSERT OR REPLACE INTO embedding_cache (
                cache_key, namespace, text_sha1, embedding, created_at
            )
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            (
                _cache_key(self.inner.cache_namespace, text),
                self.inner.cache_namespace,
                text_sha1,
                json.dumps(embedding),
            ),
        )
        self.connection.commit()


def create_embedding_provider(name: str | None = None) -> EmbeddingProvider:
    if name is None:
        load_default_env_file()
    provider_name = name or os.environ.get("SAM_EMBEDDING_PROVIDER", "local")
    if provider_name == "openai":
        provider: EmbeddingProvider = OpenAIEmbeddingProvider()
    elif provider_name in {"azure_openai", "azure"}:
        apply_provider_env_aliases(target_prefix="SAM_AZURE_EMBEDDING_")
        provider = AzureOpenAIEmbeddingProvider()
    elif provider_name in {"azure_openai_sdk", "azure_sdk"}:
        apply_provider_env_aliases(target_prefix="SAM_AZURE_EMBEDDING_")
        provider = AzureOpenAISDKEmbeddingProvider()
    elif provider_name in {"sentence_transformers", "sentence_transformer", "hf_local", "local_model"}:
        provider = SentenceTransformerEmbeddingProvider()
    elif provider_name == "local":
        provider = LocalHashEmbeddingProvider()
    else:
        raise ValueError(f"未知 embedding provider: {provider_name}")
    cache_path = os.environ.get("SAM_EMBEDDING_CACHE_PATH")
    if cache_path:
        return CachedEmbeddingProvider(provider, cache_path)
    if os.environ.get("SAM_EMBEDDING_CACHE") == "1":
        return CachedEmbeddingProvider(provider, "data/embedding_cache.sqlite")
    return provider


def inspect_embedding_provider_config(name: str | None = None) -> dict[str, object]:
    """检查 embedding 配置是否完整。

    返回值只包含变量名和开关状态，不返回任何密钥或 endpoint 明文。
    """

    if name is None:
        load_default_env_file()
    provider_name = name or os.environ.get("SAM_EMBEDDING_PROVIDER", "local")
    aliases = {"azure": "azure_openai"}
    provider_name = aliases.get(provider_name, provider_name)
    alias_sources: dict[str, str] = {}
    if provider_name == "local":
        missing: list[str] = []
        required_any_missing: list[list[str]] = []
        optional = ["SAM_EMBEDDING_CACHE", "SAM_EMBEDDING_CACHE_PATH"]
    elif provider_name in {"sentence_transformers", "sentence_transformer", "hf_local", "local_model"}:
        missing = []
        required_any_missing = []
        missing_packages = []
        if importlib.util.find_spec("sentence_transformers") is None:
            missing_packages.append("sentence-transformers")
        optional = [
            "SAM_SENTENCE_TRANSFORMER_MODEL",
            "SAM_SENTENCE_TRANSFORMER_DEVICE",
            "SAM_SENTENCE_TRANSFORMER_BATCH_SIZE",
            "SAM_SENTENCE_TRANSFORMER_NORMALIZE",
            "SAM_SENTENCE_TRANSFORMER_TRUST_REMOTE_CODE",
            "SAM_EMBEDDING_CACHE",
            "SAM_EMBEDDING_CACHE_PATH",
        ]
    elif provider_name == "openai":
        missing = _missing_env(["OPENAI_API_KEY"])
        required_any_missing = []
        optional = [
            "OPENAI_BASE_URL",
            "SAM_OPENAI_EMBEDDING_MODEL",
            "SAM_OPENAI_EMBEDDING_DIMENSIONS",
            "SAM_OPENAI_EMBEDDING_CONCURRENCY",
            "SAM_OPENAI_EMBEDDING_BATCH_SIZE",
            "SAM_EMBEDDING_CACHE",
            "SAM_EMBEDDING_CACHE_PATH",
        ]
    elif provider_name in {"azure_openai", "azure_openai_sdk", "azure_sdk"}:
        alias_sources = apply_provider_env_aliases(target_prefix="SAM_AZURE_EMBEDDING_")
        missing = _missing_env(["SAM_AZURE_EMBEDDING_API_KEY"])
        missing_packages = []
        if provider_name == "azure_openai":
            required_any_missing = [
                group
                for group in [["SAM_AZURE_EMBEDDING_ENDPOINT", "SAM_AZURE_EMBEDDING_URL"]]
                if not any(os.environ.get(item) for item in group)
            ]
        else:
            if importlib.util.find_spec("openai") is None:
                missing_packages.append("openai")
            required_any_missing = [
                group
                for group in [["SAM_AZURE_EMBEDDING_ENDPOINT"]]
                if not any(os.environ.get(item) for item in group)
            ]
        optional = [
            "SAM_AZURE_EMBEDDING_API_VERSION",
            "SAM_AZURE_EMBEDDING_MODEL",
            "SAM_AZURE_EMBEDDING_DIMENSIONS",
            "SAM_AZURE_EMBEDDING_CONCURRENCY",
            "SAM_AZURE_EMBEDDING_BATCH_SIZE",
            "SAM_AZURE_EMBEDDING_SEND_MODEL",
            "SAM_AZURE_EMBEDDING_AUTH_HEADER",
            "SAM_AZURE_EMBEDDING_TIMEOUT",
            "SAM_AZURE_EMBEDDING_MAX_RETRIES",
            "SAM_AZURE_EMBEDDING_RATE_LIMIT_RETRIES",
            "SAM_AZURE_EMBEDDING_RATE_LIMIT_SLEEP_SECONDS",
            "SAM_AZURE_EMBEDDING_RETRY_BASE_SECONDS",
            "SAM_EMBEDDING_CACHE",
            "SAM_EMBEDDING_CACHE_PATH",
        ]
    else:
        return {
            "provider": provider_name,
            "ready": False,
            "error": f"未知 embedding provider: {provider_name}",
            "missing": [],
            "missing_packages": [],
            "install_hint": "",
            "required_any_missing": [],
            "configured_optional": [],
            "cache_enabled": False,
        }
    missing_packages = locals().get("missing_packages", [])
    return {
        "provider": provider_name,
        "ready": not missing and not required_any_missing and not missing_packages,
        "missing": missing,
        "missing_packages": missing_packages,
        "install_hint": _install_hint(missing_packages),
        "required_any_missing": required_any_missing,
        "configured_optional": [key for key in optional if os.environ.get(key)],
        "cache_enabled": bool(os.environ.get("SAM_EMBEDDING_CACHE_PATH") or os.environ.get("SAM_EMBEDDING_CACHE") == "1"),
        "alias_sources": {
            key: value
            for key, value in alias_sources.items()
            if key.startswith("SAM_AZURE_EMBEDDING_")
        },
    }


def preflight_embedding_endpoint(name: str | None = None, *, timeout: float = 8.0) -> dict[str, object]:
    """用短超时检查在线 embedding endpoint 是否具备基础网络连通性。

    返回值不包含 endpoint 明文，只报告是否检查、是否通过和错误类型。
    该函数用于 provider smoke，避免公司网关不可达时真实请求长时间挂起。
    """

    if name is None:
        load_default_env_file()
    provider_name = name or os.environ.get("SAM_EMBEDDING_PROVIDER", "local")
    provider_name = {"azure": "azure_openai", "azure_sdk": "azure_openai_sdk"}.get(provider_name, provider_name)
    if provider_name in {"local", "sentence_transformers", "sentence_transformer", "hf_local", "local_model"}:
        return {"checked": False, "ok": True, "reason": "local_provider"}
    if provider_name == "openai":
        url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    elif provider_name in {"azure_openai", "azure_openai_sdk"}:
        apply_provider_env_aliases(target_prefix="SAM_AZURE_EMBEDDING_")
        url = (
            os.environ.get("SAM_AZURE_EMBEDDING_URL")
            or os.environ.get("SAM_AZURE_EMBEDDING_ENDPOINT")
            or ""
        )
    else:
        return {"checked": False, "ok": False, "error_type": "UnknownProvider"}
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if not host:
        return {"checked": False, "ok": False, "error_type": "MissingEndpoint"}
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {"checked": True, "ok": True}
    except Exception as exc:
        return {
            "checked": True,
            "ok": False,
            "error_type": type(exc).__name__,
            "message": "embedding endpoint TCP preflight failed",
        }


def _missing_env(keys: list[str]) -> list[str]:
    return [key for key in keys if _is_missing_env_value(os.environ.get(key))]


def _require_env(key: str) -> str:
    value = os.environ.get(key)
    if _is_missing_env_value(value):
        raise ValueError(f"缺少环境变量 {key}")
    return value


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "rate limit" in message or "qpm limit" in message or "429" in message or "limit" in message


def _is_missing_env_value(value: str | None) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    return not stripped or stripped.startswith("replace-with-")


def _install_hint(missing_packages: list[str]) -> str:
    if "openai" in missing_packages:
        return "python -m pip install 'openai>=1.0.0'"
    if "sentence-transformers" in missing_packages:
        return "python -m pip install 'sentence-transformers>=3.0.0'"
    return ""


def _parallel_embed(
    embed_fn,
    texts: list[str],
    max_concurrency: int,
) -> list[list[float]]:
    if max_concurrency <= 1 or len(texts) <= 1:
        return [embed_fn(text) for text in texts]
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        return list(executor.map(embed_fn, texts))


def _parallel_embed_batches(
    embed_batch_fn,
    texts: list[str],
    batch_size: int,
    max_concurrency: int,
) -> list[list[float]]:
    if not texts:
        return []
    safe_batch_size = max(1, batch_size)
    batches = [
        texts[index : index + safe_batch_size]
        for index in range(0, len(texts), safe_batch_size)
    ]
    if max_concurrency <= 1 or len(batches) <= 1:
        batch_results = list(
            progress_iter(
                (embed_batch_fn(batch) for batch in batches),
                total=len(batches),
                desc="请求embedding批次",
            )
        )
    else:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            batch_results = list(
                progress_iter(
                    executor.map(embed_batch_fn, batches),
                    total=len(batches),
                    desc="请求embedding批次",
                )
            )
    return [
        embedding
        for batch in batch_results
        for embedding in batch
    ]


def _cache_key(namespace: str, text: str) -> str:
    digest = hashlib.sha1(f"{namespace}\n{text}".encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def _to_float_list(vector: object) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]  # type: ignore[union-attr]
