from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

from sam.dataset_format import load_sam_dataset
from sam.models import DatasetDocument
from sam.text import extract_keywords


def build_embedding_run_plan(
    *,
    dataset_path: str | Path,
    provider_name: str | None = None,
    cache_path: str | Path | None = None,
    cache_namespace: str | None = None,
    batch_size: int | None = None,
    include_query_summaries: bool = True,
) -> dict[str, object]:
    """估算一次 embedding 实验会请求多少唯一文本。

    该函数只读取数据集和本地缓存，不实例化在线 provider，也不发送网络请求。
    """

    documents, _queries, payload = load_sam_dataset(dataset_path)
    document_texts = [_document_embedding_text(document) for document in documents]
    summary_texts = _summary_embedding_texts(documents) if include_query_summaries else []
    all_texts = [*document_texts, *summary_texts]
    unique_texts = list(dict.fromkeys(all_texts))
    resolved_provider = provider_name or os.environ.get("SAM_EMBEDDING_PROVIDER", "local")
    resolved_batch_size = max(1, batch_size or _provider_batch_size(resolved_provider))
    namespace = cache_namespace or _cache_namespace_from_env(resolved_provider)
    cache = _inspect_cache(
        cache_path=cache_path or os.environ.get("SAM_EMBEDDING_CACHE_PATH"),
        texts=unique_texts,
        namespace=namespace,
    )
    miss_count = len(unique_texts) - int(cache["hit_count"])
    return {
        "dataset_path": str(dataset_path),
        "dataset_name": payload.get("dataset_info", {}).get("name"),
        "provider": resolved_provider,
        "document_text_count": len(document_texts),
        "summary_text_count": len(summary_texts),
        "total_text_count": len(all_texts),
        "unique_text_count": len(unique_texts),
        "duplicate_text_count": len(all_texts) - len(unique_texts),
        "cache_path": str(cache_path or os.environ.get("SAM_EMBEDDING_CACHE_PATH") or ""),
        "cache_namespace_mode": cache["namespace_mode"],
        "cache_hit_count": cache["hit_count"],
        "cache_miss_count": miss_count,
        "batch_size": resolved_batch_size,
        "estimated_batch_count": math.ceil(miss_count / resolved_batch_size) if miss_count else 0,
        "will_call_provider": miss_count > 0,
    }


def write_embedding_run_plan(plan: dict[str, object], output_dir: str | Path) -> tuple[Path, Path]:
    """写出 embedding 运行计划 JSON 和 Markdown。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "embedding_run_plan.json"
    markdown_path = target / "embedding_run_plan.md"
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_plan_to_markdown(plan), encoding="utf-8")
    return json_path, markdown_path


def _document_embedding_text(document: DatasetDocument) -> str:
    return f"{document.title}\n{document.text}"


def _summary_embedding_texts(documents: list[DatasetDocument]) -> list[str]:
    groups: dict[str, list[DatasetDocument]] = {}
    for document in documents:
        query_id = document.metadata.get("query_id")
        if query_id:
            groups.setdefault(str(query_id), []).append(document)

    texts: list[str] = []
    for query_id, group in groups.items():
        ordered = sorted(group, key=lambda document: str(document.metadata.get("paragraph_index", document.id)))
        title_terms = [str(document.metadata.get("title") or document.title) for document in ordered]
        keyword_terms = sorted({keyword for document in ordered for keyword in document.keywords[:6]})
        summary_text = "\n".join(
            f"{document.metadata.get('title') or document.title}: {document.text[:180]}"
            for document in ordered
        )
        texts.append(
            "查询上下文摘要："
            f"{query_id}\n"
            f"候选标题：{'; '.join(title_terms)}\n"
            f"关键词：{', '.join(keyword_terms[:32])}\n"
            f"{summary_text}"
        )
    return texts


def _provider_batch_size(provider_name: str) -> int:
    if provider_name in {"openai"}:
        return int(os.environ.get("SAM_OPENAI_EMBEDDING_BATCH_SIZE", "16"))
    if provider_name in {"azure_openai", "azure", "azure_openai_sdk", "azure_sdk"}:
        return int(os.environ.get("SAM_AZURE_EMBEDDING_BATCH_SIZE", "16"))
    return 16


def _cache_namespace_from_env(provider_name: str) -> str | None:
    if provider_name == "local":
        dimensions = int(os.environ.get("SAM_LOCAL_EMBEDDING_DIMENSIONS", "256"))
        return f"local_hash:{dimensions}"
    if provider_name == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.environ.get("SAM_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        dimensions = os.environ.get("SAM_OPENAI_EMBEDDING_DIMENSIONS") or "default"
        return f"openai:{base_url}:{model}:{dimensions}"
    if provider_name in {"azure_openai_sdk", "azure_sdk"}:
        endpoint = os.environ.get("SAM_AZURE_EMBEDDING_ENDPOINT")
        if not endpoint:
            return None
        api_version = os.environ.get("SAM_AZURE_EMBEDDING_API_VERSION", "2023-07-01-preview")
        model = os.environ.get("SAM_AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
        dimensions = os.environ.get("SAM_AZURE_EMBEDDING_DIMENSIONS") or "default"
        return f"azure_sdk:{endpoint.rstrip('/')}:{api_version}:{model}:{dimensions}"
    return None


def _inspect_cache(
    *,
    cache_path: str | Path | None,
    texts: list[str],
    namespace: str | None,
) -> dict[str, object]:
    if not cache_path:
        return {"hit_count": 0, "namespace_mode": "disabled"}
    path = Path(cache_path)
    if not path.exists():
        return {"hit_count": 0, "namespace_mode": "missing_cache_file"}
    connection = sqlite3.connect(path)
    try:
        if namespace:
            keys = [_cache_key(namespace, text) for text in texts]
            hits = _count_existing_values(connection, "cache_key", keys)
            return {"hit_count": hits, "namespace_mode": "exact"}
        text_hashes = [hashlib.sha1(text.encode("utf-8")).hexdigest() for text in texts]
        hits = _count_existing_values(connection, "text_sha1", text_hashes)
        return {"hit_count": hits, "namespace_mode": "text_sha1_any_namespace"}
    finally:
        connection.close()


def _count_existing_values(connection: sqlite3.Connection, column: str, values: list[str]) -> int:
    if not values:
        return 0
    placeholders = ",".join("?" for _ in values)
    row = connection.execute(
        f"SELECT COUNT(DISTINCT {column}) FROM embedding_cache WHERE {column} IN ({placeholders})",
        values,
    ).fetchone()
    return int(row[0] if row else 0)


def _cache_key(namespace: str, text: str) -> str:
    digest = hashlib.sha1(f"{namespace}\n{text}".encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def _plan_to_markdown(plan: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Embedding 运行计划",
            "",
            f"- 数据集：{plan.get('dataset_name') or plan.get('dataset_path')}",
            f"- Provider：{plan.get('provider')}",
            f"- 文档 embedding 文本数：{plan.get('document_text_count')}",
            f"- 摘要 embedding 文本数：{plan.get('summary_text_count')}",
            f"- 唯一文本数：{plan.get('unique_text_count')}",
            f"- 缓存命中数：{plan.get('cache_hit_count')}",
            f"- 预计需要请求文本数：{plan.get('cache_miss_count')}",
            f"- Batch size：{plan.get('batch_size')}",
            f"- 预计 batch 数：{plan.get('estimated_batch_count')}",
            f"- 是否会调用在线 provider：{plan.get('will_call_provider')}",
            "",
            "该计划只读取本地数据集和缓存，不发送 embedding 请求。",
        ]
    )
