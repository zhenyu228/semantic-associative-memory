from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sam.models import DatasetDocument, EvaluationQuery


SCHEMA_VERSION = "sam-dataset-v1"


def save_sam_dataset(
    path: str | Path,
    documents: list[DatasetDocument],
    queries: list[EvaluationQuery],
    dataset_info: dict[str, Any],
    processing: dict[str, Any],
) -> Path:
    """保存 SAM 项目统一数据格式。

    所有外部数据集都应先转换成这个格式，再进入记忆系统。
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_info": dataset_info,
        "processing": processing,
        "documents": [asdict(document) for document in documents],
        "queries": [asdict(query) for query in queries],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_sam_dataset(path: str | Path) -> tuple[list[DatasetDocument], list[EvaluationQuery], dict[str, Any]]:
    """读取 SAM 项目统一数据格式。"""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"不支持的数据格式版本：{schema_version}")
    documents = [
        DatasetDocument(
            id=item["id"],
            dataset=item["dataset"],
            title=item["title"],
            text=item["text"],
            source=item["source"],
            tags=list(item["tags"]),
            keywords=list(item["keywords"]),
            metadata=dict(item.get("metadata", {})),
        )
        for item in payload["documents"]
    ]
    queries = [
        EvaluationQuery(
            id=item["id"],
            dataset=item["dataset"],
            question=item["question"],
            answer=item["answer"],
            supporting_doc_ids=list(item["supporting_doc_ids"]),
            candidate_doc_ids=list(item["candidate_doc_ids"]),
            metadata=dict(item.get("metadata", {})),
        )
        for item in payload["queries"]
    ]
    return documents, queries, payload


def summarize_sam_dataset(path: str | Path) -> dict[str, Any]:
    """返回统一数据文件的摘要，便于脚本打印和人工检查。"""

    documents, queries, payload = load_sam_dataset(path)
    supporting_ids = {
        doc_id
        for query in queries
        for doc_id in query.supporting_doc_ids
    }
    return {
        "path": str(path),
        "schema_version": payload["schema_version"],
        "dataset_name": payload["dataset_info"].get("name"),
        "document_count": len(documents),
        "query_count": len(queries),
        "supporting_document_count": len(supporting_ids),
        "processing": payload.get("processing", {}),
    }
