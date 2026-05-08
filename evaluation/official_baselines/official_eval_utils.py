from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*|[\u4e00-\u9fff]{2,}")


def load_common_inputs(prepared_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(prepared_dir)
    documents = json.loads((root / "common/documents.json").read_text(encoding="utf-8"))
    queries = json.loads((root / "common/queries.json").read_text(encoding="utf-8"))
    return documents, queries


def write_json(path: str | Path, payload: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def answer_hit(answer: str, gold_answers: list[str]) -> bool:
    normalized_answer = answer.lower()
    return any(gold and gold.lower() in normalized_answer for gold in gold_answers)


def evidence_recall(retrieved_doc_ids: list[str], supporting_doc_ids: list[str]) -> float | None:
    if not supporting_doc_ids:
        return None
    return len(set(retrieved_doc_ids) & set(supporting_doc_ids)) / len(set(supporting_doc_ids))


def summarize_answer_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    answer_hits = sum(1 for result in results if result.get("answer_hit"))
    evidence_values = [
        result["evidence_recall"]
        for result in results
        if result.get("evidence_recall") is not None
    ]
    return {
        "query_count": len(results),
        "answer_hit_count": answer_hits,
        "answer_hit_rate": answer_hits / len(results) if results else 0.0,
        "mean_evidence_recall": (
            sum(float(value) for value in evidence_values) / len(evidence_values)
            if evidence_values
            else None
        ),
    }


def rough_retrieved_doc_ids_from_text(answer_or_context: str, documents: list[dict[str, Any]], top_k: int = 8) -> list[str]:
    """从官方方法返回的自由文本中粗略反查命中的文档。

    RAPTOR/GraphRAG 的官方高层 API 往往输出答案文本而不是 doc id。
    这里仅用于结果诊断，不把它当成严格 evidence recall 的唯一依据。
    """

    tokens = set(TOKEN_RE.findall(answer_or_context.lower()))
    scored: list[tuple[float, str]] = []
    for document in documents:
        doc_tokens = set(TOKEN_RE.findall(f"{document['title']} {document['text']}".lower()))
        if not doc_tokens:
            continue
        score = len(tokens & doc_tokens) / len(doc_tokens)
        if score > 0:
            scored.append((score, str(document["id"])))
    scored.sort(reverse=True)
    return [doc_id for _, doc_id in scored[:top_k]]
