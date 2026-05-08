from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from official_eval_utils import (
    answer_hit,
    evidence_recall,
    load_common_inputs,
    summarize_answer_metrics,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 HippoRAG 官方实现运行检索/QA baseline")
    parser.add_argument("--prepared-dir", required=True, help="export_sam_for_official.py 生成的 prepared 目录")
    parser.add_argument("--external-dir", default="evaluation/external/hipporag/src", help="HippoRAG 官方源码 src 目录")
    parser.add_argument("--save-dir", default="evaluation/runs/hipporag_cache", help="HippoRAG 官方索引缓存目录")
    parser.add_argument("--llm-model-name", default="gpt-4o-mini", help="官方 HippoRAG 使用的 LLM 名称")
    parser.add_argument("--embedding-model-name", default="nvidia/NV-Embed-v2", help="官方 HippoRAG 使用的 embedding 名称")
    parser.add_argument("--llm-base-url", default=None, help="OpenAI 兼容 LLM base url")
    parser.add_argument("--embedding-base-url", default=None, help="OpenAI 兼容 embedding base url")
    parser.add_argument("--top-k", type=int, default=4, help="检索文档数")
    parser.add_argument("--limit", type=int, default=None, help="最多评测多少个问题")
    parser.add_argument("--output", default=None, help="结果 JSON 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    external_dir = ROOT / args.external_dir
    if str(external_dir) not in sys.path:
        sys.path.insert(0, str(external_dir))
    try:
        from hipporag import HippoRAG
    except Exception as exc:
        raise RuntimeError(
            "无法导入 HippoRAG 官方实现。请先运行 fetch_official_repos.py，"
            "并按官方 README 安装 hipporag 依赖。"
        ) from exc

    prepared_dir = ROOT / args.prepared_dir
    documents, queries = load_common_inputs(prepared_dir)
    if args.limit:
        queries = queries[: args.limit]

    hipporag_kwargs: dict[str, Any] = {
        "save_dir": str(ROOT / args.save_dir),
        "llm_model_name": args.llm_model_name,
        "embedding_model_name": args.embedding_model_name,
    }
    if args.llm_base_url:
        hipporag_kwargs["llm_base_url"] = args.llm_base_url
    if args.embedding_base_url:
        hipporag_kwargs["embedding_base_url"] = args.embedding_base_url

    hipporag = HippoRAG(**hipporag_kwargs)
    docs = [f"{document['title']}\n{document['text']}" for document in documents]
    doc_id_by_text = {docs[index]: str(documents[index]["id"]) for index in range(len(documents))}
    hipporag.index(docs=docs)

    questions = [query["question"] for query in queries]
    retrieval_results = hipporag.retrieve(queries=questions, num_to_retrieve=args.top_k)
    try:
        qa_results = hipporag.rag_qa(retrieval_results)
    except Exception:
        qa_results = [None for _ in queries]

    results = []
    for query, retrieved, qa_result in zip(queries, retrieval_results, qa_results, strict=False):
        retrieved_doc_ids = _doc_ids_from_hipporag_result(retrieved, doc_id_by_text)
        answer = _answer_from_qa_result(qa_result)
        results.append(
            {
                "query_id": query["id"],
                "question": query["question"],
                "gold_answers": query["answers"],
                "official_answer": answer,
                "answer_hit": answer_hit(answer, query["answers"]),
                "retrieved_doc_ids": retrieved_doc_ids,
                "evidence_recall": evidence_recall(retrieved_doc_ids, query["supporting_doc_ids"]),
                "raw_retrieval_result": retrieved,
                "raw_qa_result": qa_result,
            }
        )

    output = Path(args.output) if args.output else prepared_dir.parent / "results/hipporag_official.json"
    write_json(
        output,
        {
            "method": "hipporag_official",
            "official_repo": "https://github.com/OSU-NLP-Group/HippoRAG",
            "metrics": summarize_answer_metrics(results),
            "results": results,
        },
    )
    print(f"HippoRAG 官方评测结果：{output}")


def _doc_ids_from_hipporag_result(result: Any, doc_id_by_text: dict[str, str]) -> list[str]:
    doc_ids: list[str] = []
    items = result if isinstance(result, list) else [result]
    for item in items:
        if isinstance(item, str) and item in doc_id_by_text:
            doc_ids.append(doc_id_by_text[item])
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or item.get("doc") or "")
            if text in doc_id_by_text:
                doc_ids.append(doc_id_by_text[text])
            idx = item.get("idx")
            if idx is not None:
                doc_ids.append(str(idx))
    return list(dict.fromkeys(doc_ids))


def _answer_from_qa_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("answer") or result.get("response") or result)
    return str(result)


if __name__ == "__main__":
    main()
