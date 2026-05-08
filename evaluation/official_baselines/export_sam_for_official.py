from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.dataset_format import load_sam_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 SAM 数据集导出为官方 baseline 可读取的评测格式")
    parser.add_argument("--dataset-file", required=True, help="SAM 统一数据格式文件")
    parser.add_argument("--dataset-name", required=True, help="导出数据集名称，例如 novelqa_demo")
    parser.add_argument("--output-root", default="evaluation/runs", help="评测输出根目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents, queries, payload = load_sam_dataset(ROOT / args.dataset_file)
    run_root = ROOT / args.output_root / args.dataset_name
    prepared = run_root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)

    common_dir = prepared / "common"
    hipporag_dir = prepared / "hipporag"
    graphrag_dir = prepared / "graphrag"
    raptor_dir = prepared / "raptor"
    for directory in [common_dir, hipporag_dir, graphrag_dir / "input", raptor_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    doc_by_id = {document.id: document for document in documents}
    common_docs = [
        {
            "id": document.id,
            "title": document.title,
            "text": document.text,
            "dataset": document.dataset,
            "source": document.source,
            "metadata": document.metadata,
        }
        for document in documents
    ]
    common_queries = [
        {
            "id": query.id,
            "dataset": query.dataset,
            "question": query.question,
            "answer": query.answer,
            "answers": _answers_for_query(query.answer, query.metadata),
            "supporting_doc_ids": query.supporting_doc_ids,
            "candidate_doc_ids": query.candidate_doc_ids,
            "metadata": query.metadata,
        }
        for query in queries
    ]
    _write_json(common_dir / "documents.json", common_docs)
    _write_json(common_dir / "queries.json", common_queries)
    _write_json(common_dir / "sam_dataset_payload.json", payload)

    hipporag_corpus = [
        {
            "title": document.title,
            "text": document.text,
            "idx": index,
            "sam_doc_id": document.id,
        }
        for index, document in enumerate(documents)
    ]
    hipporag_idx_by_doc_id = {item["sam_doc_id"]: item["idx"] for item in hipporag_corpus}
    hipporag_queries = []
    for query in queries:
        paragraphs = []
        for doc_id in query.candidate_doc_ids:
            document = doc_by_id.get(doc_id)
            if not document:
                continue
            paragraphs.append(
                {
                    "title": document.title,
                    "text": document.text,
                    "is_supporting": doc_id in set(query.supporting_doc_ids),
                    "idx": hipporag_idx_by_doc_id[doc_id],
                    "sam_doc_id": doc_id,
                }
            )
        hipporag_queries.append(
            {
                "id": f"{args.dataset_name}/{query.id}.json",
                "question": query.question,
                "answer": _answers_for_query(query.answer, query.metadata),
                "answerable": bool(query.answer),
                "paragraphs": paragraphs,
                "sam_query_id": query.id,
            }
        )
    _write_json(hipporag_dir / f"{args.dataset_name}_corpus.json", hipporag_corpus)
    _write_json(hipporag_dir / f"{args.dataset_name}.json", hipporag_queries)

    for document in documents:
        safe_name = _safe_filename(document.id)
        (graphrag_dir / "input" / f"{safe_name}.txt").write_text(
            f"# {document.title}\n\n{document.text}\n",
            encoding="utf-8",
        )
    _write_json(graphrag_dir / "questions.json", common_queries)

    raptor_corpus = "\n\n".join(
        f"[{document.id}] {document.title}\n{document.text}"
        for document in documents
    )
    (raptor_dir / "corpus.txt").write_text(raptor_corpus, encoding="utf-8")
    _write_json(raptor_dir / "queries.json", common_queries)
    _write_json(
        prepared / "manifest.json",
        {
            "dataset_name": args.dataset_name,
            "source_dataset_file": args.dataset_file,
            "document_count": len(documents),
            "query_count": len(queries),
            "paths": {
                "common_documents": str(common_dir / "documents.json"),
                "common_queries": str(common_dir / "queries.json"),
                "hipporag_corpus": str(hipporag_dir / f"{args.dataset_name}_corpus.json"),
                "hipporag_queries": str(hipporag_dir / f"{args.dataset_name}.json"),
                "graphrag_input": str(graphrag_dir / "input"),
                "raptor_corpus": str(raptor_dir / "corpus.txt"),
            },
        },
    )
    print(f"官方 baseline 数据已导出：{prepared}")


def _answers_for_query(answer: str, metadata: dict[str, object]) -> list[str]:
    answers: list[str] = []
    if answer:
        answers.append(answer)
    options = metadata.get("options")
    gold = metadata.get("gold")
    if isinstance(options, dict) and gold in options:
        option_answer = str(options[gold])
        if option_answer not in answers:
            answers.append(option_answer)
    return answers


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


if __name__ == "__main__":
    main()
