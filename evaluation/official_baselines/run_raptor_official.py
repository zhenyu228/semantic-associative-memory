from __future__ import annotations

import argparse
import sys
from pathlib import Path

from official_eval_utils import (
    answer_hit,
    load_common_inputs,
    rough_retrieved_doc_ids_from_text,
    summarize_answer_metrics,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 RAPTOR 官方实现运行 QA baseline")
    parser.add_argument("--prepared-dir", required=True, help="export_sam_for_official.py 生成的 prepared 目录")
    parser.add_argument("--external-dir", default="evaluation/external/raptor", help="RAPTOR 官方仓库目录")
    parser.add_argument("--output", default=None, help="结果 JSON 路径")
    parser.add_argument("--limit", type=int, default=None, help="最多评测多少个问题")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    external_dir = ROOT / args.external_dir
    if str(external_dir) not in sys.path:
        sys.path.insert(0, str(external_dir))
    try:
        from raptor import RetrievalAugmentation
    except Exception as exc:
        raise RuntimeError(
            "无法导入 RAPTOR 官方实现。请先运行 fetch_official_repos.py，"
            "并按 evaluation/official_baselines/README.md 安装官方依赖。"
        ) from exc

    prepared_dir = ROOT / args.prepared_dir
    documents, queries = load_common_inputs(prepared_dir)
    if args.limit:
        queries = queries[: args.limit]
    corpus_path = prepared_dir / "raptor/corpus.txt"
    corpus = corpus_path.read_text(encoding="utf-8")

    retrieval_augmentation = RetrievalAugmentation()
    retrieval_augmentation.add_documents(corpus)

    results = []
    for query in queries:
        answer = retrieval_augmentation.answer_question(question=query["question"])
        retrieved_doc_ids = rough_retrieved_doc_ids_from_text(answer, documents)
        results.append(
            {
                "query_id": query["id"],
                "question": query["question"],
                "gold_answers": query["answers"],
                "official_answer": answer,
                "answer_hit": answer_hit(answer, query["answers"]),
                "retrieved_doc_ids_diagnostic": retrieved_doc_ids,
                "evidence_recall": None,
                "note": "RAPTOR 官方高层 QA API 返回答案文本；doc id 为文本反查诊断，不作为严格官方检索输出。",
            }
        )

    output = Path(args.output) if args.output else prepared_dir.parent / "results/raptor_official.json"
    write_json(
        output,
        {
            "method": "raptor_official",
            "official_repo": "https://github.com/parthsarthi03/raptor",
            "metrics": summarize_answer_metrics(results),
            "results": results,
        },
    )
    print(f"RAPTOR 官方评测结果：{output}")


if __name__ == "__main__":
    main()
