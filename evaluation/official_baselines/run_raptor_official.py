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
    parser.add_argument("--qa-model", default=None, help="公司网关中的 chat/QA 模型名，默认读取 RAPTOR_QA_MODEL 或 gpt-3.5-turbo")
    parser.add_argument("--summary-model", default=None, help="公司网关中的摘要模型名，默认读取 RAPTOR_SUMMARY_MODEL 或 qa-model")
    parser.add_argument("--embedding-model", default=None, help="公司网关中的 embedding 模型名，默认读取 RAPTOR_EMBEDDING_MODEL 或 text-embedding-ada-002")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    external_dir = ROOT / args.external_dir
    if str(external_dir) not in sys.path:
        sys.path.insert(0, str(external_dir))
    try:
        from raptor import RetrievalAugmentation
        from raptor import RetrievalAugmentationConfig
        from raptor.EmbeddingModels import OpenAIEmbeddingModel
        from raptor.QAModels import GPT3TurboQAModel
        from raptor.SummarizationModels import GPT3TurboSummarizationModel
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

    qa_model_name = args.qa_model or _env("RAPTOR_QA_MODEL", "gpt-3.5-turbo")
    summary_model_name = args.summary_model or _env("RAPTOR_SUMMARY_MODEL", qa_model_name)
    embedding_model_name = args.embedding_model or _env("RAPTOR_EMBEDDING_MODEL", "text-embedding-ada-002")
    retrieval_augmentation = RetrievalAugmentation(
        config=RetrievalAugmentationConfig(
            qa_model=GPT3TurboQAModel(model=qa_model_name),
            summarization_model=GPT3TurboSummarizationModel(model=summary_model_name),
            embedding_model=OpenAIEmbeddingModel(model=embedding_model_name),
        )
    )
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
            "models": {
                "qa_model": qa_model_name,
                "summary_model": summary_model_name,
                "embedding_model": embedding_model_name,
            },
            "metrics": summarize_answer_metrics(results),
            "results": results,
        },
    )
    print(f"RAPTOR 官方评测结果：{output}")


def _env(name: str, default: str) -> str:
    import os

    return os.getenv(name) or default


if __name__ == "__main__":
    main()
