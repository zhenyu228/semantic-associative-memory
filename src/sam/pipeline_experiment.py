from __future__ import annotations

import json
from pathlib import Path

from sam.answer_judge import AnswerJudge
from sam.embedding import EmbeddingProvider
from sam.evaluator import Evaluator
from sam.generation import (
    ContextAnswerGenerator,
    generate_answers_for_cases,
    write_generation_reports,
)
from sam.graph import GraphBuilder
from sam.llm import ChatClient
from sam.models import DatasetDocument, EvaluationQuery
from sam.store import MemoryStore


def run_retrieval_generation_pipeline(
    *,
    documents: list[DatasetDocument],
    queries: list[EvaluationQuery],
    output_dir: str | Path,
    embedding_provider: EmbeddingProvider,
    chat_client: ChatClient,
    answer_judge: AnswerJudge,
    retrieval_methods: list[str],
    generation_method: str,
    top_k: int = 4,
    seed_k: int = 1,
    hops: int = 2,
    max_context_chars: int = 6000,
) -> dict[str, object]:
    """运行检索、生成、答案判别和生成 bad case 的完整实验闭环。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(target / "pipeline.sqlite")
    try:
        graph_builder = GraphBuilder(store)
        evaluator = Evaluator(store, embedding_provider, graph_builder)
        evaluator.ingest(documents)
        retrieval_result = evaluator.evaluate(
            queries=queries,
            top_k=top_k,
            seed_k=seed_k,
            hops=hops,
            methods=retrieval_methods,
        )
        metrics_json_path, metrics_md_path = evaluator.write_reports(retrieval_result, target)
        generator = ContextAnswerGenerator(
            chat_client,
            max_context_chars=max_context_chars,
            answer_judge=answer_judge,
        )
        generated_answers = generate_answers_for_cases(
            retrieval_result.cases,
            generator,
            method=generation_method,
        )
        generated_json_path, generated_md_path = write_generation_reports(generated_answers, target)
    finally:
        store.close()

    generation_hit_count = sum(1 for answer in generated_answers if answer.answer_hit)
    summary = {
        "query_count": len(queries),
        "document_count": len(documents),
        "retrieval_methods": retrieval_methods,
        "generation_method": generation_method,
        "retrieval": {
            "metrics_json": str(metrics_json_path),
            "metrics_markdown": str(metrics_md_path),
            "method_metrics": retrieval_result.method_metrics,
        },
        "generation": {
            "generated_answers_json": str(generated_json_path),
            "generated_answers_markdown": str(generated_md_path),
            "answer_hit_count": generation_hit_count,
            "answer_hit_rate": generation_hit_count / len(generated_answers) if generated_answers else 0.0,
        },
        "outputs": {
            "cases_json": str(target / "cases.json"),
            "generation_bad_cases_json": str(target / "generation_bad_cases.json"),
            "generation_bad_cases_markdown": str(target / "generation_bad_cases.md"),
        },
    }
    (target / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (target / "pipeline_summary.md").write_text(
        _summary_to_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _summary_to_markdown(summary: dict[str, object]) -> str:
    retrieval = summary["retrieval"]
    generation = summary["generation"]
    assert isinstance(retrieval, dict)
    assert isinstance(generation, dict)
    method_metrics = retrieval["method_metrics"]
    assert isinstance(method_metrics, dict)
    lines = [
        "# SAM 端到端实验摘要",
        "",
        f"- 查询数量：{summary['query_count']}",
        f"- 文档数量：{summary['document_count']}",
        f"- 生成方法：{summary['generation_method']}",
        "",
        "## 检索指标",
        "",
        "| 方法 | 证据召回率 | 答案命中率 |",
        "| --- | ---: | ---: |",
    ]
    for method, metrics in method_metrics.items():
        if not isinstance(metrics, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(metrics.get("display_name", method)),
                    f"{float(metrics.get('evidence_recall', 0.0)):.3f}",
                    f"{float(metrics.get('answer_hit_rate', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 生成指标",
            "",
            f"- 答案命中数：{generation['answer_hit_count']}",
            f"- 答案命中率：{float(generation['answer_hit_rate']):.3f}",
            "",
        ]
    )
    return "\n".join(lines)
