from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.answer_judge import create_answer_judge  # noqa: E402
from sam.dataset_format import load_sam_dataset, summarize_sam_dataset  # noqa: E402
from sam.embedding import create_embedding_provider  # noqa: E402
from sam.experiment_audit import audit_run_directory, write_experiment_audit  # noqa: E402
from sam.llm import create_chat_client  # noqa: E402
from sam.pipeline_experiment import run_retrieval_generation_pipeline  # noqa: E402
from sam.query_planner import create_query_planner  # noqa: E402
from sam.relation_judge import create_relation_judge  # noqa: E402
from scripts.check_model_providers import build_provider_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行低额度 SAM 正式模型 smoke 实验")
    parser.add_argument("--dataset-file", default="data/processed/hotpotqa_sam_sample.json", help="SAM 统一数据格式文件")
    parser.add_argument("--output-root", default="outputs/runs", help="运行产物根目录")
    parser.add_argument("--run-name", default=None, help="本次运行名称")
    parser.add_argument("--limit", type=int, default=2, help="参与 smoke 的查询数量，建议 1-3")
    parser.add_argument("--embedding-provider", default=None, help="local、openai、azure_openai 或 azure_openai_sdk")
    parser.add_argument("--chat-provider", default=None, help="heuristic 或 azure_openai")
    parser.add_argument("--embedding-probe", default=None, help="可选：先发一条 embedding 连通性测试")
    parser.add_argument("--chat-probe", default=None, help="可选：先发一条 chat 连通性测试")
    parser.add_argument("--require", default="both", choices=["both", "embedding", "chat"], help="provider gate 要求")
    parser.add_argument("--answer-judge", default="rule", choices=["rule", "gpt54"], help="答案判别器")
    parser.add_argument("--query-planner", default="disabled", choices=["disabled", "heuristic", "gpt54"], help="查询规划器")
    parser.add_argument("--relation-judge", default="disabled", help="关系级建边判别器：disabled、gpt54 或 cached_gpt54")
    parser.add_argument("--retrieval-methods", default="embedding_topk,sam_full", help="逗号分隔的检索方法列表")
    parser.add_argument("--generation-method", default="sam_full", help="用于生成答案的检索方法")
    parser.add_argument("--top-k", type=int, default=2, help="最终返回文档数")
    parser.add_argument("--seed-k", type=int, default=1, help="SAM 种子节点数")
    parser.add_argument("--hops", type=int, default=1, help="图扩展跳数")
    parser.add_argument("--max-context-chars", type=int, default=3000, help="生成阶段每题最多上下文字符数")
    return parser.parse_args()


def run_provider_smoke_experiment(
    *,
    dataset_file: str | Path,
    output_dir: str | Path,
    limit: int,
    embedding_provider_name: str | None,
    chat_provider_name: str | None,
    answer_judge_name: str,
    query_planner_name: str,
    relation_judge_name: str,
    embedding_probe: str | None = None,
    chat_probe: str | None = None,
    required_providers: str = "both",
    retrieval_methods: list[str] | None = None,
    generation_method: str = "sam_full",
    top_k: int = 2,
    seed_k: int = 1,
    hops: int = 1,
    max_context_chars: int = 3000,
) -> dict[str, object]:
    """运行一次极小规模 provider gate + 端到端实验。

    该函数用于正式模型低额度验证。provider gate 未通过时直接失败，不进入实验。
    """

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    provider_status = build_provider_status(
        embedding_provider=embedding_provider_name,
        chat_provider=chat_provider_name,
        embedding_probe=embedding_probe,
        chat_probe=chat_probe,
        required_providers=required_providers,
    )
    (target / "provider_status.json").write_text(
        json.dumps(provider_status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not provider_status["ready"]:
        (target / "smoke_summary.md").write_text(
            _smoke_summary_markdown({"provider_status": provider_status, "pipeline": None}),
            encoding="utf-8",
        )
        raise RuntimeError("provider gate 未通过，已跳过端到端 smoke 实验")

    dataset_path = Path(dataset_file)
    documents, queries, _ = load_sam_dataset(dataset_path)
    selected_queries = queries[:limit]
    methods = retrieval_methods or ["embedding_topk", "sam_full"]
    (target / "dataset_summary.json").write_text(
        json.dumps(
            summarize_sam_dataset(dataset_path) | {"selected_query_count": len(selected_queries)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pipeline = run_retrieval_generation_pipeline(
        documents=documents,
        queries=selected_queries,
        output_dir=target,
        embedding_provider=create_embedding_provider(embedding_provider_name),
        chat_client=create_chat_client(chat_provider_name),
        answer_judge=create_answer_judge(answer_judge_name),
        retrieval_methods=methods,
        generation_method=generation_method,
        query_planner=create_query_planner(query_planner_name),
        relation_judge=create_relation_judge(relation_judge_name),
        top_k=top_k,
        seed_k=seed_k,
        hops=hops,
        max_context_chars=max_context_chars,
    )
    audit = audit_run_directory(
        target,
        primary_method=generation_method,
        baseline_method="embedding_topk",
    )
    write_experiment_audit(audit, target)
    summary = {
        "provider_status": provider_status,
        "pipeline": pipeline,
        "audit": audit,
    }
    (target / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (target / "smoke_summary.md").write_text(
        _smoke_summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_provider_smoke"
    run_dir = ROOT / args.output_root / run_name
    methods = [method.strip() for method in args.retrieval_methods.split(",") if method.strip()]
    (run_dir / "config.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args) | {"retrieval_methods": methods, "run_dir": str(run_dir)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = run_provider_smoke_experiment(
        dataset_file=ROOT / args.dataset_file,
        output_dir=run_dir,
        limit=args.limit,
        embedding_provider_name=args.embedding_provider,
        chat_provider_name=args.chat_provider,
        answer_judge_name=args.answer_judge,
        query_planner_name=args.query_planner,
        relation_judge_name=args.relation_judge,
        embedding_probe=args.embedding_probe,
        chat_probe=args.chat_probe,
        required_providers=args.require,
        retrieval_methods=methods,
        generation_method=args.generation_method,
        top_k=args.top_k,
        seed_k=args.seed_k,
        hops=args.hops,
        max_context_chars=args.max_context_chars,
    )
    pipeline = summary["pipeline"]
    assert isinstance(pipeline, dict)
    generation = pipeline["generation"]
    assert isinstance(generation, dict)
    print("SAM provider smoke 实验完成")
    print(f"运行目录：{run_dir}")
    print(f"查询数量：{pipeline['query_count']}")
    print(f"生成答案命中率：{float(generation['answer_hit_rate']):.3f}")


def _smoke_summary_markdown(summary: dict[str, object]) -> str:
    provider_status = summary["provider_status"]
    pipeline = summary.get("pipeline")
    assert isinstance(provider_status, dict)
    lines = [
        "# SAM Provider Smoke 实验摘要",
        "",
        f"- Provider gate：{'通过' if provider_status.get('ready') else '未通过'}",
        f"- Embedding provider：{provider_status.get('embedding', {}).get('provider')}",
        f"- Chat provider：{provider_status.get('chat', {}).get('provider')}",
    ]
    if not isinstance(pipeline, dict):
        lines.append("- Pipeline：未运行")
        return "\n".join(lines) + "\n"
    generation = pipeline["generation"]
    assert isinstance(generation, dict)
    lines.extend(
        [
            f"- 查询数量：{pipeline['query_count']}",
            f"- 生成方法：{pipeline['generation_method']}",
            f"- 答案命中率：{float(generation['answer_hit_rate']):.3f}",
            "",
            "## 输出文件",
            "",
            "- `provider_status.json`",
            "- `pipeline_summary.json`",
            "- `metrics.json`",
            "- `cases.json`",
            "- `generated_answers.json`",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
