from __future__ import annotations

import argparse
import os
import shutil
import subprocess
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
    parser = argparse.ArgumentParser(description="使用 Microsoft GraphRAG 官方 CLI 运行 baseline")
    parser.add_argument("--prepared-dir", required=True, help="export_sam_for_official.py 生成的 prepared 目录")
    parser.add_argument("--work-dir", default=None, help="GraphRAG 官方工作目录")
    parser.add_argument("--cli", default="evaluation/.venvs/graphrag/bin/graphrag", help="GraphRAG 官方 CLI 命令")
    parser.add_argument("--query-method", default="local", choices=["local", "global", "drift"], help="GraphRAG 查询模式")
    parser.add_argument("--model-provider", default=None, help="模型 provider，默认读取 GRAPHRAG_MODEL_PROVIDER 或 openai")
    parser.add_argument("--api-base", default=None, help="公司 OpenAI-compatible base url，默认读取 GRAPHRAG_API_BASE 或 OPENAI_BASE_URL")
    parser.add_argument("--api-version", default=None, help="Azure API version，默认读取 GRAPHRAG_API_VERSION 或 GPT54_API_VERSION")
    parser.add_argument("--chat-model", default=None, help="chat/completion 模型名，默认读取 GRAPHRAG_CHAT_MODEL")
    parser.add_argument("--embedding-model", default=None, help="embedding 模型名，默认读取 GRAPHRAG_EMBEDDING_MODEL")
    parser.add_argument("--chat-deployment", default=None, help="Azure chat deployment，默认等于 chat-model")
    parser.add_argument("--embedding-deployment", default=None, help="Azure embedding deployment，默认等于 embedding-model")
    parser.add_argument("--limit", type=int, default=None, help="最多评测多少个问题")
    parser.add_argument("--skip-index", action="store_true", help="跳过 graphrag index，仅运行 query")
    parser.add_argument("--output", default=None, help="结果 JSON 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which(args.cli) is None:
        raise RuntimeError("找不到 GraphRAG 官方 CLI。请先安装官方包，例如 pip install graphrag。")

    prepared_dir = ROOT / args.prepared_dir
    documents, queries = load_common_inputs(prepared_dir)
    if args.limit:
        queries = queries[: args.limit]
    work_dir = Path(args.work_dir) if args.work_dir else prepared_dir.parent / "graphrag_official_work"
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for source in (prepared_dir / "graphrag/input").glob("*.txt"):
        shutil.copy2(source, input_dir / source.name)

    if not (work_dir / "settings.yaml").exists():
        subprocess.run([args.cli, "init", "--root", str(work_dir), "--force"], check=True)
    _configure_graphrag_settings(work_dir, args)
    if not args.skip_index:
        subprocess.run([args.cli, "index", "--root", str(work_dir)], check=True)

    results = []
    for query in queries:
        completed = subprocess.run(
            [
                args.cli,
                "query",
                "--root",
                str(work_dir),
                "--method",
                args.query_method,
                "--query",
                query["question"],
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        answer = completed.stdout.strip()
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
                "note": "GraphRAG CLI 输出答案文本；doc id 为文本反查诊断，不作为严格官方检索输出。",
            }
        )

    output = Path(args.output) if args.output else prepared_dir.parent / f"results/graphrag_{args.query_method}_official.json"
    write_json(
        output,
        {
            "method": f"graphrag_{args.query_method}_official",
            "official_repo": "https://github.com/microsoft/graphrag",
            "work_dir": str(work_dir),
            "metrics": summarize_answer_metrics(results),
            "results": results,
        },
    )
    print(f"GraphRAG 官方评测结果：{output}")


def _configure_graphrag_settings(work_dir: Path, args: argparse.Namespace) -> None:
    import yaml

    settings_path = work_dir / "settings.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    model_provider = args.model_provider or os.getenv("GRAPHRAG_MODEL_PROVIDER") or "openai"
    api_base = args.api_base or os.getenv("GRAPHRAG_API_BASE") or os.getenv("OPENAI_BASE_URL")
    api_version = args.api_version or os.getenv("GRAPHRAG_API_VERSION") or os.getenv("GPT54_API_VERSION")
    chat_model = args.chat_model or os.getenv("GRAPHRAG_CHAT_MODEL") or "gpt-4o-mini"
    embedding_model = args.embedding_model or os.getenv("GRAPHRAG_EMBEDDING_MODEL") or "text-embedding-3-small"
    chat_deployment = args.chat_deployment or os.getenv("GRAPHRAG_CHAT_DEPLOYMENT") or chat_model
    embedding_deployment = args.embedding_deployment or os.getenv("GRAPHRAG_EMBEDDING_DEPLOYMENT") or embedding_model
    api_key_value = os.getenv("GRAPHRAG_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GPT54_API_KEY") or "<API_KEY>"

    for model_config in settings.get("completion_models", {}).values():
        model_config["model_provider"] = model_provider
        model_config["model"] = chat_model
        model_config["api_key"] = "${GRAPHRAG_API_KEY}"
        if api_base:
            model_config["api_base"] = api_base
        if api_version:
            model_config["api_version"] = api_version
        if model_provider == "azure":
            model_config["azure_deployment_name"] = chat_deployment

    for model_config in settings.get("embedding_models", {}).values():
        model_config["model_provider"] = model_provider
        model_config["model"] = embedding_model
        model_config["api_key"] = "${GRAPHRAG_API_KEY}"
        if api_base:
            model_config["api_base"] = api_base
        if api_version:
            model_config["api_version"] = api_version
        if model_provider == "azure":
            model_config["azure_deployment_name"] = embedding_deployment

    settings_path.write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8")

    dotenv_path = work_dir / ".env"
    dotenv_lines = [f"GRAPHRAG_API_KEY={api_key_value}"]
    if api_base:
        dotenv_lines.append(f"GRAPHRAG_API_BASE={api_base}")
    if api_version:
        dotenv_lines.append(f"GRAPHRAG_API_VERSION={api_version}")
    dotenv_path.write_text("\n".join(dotenv_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
