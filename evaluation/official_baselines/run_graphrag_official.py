from __future__ import annotations

import argparse
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


if __name__ == "__main__":
    main()
