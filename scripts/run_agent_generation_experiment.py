from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.agent_reuse_experiment import (  # noqa: E402
    compare_agent_generation_variants,
    load_agent_reuse_cases,
    write_agent_generation_comparison_reports,
)
from sam.agent_workflow import MultiAgentResearchWorkflow  # noqa: E402
from sam.agents import SharedMemoryCoordinator  # noqa: E402
from sam.embedding import create_embedding_provider  # noqa: E402
from sam.generation import ContextAnswerGenerator  # noqa: E402
from sam.llm import create_chat_client  # noqa: E402
from sam.store import MemoryStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行多智能体共享记忆生成对照实验")
    parser.add_argument("--cases-file", required=True, help="cases.json 或 memory_reuse_results.json")
    parser.add_argument("--all-cases-file", default=None, help="用于类比提示检索的完整 cases 文件")
    parser.add_argument("--method", default="sam_full", help="使用哪个检索方法的上下文")
    parser.add_argument("--chat-provider", default=None, help="heuristic 或 azure_openai")
    parser.add_argument("--embedding-provider", default=None, help="local、openai 或 azure_openai")
    parser.add_argument("--limit", type=int, default=None, help="最多运行多少条 case")
    parser.add_argument("--analogy-top-k", type=int, default=2, help="每条样本最多使用多少个类比案例")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认写到 cases 文件所在目录的 agent_generation_comparison 子目录",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="共享记忆 SQLite 路径，默认写到输出目录 agent_generation.sqlite",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_path = _resolve_path(args.cases_file)
    output_dir = (
        _resolve_path(args.output_dir)
        if args.output_dir
        else cases_path.parent / "agent_generation_comparison"
    )
    db_path = _resolve_path(args.db) if args.db else output_dir / "agent_generation.sqlite"
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_agent_reuse_cases(cases_path)
    all_cases = (
        load_agent_reuse_cases(_resolve_path(args.all_cases_file))
        if args.all_cases_file
        else cases
    )
    embedding_provider = create_embedding_provider(args.embedding_provider)
    chat_client = create_chat_client(args.chat_provider)
    store = MemoryStore(db_path)
    try:
        coordinator = SharedMemoryCoordinator(store, embedding_provider)
        generator = ContextAnswerGenerator(chat_client)
        workflow = MultiAgentResearchWorkflow(
            coordinator=coordinator,
            generator=generator,
            method=args.method,
        )
        result = compare_agent_generation_variants(
            cases,
            all_cases=all_cases,
            workflow=workflow,
            generator=generator,
            method=args.method,
            limit=args.limit,
            analogy_top_k=args.analogy_top_k,
        )
        json_path, markdown_path = write_agent_generation_comparison_reports(result, output_dir)
    finally:
        store.close()

    variants = result["variants"]
    print(f"多智能体生成对照实验完成：{result['query_count']} 条")
    print(f"baseline 答案命中率：{variants['baseline']['answer_hit_rate']:.3f}")
    print(f"shared_memory 答案命中率：{variants['shared_memory']['answer_hit_rate']:.3f}")
    print(
        "shared_memory_with_analogy 答案命中率："
        f"{variants['shared_memory_with_analogy']['answer_hit_rate']:.3f}"
    )
    print(f"JSON：{json_path}")
    print(f"Markdown：{markdown_path}")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
