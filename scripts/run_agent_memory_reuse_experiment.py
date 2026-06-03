from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.agent_reuse_experiment import (  # noqa: E402
    load_agent_reuse_cases,
    run_agent_memory_reuse_probe,
    write_agent_memory_reuse_reports,
)
from sam.agent_workflow import MultiAgentResearchWorkflow  # noqa: E402
from sam.agents import SharedMemoryCoordinator  # noqa: E402
from sam.embedding import create_embedding_provider  # noqa: E402
from sam.generation import ContextAnswerGenerator  # noqa: E402
from sam.llm import create_chat_client  # noqa: E402
from sam.store import MemoryStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行多智能体共享记忆复用实验")
    parser.add_argument(
        "--cases-file",
        required=True,
        help="cases.json 或 memory_reuse_results.json",
    )
    parser.add_argument(
        "--method",
        default="sam_no_feedback",
        help="作为 SAM 侧输入的检索方法",
    )
    parser.add_argument(
        "--baseline-method",
        default="embedding_topk",
        help="用于比较支持证据增益的 baseline 方法",
    )
    parser.add_argument("--chat-provider", default=None, help="heuristic 或 azure_openai")
    parser.add_argument("--embedding-provider", default=None, help="local、openai、azure_openai 或 azure_openai_sdk")
    parser.add_argument("--limit", type=int, default=None, help="最多运行多少条 case")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认写到 cases 文件所在目录的 agent_memory_reuse 子目录",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="共享记忆 SQLite 路径，默认写到输出目录 agent_memory_reuse.sqlite",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_path = _resolve_path(args.cases_file)
    output_dir = (
        _resolve_path(args.output_dir)
        if args.output_dir
        else cases_path.parent / "agent_memory_reuse"
    )
    db_path = _resolve_path(args.db) if args.db else output_dir / "agent_memory_reuse.sqlite"
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_agent_reuse_cases(cases_path)
    embedding_provider = create_embedding_provider(args.embedding_provider)
    chat_client = create_chat_client(args.chat_provider)
    store = MemoryStore(db_path)
    try:
        coordinator = SharedMemoryCoordinator(store, embedding_provider)
        workflow = MultiAgentResearchWorkflow(
            coordinator=coordinator,
            generator=ContextAnswerGenerator(chat_client),
            method=args.method,
        )
        result = run_agent_memory_reuse_probe(
            cases,
            workflow=workflow,
            method=args.method,
            baseline_method=args.baseline_method,
            limit=args.limit,
        )
        json_path, markdown_path = write_agent_memory_reuse_reports(result, output_dir)
    finally:
        store.close()

    summary = result["summary"]
    print(f"多智能体共享记忆复用实验完成：{summary['query_count']} 条")
    print(f"支持证据增益总数：{summary['support_gain_total']}")
    print(f"多智能体复用链路成功率：{summary['multi_agent_reuse_success_rate']:.3f}")
    print(f"JSON：{json_path}")
    print(f"Markdown：{markdown_path}")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
