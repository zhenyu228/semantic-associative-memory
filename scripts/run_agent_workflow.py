from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.agent_workflow import (  # noqa: E402
    MultiAgentResearchWorkflow,
    run_agent_workflow_for_cases,
    write_agent_workflow_reports,
)
from sam.agents import SharedMemoryCoordinator  # noqa: E402
from sam.embedding import create_embedding_provider  # noqa: E402
from sam.generation import ContextAnswerGenerator  # noqa: E402
from sam.llm import create_chat_client  # noqa: E402
from sam.store import MemoryStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行多智能体共享记忆协作实验")
    parser.add_argument("--cases-file", required=True, help="run 目录中的 cases.json")
    parser.add_argument("--method", default="sam_full", help="使用哪个检索方法的上下文")
    parser.add_argument("--chat-provider", default=None, help="heuristic 或 azure_openai")
    parser.add_argument("--embedding-provider", default=None, help="local、openai 或 azure_openai")
    parser.add_argument("--limit", type=int, default=None, help="最多运行多少条 case")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认写到 cases.json 所在目录")
    parser.add_argument("--db", default=None, help="共享记忆 SQLite 路径，默认写到输出目录 agent_workflow.sqlite")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_path = ROOT / args.cases_file if not Path(args.cases_file).is_absolute() else Path(args.cases_file)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    output_dir = (
        ROOT / args.output_dir
        if args.output_dir and not Path(args.output_dir).is_absolute()
        else Path(args.output_dir) if args.output_dir else cases_path.parent
    )
    db_path = (
        ROOT / args.db
        if args.db and not Path(args.db).is_absolute()
        else Path(args.db) if args.db else output_dir / "agent_workflow.sqlite"
    )
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
        results = run_agent_workflow_for_cases(cases, workflow, limit=args.limit)
        json_path, markdown_path = write_agent_workflow_reports(results, output_dir)
    finally:
        store.close()
    passed = sum(
        1 for result in results
        if result.get("verifier", {}).get("status") == "passed"
    )
    print(f"多智能体协作实验完成：{len(results)} 条")
    print(f"验证通过率：{passed / len(results):.3f}" if results else "验证通过率：N/A")
    print(f"JSON：{json_path}")
    print(f"Markdown：{markdown_path}")


if __name__ == "__main__":
    main()
