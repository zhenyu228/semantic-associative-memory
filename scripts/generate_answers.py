from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.generation import (  # noqa: E402
    CaseAnalogyHintBuilder,
    ContextAnswerGenerator,
    compare_generation_variants,
    generate_answers_for_cases,
    write_generation_comparison_reports,
    write_generation_reports,
)
from sam.answer_judge import create_answer_judge  # noqa: E402
from sam.env import load_env_file  # noqa: E402
from sam.llm import create_chat_client  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 cases.json 生成最终答案并评测")
    parser.add_argument("--cases-file", required=True, help="run 目录中的 cases.json")
    parser.add_argument("--method", default="sam_full", help="用于生成答案的检索方法")
    parser.add_argument("--env-file", default=None, help="可选：加载本地 .env.local；文件已被 gitignore 忽略")
    parser.add_argument("--chat-provider", default=None, help="heuristic 或 azure_openai")
    parser.add_argument("--answer-judge", default="rule", choices=["rule", "gpt54"], help="答案命中判别器：rule 或 gpt54")
    parser.add_argument("--limit", type=int, default=None, help="最多生成多少条")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认写到 cases.json 所在目录")
    parser.add_argument("--max-context-chars", type=int, default=6000, help="每条样本最多使用的上下文字符数")
    parser.add_argument("--use-analogy-hints", action="store_true", help="从 cases.json 中检索相似历史案例并加入类比提示")
    parser.add_argument("--analogy-top-k", type=int, default=2, help="每条样本最多使用多少条类比提示")
    parser.add_argument("--compare-analogy", action="store_true", help="同时运行无类比提示和有类比提示两种生成，并输出对照报告")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env_file:
        load_env_file(ROOT / args.env_file)
    cases_path = ROOT / args.cases_file if not Path(args.cases_file).is_absolute() else Path(args.cases_file)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    output_dir = (
        ROOT / args.output_dir
        if args.output_dir and not Path(args.output_dir).is_absolute()
        else Path(args.output_dir) if args.output_dir else cases_path.parent
    )
    chat_client = create_chat_client(args.chat_provider)
    generator = ContextAnswerGenerator(
        chat_client,
        max_context_chars=args.max_context_chars,
        answer_judge=create_answer_judge(args.answer_judge),
    )
    selected_cases = cases[:args.limit] if args.limit is not None else cases
    if args.compare_analogy:
        comparison = compare_generation_variants(
            selected_cases,
            all_cases=cases,
            generator=generator,
            method=args.method,
            analogy_top_k=args.analogy_top_k,
        )
        json_path, markdown_path = write_generation_comparison_reports(comparison, output_dir)
        delta = comparison["delta"]
        print(f"对照完成：{comparison['query_count']} 条")
        print(f"答案命中率变化：{delta['answer_hit_rate']:.3f}")
        print(f"JSON：{json_path}")
        print(f"Markdown：{markdown_path}")
        return

    analogy_hint_builder = (
        CaseAnalogyHintBuilder(cases, method=args.method)
        if args.use_analogy_hints
        else None
    )
    answers = generate_answers_for_cases(
        cases,
        generator,
        method=args.method,
        limit=args.limit,
        analogy_hint_builder=analogy_hint_builder,
        analogy_top_k=args.analogy_top_k,
    )
    json_path, markdown_path = write_generation_reports(answers, output_dir)
    hit_count = sum(1 for answer in answers if answer.answer_hit)
    hit_rate = hit_count / len(answers) if answers else 0.0
    print(f"生成完成：{len(answers)} 条")
    print(f"答案命中率：{hit_rate:.3f}")
    print(f"JSON：{json_path}")
    print(f"Markdown：{markdown_path}")


if __name__ == "__main__":
    main()
