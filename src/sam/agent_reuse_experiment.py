from __future__ import annotations

import json
from pathlib import Path

from sam.agent_workflow import MultiAgentResearchWorkflow
from sam.badcase import GenerationBadCaseAnalyzer, write_generation_bad_case_reports
from sam.generation import CaseAnalogyHintBuilder, ContextAnswerGenerator, GeneratedAnswer


def run_agent_memory_reuse_probe(
    cases: list[dict[str, object]],
    *,
    workflow: MultiAgentResearchWorkflow,
    method: str,
    baseline_method: str = "embedding_topk",
    limit: int | None = None,
) -> dict[str, object]:
    """评估多智能体共享记忆是否把 SAM 检索增益传递到后续角色。

    该实验不重新定义检索方法，而是读取已有 `cases.json` 或连续记忆复用
    实验中的 `probe_cases`。每条 case 会经过 planner、retriever、writer、
    verifier 四个角色，随后统计 writer 是否读取了 retriever handoff，
    verifier 是否读取了 writer handoff，以及 SAM 相对 baseline 的支持证据增益。
    """

    selected_cases = cases[:limit] if limit is not None else cases
    results: list[dict[str, object]] = []
    for case in selected_cases:
        workflow_result = workflow.run_case(case)
        method_support_hits = _support_hits(case, method)
        baseline_support_hits = _support_hits(case, baseline_method)
        support_gain = max(0, method_support_hits - baseline_support_hits)
        writer_used_retriever_handoff = _has_agent_handoff(
            workflow_result.get("writer_memory", []),
            source_agent_id="retriever",
            target_agent_id="writer",
        )
        verifier_used_writer_handoff = _has_agent_handoff(
            workflow_result.get("verifier_memory", []),
            source_agent_id="writer",
            target_agent_id="verifier",
        )
        final_answer = workflow_result.get("final_answer", {})
        answer_hit = (
            bool(final_answer.get("answer_hit"))
            if isinstance(final_answer, dict)
            else False
        )
        multi_agent_reuse_success = (
            support_gain > 0
            and writer_used_retriever_handoff
            and verifier_used_writer_handoff
        )
        results.append(
            {
                "query_id": workflow_result.get("query_id", case.get("query_id", "")),
                "question": workflow_result.get("question", case.get("question", "")),
                "method": method,
                "baseline_method": baseline_method,
                "method_support_hits": method_support_hits,
                "baseline_support_hits": baseline_support_hits,
                "support_gain": support_gain,
                "writer_used_retriever_handoff": writer_used_retriever_handoff,
                "verifier_used_writer_handoff": verifier_used_writer_handoff,
                "answer_hit": answer_hit,
                "multi_agent_reuse_success": multi_agent_reuse_success,
                "workflow": workflow_result,
            }
        )
    return {
        "summary": _summarize_agent_memory_reuse(results),
        "cases": results,
    }


def compare_agent_generation_variants(
    cases: list[dict[str, object]],
    *,
    workflow: MultiAgentResearchWorkflow,
    generator: ContextAnswerGenerator,
    method: str,
    all_cases: list[dict[str, object]] | None = None,
    limit: int | None = None,
    analogy_top_k: int = 2,
) -> dict[str, object]:
    """对比无共享记忆、共享记忆、共享记忆+类比提示三种生成设置。"""

    selected_cases = cases[:limit] if limit is not None else cases
    hint_builder = CaseAnalogyHintBuilder(all_cases or selected_cases, method=method)
    baseline_answers: list[GeneratedAnswer] = []
    shared_answers: list[dict[str, object]] = []
    shared_analogy_answers: list[GeneratedAnswer] = []
    case_deltas: list[dict[str, object]] = []
    for case in selected_cases:
        baseline = generator.generate_for_case(case, method=method)
        workflow_result = workflow.run_case(case)
        shared_answer = workflow_result.get("final_answer", {})
        shared_memory_hints = _shared_memory_hints(workflow_result)
        shared_memory_contexts = _shared_memory_contexts(workflow_result)
        analogy_hints = [
            *shared_memory_hints,
            *hint_builder.hints_for(case, top_k=analogy_top_k),
        ]
        with_analogy = generator.generate_for_case(
            case,
            method=method,
            analogy_hints=analogy_hints,
            supplemental_contexts=shared_memory_contexts,
        )
        baseline_answers.append(baseline)
        shared_answers.append(shared_answer if isinstance(shared_answer, dict) else {})
        shared_analogy_answers.append(with_analogy)
        case_deltas.append(
            _agent_generation_case_delta(
                case=case,
                baseline=baseline,
                shared_memory=shared_answers[-1],
                shared_memory_with_analogy=with_analogy,
            )
        )

    baseline_metrics = _answer_metrics([answer.to_dict() for answer in baseline_answers])
    shared_metrics = _answer_metrics(shared_answers)
    shared_analogy_metrics = _answer_metrics(
        [answer.to_dict() for answer in shared_analogy_answers]
    )
    return {
        "method": method,
        "query_count": len(selected_cases),
        "variants": {
            "baseline": baseline_metrics,
            "shared_memory": shared_metrics,
            "shared_memory_with_analogy": shared_analogy_metrics,
        },
        "delta": {
            "shared_memory_vs_baseline_answer_hits": (
                int(shared_metrics["answer_hit_count"])
                - int(baseline_metrics["answer_hit_count"])
            ),
            "shared_memory_with_analogy_vs_baseline_answer_hits": (
                int(shared_analogy_metrics["answer_hit_count"])
                - int(baseline_metrics["answer_hit_count"])
            ),
            "shared_memory_with_analogy_vs_shared_memory_answer_hits": (
                int(shared_analogy_metrics["answer_hit_count"])
                - int(shared_metrics["answer_hit_count"])
            ),
        },
        "case_deltas": case_deltas,
        "answers": {
            "baseline": [answer.to_dict() for answer in baseline_answers],
            "shared_memory": shared_answers,
            "shared_memory_with_analogy": [
                answer.to_dict() for answer in shared_analogy_answers
            ],
        },
    }


def write_agent_generation_comparison_reports(
    result: dict[str, object],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """写出多智能体生成对照实验结果。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "agent_generation_comparison.json"
    markdown_path = target / "agent_generation_comparison.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_agent_generation_markdown(result), encoding="utf-8")
    bad_cases = GenerationBadCaseAnalyzer().analyze(_flatten_variant_answers(result))
    write_generation_bad_case_reports(bad_cases, target / "generation_bad_cases")
    return json_path, markdown_path


def write_agent_memory_reuse_reports(
    result: dict[str, object],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """写出多智能体共享记忆复用实验结果。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "agent_memory_reuse_results.json"
    markdown_path = target / "agent_memory_reuse_results.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_agent_memory_reuse_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def _agent_generation_markdown(result: dict[str, object]) -> str:
    variants = result.get("variants", {})
    if not isinstance(variants, dict):
        variants = {}
    lines = [
        "# 多智能体共享记忆生成对照实验",
        "",
        f"- 方法：{result.get('method', '')}",
        f"- 查询数量：{result.get('query_count', 0)}",
        "",
        "| 变体 | 答案命中数 | 答案命中率 | 上下文含答案数 | 有证据但生成失败数 | 平均 prompt token 估计 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["baseline", "shared_memory", "shared_memory_with_analogy"]:
        metric = variants.get(name, {})
        if not isinstance(metric, dict):
            metric = {}
        lines.append(
            f"| {name} | "
            f"{metric.get('answer_hit_count', 0)} | "
            f"{float(metric.get('answer_hit_rate', 0.0)):.3f} | "
            f"{metric.get('context_answer_hit_count', 0)} | "
            f"{metric.get('generation_failure_with_context_count', 0)} | "
            f"{float(metric.get('average_prompt_tokens_estimate', 0.0)):.1f} |"
        )
    lines.extend(
        [
            "",
            "## 样本变化",
            "",
            "| Query | baseline | shared memory | shared memory + analogy | 状态 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for case in result.get("case_deltas", []):
        if not isinstance(case, dict):
            continue
        lines.append(
            f"| {case.get('query_id', '')} | "
            f"{'命中' if case.get('baseline_answer_hit') else '未命中'} | "
            f"{'命中' if case.get('shared_memory_answer_hit') else '未命中'} | "
            f"{'命中' if case.get('shared_memory_with_analogy_answer_hit') else '未命中'} | "
            f"{case.get('shared_memory_status', '')} |"
        )
    return "\n".join(lines)


def load_agent_reuse_cases(path: str | Path) -> list[dict[str, object]]:
    """读取普通 `cases.json` 或 `memory_reuse_results.json`。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("probe_cases"), list):
        return payload["probe_cases"]
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload["cases"]
    raise ValueError(f"无法从 {path} 读取 case 列表")


def _shared_memory_hints(workflow_result: dict[str, object]) -> list[str]:
    hints: list[str] = []
    for memory in workflow_result.get("writer_memory", []):
        if isinstance(memory, dict) and memory.get("text"):
            hints.append(f"{memory.get('agent_id', '')} 共享记忆：{memory.get('text')}")
    return hints


def _shared_memory_contexts(workflow_result: dict[str, object]) -> list[dict[str, object]]:
    contexts: list[dict[str, object]] = []
    for memory in workflow_result.get("writer_memory", []):
        if not isinstance(memory, dict) or not memory.get("text"):
            continue
        contexts.append(
            {
                "node_id": memory.get("node_id", ""),
                "title": f"共享记忆:{memory.get('agent_id', '')}:{memory.get('layer', '')}",
                "text": str(memory.get("text", "")),
                "reason": "multi_agent_shared_memory",
            }
        )
    return contexts


def _agent_generation_case_delta(
    *,
    case: dict[str, object],
    baseline: GeneratedAnswer,
    shared_memory: dict[str, object],
    shared_memory_with_analogy: GeneratedAnswer,
) -> dict[str, object]:
    baseline_hit = baseline.answer_hit
    shared_hit = bool(shared_memory.get("answer_hit"))
    analogy_hit = shared_memory_with_analogy.answer_hit
    return {
        "query_id": case.get("query_id", baseline.query_id),
        "baseline_answer_hit": baseline_hit,
        "shared_memory_answer_hit": shared_hit,
        "shared_memory_with_analogy_answer_hit": analogy_hit,
        "shared_memory_status": _delta_status(baseline_hit, shared_hit),
        "shared_memory_with_analogy_status": _delta_status(shared_hit, analogy_hit),
    }


def _answer_metrics(answers: list[dict[str, object]]) -> dict[str, object]:
    count = len(answers)
    answer_hit_count = sum(1 for answer in answers if answer.get("answer_hit"))
    context_answer_hit_count = sum(
        1
        for answer in answers
        if _nested_bool(answer, "metadata", "context_answer_judgment", "answer_hit")
    )
    ungrounded_answer_hit_count = sum(
        1
        for answer in answers
        if _nested_bool(answer, "metadata", "ungrounded_answer_hit")
    )
    generation_failure_with_context_count = sum(
        1
        for answer in answers
        if (
            not answer.get("answer_hit")
            and _nested_bool(answer, "metadata", "context_answer_judgment", "answer_hit")
            and not _nested_bool(answer, "metadata", "ungrounded_answer_hit")
        )
    )
    token_values = [
        int(answer.get("prompt_tokens_estimate", 0))
        for answer in answers
        if isinstance(answer.get("prompt_tokens_estimate", 0), int | float)
    ]
    supplemental_counts = [
        int(answer.get("metadata", {}).get("supplemental_context_count", 0))
        for answer in answers
        if isinstance(answer.get("metadata"), dict)
    ]
    return {
        "answer_count": count,
        "answer_hit_count": answer_hit_count,
        "answer_hit_rate": _rate(answer_hit_count, count),
        "context_answer_hit_count": context_answer_hit_count,
        "context_answer_hit_rate": _rate(context_answer_hit_count, count),
        "generation_failure_with_context_count": generation_failure_with_context_count,
        "generation_failure_with_context_rate": _rate(generation_failure_with_context_count, count),
        "ungrounded_answer_hit_count": ungrounded_answer_hit_count,
        "average_supplemental_context_count": (
            sum(supplemental_counts) / len(supplemental_counts)
            if supplemental_counts
            else 0.0
        ),
        "average_prompt_tokens_estimate": (
            sum(token_values) / len(token_values) if token_values else 0.0
        ),
    }


def _delta_status(before: bool, after: bool) -> str:
    if before == after:
        return "unchanged"
    return "improved" if after else "regressed"


def _summarize_agent_memory_reuse(
    results: list[dict[str, object]],
) -> dict[str, object]:
    query_count = len(results)
    support_gain_count = sum(1 for result in results if int(result.get("support_gain", 0)) > 0)
    writer_handoff_used_count = sum(
        1 for result in results if result.get("writer_used_retriever_handoff")
    )
    verifier_handoff_used_count = sum(
        1 for result in results if result.get("verifier_used_writer_handoff")
    )
    answer_hit_count = sum(1 for result in results if result.get("answer_hit"))
    multi_agent_reuse_success_count = sum(
        1 for result in results if result.get("multi_agent_reuse_success")
    )
    method_support_hits = sum(int(result.get("method_support_hits", 0)) for result in results)
    baseline_support_hits = sum(int(result.get("baseline_support_hits", 0)) for result in results)
    support_gain_total = sum(int(result.get("support_gain", 0)) for result in results)
    return {
        "query_count": query_count,
        "method_support_hits": method_support_hits,
        "baseline_support_hits": baseline_support_hits,
        "support_gain_total": support_gain_total,
        "support_gain_count": support_gain_count,
        "writer_handoff_used_count": writer_handoff_used_count,
        "verifier_handoff_used_count": verifier_handoff_used_count,
        "answer_hit_count": answer_hit_count,
        "multi_agent_reuse_success_count": multi_agent_reuse_success_count,
        "support_gain_rate": _rate(support_gain_count, query_count),
        "writer_handoff_used_rate": _rate(writer_handoff_used_count, query_count),
        "verifier_handoff_used_rate": _rate(verifier_handoff_used_count, query_count),
        "answer_hit_rate": _rate(answer_hit_count, query_count),
        "multi_agent_reuse_success_rate": _rate(multi_agent_reuse_success_count, query_count),
    }


def _agent_memory_reuse_markdown(result: dict[str, object]) -> str:
    summary = result.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    lines = [
        "# 多智能体共享记忆复用实验",
        "",
        f"- 查询数量：{summary.get('query_count', 0)}",
        f"- Baseline 支持证据命中数：{summary.get('baseline_support_hits', 0)}",
        f"- SAM 支持证据命中数：{summary.get('method_support_hits', 0)}",
        f"- 支持证据增益总数：{summary.get('support_gain_total', 0)}",
        f"- 存在支持证据增益的样本数：{summary.get('support_gain_count', 0)}",
        f"- writer 使用 retriever 共享记忆次数：{summary.get('writer_handoff_used_count', 0)}",
        f"- verifier 使用 writer 共享记忆次数：{summary.get('verifier_handoff_used_count', 0)}",
        f"- 多智能体复用链路成功数：{summary.get('multi_agent_reuse_success_count', 0)}",
        f"- 多智能体复用链路成功率：{float(summary.get('multi_agent_reuse_success_rate', 0.0)):.3f}",
        "",
        "## 样本明细",
        "",
        "| Query | Baseline 命中 | SAM 命中 | 增益 | writer handoff | verifier handoff |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for case in result.get("cases", []):
        if not isinstance(case, dict):
            continue
        lines.append(
            f"| {case.get('query_id', '')} | "
            f"{case.get('baseline_support_hits', 0)} | "
            f"{case.get('method_support_hits', 0)} | "
            f"{case.get('support_gain', 0)} | "
            f"{'是' if case.get('writer_used_retriever_handoff') else '否'} | "
            f"{'是' if case.get('verifier_used_writer_handoff') else '否'} |"
        )
    return "\n".join(lines)


def _support_hits(case: dict[str, object], method: str) -> int:
    support_hits = case.get("support_hits_by_method", {})
    if isinstance(support_hits, dict):
        value = support_hits.get(method, 0)
        return int(value) if isinstance(value, int | float) else 0
    legacy_key = "vector_support_hits" if method == "embedding_topk" else "associative_support_hits"
    value = case.get(legacy_key, 0)
    return int(value) if isinstance(value, int | float) else 0


def _has_agent_handoff(
    memories: object,
    *,
    source_agent_id: str,
    target_agent_id: str,
) -> bool:
    if not isinstance(memories, list):
        return False
    return any(
        isinstance(memory, dict)
        and memory.get("agent_id") == source_agent_id
        and memory.get("target_agent_id") == target_agent_id
        for memory in memories
    )


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _nested_bool(payload: dict[str, object], *keys: str) -> bool:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return False
        current = current.get(key)
    return bool(current)


def _flatten_variant_answers(result: dict[str, object]) -> list[dict[str, object]]:
    answers = result.get("answers", {})
    if not isinstance(answers, dict):
        return []
    flattened: list[dict[str, object]] = []
    for variant_name, variant_answers in answers.items():
        if not isinstance(variant_answers, list):
            continue
        for answer in variant_answers:
            if not isinstance(answer, dict):
                continue
            item = dict(answer)
            metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}
            metadata["agent_generation_variant"] = variant_name
            item["metadata"] = metadata
            item["method"] = f"{item.get('method', '')}:{variant_name}"
            flattened.append(item)
    return flattened
