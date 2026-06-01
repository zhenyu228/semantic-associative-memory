from __future__ import annotations

import json
from pathlib import Path

from sam.agent_workflow import MultiAgentResearchWorkflow


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
