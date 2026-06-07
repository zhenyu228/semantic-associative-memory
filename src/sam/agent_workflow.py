from __future__ import annotations

import json
from pathlib import Path

from sam.agents import SharedMemoryCoordinator
from sam.generation import ContextAnswerGenerator, GeneratedAnswer
from sam.models import MemoryNode


class MultiAgentResearchWorkflow:
    """多智能体共享记忆协作流程。

    当前版本聚焦开题报告中的协作记忆主线：规划、检索、写作、验证四个
    角色通过同一个 MemoryStore 传递中间结论，形成可检查的共享记忆轨迹。
    """

    def __init__(
        self,
        *,
        coordinator: SharedMemoryCoordinator,
        generator: ContextAnswerGenerator,
        method: str,
    ) -> None:
        self.coordinator = coordinator
        self.generator = generator
        self.method = method

    def run_case(self, case: dict[str, object]) -> dict[str, object]:
        query_id = str(case.get("query_id", ""))
        question = str(case.get("question", ""))
        task_id = f"agent_workflow:{query_id}"
        session_id = query_id
        steps: list[dict[str, object]] = []
        memory_node_ids: list[str] = []

        planner_record = self.coordinator.write_memory(
            agent_id="planner",
            layer="global_insight",
            session_id=session_id,
            text=(
                f"任务 {query_id} 需要先确认检索证据，再由写作智能体基于共享记忆组织答案。"
                f"问题：{question}"
            ),
            metadata={"task_id": task_id, "workflow_role": "planning"},
        )
        memory_node_ids.append(planner_record.node_id)
        steps.append(_step("planner", "制定证据组织计划", [planner_record.node_id]))

        hits = _case_hits(case, self.method)
        retriever_text = _retriever_handoff_text(case, self.method, hits)
        retriever_record = self.coordinator.write_handoff(
            source_agent_id="retriever",
            target_agent_id="writer",
            session_id=session_id,
            task_id=task_id,
            text=retriever_text,
        )
        memory_node_ids.append(retriever_record.node_id)
        steps.append(_step("retriever", "向 writer 交接检索证据和路径", [retriever_record.node_id]))

        writer_memory = self.coordinator.query_memory(
            question,
            layers={"global_insight", "session"},
            session_id=session_id,
            task_id=task_id,
            include_other_sessions=False,
            agent_id="writer",
            latest_version_only=True,
        )
        answer = self.generator.generate_for_case(
            case,
            method=self.method,
            analogy_hints=[_memory_hint(node) for node in writer_memory],
            supplemental_contexts=[_memory_context(node) for node in writer_memory],
        )
        writer_record = self.coordinator.write_handoff(
            source_agent_id="writer",
            target_agent_id="verifier",
            session_id=session_id,
            task_id=task_id,
            text=f"writer 生成答案：{answer.generated_answer}",
            confidence=0.9 if answer.answer_hit else 0.58,
        )
        memory_node_ids.append(writer_record.node_id)
        steps.append(_step("writer", "读取共享记忆并生成答案", [node.id for node in writer_memory]))

        verifier_memory = self.coordinator.query_memory(
            f"验证答案 {answer.generated_answer}",
            layers={"session"},
            session_id=session_id,
            task_id=task_id,
            include_other_sessions=False,
            agent_id="verifier",
            latest_version_only=True,
        )
        verifier = _verify_answer(answer, verifier_memory)
        verifier_record = self.coordinator.write_memory(
            agent_id="verifier",
            layer="session",
            session_id=session_id,
            text=f"验证结果：{verifier['status']}；命中答案={verifier['answer_hit']}",
            tags=["verification"],
            metadata={
                "task_id": task_id,
                "workflow_role": "verification",
                "answer_hit": verifier["answer_hit"],
            },
        )
        memory_node_ids.append(verifier_record.node_id)
        steps.append(_step("verifier", "验证答案是否覆盖标准答案", [node.id for node in verifier_memory]))
        conflict_resolution_node_ids: list[str] = []
        if not answer.answer_hit:
            conflict_record = self.coordinator.resolve_conflict(
                resolver_agent_id="verifier",
                session_id=session_id,
                task_id=task_id,
                topic="answer_generation",
                candidate_node_ids=[retriever_record.node_id, writer_record.node_id],
            )
            memory_node_ids.append(conflict_record.node_id)
            conflict_resolution_node_ids.append(conflict_record.node_id)
            steps.append(_step("verifier", "裁决检索交接与写作答案的冲突", [conflict_record.node_id]))
        collaboration_metrics = self.coordinator.collaboration_metrics(
            session_id=session_id,
            task_id=task_id,
        )

        return {
            "query_id": query_id,
            "question": question,
            "method": self.method,
            "agent_steps": steps,
            "shared_memory_node_ids": memory_node_ids,
            "writer_memory": [_serialize_memory(node) for node in writer_memory],
            "verifier_memory": [_serialize_memory(node) for node in verifier_memory],
            "final_answer": answer.to_dict(),
            "verifier": verifier,
            "conflict_resolution_node_ids": conflict_resolution_node_ids,
            "collaboration_metrics": collaboration_metrics,
        }


def run_agent_workflow_for_cases(
    cases: list[dict[str, object]],
    workflow: MultiAgentResearchWorkflow,
    *,
    limit: int | None = None,
) -> list[dict[str, object]]:
    selected_cases = cases[:limit] if limit is not None else cases
    return [workflow.run_case(case) for case in selected_cases]


def write_agent_workflow_reports(
    results: list[dict[str, object]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "agent_workflow.json"
    markdown_path = target / "agent_workflow.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(
        1
        for result in results
        if isinstance(result.get("verifier"), dict)
        and result["verifier"].get("status") == "passed"
    )
    lines = [
        "# 多智能体共享记忆协作实验",
        "",
        f"- 样本数：{len(results)}",
        f"- 验证通过数：{passed}",
        f"- 验证通过率：{passed / len(results):.3f}" if results else "- 验证通过率：N/A",
        "",
        "| Query | 最终状态 | 共享记忆数 | Handoff 数 | 冲突裁决数 | 最大版本 | 角色步骤 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        steps = result.get("agent_steps", [])
        step_text = " -> ".join(
            str(step.get("agent_id", ""))
            for step in steps
            if isinstance(step, dict)
        )
        verifier = result.get("verifier", {})
        status = verifier.get("status", "") if isinstance(verifier, dict) else ""
        metrics = result.get("collaboration_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        lines.append(
            f"| {result.get('query_id', '')} | {status} | "
            f"{len(result.get('shared_memory_node_ids', []))} | "
            f"{metrics.get('handoff_count', 0)} | "
            f"{metrics.get('conflict_resolution_count', 0)} | "
            f"{metrics.get('max_memory_version', 0)} | "
            f"{step_text} |"
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    write_agent_workflow_audit(audit_agent_workflow_results(results), target)
    return json_path, markdown_path


def audit_agent_workflow_results(results: list[dict[str, object]]) -> dict[str, object]:
    """审计多智能体 workflow 中的共享记忆使用和污染风险。"""

    cases: list[dict[str, object]] = []
    passed_count = 0
    total_handoff_count = 0
    total_conflict_resolution_count = 0
    total_memory_count = 0
    rejected_memory_used_count = 0
    contaminated_case_count = 0

    for result in results:
        verifier = result.get("verifier", {})
        status = verifier.get("status") if isinstance(verifier, dict) else None
        if status == "passed":
            passed_count += 1
        metrics = result.get("collaboration_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        memory_count = int(metrics.get("memory_count", len(result.get("shared_memory_node_ids", [])) or 0))
        handoff_count = int(metrics.get("handoff_count", 0))
        conflict_resolution_count = int(metrics.get("conflict_resolution_count", 0))
        total_memory_count += memory_count
        total_handoff_count += handoff_count
        total_conflict_resolution_count += conflict_resolution_count

        writer_memory = _memory_list(result.get("writer_memory"))
        verifier_memory = _memory_list(result.get("verifier_memory"))
        rejected_ids = [
            str(memory.get("node_id"))
            for memory in [*writer_memory, *verifier_memory]
            if memory.get("conflict_status") == "rejected" and memory.get("node_id")
        ]
        selected_ids = [
            str(memory.get("node_id"))
            for memory in [*writer_memory, *verifier_memory]
            if memory.get("conflict_status") == "selected" and memory.get("node_id")
        ]
        rejected_memory_used_count += len(rejected_ids)
        contaminated = bool(rejected_ids)
        if contaminated:
            contaminated_case_count += 1
        cases.append(
            {
                "query_id": result.get("query_id", ""),
                "status": status or "",
                "memory_count": memory_count,
                "handoff_count": handoff_count,
                "conflict_resolution_count": conflict_resolution_count,
                "max_memory_version": int(metrics.get("max_memory_version", 0)),
                "writer_memory_count": len(writer_memory),
                "verifier_memory_count": len(verifier_memory),
                "selected_memory_node_ids": selected_ids,
                "rejected_memory_node_ids": rejected_ids,
                "contaminated_by_rejected_memory": contaminated,
            }
        )

    case_count = len(results)
    return {
        "summary": {
            "case_count": case_count,
            "passed_count": passed_count,
            "failed_count": case_count - passed_count,
            "pass_rate": passed_count / case_count if case_count else 0.0,
            "memory_count": total_memory_count,
            "average_memory_count": total_memory_count / case_count if case_count else 0.0,
            "handoff_count": total_handoff_count,
            "conflict_resolution_count": total_conflict_resolution_count,
            "rejected_memory_used_count": rejected_memory_used_count,
            "contaminated_case_count": contaminated_case_count,
            "contaminated_case_rate": contaminated_case_count / case_count if case_count else 0.0,
        },
        "cases": cases,
    }


def write_agent_workflow_audit(
    audit: dict[str, object],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """写出多智能体共享记忆审计 JSON 和 Markdown。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "agent_workflow_audit.json"
    markdown_path = target / "agent_workflow_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = audit.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    lines = [
        "# 多智能体共享记忆审计",
        "",
        f"- 样本数：{summary.get('case_count', 0)}",
        f"- 验证通过数：{summary.get('passed_count', 0)}",
        f"- Handoff 总数：{summary.get('handoff_count', 0)}",
        f"- 冲突裁决总数：{summary.get('conflict_resolution_count', 0)}",
        f"- 共享记忆污染案例数：{summary.get('contaminated_case_count', 0)}",
        f"- 被 rejected 记忆污染次数：{summary.get('rejected_memory_used_count', 0)}",
        "",
        "| Query | 状态 | 共享记忆数 | Handoff | 冲突裁决 | Rejected 记忆数 | 污染 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    cases = audit.get("cases", [])
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict):
                continue
            rejected_ids = case.get("rejected_memory_node_ids", [])
            rejected_count = len(rejected_ids) if isinstance(rejected_ids, list) else 0
            lines.append(
                f"| {case.get('query_id', '')} | {case.get('status', '')} | "
                f"{case.get('memory_count', 0)} | {case.get('handoff_count', 0)} | "
                f"{case.get('conflict_resolution_count', 0)} | {rejected_count} | "
                f"{'是' if case.get('contaminated_by_rejected_memory') else '否'} |"
            )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def _case_hits(case: dict[str, object], method: str) -> list[dict[str, object]]:
    methods = case.get("methods", {})
    if not isinstance(methods, dict):
        return []
    hits = methods.get(method, [])
    return hits if isinstance(hits, list) else []


def _retriever_handoff_text(
    case: dict[str, object],
    method: str,
    hits: list[dict[str, object]],
) -> str:
    titles = [
        str(hit.get("title") or hit.get("node_id") or "")
        for hit in hits
        if isinstance(hit, dict)
    ]
    support_hits = case.get("support_hits_by_method", {})
    support_count = (
        support_hits.get(method, 0)
        if isinstance(support_hits, dict)
        else 0
    )
    evidence_snippets = []
    for index, hit in enumerate(hits[:3], start=1):
        if not isinstance(hit, dict):
            continue
        title = str(hit.get("title") or hit.get("node_id") or f"candidate_{index}")
        text = _compact_text(str(hit.get("text") or ""), limit=260)
        reason = _compact_text(str(hit.get("reason") or ""), limit=120)
        evidence_snippets.append(
            f"[{index}] {title}：{text}。依据：{reason}"
        )
    evidence_text = " ".join(evidence_snippets)
    return (
        f"retriever handoff：{method} 返回 {len(hits)} 条候选，"
        f"命中支持证据 {support_count} 条。候选标题：{'; '.join(titles[:5])}。"
        f"关键证据片段：{evidence_text}"
    )


def _memory_hint(node: MemoryNode) -> str:
    return f"{node.metadata.get('agent_id', '')} 共享记忆：{node.text}"


def _memory_context(node: MemoryNode) -> dict[str, object]:
    return {
        "node_id": node.id,
        "title": f"共享记忆:{node.metadata.get('agent_id', '')}:{node.metadata.get('memory_layer', '')}",
        "text": node.text,
        "reason": "multi_agent_shared_memory",
    }


def _compact_text(text: str, *, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def _verify_answer(
    answer: GeneratedAnswer,
    verifier_memory: list[MemoryNode],
) -> dict[str, object]:
    return {
        "status": "passed" if answer.answer_hit else "failed",
        "answer_hit": answer.answer_hit,
        "verifier_memory_count": len(verifier_memory),
    }


def _serialize_memory(node: MemoryNode) -> dict[str, object]:
    return {
        "node_id": node.id,
        "agent_id": node.metadata.get("agent_id"),
        "source_agent_id": node.metadata.get("source_agent_id"),
        "target_agent_id": node.metadata.get("target_agent_id"),
        "layer": node.metadata.get("memory_layer"),
        "task_id": node.metadata.get("task_id"),
        "memory_version": node.metadata.get("memory_version"),
        "conflict_status": node.metadata.get("conflict_status"),
        "usage_count": node.usage_count,
        "text": node.text,
    }


def _memory_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _step(agent_id: str, action: str, memory_node_ids: list[str]) -> dict[str, object]:
    return {
        "agent_id": agent_id,
        "action": action,
        "memory_node_ids": memory_node_ids,
    }
