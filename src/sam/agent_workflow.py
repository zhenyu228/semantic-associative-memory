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
            include_other_sessions=False,
            agent_id="writer",
        )
        answer = self.generator.generate_for_case(
            case,
            method=self.method,
            analogy_hints=[_memory_hint(node) for node in writer_memory],
        )
        writer_record = self.coordinator.write_handoff(
            source_agent_id="writer",
            target_agent_id="verifier",
            session_id=session_id,
            task_id=task_id,
            text=f"writer 生成答案：{answer.generated_answer}",
        )
        memory_node_ids.append(writer_record.node_id)
        steps.append(_step("writer", "读取共享记忆并生成答案", [node.id for node in writer_memory]))

        verifier_memory = self.coordinator.query_memory(
            f"验证答案 {answer.generated_answer}",
            layers={"session"},
            session_id=session_id,
            include_other_sessions=False,
            agent_id="verifier",
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
    return (
        f"retriever handoff：{method} 返回 {len(hits)} 条候选，"
        f"命中支持证据 {support_count} 条。候选标题：{'; '.join(titles[:5])}。"
    )


def _memory_hint(node: MemoryNode) -> str:
    return f"{node.metadata.get('agent_id', '')} 共享记忆：{node.text}"


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
        "target_agent_id": node.metadata.get("target_agent_id"),
        "layer": node.metadata.get("memory_layer"),
        "usage_count": node.usage_count,
        "text": node.text,
    }


def _step(agent_id: str, action: str, memory_node_ids: list[str]) -> dict[str, object]:
    return {
        "agent_id": agent_id,
        "action": action,
        "memory_node_ids": memory_node_ids,
    }
