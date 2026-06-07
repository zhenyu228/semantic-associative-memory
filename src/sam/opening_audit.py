from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvidenceSpec:
    """开题模块进度审计中的一项证据。"""

    label: str
    path: str
    kind: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    """开题计划中的一个模块。"""

    module_id: str
    title: str
    opening_requirement: str
    target_progress: int
    code_evidence: list[EvidenceSpec] = field(default_factory=list)
    experiment_evidence: list[EvidenceSpec] = field(default_factory=list)
    remaining_work: list[str] = field(default_factory=list)


OPENING_MODULE_SPECS: list[ModuleSpec] = [
    ModuleSpec(
        module_id="dynamic_graph_memory",
        title="知识提取与动态知识图谱构建",
        opening_requirement="抽取关键信息单元及语义关系，将知识表示为带属性的记忆节点，并支持图谱动态生长、更新和记忆重构。",
        target_progress=74,
        code_evidence=[
            EvidenceSpec("MemoryNode / MemoryEdge 数据结构", "src/sam/models.py", "code"),
            EvidenceSpec("SQLite 记忆存储与事件表", "src/sam/store.py", "code"),
            EvidenceSpec("按需建图与边质量控制", "src/sam/graph.py", "code"),
            EvidenceSpec("图边质量审计", "src/sam/edge_audit.py", "code"),
            EvidenceSpec("反馈更新", "src/sam/feedback.py", "code"),
            EvidenceSpec("记忆巩固", "src/sam/consolidation.py", "code"),
            EvidenceSpec("GPT-5.4 关系判别接口", "src/sam/relation_judge.py", "code"),
        ],
        experiment_evidence=[
            EvidenceSpec("记忆事件 smoke", "outputs/runs/memory_events_30_smoke/metrics.json", "experiment"),
            EvidenceSpec("记忆巩固实验", "outputs/runs/memory_consolidation_hotpotqa30_v2/metrics.json", "experiment"),
            EvidenceSpec("弱关系惩罚实验", "outputs/runs/weak_relation_penalty_hotpotqa30/metrics.json", "experiment"),
            EvidenceSpec("图边质量审计 smoke", "outputs/runs/weak_relation_penalty_hotpotqa30/edge_quality_audit.json", "experiment"),
            EvidenceSpec("GPT-5.4 关系判别 30 条对照", "outputs/runs/relation_compare_risky_q30_budget20_fixed/metrics.json", "experiment"),
            EvidenceSpec("GPT-5.4 关系判别使用统计", "outputs/runs/relation_compare_risky_q30_budget20_fixed/relation_judge_usage.json", "experiment"),
        ],
        remaining_work=[
            "GPT-5.4 RelationJudge 已完成 30 条对照，但尚未形成 300 条高预算正式实验。",
            "图谱边权仍以经验公式为主，缺少学习式或系统化参数搜索。",
            "记忆重构需要更多跨任务连续验证。",
        ],
    ),
    ModuleSpec(
        module_id="associative_retrieval",
        title="语义激活与联想检索机制",
        opening_requirement="先用语义相似度锁定候选，再沿知识图谱关联路径扩展邻近记忆，形成与当前问题相关的记忆子图。",
        target_progress=78,
        code_evidence=[
            EvidenceSpec("两阶段检索与消融模式", "src/sam/retriever.py", "code"),
            EvidenceSpec("路径重排", "src/sam/reranker.py", "code"),
            EvidenceSpec("审计驱动关系噪声惩罚", "src/sam/reranker.py", "code"),
            EvidenceSpec("查询规划", "src/sam/query_planner.py", "code"),
            EvidenceSpec("评测器", "src/sam/evaluator.py", "code"),
            EvidenceSpec("主实验入口", "scripts/run_demo.py", "code"),
        ],
        experiment_evidence=[
            EvidenceSpec("HotpotQA 300 条候选集隔离实验", "outputs/runs/lexical_isolated_hotpotqa300/metrics.json", "experiment"),
            EvidenceSpec("HotpotQA 300 条消融", "outputs/runs/fair_ablation_hotpotqa_300/ablation_metrics.json", "experiment"),
            EvidenceSpec("反馈消融 300 条", "outputs/runs/feedback_ablation_hotpotqa_300_isolated/ablation_metrics.json", "experiment"),
            EvidenceSpec("PathReranker 300 条 profile 对比", "outputs/runs/reranker_profile_hotpotqa300_noise_penalty/reranker_profile_comparison.json", "experiment"),
            EvidenceSpec("Edge-audit 惩罚 30 条 smoke", "outputs/runs/edge_audit_penalty_hotpotqa30/metrics.json", "experiment"),
        ],
        remaining_work=[
            "正式 embedding 尚未重跑 HotpotQA 300 条和 NovelQA。",
            "多路径与记忆状态需要在连续任务中继续拉开贡献差异。",
            "仍需进一步降低图噪声和缺失支持证据问题。",
        ],
    ),
    ModuleSpec(
        module_id="analogy_reasoning",
        title="类比推理触发与应用",
        opening_requirement="在新问题激活子图与历史问题-解答链条结构相似时触发类比，检索类似案例并向 LLM 提供提示。",
        target_progress=58,
        code_evidence=[
            EvidenceSpec("类比检索引擎", "src/sam/analogy.py", "code"),
            EvidenceSpec("类比复用实验逻辑", "src/sam/analogy_experiment.py", "code"),
            EvidenceSpec("类比提示注入", "src/sam/generation.py", "code"),
            EvidenceSpec("类比支持证据注入检索排序", "src/sam/retriever.py", "code"),
            EvidenceSpec("类比复用脚本", "scripts/run_analogy_reuse_experiment.py", "code"),
        ],
        experiment_evidence=[
            EvidenceSpec("类比复用 30 条", "outputs/runs/analogy_reuse_hotpotqa30/analogy_reuse_results.json", "experiment"),
            EvidenceSpec("类比生成 smoke", "outputs/runs/analogy_generation_smoke/metrics.json", "experiment"),
            EvidenceSpec("类比检索排序 30 条 smoke", "outputs/runs/analogy_retrieval_smoke/metrics.json", "experiment"),
            EvidenceSpec("结构性巩固类比复用 30 条", "outputs/runs/analogy_structural_consolidation_hotpotqa30/analogy_reuse_results.json", "experiment"),
        ],
        remaining_work=[
            "类比提示对最终答案质量的提升尚未用 GPT-5.4 正式验证。",
            "类比支持证据注入检索排序仍需正式规模实验验证。",
            "结构路径匹配已经进入实验，但仍需要在未知来源案例和真实多轮任务中验证泛化能力。",
            "结构性巩固已覆盖来源案例，但仍有样本缺少真实支持证据重叠，需要继续改进检索和建边质量。",
        ],
    ),
    ModuleSpec(
        module_id="multi_agent_shared_memory",
        title="多智能体语义记忆协调机制",
        opening_requirement="构建全局洞察层、会话层、交互细节层，支持多智能体共享中间结果和经验以重建推理链。",
        target_progress=52,
        code_evidence=[
            EvidenceSpec("共享记忆协调器", "src/sam/agents.py", "code"),
            EvidenceSpec("共享记忆冲突裁决与版本指标", "src/sam/agents.py", "code"),
            EvidenceSpec("多智能体研究流程", "src/sam/agent_workflow.py", "code"),
            EvidenceSpec("多智能体复用实验", "src/sam/agent_reuse_experiment.py", "code"),
            EvidenceSpec("多智能体 workflow 脚本", "scripts/run_agent_workflow.py", "code"),
            EvidenceSpec("共享记忆复用脚本", "scripts/run_agent_memory_reuse_experiment.py", "code"),
        ],
        experiment_evidence=[
            EvidenceSpec("多智能体共享记忆复用", "outputs/runs/agent_memory_reuse_hotpotqa30/agent_memory_reuse_results.json", "experiment"),
            EvidenceSpec("多智能体生成对照 smoke", "outputs/runs/agent_generation_hotpotqa30_smoke/agent_generation_comparison.json", "experiment"),
            EvidenceSpec("多智能体 workflow 自动冲突裁决 smoke", "outputs/runs/agent_workflow_conflict_smoke/agent_workflow.json", "experiment"),
        ],
        remaining_work=[
            "当前多智能体实验仍偏受控流程，不是完整 Deep Research 任务。",
            "workflow 已能在答案验证失败时自动触发冲突裁决，但仍需设计更真实的多角色分歧任务集。",
            "需要用 GPT-5.4 比较共享记忆与类比提示对最终答案质量的影响。",
        ],
    ),
    ModuleSpec(
        module_id="evaluation_and_generation",
        title="评测体系与检索-生成闭环",
        opening_requirement="设计正式实验和评测体系，覆盖跨文档语义整合、推理链重建、多智能体协作和生成结果反馈。",
        target_progress=60,
        code_evidence=[
            EvidenceSpec("端到端实验入口", "scripts/run_end_to_end_experiment.py", "code"),
            EvidenceSpec("答案判别", "src/sam/answer_judge.py", "code"),
            EvidenceSpec("Bad Case 分析", "src/sam/badcase.py", "code"),
            EvidenceSpec("GPT-5.4 SDK provider", "src/sam/llm.py", "code"),
            EvidenceSpec("Embedding provider 与缓存", "src/sam/embedding.py", "code"),
            EvidenceSpec("Embedding 正式运行前请求量规划", "scripts/plan_embedding_run.py", "code"),
            EvidenceSpec("Embedding cache 预热入口", "scripts/warm_embedding_cache.py", "code"),
            EvidenceSpec("官方 baseline 就绪审计", "evaluation/official_baselines/audit_official_baselines.py", "code"),
        ],
        experiment_evidence=[
            EvidenceSpec("GPT-5.4 grounded 生成闭环", "outputs/runs/e2e_gpt54_generation_q3_grounded_v2/pipeline_summary.json", "experiment"),
            EvidenceSpec("NovelQA 小样本", "outputs/runs/novelqa_demo_eval12_edge_filter/metrics.json", "experiment"),
            EvidenceSpec("端到端本地 smoke", "outputs/runs/end_to_end_smoke/pipeline_summary.json", "experiment"),
            EvidenceSpec("HotpotQA embedding 请求量计划", "outputs/plans/hotpotqa_embedding_plan/embedding_run_plan.json", "experiment"),
            EvidenceSpec("HotpotQA embedding cache 本地预热 smoke", "outputs/plans/hotpotqa_local_warmup/embedding_cache_warmup.json", "experiment"),
            EvidenceSpec("官方 baseline 就绪状态审计", "docs/official_baseline_audit.json", "experiment"),
        ],
        remaining_work=[
            "正式 embedding endpoint/key 已在本地安全配置中提供，但当前真实 probe 返回 TimeoutError，HotpotQA 300 条和 NovelQA 正式在线 embedding 主实验尚未完成。",
            "GPT-5.4 grounded 生成闭环已跑通低额度实验，但需要扩大到更多样本并接入语义答案判别。",
            "官方 baseline 中 GraphRAG 已达到本地 ready 状态，RAPTOR 和 HippoRAG 仍需修复官方依赖后再跑正式分数。",
        ],
    ),
]


def build_opening_plan_audit(root: str | Path) -> dict[str, Any]:
    """构建开题计划进度审计。"""

    project_root = Path(root)
    modules = [_audit_module(project_root, spec) for spec in OPENING_MODULE_SPECS]
    overall_progress = round(
        sum(module["estimated_progress"] for module in modules) / max(1, len(modules)),
        1,
    )
    return {
        "title": "SAM 开题计划进度审计",
        "overall_progress": overall_progress,
        "module_count": len(modules),
        "modules": modules,
        "next_actions": _next_actions(modules),
    }


def write_opening_plan_audit(audit: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """写入开题计划进度审计 JSON 和 Markdown。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "opening_plan_audit.json"
    markdown_path = target / "opening_plan_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_audit_markdown(audit), encoding="utf-8")
    return json_path, markdown_path


def _audit_module(project_root: Path, spec: ModuleSpec) -> dict[str, Any]:
    code = [_audit_evidence(project_root, item) for item in spec.code_evidence]
    experiments = [_audit_evidence(project_root, item) for item in spec.experiment_evidence]
    code_ratio = _presence_ratio(code)
    experiment_ratio = _presence_ratio(experiments)
    estimated = round(spec.target_progress * (0.55 + 0.25 * code_ratio + 0.20 * experiment_ratio), 1)
    estimated = min(spec.target_progress, estimated)
    status = "已完成阶段性目标" if experiment_ratio >= 0.8 and code_ratio >= 0.9 else "部分完成"
    if experiment_ratio < 0.4:
        status = "实现已有，实验不足"
    return {
        "module_id": spec.module_id,
        "title": spec.title,
        "opening_requirement": spec.opening_requirement,
        "status": status,
        "target_progress": spec.target_progress,
        "estimated_progress": estimated,
        "code_evidence": code,
        "experiment_evidence": experiments,
        "remaining_work": spec.remaining_work,
    }


def _audit_evidence(project_root: Path, spec: EvidenceSpec) -> dict[str, Any]:
    path = project_root / spec.path
    exists = path.exists()
    payload: dict[str, Any] = {
        "label": spec.label,
        "path": spec.path,
        "kind": spec.kind,
        "exists": exists,
    }
    if spec.note:
        payload["note"] = spec.note
    if exists and path.suffix == ".json":
        summary = _json_summary(path)
        if summary:
            payload["summary"] = summary
    return payload


def _json_summary(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        if "method_metrics" in data and isinstance(data["method_metrics"], dict):
            methods = data["method_metrics"]
            sam = methods.get("sam_full") or methods.get("sam")
            embedding = methods.get("embedding_topk")
            summary: dict[str, Any] = {}
            if isinstance(sam, dict):
                summary.update(
                    {
                        "sam_evidence_recall": _round_or_none(sam.get("evidence_recall")),
                        "sam_answer_hit_rate": _round_or_none(sam.get("answer_hit_rate")),
                    }
                )
            if isinstance(embedding, dict):
                summary.update(
                    {
                        "embedding_evidence_recall": _round_or_none(embedding.get("evidence_recall")),
                        "embedding_answer_hit_rate": _round_or_none(embedding.get("answer_hit_rate")),
                    }
                )
            return summary
        if "sam_full" in data and isinstance(data["sam_full"], dict):
            sam = data["sam_full"]
            return {
                "sam_evidence_recall": _round_or_none(sam.get("evidence_recall")),
                "sam_answer_hit_rate": _round_or_none(sam.get("answer_hit_rate")),
            }
        if "summary" in data and isinstance(data["summary"], dict):
            summary = data["summary"]
            return {
                key: summary.get(key)
                for key in [
                    "warmup_consolidated_count",
                    "sam_support_hits",
                    "support_hit_gain",
                    "analogy_case_hit_rate",
                    "shared_memory_chain_success_rate",
                    "ready_count",
                    "partial_count",
                    "prepared_dataset_count",
                ]
                if key in summary
            }
        if data.get("type") == "CachedRelationJudge":
            base = data.get("base", {})
            if isinstance(base, dict):
                budget = base.get("base", {}) if base.get("type") == "BudgetedRelationJudge" else {}
                return {
                    "cache_hits": data.get("cache_hits"),
                    "cache_misses": data.get("cache_misses"),
                    "calls_made": base.get("calls_made"),
                    "skipped_count": base.get("skipped_count"),
                    "chat_provider": budget.get("chat_provider") if isinstance(budget, dict) else None,
                }
        if "generation" in data and isinstance(data["generation"], dict):
            generation = data["generation"]
            return {
                "answer_hit_rate": _round_or_none(generation.get("answer_hit_rate")),
            }
        if "profile_results" in data and isinstance(data["profile_results"], dict):
            return {
                "best_profile": data.get("best_profile"),
                "profile_count": len(data["profile_results"]),
            }
    return {}


def _presence_ratio(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if item["exists"]) / len(items)


def _next_actions(modules: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for module in sorted(modules, key=lambda item: float(item["estimated_progress"])):
        if module["remaining_work"]:
            actions.append(f"{module['title']}：{module['remaining_work'][0]}")
    return actions[:6]


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# SAM 开题计划进度审计",
        "",
        f"- 模块数量：{audit['module_count']}",
        f"- 估算总体进度：{audit['overall_progress']}%",
        "",
        "| 模块 | 状态 | 估算进度 | 代码证据 | 实验证据 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for module in audit["modules"]:
        code_count = sum(1 for item in module["code_evidence"] if item["exists"])
        experiment_count = sum(1 for item in module["experiment_evidence"] if item["exists"])
        lines.append(
            "| "
            + " | ".join(
                [
                    module["title"],
                    module["status"],
                    f"{module['estimated_progress']}%",
                    f"{code_count}/{len(module['code_evidence'])}",
                    f"{experiment_count}/{len(module['experiment_evidence'])}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## 模块明细", ""])
    for module in audit["modules"]:
        lines.extend(
            [
                f"### {module['title']}",
                "",
                f"- 开题要求：{module['opening_requirement']}",
                f"- 当前状态：{module['status']}，估算进度 {module['estimated_progress']}%",
                "- 代码证据："
            ]
        )
        for item in module["code_evidence"]:
            marker = "已存在" if item["exists"] else "缺失"
            lines.append(f"  - {marker}：`{item['path']}`，{item['label']}")
        lines.append("- 实验证据：")
        for item in module["experiment_evidence"]:
            marker = "已存在" if item["exists"] else "缺失"
            summary = (
                "，摘要："
                + json.dumps(item["summary"], ensure_ascii=False)
                if item.get("summary") else ""
            )
            lines.append(f"  - {marker}：`{item['path']}`，{item['label']}{summary}")
        lines.append("- 剩余工作：")
        for work in module["remaining_work"]:
            lines.append(f"  - {work}")
        lines.append("")
    lines.extend(["## 下一步优先事项", ""])
    for action in audit["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def _round_or_none(value: object) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None
