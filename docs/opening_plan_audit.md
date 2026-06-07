# SAM 开题计划进度审计

- 模块数量：5
- 估算总体进度：63.6%

| 模块 | 状态 | 估算进度 | 代码证据 | 实验证据 |
| --- | --- | ---: | ---: | ---: |
| 知识提取与动态知识图谱构建 | 已完成阶段性目标 | 72% | 7/7 | 5/5 |
| 语义激活与联想检索机制 | 已完成阶段性目标 | 78% | 6/6 | 5/5 |
| 类比推理触发与应用 | 已完成阶段性目标 | 58% | 5/5 | 4/4 |
| 多智能体语义记忆协调机制 | 已完成阶段性目标 | 52% | 6/6 | 3/3 |
| 评测体系与检索-生成闭环 | 已完成阶段性目标 | 58% | 8/8 | 6/6 |

## 模块明细

### 知识提取与动态知识图谱构建

- 开题要求：抽取关键信息单元及语义关系，将知识表示为带属性的记忆节点，并支持图谱动态生长、更新和记忆重构。
- 当前状态：已完成阶段性目标，估算进度 72%
- 代码证据：
  - 已存在：`src/sam/models.py`，MemoryNode / MemoryEdge 数据结构
  - 已存在：`src/sam/store.py`，SQLite 记忆存储与事件表
  - 已存在：`src/sam/graph.py`，按需建图与边质量控制
  - 已存在：`src/sam/edge_audit.py`，图边质量审计
  - 已存在：`src/sam/feedback.py`，反馈更新
  - 已存在：`src/sam/consolidation.py`，记忆巩固
  - 已存在：`src/sam/relation_judge.py`，GPT-5.4 关系判别接口
- 实验证据：
  - 已存在：`outputs/runs/memory_events_30_smoke/metrics.json`，记忆事件 smoke，摘要：{"sam_evidence_recall": 0.5167, "sam_answer_hit_rate": 0.4, "embedding_evidence_recall": 0.4833, "embedding_answer_hit_rate": 0.4}
  - 已存在：`outputs/runs/memory_consolidation_hotpotqa30_v2/metrics.json`，记忆巩固实验，摘要：{"sam_evidence_recall": 0.5167, "sam_answer_hit_rate": 0.4, "embedding_evidence_recall": 0.4833, "embedding_answer_hit_rate": 0.4}
  - 已存在：`outputs/runs/weak_relation_penalty_hotpotqa30/metrics.json`，弱关系惩罚实验，摘要：{"sam_evidence_recall": 0.6167, "sam_answer_hit_rate": 0.6667, "embedding_evidence_recall": 0.5, "embedding_answer_hit_rate": 0.5667}
  - 已存在：`outputs/runs/weak_relation_penalty_hotpotqa30/edge_quality_audit.json`，图边质量审计 smoke
  - 已存在：`outputs/runs/relation_judge_gpt54_querylimit5_smoke/relation_judge_usage.json`，GPT-5.4 关系判别 smoke，摘要：{"cache_hits": 96, "cache_misses": 0, "calls_made": 0, "skipped_count": 0, "chat_provider": "azure_openai_sdk"}
- 剩余工作：
  - GPT-5.4 RelationJudge 已完成低预算 smoke，但尚未形成正式规模实验。
  - 图谱边权仍以经验公式为主，缺少学习式或系统化参数搜索。
  - 记忆重构需要更多跨任务连续验证。

### 语义激活与联想检索机制

- 开题要求：先用语义相似度锁定候选，再沿知识图谱关联路径扩展邻近记忆，形成与当前问题相关的记忆子图。
- 当前状态：已完成阶段性目标，估算进度 78%
- 代码证据：
  - 已存在：`src/sam/retriever.py`，两阶段检索与消融模式
  - 已存在：`src/sam/reranker.py`，路径重排
  - 已存在：`src/sam/reranker.py`，审计驱动关系噪声惩罚
  - 已存在：`src/sam/query_planner.py`，查询规划
  - 已存在：`src/sam/evaluator.py`，评测器
  - 已存在：`scripts/run_demo.py`，主实验入口
- 实验证据：
  - 已存在：`outputs/runs/lexical_isolated_hotpotqa300/metrics.json`，HotpotQA 300 条候选集隔离实验，摘要：{"sam_evidence_recall": 0.6617, "sam_answer_hit_rate": 0.75, "embedding_evidence_recall": 0.57, "embedding_answer_hit_rate": 0.6467}
  - 已存在：`outputs/runs/fair_ablation_hotpotqa_300/ablation_metrics.json`，HotpotQA 300 条消融，摘要：{"sam_evidence_recall": 0.6033, "sam_answer_hit_rate": 0.5967}
  - 已存在：`outputs/runs/feedback_ablation_hotpotqa_300_isolated/ablation_metrics.json`，反馈消融 300 条，摘要：{"sam_evidence_recall": 0.6033, "sam_answer_hit_rate": 0.5967}
  - 已存在：`outputs/runs/reranker_profile_hotpotqa300_noise_penalty/reranker_profile_comparison.json`，PathReranker 300 条 profile 对比，摘要：{"best_profile": "semantic_heavy", "profile_count": 4}
  - 已存在：`outputs/runs/edge_audit_penalty_hotpotqa30/metrics.json`，Edge-audit 惩罚 30 条 smoke，摘要：{"sam_evidence_recall": 0.6, "sam_answer_hit_rate": 0.7333, "embedding_evidence_recall": 0.5167, "embedding_answer_hit_rate": 0.5667}
- 剩余工作：
  - 正式 embedding 尚未重跑 HotpotQA 300 条和 NovelQA。
  - 多路径与记忆状态需要在连续任务中继续拉开贡献差异。
  - 仍需进一步降低图噪声和缺失支持证据问题。

### 类比推理触发与应用

- 开题要求：在新问题激活子图与历史问题-解答链条结构相似时触发类比，检索类似案例并向 LLM 提供提示。
- 当前状态：已完成阶段性目标，估算进度 58%
- 代码证据：
  - 已存在：`src/sam/analogy.py`，类比检索引擎
  - 已存在：`src/sam/analogy_experiment.py`，类比复用实验逻辑
  - 已存在：`src/sam/generation.py`，类比提示注入
  - 已存在：`src/sam/retriever.py`，类比支持证据注入检索排序
  - 已存在：`scripts/run_analogy_reuse_experiment.py`，类比复用脚本
- 实验证据：
  - 已存在：`outputs/runs/analogy_reuse_hotpotqa30/analogy_reuse_results.json`，类比复用 30 条
  - 已存在：`outputs/runs/analogy_generation_smoke/metrics.json`，类比生成 smoke，摘要：{"sam_evidence_recall": 1.0, "sam_answer_hit_rate": 1.0, "embedding_evidence_recall": 0.6667, "embedding_answer_hit_rate": 0.6667}
  - 已存在：`outputs/runs/analogy_retrieval_smoke/metrics.json`，类比检索排序 30 条 smoke，摘要：{"sam_evidence_recall": 0.55, "sam_answer_hit_rate": 0.6333, "embedding_evidence_recall": 0.4833, "embedding_answer_hit_rate": 0.5}
  - 已存在：`outputs/runs/analogy_structural_consolidation_hotpotqa30/analogy_reuse_results.json`，结构性巩固类比复用 30 条
- 剩余工作：
  - 类比提示对最终答案质量的提升尚未用 GPT-5.4 正式验证。
  - 类比支持证据注入检索排序仍需正式规模实验验证。
  - 结构路径匹配已经进入实验，但仍需要在未知来源案例和真实多轮任务中验证泛化能力。
  - 结构性巩固已覆盖来源案例，但仍有样本缺少真实支持证据重叠，需要继续改进检索和建边质量。

### 多智能体语义记忆协调机制

- 开题要求：构建全局洞察层、会话层、交互细节层，支持多智能体共享中间结果和经验以重建推理链。
- 当前状态：已完成阶段性目标，估算进度 52%
- 代码证据：
  - 已存在：`src/sam/agents.py`，共享记忆协调器
  - 已存在：`src/sam/agents.py`，共享记忆冲突裁决与版本指标
  - 已存在：`src/sam/agent_workflow.py`，多智能体研究流程
  - 已存在：`src/sam/agent_reuse_experiment.py`，多智能体复用实验
  - 已存在：`scripts/run_agent_workflow.py`，多智能体 workflow 脚本
  - 已存在：`scripts/run_agent_memory_reuse_experiment.py`，共享记忆复用脚本
- 实验证据：
  - 已存在：`outputs/runs/agent_memory_reuse_hotpotqa30/agent_memory_reuse_results.json`，多智能体共享记忆复用
  - 已存在：`outputs/runs/agent_generation_hotpotqa30_smoke/agent_generation_comparison.json`，多智能体生成对照 smoke
  - 已存在：`outputs/runs/agent_workflow_conflict_smoke/agent_workflow.json`，多智能体 workflow 自动冲突裁决 smoke
- 剩余工作：
  - 当前多智能体实验仍偏受控流程，不是完整 Deep Research 任务。
  - workflow 已能在答案验证失败时自动触发冲突裁决，但仍需设计更真实的多角色分歧任务集。
  - 需要用 GPT-5.4 比较共享记忆与类比提示对最终答案质量的影响。

### 评测体系与检索-生成闭环

- 开题要求：设计正式实验和评测体系，覆盖跨文档语义整合、推理链重建、多智能体协作和生成结果反馈。
- 当前状态：已完成阶段性目标，估算进度 58%
- 代码证据：
  - 已存在：`scripts/run_end_to_end_experiment.py`，端到端实验入口
  - 已存在：`src/sam/answer_judge.py`，答案判别
  - 已存在：`src/sam/badcase.py`，Bad Case 分析
  - 已存在：`src/sam/llm.py`，GPT-5.4 SDK provider
  - 已存在：`src/sam/embedding.py`，Embedding provider 与缓存
  - 已存在：`scripts/plan_embedding_run.py`，Embedding 正式运行前请求量规划
  - 已存在：`scripts/warm_embedding_cache.py`，Embedding cache 预热入口
  - 已存在：`evaluation/official_baselines/audit_official_baselines.py`，官方 baseline 就绪审计
- 实验证据：
  - 已存在：`outputs/runs/provider_smoke_gpt54_sdk_hotpotqa1/pipeline_summary.json`，GPT-5.4 SDK HotpotQA smoke，摘要：{"answer_hit_rate": 0.0}
  - 已存在：`outputs/runs/novelqa_demo_eval12_edge_filter/metrics.json`，NovelQA 小样本，摘要：{"sam_evidence_recall": 0.1429, "sam_answer_hit_rate": 0.0833, "embedding_evidence_recall": 0.1429, "embedding_answer_hit_rate": 0.0}
  - 已存在：`outputs/runs/end_to_end_smoke/pipeline_summary.json`，端到端本地 smoke，摘要：{"answer_hit_rate": 0.0}
  - 已存在：`outputs/plans/hotpotqa_embedding_plan/embedding_run_plan.json`，HotpotQA embedding 请求量计划
  - 已存在：`outputs/plans/hotpotqa_local_warmup/embedding_cache_warmup.json`，HotpotQA embedding cache 本地预热 smoke
  - 已存在：`docs/official_baseline_audit.json`，官方 baseline 就绪状态审计，摘要：{"ready_count": 2, "partial_count": 1, "prepared_dataset_count": 1}
- 剩余工作：
  - 正式 embedding endpoint/key 已在本地安全配置中提供，但当前真实 probe 返回 TimeoutError，HotpotQA 300 条和 NovelQA 正式在线 embedding 主实验尚未完成。
  - GPT-5.4 生成和答案判别需要扩大到多样本正式结果。
  - 官方 baseline 中 GraphRAG 已达到本地 ready 状态，RAPTOR 和 HippoRAG 仍需修复官方依赖后再跑正式分数。

## 下一步优先事项

- 多智能体语义记忆协调机制：当前多智能体实验仍偏受控流程，不是完整 Deep Research 任务。
- 类比推理触发与应用：类比提示对最终答案质量的提升尚未用 GPT-5.4 正式验证。
- 评测体系与检索-生成闭环：正式 embedding endpoint/key 已在本地安全配置中提供，但当前真实 probe 返回 TimeoutError，HotpotQA 300 条和 NovelQA 正式在线 embedding 主实验尚未完成。
- 知识提取与动态知识图谱构建：GPT-5.4 RelationJudge 已完成低预算 smoke，但尚未形成正式规模实验。
- 语义激活与联想检索机制：正式 embedding 尚未重跑 HotpotQA 300 条和 NovelQA。
