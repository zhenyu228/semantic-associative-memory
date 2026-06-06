# 开题计划进度对齐说明

本文档用于把开题报告详情表中的技术路线与当前 SAM 仓库实现逐项对齐。它的用途不是宣传项目已经完成，而是明确：哪些模块已有代码和实验支撑，哪些模块还只是初版，哪些内容仍需继续补齐。

## 1. 开题计划中的核心任务

根据《复旦大学研究生学位论文开题报告记录表（详情表）.pdf》，本课题需要解决四个关键问题：

1. 如何构建和更新动态知识图谱，以存储并有效表示知识间的语义关系。
2. 如何设计语义激活与回忆机制，使系统能够从海量过往记忆中联想出相关信息。
3. 如何实现多智能体之间的语义记忆共享与协调，支持协作过程中的经验交换。
4. 如何触发和应用类比推理，在遇到新问题时检索类似情境的记忆并启发求解。

对应的技术路线包括四个主模块：

1. 知识提取与动态知识图谱构建。
2. 语义激活与记忆检索机制。
3. 类比推理触发与应用。
4. 多智能体语义记忆协调机制。

开题进度安排中，2026 年 2 月至 5 月的目标是完成主体框架，包括知识图谱构建模块、语义扩散激活与联想检索模块、多智能体记忆共享机制、类比推理触发器，并设计正式实验、构建评测体系、收集阶段性结果。

## 2. 当前总体进度

当前项目已经完成可运行原型和若干阶段性实验，但还没有达到“完整毕业论文系统”的状态。总体判断如下：

| 模块 | 当前状态 | 完成度判断 | 主要证据 |
| --- | --- | --- | --- |
| 统一数据格式与数据集适配 | 已完成初版 | 80% | HotpotQA、NovelQA 已转换为 `sam-dataset-v1` |
| 动态知识图谱构建与更新 | 已完成主体框架 | 70% | `MemoryStore`、`GraphBuilder`、节点/边状态、事件流、按需建边 |
| 语义激活与联想检索 | 已完成主体框架 | 75% | `Retriever`、多跳扩展、消融模式、路径重排 |
| 评测体系与实验产物 | 已完成阶段性版本 | 70% | HotpotQA 300 条主实验、NovelQA 小样本实验、bad case 分析 |
| 类比推理触发 | 已完成初版 | 45% | `AnalogyEngine`、连续复用与类比复用实验 |
| 多智能体记忆共享 | 已完成初版 | 45% | `SharedMemoryCoordinator`、四角色 workflow、共享记忆复用实验 |
| 检索-生成-判别闭环 | 已有实验入口 | 40% | `run_end_to_end_experiment.py`、AnswerJudge、生成 bad case 分析 |
| 正式 embedding 与大模型实验 | 接口完成，正式结果不足 | 30% | Azure embedding provider、GPT-5.4 provider 检查脚本 |
| 官方 baseline 严格复现 | 目录和适配脚本已有，尚未完成 | 25% | `evaluation/official_baselines/` |

## 3. 模块进度明细

### 3.1 知识提取与动态知识图谱构建

开题要求：从原始语料中抽取关键信息单元及其语义关系，将知识表示为带属性的记忆节点，并根据语义相似度自动建立新旧节点间的关联，实现图谱动态生长与更新。

当前已实现：

- `MemoryNode`：保存文本、标题、摘要、关键词、来源、时间戳、使用次数、置信度、embedding 等信息。
- `MemoryEdge`：保存起点、终点、关系类型、权重、建边原因、创建模块、激活时间等信息。
- `MemoryStore`：使用 SQLite 保存节点、边、检索日志和记忆事件。
- `GraphBuilder`：支持按需建边，不做全量两两建图；围绕初始种子节点和后续桥接节点逐步补边。
- 建边原因拆分：实体重叠、关键词重叠、语义相似、上下文共现、反馈强化、摘要层级边。
- 边质量控制：过滤低信息关键词边，并对弱关系二跳路径加入排序惩罚。
- 关系判别接口：已实现 `RelationJudge` 和 `CachedRelationJudge`，可通过 GPT-5.4 判断候选边是否具备真实语义关系。

对应代码：

- `src/sam/models.py`
- `src/sam/store.py`
- `src/sam/graph.py`
- `src/sam/relation_judge.py`
- `src/sam/feedback.py`

当前不足：

- 关系抽取仍主要依赖规则、关键词和相似度，GPT-5.4 关系判别尚未形成正式规模实验。
- 记忆重构已有 `MemoryConsolidator` 初版，但还没有形成长期、多轮、跨任务的大规模验证。
- 图谱动态更新策略仍是经验公式，尚未加入学习式权重更新。

### 3.2 语义激活与记忆检索机制

开题要求：先用向量相似度锁定候选集合，再沿知识图谱链接扩展邻近关联记忆，形成与当前问题相关的记忆子图，支持复杂推理中的推理链重建。

当前已实现：

- `Retriever` 支持两阶段检索：初始召回加图扩展。
- 支持一跳和两跳联想扩展，并通过 seed-k、top-k、hops 控制成本。
- 支持桥接节点继续按需建边，使路径能够覆盖“问题相关种子文档 -> 桥接实体文档 -> 答案证据文档”。
- `PathReranker` 将语义分、图路径分、多路径支持分、记忆状态分和反馈分拆开计算。
- 已实现多种消融模式：`sam_full`、`sam_no_graph`、`sam_no_multipath`、`sam_no_memory_state`、`sam_static_graph`、`sam_no_feedback`、`sam_with_summary` 等。
- 已实现 `QueryPlanner`，支持启发式查询规划和 GPT-5.4 查询规划接口。

对应代码：

- `src/sam/retriever.py`
- `src/sam/reranker.py`
- `src/sam/query_planner.py`
- `src/sam/evaluator.py`
- `scripts/run_demo.py`

主要实验结果：

- HotpotQA bridge-style 300 条主实验中，SAM-full 的证据召回率为 0.603，答案命中率为 0.597；Embedding Top-k 为 0.572 和 0.547。
- 去掉图扩展后，`sam_no_graph` 为 0.578 和 0.553，说明当前主要增益来自图扩展。
- 桥接节点按需建边 30 条回归实验中，SAM-full 证据召回率为 0.617，Embedding Top-k 和 SAM-no-graph 均为 0.500。

当前不足：

- 多路径和记忆状态在 300 条单轮 HotpotQA 实验中没有明显拉开差距。
- 当前默认 embedding 仍限制初始召回质量，需要用正式 embedding 重跑主实验。
- 图扩展仍存在噪声路径，需要进一步使用实体消歧、关系判别和更强重排策略。

### 3.3 类比推理触发与应用

开题要求：当新问题激活的节点子图与历史问题-解答链条存在相似关系结构时，触发类比机制，检索结构相似或情境相似的过去案例，并以提示形式提供给 LLM。

当前已实现：

- `AnalogyEngine` 能从历史巩固记忆中检索相似案例。
- 支持基于问题相似、关键词重叠、关系路径模式的案例匹配。
- `MemoryConsolidator` 会把成功检索的证据链沉淀为 `consolidated_memory`，供后续类比检索。
- `run_analogy_reuse_experiment.py` 已能评估 masked probe 查询是否命中 warmup 阶段形成的历史案例。
- 生成阶段已有类比提示注入入口，可对比无类比提示和有类比提示两种设置。

对应代码：

- `src/sam/analogy.py`
- `src/sam/consolidation.py`
- `src/sam/analogy_experiment.py`
- `scripts/run_analogy_reuse_experiment.py`
- `scripts/generate_answers.py`

主要实验结果：

- 30 条 HotpotQA 类比复用实验中，warmup 阶段生成 22 个巩固记忆节点。
- 巩固案例命中率为 0.733，支持证据重叠命中率为 0.800。

当前不足：

- 类比推理目前主要验证“能否找回历史证据链”，还没有充分验证“是否提升最终答案质量”。
- 类比触发条件仍偏规则化，尚未使用更强的子图匹配或学习式匹配方法。
- GPT-5.4 类比提示生成实验尚未形成正式结果。

### 3.4 多智能体语义记忆协调机制

开题要求：设计全局洞察层、会话层和交互细节层，支持多智能体查询公共知识、访问彼此中间结果，并通过共享记忆重建完整推理链。

当前已实现：

- `SharedMemoryCoordinator` 支持 `global_insight`、`session`、`interaction` 三层共享记忆。
- 支持不同智能体写入和查询共享记忆。
- 支持 handoff 机制，使 retriever 的证据可以传递给 writer，writer 的结果可以传递给 verifier。
- `MultiAgentResearchWorkflow` 已串联 planner、retriever、writer、verifier 四个角色。
- `run_agent_memory_reuse_experiment.py` 已用于检查 SAM 记忆复用增益是否能通过共享记忆传递到后续智能体角色。
- `run_agent_generation_experiment.py` 已支持 baseline、shared memory、shared memory with analogy 三种生成设置。

对应代码：

- `src/sam/agents.py`
- `src/sam/agent_workflow.py`
- `src/sam/agent_reuse_experiment.py`
- `scripts/run_agent_workflow.py`
- `scripts/run_agent_memory_reuse_experiment.py`
- `scripts/run_agent_generation_experiment.py`

主要实验结果：

- 30 条 HotpotQA 多智能体共享记忆复用实验中，Embedding Top-k 支持证据命中数为 0，SAM 支持证据命中数为 31。
- 存在支持证据增益的样本数为 22，多智能体复用链路成功率为 0.733。
- 多智能体生成 smoke run 已验证共享记忆和类比提示能进入生成上下文，但本地启发式生成器的答案质量不作为正式结论。

当前不足：

- 目前多智能体实验更偏受控流程验证，不是完整 Deep Research 真实任务。
- 多智能体协作效率、跨角色冲突解决、版本管理策略尚未充分实现。
- 需要接入 GPT-5.4 生成器，正式比较无共享记忆、共享记忆、共享记忆加类比提示三种设置下的答案质量。

## 4. 实验进度

### 4.1 HotpotQA

已完成 bridge-style 300 条主实验：

- 查询数量：300。
- 候选文档节点数量：2992。
- 摘要记忆节点数量：300。
- Gold 支持证据数量：600。
- 默认参数：top-k=4、seed-k=1、hops=2。

阶段结果：

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.572 | 0.547 |
| RAPTOR | 0.635 | 0.613 |
| GraphRAG | 0.562 | 0.547 |
| HippoRAG | 0.587 | 0.553 |
| SAM-full | 0.603 | 0.597 |
| SAM-no-graph | 0.578 | 0.553 |

该结果说明当前 SAM 已经具备比普通 embedding 检索更强的图联想补证据能力，但相比 RAPTOR 仍存在差距，尤其需要优化摘要记忆、初始召回和路径重排。

### 4.2 NovelQA

已完成 NovelQA demonstration 小样本接入：

- 数据来源：本地 `data/raw/NovelQA.zip`。
- 当前使用 Frankenstein demonstration 子集。
- 查询数量：12。
- 小说 chunk 数量：120。
- 可映射 gold evidence chunk 数量：20。

当前最好的一组小样本结果显示：

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.143 | 0.000 |
| RAPTOR | 0.143 | 0.000 |
| GraphRAG | 0.357 | 0.083 |
| HippoRAG | 0.143 | 0.000 |
| SAM-full | 0.143 | 0.083 |
| SAM-no-graph | 0.143 | 0.000 |

NovelQA 暴露的主要问题是长文本 chunk 定位困难、实体消歧不足、答案表述不一定精确出现在原文中，以及当前 embedding 表示能力不足。

### 4.3 连续记忆复用与类比实验

已完成 30 条 HotpotQA 受控连续复用实验：

- Warmup 阶段生成 22 个巩固记忆节点和 62 条巩固相关边。
- Probe 阶段移除 gold 支持文档后，Embedding Top-k 支持证据命中数为 0，SAM 支持证据命中数为 31。
- 类比复用实验中，巩固案例命中率为 0.733，支持证据重叠命中率为 0.800。

这部分结果证明当前系统已经具备“成功检索 -> 记忆巩固 -> 后续候选补全 -> 类比复用”的初步闭环。

## 5. 和开题时间安排的对齐情况

### 2025.11-2025.12：调研与框架拆解

当前状态：已完成。

已有 README、系统设计文档、实验协议、方法对比说明和开题中期材料草稿。核心模块已经拆为 Dataset Adapter、EmbeddingProvider、MemoryStore、GraphBuilder、Retriever、Evaluator、QueryPlanner、RelationJudge、FeedbackUpdater、AnalogyEngine、SharedMemoryCoordinator 等。

### 2025.12-2026.01：最小可行原型与小规模实验

当前状态：已完成。

已有可运行 demo、HotpotQA 小样本、NovelQA 小样本、图谱可视化和基础实验结果。

### 2026.02-2026.05：主体框架、正式实验、评测体系

当前状态：主体框架已完成，正式实验部分完成，评测体系仍需增强。

已完成：

- 动态知识图谱构建模块。
- 语义扩散激活与联想检索模块。
- 类比推理触发器初版。
- 多智能体共享记忆初版。
- HotpotQA 300 条主实验。
- 消融实验和 bad case 分析。

未完成或不足：

- NovelQA 还不是完整规模实验。
- 官方 RAPTOR、GraphRAG、HippoRAG 严格复现尚未完成。
- GPT-5.4 生成和判别实验尚未形成正式结果。
- 推理链完整性、跨文档一致性、智能体协作效率等指标还不够系统。

### 2026.05-2026.06：系统优化与最终对比实验

当前状态：已经开始，但未完成。

已做优化：

- 桥接节点按需建边。
- 弱关键词边过滤。
- 弱关系二跳路径惩罚。
- 过密路径惩罚。
- reranker profile 对比。
- 记忆巩固和连续复用实验。

仍需继续：

- 接入正式 embedding 并重跑主实验。
- 使用 GPT-5.4 RelationJudge 进行边质量实验。
- 扩大 NovelQA 样本。
- 完成官方 baseline 或更严格对照。
- 完成检索-生成-判别端到端正式实验。

## 6. 当前最需要补的内容

按论文说服力排序，下一步建议优先做以下工作：

1. 使用正式 embedding 重跑 HotpotQA 300 条和 NovelQA 小样本实验，确认当前结论是否仍成立。
2. 启用 GPT-5.4 RelationJudge，对候选边做语义关系判别，验证是否能降低图噪声。
3. 扩大 NovelQA 样本规模，至少从 12 条 demonstration 提升到可稳定复现的几十条或上百条。
4. 把连续记忆复用实验从受控 mask probe 扩展到同主题多轮问答，验证反馈和巩固记忆在真实连续任务中的收益。
5. 接入 GPT-5.4 生成和答案判别，完成检索-生成-判别端到端对照。
6. 推进官方 baseline 目录中的 RAPTOR、GraphRAG、HippoRAG 评测，使对比实验更可辩护。

## 7. 当前结论

当前项目已经实现了开题计划中最核心的两个模块：动态知识图谱构建与语义联想检索，并完成 HotpotQA 300 条阶段性实验。类比推理和多智能体共享记忆也已经有可运行原型和受控实验结果，但还没有达到正式论文主实验的强度。

因此，当前进度可以表述为：已完成动态知识图谱记忆系统的主体原型、统一数据格式、按需建图、联想检索、消融评测、记忆巩固、类比复用和多智能体共享记忆初版；下一阶段重点是使用正式 embedding 与 GPT-5.4 完成更强实验，扩大 NovelQA，并补齐官方 baseline 与端到端生成评测。
