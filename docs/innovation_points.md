# SAM 创新点闭环记录

本文档用于把当前代码实现、实验结果和论文创新点对应起来。运行产物默认写入 `outputs/runs/`，该目录不进入 Git 仓库；本文只记录可复现实验路径和阶段结论。

## 1. 动态按需知识图谱记忆构建

当前系统没有在数据集载入时构建完整图谱，而是在检索发生后围绕激活节点局部建边。流程为：先将候选文档写入 `MemoryStore`，再由 `Retriever` 选择种子节点，随后 `GraphBuilder` 只在种子节点、候选节点和扩展路径周围计算实体、关键词、语义相似等关系，最后将建边结果、边权和访问日志写回存储层。

对应实现：

- `MemoryStore`：保存记忆节点、语义边、检索日志、反馈事件。
- `GraphBuilder`：执行按需建边、关系类型判定、边质量过滤、建边原因记录。
- `graph_cost_audit`：统计每次实验中新建边数量、关系类型分布、局部建图与全量建图的理论成本差异。
- `run_demo.py`：每次实验输出 `graphs/edge_creation_log.json`、`graphs/graph_build_cost_audit.json` 和 `graphs/graph_build_cost_audit.md`。

当前证据：

- HotpotQA 1 条真实 embedding smoke 已输出建图成本审计，说明在线 embedding 链路下也能完成写入、检索、局部建边和审计。
- HotpotQA 30 条真实 embedding smoke `outputs/runs/hotpotqa30_real_embedding_smoke_v2/` 中，300 个文档节点的全量建图理论边数为 44850，SAM 实际唯一新建无向节点对为 1164，占比 0.025953，估算节省比例为 0.974047。
- 300 条 HotpotQA 真实 embedding 主实验 `outputs/runs/hotpotqa300_real_embedding_main_v4_hops1/` 中，2992 个文档节点的全量建图理论边数为 4474536，SAM 实际唯一新建无向节点对为 2347，占比 0.000525，估算节省比例为 0.999475，平均每个 query 新建无向节点对 7.823。
- 300 条 HotpotQA 主实验已经证明 SAM-full 相比 Embedding Top-k 有更高证据召回率；图扩展关闭后，SAM-no-graph 指标下降，说明局部图谱确实参与了检索。

该创新点对应专家提出的“建图成本可能很高”的问题。当前回答是：SAM 不追求一次性全量建图，而是将建图动作推迟到查询发生时，只围绕被激活记忆进行局部关系补全，并将新增边和触达边记录为可审计产物。

## 2. 语义联想路径检索

当前系统不是只做 embedding top-k，而是采用“初始召回 + 局部建图 + 路径扩展 + 多信号重排”的检索流程。Embedding 召回只负责提供入口节点，最终结果还会参考实体共享、关键词重叠、语义相似边、路径长度、路径支持分、记忆状态和反馈信号。

对应实现：

- `Retriever`：支持 Embedding Top-k、RAPTOR、GraphRAG、HippoRAG、SAM-full、SAM-no-graph 等方法。
- SAM 消融模式：`sam_full`、`sam_no_graph`、`sam_no_multipath`、`sam_no_memory_state`、`sam_static_graph`、`sam_no_feedback`。
- `Evaluator`：统计证据召回率、答案命中率、平均路径长度、平均候选路径数等指标。

当前主要结果：

- `outputs/runs/hotpotqa30_real_embedding_smoke_v2/`：使用公司可用 embedding endpoint 跑通 30 条 HotpotQA。Embedding Top-k 证据召回率 0.867，RAPTOR 0.900，GraphRAG 0.767，HippoRAG 0.900，SAM-full 0.883，SAM-no-graph 0.867。
- `outputs/runs/hotpotqa300_real_embedding_main_v4_hops1/`：使用公司可用 embedding endpoint 跑通 300 条 HotpotQA。Embedding Top-k 证据召回率 0.877，RAPTOR 0.890，GraphRAG 0.795，HippoRAG 0.882，SAM-full 0.890，SAM-no-graph 0.877。SAM-full 相比 Embedding Top-k 和 SAM-no-graph 多命中 8 个支持证据。
- `outputs/runs/fair_ablation_hotpotqa_300/`：Embedding Top-k 证据召回率 0.572，SAM-full 0.603，SAM-no-graph 0.578。
- `outputs/runs/lexical_isolated_hotpotqa300/`：Embedding Top-k 证据召回率 0.570，SAM-full 0.662，SAM-with-lexical-activation 0.670。
- `outputs/runs/feedback_ablation_hotpotqa_300_isolated/`：Embedding Top-k 证据召回率 0.572，RAPTOR 0.635，GraphRAG 0.562，HippoRAG 0.587，SAM-full 0.603，SAM-no-graph 0.572。

阶段结论：图联想扩展能补回一部分纯向量召回遗漏的支持证据，尤其在 HotpotQA bridge-style 多跳问题中体现更明显。真实 embedding 的 300 条主实验中，SAM-full 与 RAPTOR 的证据召回率同为 0.890，高于 Embedding Top-k、SAM-no-graph、HippoRAG 和 GraphRAG。二跳扩展在真实 embedding 下曾出现噪声路径，因此当前稳定主实验采用一跳局部联想；二跳扩展需要在 RelationJudge 和路径重排加强后再进入主线。

## 3. 状态与反馈驱动的记忆演化

当前系统已经将“记忆会随使用而变化”落到存储和检索逻辑中。检索命中节点后，系统会更新节点使用次数、最近访问时间；路径经过边后，会更新边激活次数；评测阶段会根据支持证据命中、答案命中和无效路径写入反馈事件，并通过 `FeedbackUpdater` 调整后续检索可使用的边权信号。

对应实现：

- `MemoryStore`：记录节点使用状态、边激活状态和事件日志。
- `FeedbackUpdater`：根据检索结果写入反馈事件，并对有效路径和无效路径进行强化或抑制。
- `MemoryConsolidator`：将命中支持证据的问答过程沉淀为长期记忆节点，连接回原始证据。
- 连续记忆复用实验：验证历史记忆是否能被后续查询再次读取。

当前证据：

- `outputs/runs/feedback_ablation_hotpotqa_300_isolated/` 中，SAM-full 写入了节点访问、边经过、证据命中、答案命中和路径拒绝事件。
- `outputs/runs/hotpotqa30_real_embedding_smoke_v2/` 中，SAM-full 写入 `node_retrieved` 120 条、`edge_traversed` 141 条、`support_hit` 53 条、`answer_hit` 24 条、`path_rejected` 66 条、`memory_consolidated` 30 条。
- `outputs/runs/hotpotqa300_real_embedding_main_v4_hops1/` 中，SAM-full 写入 `node_retrieved` 1198 条、`edge_traversed` 898 条、`support_hit` 534 条、`answer_hit` 240 条、`path_rejected` 624 条、`memory_consolidated` 300 条。
- `outputs/runs/agent_memory_reuse_shared_context_hotpotqa300/` 显示共享上下文下存在支持证据增益，说明历史记忆可以被后续流程读取。

阶段结论：动态状态已经不是静态字段，而是会在检索后写回系统，并进入后续排序和共享记忆流程。当前版本还修正了反馈阶段的 embedding 成本问题：长期巩固记忆的向量由命中证据节点向量加权合成，不再为每条反馈额外调用在线 embedding。当前不足是 HotpotQA 独立样本之间共享实体和重复查询有限，反馈机制在单轮指标上与 SAM-full 尚未拉开明显差距；后续应设计同主题连续任务来放大记忆演化效果。

## 4. 多智能体共享记忆机制

当前系统已经实现 planner、retriever、writer、verifier 四角色工作流。不同 agent 不再只通过函数返回值传递信息，而是围绕共享记忆进行写入、读取、版本过滤和冲突裁决。

对应实现：

- planner：写入任务计划和检索目标。
- retriever：写入证据记忆和检索路径。
- writer：读取 retriever 证据并生成候选回答记忆。
- verifier：读取 writer 输出，判断是否存在冲突、污染或需要拒绝的记忆版本。
- `agent_workflow_audit`：统计 handoff 数量、冲突裁决数量、被拒绝记忆是否污染后续上下文等。

当前证据：

- `outputs/runs/agent_workflow_audit_smoke/`：产生 planner、retriever、writer、verifier 多角色记忆，包含 handoff、冲突裁决和 rejected 版本过滤。
- `outputs/runs/agent_memory_reuse_shared_context_hotpotqa300/`：writer/verifier 对共享记忆的读取率为 1.0，说明共享记忆确实被后续 agent 使用。

阶段结论：多智能体共享记忆的底座已完成，可以支撑“证据交接、答案交接、冲突裁决、版本过滤”四类核心行为。下一阶段需要接入 GPT-5.4 做生成式对照，比较无共享记忆、有共享记忆、有共享记忆并带类比提示三种设置。

## 5. 类比推理机制

类比推理目前作为扩展创新点推进。系统已具备长期记忆节点和历史案例检索基础，但关系路径模式匹配、类比提示生成和 GPT-5.4 生成评测还没有形成稳定主实验。

当前定位：

- 输入：历史 consolidated memory 和当前问题的候选路径。
- 目标：从历史问答中找到结构相似的证据组织方式。
- 输出：给 writer 或 verifier 的类比提示，帮助组织多跳证据链。

阶段结论：类比推理已有数据基础，但还不能作为当前最核心实验结论。短期应先完成 3 个可解释类比案例和失败案例，再决定是否进入中期答辩主线。

## 当前优先级

第一优先级是继续巩固 HotpotQA 300 条主实验：围绕一跳联想的成功结果补充典型案例，并进一步优化二跳路径的噪声控制。第二优先级是连续任务实验：让同一主题记忆在多轮查询中反复被激活，验证反馈和记忆状态是否能影响后续排序。第三优先级是 GPT-5.4 多智能体生成对照与类比案例。
