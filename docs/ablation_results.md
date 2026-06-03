# SAM 300 条消融实验记录

本文档记录当前阶段的 HotpotQA 300 条主实验。运行产物位于 `outputs/runs/fair_ablation_hotpotqa_300/`，该目录不进入 Git 仓库。

## 1. 实验设置

- 数据集：HotpotQA dev distractor
- 样本类型：bridge-style 多跳问答
- 查询数量：300
- 候选文档节点数量：2992
- 摘要记忆节点数量：300
- 记忆节点总数：3292
- Gold 支持证据数量：600
- 默认参数：`top-k=4`，`seed-k=1`，`hops=2`
- 统一格式文件：`data/processed/hotpotqa_midterm300_sam_sample.json`

复现命令：

```bash
conda run -n sam python scripts/run_demo.py \
  --reset \
  --db outputs/runs/fair_ablation_hotpotqa_300/sam.sqlite \
  --dataset hotpotqa \
  --dataset-file data/processed/hotpotqa_midterm300_sam_sample.json \
  --rebuild-dataset \
  --sample-size 300 \
  --max-scan 100000 \
  --run-name fair_ablation_hotpotqa_300 \
  --methods embedding_topk,raptor_style,graphrag_style,hipporag_style,sam_full,sam_no_multipath,sam_no_memory_state,sam_no_graph,sam_static_graph,sam_with_summary \
  --top-k 4 \
  --seed-k 1 \
  --hops 2
```

## 2. 方法对比

| 方法 | 证据命中数 | 证据召回率 | 答案命中数 | 答案命中率 |
| --- | ---: | ---: | ---: | ---: |
| Embedding Top-k | 343 | 0.572 | 164 | 0.547 |
| RAPTOR | 381 | 0.635 | 184 | 0.613 |
| GraphRAG | 337 | 0.562 | 164 | 0.547 |
| HippoRAG | 352 | 0.587 | 166 | 0.553 |
| SAM-full | 362 | 0.603 | 179 | 0.597 |
| SAM-no-multipath | 362 | 0.603 | 179 | 0.597 |
| SAM-no-memory-state | 362 | 0.603 | 179 | 0.597 |
| SAM-no-graph | 347 | 0.578 | 166 | 0.553 |
| SAM-static-graph | 362 | 0.603 | 179 | 0.597 |
| SAM-with-summary | 355 | 0.592 | 171 | 0.570 |

## 3. SAM 消融结果

| 方法 | 证据命中数 | 证据召回率 | 答案命中率 | 平均路径长度 | 平均候选路径数 | 平均路径支持分 | 平均边记忆分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAM-full | 362 | 0.603 | 0.597 | 1.75 | 3.94 | 0.709 | 0.000 |
| SAM-no-multipath | 362 | 0.603 | 0.597 | 1.75 | 1.00 | 0.000 | 0.000 |
| SAM-no-memory-state | 362 | 0.603 | 0.597 | 1.75 | 3.99 | 0.710 | 0.000 |
| SAM-no-graph | 347 | 0.578 | 0.553 | 1.00 | 1.25 | 0.000 | 0.000 |
| SAM-static-graph | 362 | 0.603 | 0.597 | 1.75 | 3.99 | 0.710 | 0.173 |
| SAM-with-summary | 355 | 0.592 | 0.570 | 2.41 | 6.93 | 0.749 | 0.164 |

## 4. 阶段结论

SAM-full 相比 Embedding Top-k 多命中 19 个支持证据，证据召回率从 0.572 提升到 0.603，答案命中率从 0.547 提升到 0.597。这个结果说明系统已经不是单纯的向量 top-k，而是可以通过图扩展补充一部分间接证据。

`sam_no_graph` 的平均路径长度为 1.00，证据召回率为 0.578，答案命中率为 0.553，均低于 SAM-full。该差异说明图扩展对最终答案上下文有贡献，尤其在 bridge-style 问题中，单个向量种子无法稳定覆盖完整证据链。

`sam_no_multipath` 与 SAM-full 的差距较小，说明当前多路径信号已经进入排序，但权重还不够强，或者候选图中多条有效路径的区分度不足。后续应重点优化路径支持分的归一化方式和 beam 搜索策略。

`sam_no_multipath`、`sam_no_memory_state` 和 `sam_static_graph` 与 SAM-full 的总体分数相同，说明当前 300 条实验中的主要增益来自“图扩展是否打开”，而多路径累积、记忆状态和静态/动态更新还没有充分拉开差距。下一阶段应加入更明确的时间衰减函数、任务反馈强化和跨查询共激活边。

RAPTOR 在本轮实验中表现最好，证据召回率达到 0.635，说明摘要层级结构对当前 HotpotQA 候选集有效。但 `sam_with_summary` 的结果低于 SAM-full，说明简单的 query-level summary node 会引入噪声。SAM 后续需要更细粒度的摘要节点和路径重排，而不是把所有同题候选直接接到一个摘要中心。

当前最弱环节主要有三点：第一，embedding 表示仍然限制了初始种子的质量；第二，建边 scorer 主要依赖实体、关键词和相似度，尚未利用 LLM 判断关系类型；第三，SAM 重排参数仍是经验设定，需要进一步通过验证集或学习式方法调优。

## 5. 后续改造：摘要记忆节点

基于本轮实验中 RAPTOR 表现较强这一现象，系统已进一步加入 query summary memory node。每个查询候选上下文会生成一个摘要记忆节点，该节点覆盖同一问题下的候选文档标题、关键词和摘要，并通过 `summary_parent`、`summary_child` 两类层级边接入动态记忆图。

该节点不参与 gold evidence 计分，SAM 最终 top-k 默认过滤摘要节点，只允许它作为中间联想路径存在。这样可以避免把摘要当作证据刷分，同时保留摘要层级对多跳扩展的帮助。当前稳定主方法暂不默认启用摘要节点，而是通过 `sam_with_summary` 单独评测。

30 条 HotpotQA smoke run 结果如下：

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.483 | 0.400 |
| RAPTOR | 0.650 | 0.533 |
| GraphRAG | 0.483 | 0.367 |
| HippoRAG | 0.517 | 0.367 |
| SAM-full | 0.517 | 0.400 |
| SAM-with-summary | 0.483 | 0.367 |
| SAM-no-graph | 0.517 | 0.467 |

该 smoke run 的 `cases.json` 中已经出现经过 `summary_*` 节点的检索路径，例如：

```text
种子文档 -> summary_child -> query_summary -> summary_parent -> 候选文档
```

隔离评测后可以看到，摘要节点虽然已经进入图扩展路径，但当前版本会降低 30 条 smoke run 的证据召回率。这说明简单地把同题候选文档全部接到一个摘要节点，会造成过宽的上下文跳转。下一步应保留 `sam_with_summary` 作为实验分支，继续优化摘要节点粒度、summary-edge 权重和路径重排，而不是直接把它并入稳定主方法。

本轮也修正了评测协议：不同方法现在从同一份初始记忆库快照开始运行，避免一个方法的动态图更新影响另一个方法的分数。后续所有消融实验都应采用这个隔离评测方式。

## 6. 反馈机制消融

在事件化动态记忆实现后，系统增加了 `sam_no_feedback`，用于单独关闭评测反馈强化和噪声路径抑制。该模式保留 SAM-full 的检索流程、图扩展和路径重排，但不执行 `FeedbackUpdater`，因此不会写入 `support_hit`、`answer_hit`、`path_rejected` 这类反馈事件，也不会根据结果调整路径边权。

本轮重新修正了评测隔离方式：先保存初始记忆库快照，然后每个方法都从该快照复制独立临时库，避免展示用的 SAM-full 动态状态污染后续方法。隔离后的 300 条反馈消融 run 位于 `outputs/runs/feedback_ablation_hotpotqa_300_isolated/`，主要结果如下：

| 方法 | 证据召回率 | 答案命中率 | 平均路径长度 | 平均候选路径数 | 平均边记忆分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Embedding Top-k | 0.572 | 0.547 | - | - | - |
| RAPTOR | 0.635 | 0.613 | - | - | - |
| GraphRAG | 0.562 | 0.547 | - | - | - |
| HippoRAG | 0.587 | 0.553 | - | - | - |
| SAM-full | 0.603 | 0.597 | 1.75 | 3.94 | 0.000 |
| SAM-no-feedback | 0.603 | 0.597 | 1.75 | 3.94 | 0.000 |
| SAM-no-graph | 0.572 | 0.547 | 1.00 | 1.25 | 0.000 |

该结果说明：在 HotpotQA 当前 300 条设置中，SAM 相比 Embedding Top-k 的提升主要来自图扩展，反馈机制没有直接体现在单轮指标上。原因是当前样本按问题组织，每个问题的候选文档相对独立，查询之间共享节点和共享边较少；反馈边权即使被更新，也很难在后续问题中被再次使用。

不过，反馈机制本身已经产生可检查的事件流。300 条 run 中，SAM-full 写入 `node_retrieved` 1198 条、`edge_traversed` 898 条、`support_hit` 362 条、`answer_hit` 179 条、`path_rejected` 705 条。这表明系统已经能把“哪些节点被访问、哪些路径命中证据、哪些路径没有贡献”记录下来，并用于后续边权调整。下一步需要构造共享实体或同主题连续问答实验，让同一批记忆节点和边在多轮查询中反复被激活，从而检验反馈机制对后续排序的影响。

## 7. Bad Case 驱动的向量锚点改造

HotpotQA 30 条 bad case 显示，部分失败样本属于“图扩展或重排把有效向量候选挤出 top-k”。据此增加了两个实验模式：

- `sam_vector_anchor`：固定保留更多初始向量候选。
- `sam_adaptive_anchor`：根据扩展路径平均支持分动态决定是否保留更多向量候选，并在 `cases.json` 中记录触发原因。

30 条对比 run 位于 `outputs/runs/adaptive_anchor_hotpotqa30_v2/`，结果如下：

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.483 | 0.400 |
| SAM-full | 0.517 | 0.400 |
| SAM-vector-anchor | 0.500 | 0.433 |
| SAM-adaptive-anchor | 0.517 | 0.400 |

`sam_adaptive_anchor` 共触发 36 个 `weak_graph_paths` 检索结果和 84 个 `strong_graph_paths` 检索结果。结果说明，当前自适应锚点机制能够记录并区分路径置信状态，但仅依赖 path support 分数还不能稳定提升答案命中率。后续应把 bad case 中的 `graph_noise` 进一步转化为边质量约束，例如降低弱关键词边、加入实体类型匹配，或使用 GPT-5.4 对候选边进行关系有效性判断。

## 8. 边质量约束改造

针对 bad case 中出现的图噪声问题，系统进一步在 `GraphBuilder` 中增加了低信息关键词过滤。过去只要两个节点共享足够数量的关键词，就可能建立 `keyword_overlap` 边；但在真实任务中，`system`、`report`、`data`、`question`、`evidence` 这类词即使重复出现，也不能说明两个记忆节点之间存在有效语义关系。

改造后，按需建边会先计算 `edge_quality`。如果候选边只依赖低信息关键词重叠，系统会跳过建边，并在候选边打分结果中记录原因。这一改造直接对应当前架构中的弱点：动态图谱不能只追求“多连边”，还必须控制边的语义质量，否则联想扩展会把噪声路径误认为推理链。

30 条 HotpotQA 验证 run 位于 `outputs/runs/edge_quality_hotpotqa30/`，结果如下：

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.483 | 0.400 |
| SAM-full | 0.517 | 0.400 |
| SAM-adaptive-anchor | 0.517 | 0.400 |

本轮指标与改造前基本一致，原因是该 30 条样本中实际创建的 392 条边均被判定为正常边，没有触发低信息关键词过滤。这个结果说明该改造更像是边质量的安全约束和回归防护，尚未解决主要性能瓶颈。

根据该结论，系统已经进一步实现 `RelationJudge` 关系级建边判别接口。后续实验可以通过 `--relation-judge gpt54` 启用 GPT-5.4 关系判别，使候选边在写入动态图谱前经过一次“是否存在真实语义关系”的判断。该模块是下一轮图谱质量实验的核心变量。

## 9. 记忆巩固机制验证

为推进开题计划中的“记忆重构”目标，系统新增 `MemoryConsolidator`。当 SAM 检索命中支持证据时，系统不再只记录访问日志和边权变化，而是生成一个 `consolidated_memory` 长期记忆节点，保存问题、答案、检索方法、答案状态和支持证据摘要，并通过 `consolidates_support` 边连接回原始证据节点。

30 条 HotpotQA 验证 run 位于 `outputs/runs/memory_consolidation_hotpotqa30_v2/`，主要结果如下：

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.483 | 0.400 |
| SAM-full | 0.517 | 0.400 |
| SAM-no-feedback | 0.517 | 0.400 |

该 run 中，`memory_events.json` 记录了 22 条 `memory_consolidated` 事件；图谱 JSON 中包含 22 个 `consolidated_memory` 节点，并生成 31 条 `consolidates_support` 边和 31 条 `support_consolidated_by` 反向边。由于这一步发生在检索反馈之后，它不会直接改变同一轮的证据召回率和答案命中率；它的作用是把已经验证有效的证据链沉淀为后续任务可复用的长期记忆。

随后系统继续补充了巩固记忆复用逻辑：SAM 方法会把已有 `consolidated_memory` 加入后续候选池，但最终 top-k 会过滤巩固节点，只允许它作为中间联想路径出现。`outputs/runs/consolidated_reuse_hotpotqa30/` 显示 30 条 HotpotQA 中仍生成了 22 个巩固节点和 62 条巩固边，但没有出现经过巩固节点的最终路径。这个结果符合 HotpotQA 当前设置的特点：每个问题的候选文档相对独立，不是连续任务场景，前一题巩固出的经验很难被后一题自然复用。下一阶段需要构造连续问答或跨任务复用实验，验证这些巩固记忆是否能提升后续查询效率和类比推理质量。

## 10. 连续记忆复用实验

为验证巩固记忆是否能在后续任务中发挥作用，系统新增连续记忆复用实验脚本 `scripts/run_memory_reuse_experiment.py`。实验分为两个阶段：第一阶段使用正常候选集进行 warmup，使 SAM 生成巩固记忆；第二阶段构造 probe 查询，将每个问题的 gold 支持文档从候选集中移除，再比较 Embedding Top-k 与 SAM 的表现。

30 条 HotpotQA 受控复用实验位于 `outputs/runs/memory_reuse_hotpotqa30/`，结果如下：

| 方法 | 支持证据命中数 | 证据召回率 |
| --- | ---: | ---: |
| Embedding Top-k | 0 | 0.000 |
| SAM | 31 | 0.517 |

Warmup 阶段生成了 22 个巩固记忆节点和 62 条巩固相关边。Probe 阶段中，Embedding Top-k 无法访问被移除的 gold 支持文档，因此支持证据命中数为 0；SAM 则通过巩固记忆中记录的 `support_node_ids` 将历史支持证据带回候选池，最终命中 31 条支持证据。这个结果说明当前系统已经形成“成功检索 -> 记忆巩固 -> 后续候选补全 -> 联想检索”的闭环。

需要注意，该实验是受控连续复用实验，不等同于普通 HotpotQA 排行榜式评测。它验证的是动态记忆系统在信息缺失场景下利用历史经验补全证据的能力。后续应扩展到真实连续任务，例如同一主题多轮问答、跨文档研究任务和多智能体协作任务。

## 11. 类比复用实验

为推进开题计划中的类比推理模块，系统进一步让 `AnalogyEngine` 识别巩固记忆案例，并返回案例答案、支持证据节点、支持证据标题和匹配关系路径。新增脚本 `scripts/run_analogy_reuse_experiment.py` 用于评估 masked probe 查询能否类比到 warmup 阶段形成的成功经验。

30 条 HotpotQA 类比复用实验位于 `outputs/runs/analogy_reuse_hotpotqa30/`，结果如下：

| 指标 | 数值 |
| --- | ---: |
| 查询数量 | 30 |
| Warmup 巩固记忆节点数 | 22 |
| 巩固案例命中数 | 22 |
| 巩固案例命中率 | 0.733 |
| 支持证据重叠命中数 | 24 |
| 支持证据重叠命中率 | 0.800 |

该结果说明，当前系统已经能够从已巩固的成功检索中提取可类比案例，并在后续信息缺失的 probe 查询中找回相同或相关的证据链。8 个未命中巩固案例的样本主要来自 warmup 阶段未生成巩固记忆的查询，说明类比模块的上限仍受基础 SAM 召回和巩固触发条件影响。下一步应优化巩固条件，并在多轮任务中评估类比提示对答案生成质量的影响。

## 12. 多智能体共享记忆复用实验

为推进开题计划中的“多智能体语义记忆共享结构”，系统新增 `scripts/run_agent_memory_reuse_experiment.py`。该实验读取连续记忆复用实验中的 probe cases，检查 SAM 通过巩固记忆补回的支持证据，是否能够继续被 retriever 写入共享记忆，并被 writer 和 verifier 跨角色使用。

30 条 HotpotQA 多智能体共享记忆复用实验位于 `outputs/runs/agent_memory_reuse_hotpotqa30/`，结果如下：

| 指标 | 数值 |
| --- | ---: |
| 查询数量 | 30 |
| Embedding Top-k 支持证据命中数 | 0 |
| SAM 支持证据命中数 | 31 |
| 支持证据增益总数 | 31 |
| 存在支持证据增益的样本数 | 22 |
| writer 使用 retriever 共享记忆次数 | 30 |
| verifier 使用 writer 共享记忆次数 | 30 |
| 多智能体复用链路成功数 | 22 |
| 多智能体复用链路成功率 | 0.733 |

该结果说明，多智能体模块已经从流程级 demo 推进到可量化实验：在信息缺失场景下，SAM 的历史记忆复用增益可以通过共享记忆 handoff 传递到后续智能体角色。当前实验不把本地启发式生成器的答案命中率作为结论，后续需要接入 GPT-5.4，比较无共享记忆、有共享记忆、有共享记忆和类比提示三种设置下的最终答案质量。

## 13. 多智能体生成对照实验

为进一步评估共享记忆是否能作用到最终答案生成阶段，系统新增 `scripts/run_agent_generation_experiment.py`。该脚本在同一批 case 上运行三种设置：

- `baseline`：只使用检索上下文生成答案。
- `shared_memory`：通过 planner、retriever、writer、verifier 的共享记忆流程生成答案。
- `shared_memory_with_analogy`：在共享记忆基础上加入历史案例类比提示。

30 条 HotpotQA smoke run 位于 `outputs/runs/agent_generation_hotpotqa30_smoke/`，结果如下：

| 变体 | 答案命中率 | 平均 prompt token 估计 |
| --- | ---: | ---: |
| baseline | 0.000 | 698.9 |
| shared_memory | 0.000 | 790.2 |
| shared_memory_with_analogy | 0.000 | 923.2 |

该 smoke run 使用本地启发式生成器，目的不是报告模型效果，而是验证三种生成设置、共享记忆注入和类比提示注入是否能稳定生成产物。从 prompt token 估计可以看到，共享记忆和类比提示已经进入生成上下文。正式实验需要使用 GPT-5.4 运行同一脚本，重点比较三种设置的答案命中率、证据引用完整性和 bad case 恢复情况。

## 14. NovelQA 小样本实验与查询扩展消融

为验证系统不只围绕 HotpotQA 的“标题 + 段落”结构运行，当前使用本地 `NovelQA.zip` 的 demonstration 子集构造了 12 条 Frankenstein 长文本问答样本。数据转换输出为 `data/processed/novelqa_demo_eval_sam_sample.json`，包含 120 个小说 chunk、12 个问题和 20 个可映射 gold evidence chunk。

默认原问题检索 run 位于 `outputs/runs/novelqa_demo_eval12_default_query_policy/`，结果如下：

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.045 | 0.000 |
| RAPTOR | 0.045 | 0.000 |
| GraphRAG | 0.136 | 0.083 |
| HippoRAG | 0.045 | 0.000 |
| SAM-full | 0.091 | 0.000 |
| SAM-no-graph | 0.045 | 0.000 |

该结果说明，SAM-full 相比 SAM-no-graph 在 NovelQA 上仍能通过图扩展补回少量 evidence，但整体效果明显低于 HotpotQA。bad case 显示主要问题包括：小说长文本中的同名人物和泛化代词过多，弱关键词边容易把检索带向相邻但无关情节；部分 NovelQA 答案不是原文字符串，当前 answer hit 指标偏严格；同时，本地哈希 embedding 对长文本语义定位能力不足。

为进一步验证 query expansion，系统新增 `metadata.retrieval_query` 和 `--use-retrieval-query` 开关。第一版 NovelQA adapter 曾把原问题、Aspect、Complexity 和全部 Options 写入 `retrieval_query`，但默认不启用。启用全量选项拼接的 run 位于 `outputs/runs/novelqa_demo_eval12_retrieval_query_policy/`，结果如下：

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.000 | 0.000 |
| RAPTOR | 0.000 | 0.000 |
| GraphRAG | 0.000 | 0.000 |
| HippoRAG | 0.000 | 0.000 |
| SAM-full | 0.091 | 0.000 |
| SAM-no-graph | 0.000 | 0.000 |

该对照说明，直接把所有选项文本拼接进查询会引入明显噪声，尤其会干扰 baseline 的相似度排序。因此当前决策是：保留 `retrieval_query` 作为可控实验变量，但主实验默认使用原始 question。

随后系统将 NovelQA `retrieval_query` 改为启发式 query plan：只保留原问题、问题关键词、Aspect 和 Complexity，不再拼接全部选项答案。启发式 query plan run 位于 `outputs/runs/novelqa_demo_eval12_query_plan/`，结果如下：

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.071 | 0.000 |
| RAPTOR | 0.071 | 0.000 |
| GraphRAG | 0.286 | 0.000 |
| HippoRAG | 0.071 | 0.000 |
| SAM-full | 0.143 | 0.083 |
| SAM-no-graph | 0.071 | 0.000 |

启发式 query plan 比全量 options 拼接更稳定，SAM-full 保持了图扩展带来的证据召回和答案命中优势；但它仍低于默认原始 question 下 GraphRAG 的证据召回。因此该策略暂时作为消融变量保留，后续应使用 GPT-5.4 生成更精细的 query plan，例如只保留角色实体、事件触发词和需要验证的关系。

## 15. NovelQA 弱关键词边过滤实验

NovelQA bad case 显示，小说文本中大量代词、助动词和结构词会造成虚假的关键词边，例如 `she`、`her`、`had`、`his`、`chunk`、`Frankenstein`。这些词在同一本小说的许多 chunk 中频繁出现，但并不能说明两个记忆节点存在有效语义关系。为此，系统进一步增强了停用词和建边过滤逻辑：

- 关键词抽取阶段过滤常见代词、助动词和弱连接词。
- 建边阶段过滤 `chunk`、`chapter`、`letter` 等长文本切块结构词。
- 如果两个节点来自同一本书，则该书名本身不再作为关键词边依据。

过滤后的 NovelQA 12 条 run 位于 `outputs/runs/novelqa_demo_eval12_edge_filter/`，结果如下：

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.143 | 0.000 |
| RAPTOR | 0.143 | 0.000 |
| GraphRAG | 0.357 | 0.083 |
| HippoRAG | 0.143 | 0.000 |
| SAM-full | 0.143 | 0.083 |
| SAM-no-graph | 0.143 | 0.000 |

与上一轮默认检索相比，过滤后 SAM-full 的答案命中率从 0.000 提升到 0.083，并且不再明显低于 SAM-no-graph。该结果说明弱关键词边过滤能够缓解一部分图扩展噪声，但还没有解决 NovelQA 的核心困难。GraphRAG 仍然表现最好，说明长文本小说问答更依赖实体级局部图和更强语义表示。后续应继续引入实体消歧、GPT-5.4 关系判别和正式 embedding 模型。

## 16. QueryPlanner 运行时查询规划

为避免查询扩展逻辑固化在 NovelQA adapter 中，系统新增 `QueryPlanner` 模块。该模块在 Evaluator 调用检索器之前运行，输出 `retrieval_query`、关键词、实体和规划原因，并写入每条样本的 `cases.json`。这样同一份 SAM 数据集可以在不重新转换数据的情况下，直接对比原始问题、数据集静态 `retrieval_query`、启发式 QueryPlanner 和 GPT-5.4 QueryPlanner。

当前启发式 QueryPlanner 会使用原始问题、问题关键词、Aspect 和 Complexity，但不会把所有候选选项拼接进检索文本。本轮 smoke run 位于 `outputs/runs/query_planner_smoke/`，使用 `data/processed/novelqa_demo_eval_sam_sample.json` 的 12 条 Frankenstein demonstration 样本，命令如下：

```bash
conda run -n sam python scripts/run_demo.py \
  --reset \
  --dataset novelqa \
  --dataset-file data/processed/novelqa_demo_eval_sam_sample.json \
  --methods embedding_topk,sam \
  --top-k 3 \
  --seed-k 1 \
  --hops 2 \
  --query-planner heuristic
```

结果为：Embedding Top-k 证据召回率 0.071、答案命中率 0.000；SAM 动态联想检索证据召回率 0.071、答案命中率 0.083。该结果说明在当前 NovelQA 小样本中，查询规划模块已经能稳定进入评测闭环，SAM 的图扩展仍能带来少量答案覆盖增益。但证据召回没有明显提升，说明启发式规划还不足以解决长文本小说中的实体消歧和事件定位问题。下一步应使用 GPT-5.4 QueryPlanner 生成更细粒度的角色实体、事件触发词和关系约束，再与 Qwen3-Embedding 或公司 embedding 接口结合重跑同一套实验。

## 17. PathReranker 权重配置化

为支持 bad case 后的可控架构调整，系统将 SAM 路径重排权重进一步配置化。当前 `PathReranker` 支持四种 profile：`balanced`、`semantic_heavy`、`graph_heavy`、`memory_heavy`。它们分别提高语义相似度、图路径、多路径和历史记忆状态在最终排序中的权重。运行脚本新增 `--reranker-profile` 参数，每条 SAM 检索结果会在 `cases.json` 中记录实际使用的 profile 和 `score_breakdown`。

smoke run 位于 `outputs/runs/reranker_profile_smoke/`，命令如下：

```bash
conda run -n sam python scripts/run_demo.py \
  --reset \
  --dataset builtin \
  --methods embedding_topk,sam_full \
  --top-k 2 \
  --seed-k 1 \
  --hops 2 \
  --reranker-profile graph_heavy
```

该 run 中 Embedding Top-k 与 SAM-full 的证据召回率均为 0.667，答案命中率均为 0.667。该结果不用于证明方法收益，主要验证 profile 已进入完整运行链路。后续在 HotpotQA 300 条和 NovelQA 上可分别对比 `semantic_heavy` 与 `graph_heavy`：如果 bad case 主要来自弱图边和错误扩展，应提升语义权重；如果主要来自间接证据漏召回，则应提升图路径和多路径支持权重。

进一步新增 `scripts/run_reranker_profile_experiment.py`，用于一次性比较多个 profile。HotpotQA 8 条 smoke run 位于 `outputs/runs/reranker_profile_hotpotqa8_smoke/`，命令如下：

```bash
conda run -n sam python scripts/run_reranker_profile_experiment.py \
  --dataset-file data/processed/hotpotqa_sam_sample.json \
  --limit 8 \
  --profiles balanced,semantic_heavy,graph_heavy,memory_heavy \
  --run-name reranker_profile_hotpotqa8_smoke
```

本次 8 条 smoke 中四种 profile 的证据召回率均为 0.625，答案命中率均为 0.750，脚本按 tie-break 选择 `graph_heavy`。这说明在小样本和当前本地 embedding 下，profile 权重还没有形成显著差异；但实验产物已经包含每种 profile 的指标、平均路径长度和 bad case 类型统计。下一步应在 HotpotQA 300 条和 NovelQA demonstration 上复跑，以观察不同 profile 对图噪声和间接证据召回的影响。

HotpotQA 300 条正式 profile 对比 run 位于 `outputs/runs/reranker_profile_hotpotqa300/`。实验设置为 top-k=4、seed-k=1、hops=2、方法为 `sam_full`，候选文档节点数量为 2992。结果如下：

| Profile | 证据召回率 | 答案命中率 | 平均路径长度 | Bad case 数量 |
| --- | ---: | ---: | ---: | ---: |
| balanced | 0.482 | 0.567 | 1.98 | 217 |
| semantic_heavy | 0.522 | 0.617 | 1.88 | 215 |
| graph_heavy | 0.490 | 0.593 | 2.04 | 212 |
| memory_heavy | 0.440 | 0.550 | 2.10 | 236 |

该实验显示，`semantic_heavy` 相比原默认 `balanced` 证据召回率提升 0.040，答案命中率提升 0.050；`memory_heavy` 明显下降，说明当前阶段的历史记忆和边激活信号还不能承担过高排序权重。Bad case 统计中，所有 profile 的主要失败类型仍集中在 `missing_support_evidence` 和 `graph_noise`，其中 `memory_heavy` 的图噪声和缺失证据数量最多。这说明 SAM 当前最需要控制的是噪声路径进入 top-k，而不是进一步放大历史激活。

基于该结果，系统默认 reranker profile 已从 `balanced` 调整为 `semantic_heavy`。该调整使主流程更保守地依赖语义相似度，同时仍保留图路径和多路径支持作为补充信号。后续 NovelQA 长文本实验仍需要单独复跑 profile 对比，因为小说场景可能更依赖实体关系和长程图路径。

## 18. 过密路径惩罚与长文本图噪声控制

NovelQA profile 对比显示，长文本小说场景中的平均候选路径数明显高于 HotpotQA。以 `outputs/runs/reranker_profile_novelqa12/` 为例，`semantic_heavy` 的平均候选路径数为 19.67，`memory_heavy` 达到 30.50。大量路径并不一定代表强证据，反而可能来自同一本小说中相邻 chunk、人物代词、弱关键词和结构词造成的 hub 噪声。

为此，`PathReranker` 新增 `path_noise_penalty`：当某个候选节点由过多非种子路径同时到达时，系统会对最终分数进行惩罚，避免“路径数量多”被直接解释为“证据支持强”。该惩罚从 12 条候选路径后开始生效，最高扣分 0.12，并写入每条命中的 `score_breakdown`。

加入过密路径惩罚后，HotpotQA 300 条回归实验位于 `outputs/runs/reranker_profile_hotpotqa300_noise_penalty/`，结果如下：

| Profile | 证据召回率 | 答案命中率 | 平均候选路径数 | Bad case 数量 |
| --- | ---: | ---: | ---: | ---: |
| balanced | 0.485 | 0.567 | 4.29 | 216 |
| semantic_heavy | 0.523 | 0.620 | 4.08 | 215 |
| graph_heavy | 0.490 | 0.597 | 4.29 | 211 |
| memory_heavy | 0.440 | 0.550 | 4.09 | 236 |

与未加入惩罚前相比，HotpotQA 上没有出现负向影响，`semantic_heavy` 仍为最优，证据召回率从 0.522 小幅提升到 0.523，答案命中率从 0.617 小幅提升到 0.620。

NovelQA 12 条回归实验位于 `outputs/runs/reranker_profile_novelqa12_noise_penalty/`，结果如下：

| Profile | 证据召回率 | 答案命中率 | 平均候选路径数 | Bad case 数量 |
| --- | ---: | ---: | ---: | ---: |
| balanced | 0.214 | 0.000 | 9.56 | 12 |
| semantic_heavy | 0.214 | 0.000 | 7.96 | 12 |
| graph_heavy | 0.286 | 0.000 | 10.54 | 12 |
| memory_heavy | 0.071 | 0.083 | 13.33 | 12 |

NovelQA 上，过密路径惩罚显著降低了平均候选路径数，并使 `graph_heavy` 的证据召回率达到 0.286。但答案命中率仍然很低，说明当前主要瓶颈已经不只是图路径排序，而是长文本 chunk 定位、答案表述匹配和生成式答案判断。后续 NovelQA 实验需要结合更强 embedding、GPT-5.4 QueryPlanner、实体消歧和生成式答案评估。

## 19. 生成式答案判别接口

为解决 NovelQA 等长答案场景中“检索证据命中但答案字符串不完全一致”的问题，系统新增 `AnswerJudge` 接口，并接入 `scripts/generate_answers.py`。当前支持两种判别方式：

- `rule`：默认本地规则，检查标准答案字符串包含关系和关键内容词覆盖。
- `gpt54`：通过聊天模型判断生成答案是否和标准答案语义等价，适合长答案、选项题和表述变化较大的样本。

生成结果中的每条样本都会写入 `answer_judgment`，包含 `answer_hit`、`status`、`score`、`reason` 和判别器元信息。这样后续 bad case 分析可以区分两类问题：第一，检索上下文确实没有覆盖答案；第二，检索上下文存在答案依据，但字符串级指标无法识别语义等价。

本地 smoke run 位于 `outputs/runs/answer_judge_smoke/`，命令如下：

```bash
conda run -n sam python scripts/generate_answers.py \
  --cases-file outputs/runs/default_semantic_reranker_smoke/cases.json \
  --method sam_full \
  --chat-provider heuristic \
  --answer-judge rule \
  --limit 2 \
  --output-dir outputs/runs/answer_judge_smoke
```

该 run 用本地规则判别器验证链路，输出 `generated_answers.json` 和 `generated_answers.md`。正式实验中可以把 `--chat-provider` 和 `--answer-judge` 同时切换为 GPT-5.4 配置，用于评估 SAM 检索上下文支持下的最终答案质量。

随后系统进一步加入 `GenerationBadCaseAnalyzer`，生成脚本会自动输出 `generation_bad_cases.json` 和 `generation_bad_cases.md`。本地 smoke run 位于 `outputs/runs/generation_badcase_smoke_fixed/`，其中启发式生成器在证据不足时返回“证据不足”，不再误把 system prompt 当作答案。生成 bad case 报告会把失败样本归为 `generated_answer_not_equivalent`、`judge_low_confidence` 和 `context_available_but_generation_failed` 等类型，便于区分检索召回问题和答案生成问题。

## 20. 检索-生成-判别端到端实验入口

为减少正式实验中手动串联脚本造成的配置误差，系统新增 `scripts/run_end_to_end_experiment.py`。该脚本在同一个 run 目录内完成检索评测、答案生成、答案判别和生成 bad case 分析，并输出 `pipeline_summary.json` 与 `pipeline_summary.md`。

本地 smoke run 位于 `outputs/runs/end_to_end_smoke/`，命令如下：

```bash
conda run -n sam python scripts/run_end_to_end_experiment.py \
  --dataset-file data/processed/hotpotqa_sam_sample.json \
  --limit 3 \
  --retrieval-methods embedding_topk,sam_full \
  --generation-method sam_full \
  --chat-provider heuristic \
  --answer-judge rule \
  --top-k 2 \
  --seed-k 1 \
  --hops 2 \
  --run-name end_to_end_smoke
```

该 run 的检索阶段中，Embedding Top-k 与 SAM-full 的证据召回率均为 0.500，检索答案命中率均为 0.667；生成阶段由于使用本地启发式生成器，答案命中率为 0.000。该结果不用于汇报模型效果，主要验证端到端产物完整性。run 目录中同时包含 `metrics.json`、`cases.json`、`generated_answers.json`、`generation_bad_cases.json`、`pipeline_summary.json` 和 `pipeline_summary.md`。后续正式实验可以在该入口中切换 Azure embedding、GPT-5.4 生成和 GPT-5.4 答案判别。

端到端入口随后补充了 `--query-planner`、`--relation-judge` 和 `--reranker-profile` 参数。高级参数 smoke run 位于 `outputs/runs/end_to_end_advanced_smoke/`，其中 `--query-planner heuristic` 成功写入 `cases.json` 的 `query_plan`，`--reranker-profile graph_heavy` 成功写入 SAM 命中结果。该 run 说明正式实验入口已经能够同时控制查询规划、建边判别、路径重排、生成模型和答案判别器。

## 21. 桥接节点按需建边回归实验

实验审计模块显示，前一阶段本地 smoke 的主要瓶颈是 `missing_support_evidence` 和 `weak_graph_gain`。进一步检查路径后发现，旧版 SAM 只在检索开始时围绕初始向量种子节点建边；当联想检索扩展到桥接节点后，不会继续围绕该桥接节点按需建边。这会影响典型 bridge-style 多跳问题中的路径：

```text
问题相关种子文档 -> 桥接实体文档 -> 答案证据文档
```

因此系统将按需建图机制改为“沿联想路径逐步触发”：初始阶段仍只围绕少量种子节点建边；BFS 扩展到新的桥接节点后，如果该节点本轮尚未触发过建边，则围绕该节点继续补边。同时，`PathReranker` 增加强图路径语义保底，避免高置信二跳路径因为答案文档和原问题表面语义不直接相似而被负向量相似度压低。

HotpotQA 30 条回归 run 位于 `outputs/runs/bridge_expansion_hotpotqa30/`，实验设置为 `top-k=4`、`seed-k=1`、`hops=2`，并比较 Embedding Top-k、SAM-full 和 SAM-no-graph。结果如下：

| 方法 | 证据命中数 | 证据召回率 | 答案命中数 | 答案命中率 |
| --- | ---: | ---: | ---: | ---: |
| Embedding Top-k | 30 | 0.500 | 17 | 0.567 |
| SAM-full | 37 | 0.617 | 20 | 0.667 |
| SAM-no-graph | 30 | 0.500 | 17 | 0.567 |

该结果说明，桥接节点按需建边后，SAM-full 相比 Embedding Top-k 和 SAM-no-graph 多命中 7 条支持证据，证据召回率提升 0.117，答案命中率提升 0.100。平均路径长度提升到 2.48，说明更多结果确实通过多跳路径进入最终候选，而不是停留在一跳向量召回。

审计报告 `experiment_audit.md` 仍识别出两个主要瓶颈：`graph_noise` 和 `missing_support_evidence`。30 条样本中仍有 19 个检索 bad case 缺失支持证据，且 19 个 bad case 包含图噪声。这说明桥接按需建边已经有效提升图扩展召回，但下一阶段重点应转向图边质量控制：使用 GPT-5.4 RelationJudge 过滤弱关键词边和偶然相似边，或降低 `context_cooccurrence` 与弱 `embedding_similarity` 边在二跳路径中的权重。
