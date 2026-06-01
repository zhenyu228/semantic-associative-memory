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
