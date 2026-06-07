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

## 4.1 候选集隔离与初始词项激活修正

在后续实验中发现，早期 `sam_full` 会在普通单轮 HotpotQA 评测中复用历史 `consolidated_memory` 候选。这一行为适合连续记忆复用和类比实验，但不适合独立样本评测，因为它会把上一条样本形成的长期记忆注入下一条样本，导致普通检索指标受到跨样本状态污染。当前已将该逻辑收窄：只有显式启用类比或记忆复用的 SAM 方法才会把巩固记忆加入候选池，`sam_full` 默认只在当前样本候选集内进行按需建图和联想扩展。

同时，系统新增 `sam_with_lexical_activation`，用于测试“初始激活阶段”是否应补充问题词项、标题短语和实体线索。该信号只用于选择初始种子节点，并在最终解释中记录为 `initial_lexical_activation_score`；它不作为最终重排分数直接加权，避免把词面匹配变成不可解释的结果偏置。

隔离后的 300 条 HotpotQA clean run 位于 `outputs/runs/lexical_isolated_hotpotqa300/`，实验规模如下：

| 项目 | 数值 |
| --- | ---: |
| 查询数量 | 300 |
| 候选文档数量 | 3592 |
| Gold 支持证据数量 | 600 |
| 参数 | `top-k=4, seed-k=1, hops=2` |

主要结果如下：

| 方法 | 支持证据命中数 | 证据召回率 | 答案命中数 | 答案命中率 | 平均路径长度 | 平均候选路径数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Embedding Top-k | 342 | 0.570 | 194 | 0.647 | 1.00 | 1.00 |
| SAM-full | 397 | 0.662 | 225 | 0.750 | 2.42 | 15.61 |
| SAM-with-lexical-activation | 402 | 0.670 | 226 | 0.753 | 2.42 | 15.59 |

该结果说明，在候选集隔离后，SAM-full 相比 Embedding Top-k 多命中 55 条支持证据，证据召回率从 0.570 提升到 0.662，答案命中率从 0.647 提升到 0.750。新增的初始词项激活进一步多命中 5 条支持证据，说明问题词项、实体和标题短语在选择种子节点时有帮助，但增益幅度有限，不能替代后续图扩展和路径重排。

这次修正也明确了两个实验边界：普通 HotpotQA 独立样本实验用于验证 SAM 的按需建图和联想扩展；连续记忆复用实验才用于验证 `consolidated_memory`、历史经验和类比案例是否能跨任务发挥作用。后续所有主实验都应保持这两个设置分离。

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

根据该结论，系统已经进一步实现 `RelationJudge` 关系级建边判别接口。运行时可以通过 `--relation-judge cached_gpt54` 启用带缓存的 GPT-5.4 关系判别，使候选边在写入动态图谱前经过一次“是否存在真实语义关系”的判断。判别结果会缓存到 `outputs/cache/relation_judge_cache.json`，避免同一候选边在多轮实验中重复调用模型。

当前已完成一次低预算 GPT-5.4 关系判别 smoke。5 条 HotpotQA query-limit run 位于 `outputs/runs/relation_judge_gpt54_querylimit5_smoke/`，配置为 `relation-judge-policy=risky`、`relation-judge-max-calls=3`。由于已有缓存覆盖了该批候选边，本次运行实际产生 96 次 cache hit、0 次新模型调用；这说明关系判别缓存已经可以参与图构建流程，并避免重复消耗模型额度。

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.600 | 0.600 |
| SAM-full + RelationJudge | 0.700 | 0.800 |

同时还保留了一次 300 条低预算 run：`outputs/runs/relation_judge_gpt54_tiny_smoke/`。该 run 在 300 条 HotpotQA 上使用 `max_calls=3`，实际 3 次 GPT-5.4 调用、249 次预算耗尽跳过、10242 次缓存命中。指标为 Embedding Top-k 证据召回率 0.570、答案命中率 0.647；SAM-full 证据召回率 0.637、答案命中率 0.720。该结果证明 RelationJudge 已能进入正式实验链路，但由于调用预算极低，还不能作为完整关系判别主实验结论。

## 8.1 RelationJudge 策略对照与预算耗尽修正

在扩大到 30 条对照时发现一个实现问题：当 RelationJudge 调用预算耗尽且策略为 `skip` 时，系统虽然保留候选边，但会把边类型写成 `budget_exhausted`。这会污染图谱关系类型，使后续路径重排无法区分原本的 `keyword_overlap`、`embedding_similarity` 或 `shared_entity`。当前已修正为：预算耗尽时保留原始候选关系类型，只在判别原因中记录预算耗尽；旧缓存中的 `budget_exhausted` 结果也会在读取时自动归一化。

修正后运行 30 条 HotpotQA query-limit 对照，固定使用 local embedding、`top-k=4`、`seed-k=1`、`hops=2`：

| 设置 | Run 目录 | 证据召回率 | 答案命中率 | 支持证据命中数 | 建边数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 不启用 RelationJudge | `outputs/runs/relation_compare_disabled_q30/` | 0.617 | 0.667 | 37 | 1620 |
| `risky` + budget 20 | `outputs/runs/relation_compare_risky_q30_budget20_fixed/` | 0.633 | 0.700 | 38 | 1620 |
| `all` + budget 20 | `outputs/runs/relation_compare_all_q30_budget20_fixed/` | 0.600 | 0.667 | 36 | 1608 |

对照结果说明：`risky` 策略更适合当前阶段。它只对高风险的关键词重叠边和 embedding 相似边走关系判别，保留共享实体边的确定性，因此在 30 条实验中比无判别多命中 1 条支持证据，答案命中率提高 0.033。`all` 策略会对共享实体边也做判别，在低预算和缓存覆盖不完整时反而减少有效边，导致召回下降。

边日志也支持这一判断。无判别设置创建 726 条 `shared_entity` 边、542 条 `keyword_overlap` 边和 352 条 `embedding_similarity` 边；修正后的 `risky` 设置保持相同关系类型分布，不再出现 `budget_exhausted` 边。`all` 设置只创建 714 条 `shared_entity` 边，说明严格判别会过滤掉一部分原本有用的实体边。

因此，当前默认策略仍应保持 `risky`，并把 `all` 作为边质量审计或高预算实验分支，而不是默认主方法。下一步若要把 RelationJudge 纳入 300 条正式结论，需要保证两点：第一，调用预算足够覆盖高风险边；第二，RelationJudge 返回的拒绝边能够在 bad case 中证明确实是噪声，而不是误删桥接证据。

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

## 22. 弱关系二跳路径惩罚

桥接节点按需建边后，SAM 能覆盖更多二跳证据，但审计结果也显示图噪声仍然较高。进一步检查 bad case 后发现，一部分错误路径来自二跳中的弱 `embedding_similarity`、`context_cooccurrence` 或低质量 `keyword_overlap` 边。这类边在一跳召回中可以作为候选补充，但如果出现在二跳路径末端，容易把主题相近但不能回答问题的文档推入 top-k。

因此 `PathReranker` 新增 `weak_relation_penalty`：当候选结果通过二跳弱关系边到达时，系统会对最终排序分进行小幅扣分，并在 `score_breakdown` 中记录该项。`shared_entity` 等强实体关系不受该惩罚影响。该机制不是直接删除弱边，而是在保留探索能力的同时降低弱路径对最终排序的影响。

HotpotQA 30 条回归 run 位于 `outputs/runs/weak_relation_penalty_hotpotqa30/`。在同样的 `top-k=4`、`seed-k=1`、`hops=2` 设置下，结果如下：

| 方法 | 证据命中数 | 证据召回率 | 答案命中数 | 答案命中率 |
| --- | ---: | ---: | ---: | ---: |
| Embedding Top-k | 30 | 0.500 | 17 | 0.567 |
| SAM-full | 37 | 0.617 | 20 | 0.667 |
| SAM-no-graph | 30 | 0.500 | 17 | 0.567 |

本轮指标与桥接扩展回归保持一致，说明弱关系惩罚没有损伤当前已获得的召回收益。审计报告仍显示 `graph_noise` 和 `missing_support_evidence` 是主要瓶颈，因此后续需要继续推进两类改造：第一，接入更强 embedding 模型提升初始种子质量；第二，使用 GPT-5.4 RelationJudge 对候选边进行语义关系判别，从建边阶段减少噪声。

## 23. GPT-5.4 SDK 低额度端到端 smoke

为对齐公司网关的 SDK 调用方式，系统新增 `azure_openai_sdk` 聊天模型 provider，通过 OpenAI SDK 的 `AzureOpenAI` 接入 GPT-5.4。该 provider 支持从官方 baseline 使用的 `GPT54_API_KEY`、`GPT54_BASE_URL`、`GPT54_API_VERSION`、`GPT54_MODEL` 自动映射到 SAM 的 `SAM_AZURE_CHAT_*` 配置。

配置级诊断已验证：`azure_openai_sdk` provider 可以读取本地 official baseline env，并在最小 probe 中返回 `2`。随后运行 1 条 HotpotQA 低额度端到端 smoke，路径为 `outputs/runs/provider_smoke_gpt54_sdk_hotpotqa1/`。该 run 使用 local embedding、GPT-5.4 SDK 生成、规则答案判别，结果如下：

| 指标 | 数值 |
| --- | ---: |
| 查询数量 | 1 |
| 文档数量 | 300 |
| Embedding Top-k 证据召回率 | 0.500 |
| SAM-full 证据召回率 | 0.500 |
| 生成答案命中率 | 0.000 |

该样本的问题是：`What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?`，标准答案为 `Chief of Protocol`。GPT-5.4 生成答案为 `United States Ambassador to Ghana`，说明模型根据上下文识别出 Shirley Temple，但检索上下文没有覆盖其担任 `Chief of Protocol` 的支持证据。将 `top-k` 提升到 4、`hops` 提升到 2 后仍未补回完整证据链，bad case 被归因为 `missing_support_evidence`、`answer_not_covered` 和 `graph_noise`。

该结果说明，GPT-5.4 生成链路已经进入实验闭环，但当前瓶颈仍在检索侧：local embedding 初始召回不足，图扩展没有稳定补回第二条支持证据。下一步应优先接入正式 embedding，并使用 GPT-5.4 RelationJudge 对候选边进行关系有效性判别，再重跑同一条样本和 HotpotQA 300 条主实验。

## 24. 类比案例进入检索排序层

为继续推进开题计划中的类比推理模块，系统新增 `SAM-with-analogy` 检索模式。旧版本的类比主要有两类作用：一是从历史巩固记忆中找相似案例，二是在生成阶段把历史案例写入提示词。这个流程可以解释“当前问题像哪个历史问题”，但历史成功案例还没有直接参与检索排序。

新模式将历史巩固记忆中的 `support_node_ids` 转换为当前检索中的类比路径信号：当 `AnalogyEngine` 判断当前问题与某个历史成功案例相似时，`Retriever` 会把该历史案例对应的支持证据节点加入候选路径，并在 `score_breakdown` 中记录 `analogy_component`。最终结果会保留 `analogy_case_id`、`analogy_support_node_id` 和类比提示文本，便于追踪这个证据为什么被召回。

该改造使类比推理从“生成提示层”前移到“检索排序层”。它对应的直接收益不是让模型多看到一段提示，而是让系统能够复用历史证据链，把过去已经验证过的支持证据作为当前问题的候选路径。当前已补充单元测试 `test_sam_with_analogy_reuses_consolidated_support_as_retrieval_signal`，验证历史巩固案例的支持证据可以通过 `SAM-with-analogy` 被召回，并且结果中保留类比来源解释。

随后运行 HotpotQA 30 条 smoke，run 位于 `outputs/runs/analogy_retrieval_smoke/`，比较 Embedding Top-k、SAM-full 和 SAM-with-analogy。结果如下：

| 方法 | 证据命中数 | 证据召回率 | 答案命中数 | 答案命中率 |
| --- | ---: | ---: | ---: | ---: |
| Embedding Top-k | 29 | 0.483 | 15 | 0.500 |
| SAM-full | 33 | 0.550 | 19 | 0.633 |
| SAM-with-analogy | 34 | 0.567 | 19 | 0.633 |

这个结果说明，在当前 30 条连续评测设置中，类比支持证据注入比 SAM-full 多命中 1 条支持证据，但答案命中数暂时没有变化。原因是该实验仍然以 HotpotQA 独立样本为主，历史案例之间的可复用性有限；类比模块已经能进入检索排序层，但它更适合在连续任务、同主题追问或跨任务经验复用场景中验证。

下一步需要将 `SAM-with-analogy` 加入连续任务实验，与 `SAM-full`、`SAM-no-graph` 和多智能体共享记忆方法一起比较。重点指标应包括类比支持证据命中数、答案命中率变化、错误类比比例，以及类比路径是否降低多跳证据缺失。

## 25. 多智能体共享记忆冲突裁决

为补齐开题计划中“多智能体语义记忆协调机制”的协作控制部分，`SharedMemoryCoordinator` 新增冲突裁决和版本统计能力。此前系统可以让 planner、retriever、writer、verifier 在全局洞察层、会话层和交互层写入共享记忆，也可以按目标 agent 查询 handoff；但当两个角色对同一任务给出不一致结论时，系统没有明确记录哪个版本被采纳、哪个版本被废弃。

新增接口 `resolve_conflict` 会读取同一任务下的候选记忆节点，按置信度和版本号选择当前采用版本，并写入一个 `agent_conflict_resolution` 记忆节点。该节点记录冲突主题、候选节点、采纳节点、废弃节点和裁决 agent。原候选记忆也会被更新为 selected 或 rejected，并保留 `resolved_by_node_id`，从而形成可追踪的版本链。

同时新增 `collaboration_metrics`，用于统计某个 session 或 task 内的共享记忆数量、handoff 数、冲突裁决数量、最大版本号和参与 agent 数。这一步让多智能体协作不只是“共享了几段文本”，而是能够追踪任务过程中的角色分歧、版本演化和裁决结果。

当前已补充单元测试 `test_shared_memory_coordinator_resolves_conflicting_handoffs_with_versions`，验证两个 agent 给 writer 的冲突 handoff 可以被 verifier 裁决，且指标能正确统计 handoff 数、冲突裁决数、最大记忆版本和参与 agent 数。下一步需要把该机制接入 `MultiAgentResearchWorkflow` 的完整运行结果中，让 workflow 报告自动输出冲突案例和协作效率指标。

随后系统将 `collaboration_metrics` 接入 `MultiAgentResearchWorkflow`。每条 case 运行结束后，workflow 会输出该任务内的共享记忆数量、handoff 数、冲突裁决数、最大版本号和参与 agent 列表。报告表格也新增 Handoff 数、冲突裁决数和最大版本。

5 条 HotpotQA workflow smoke 位于 `outputs/runs/agent_workflow_metrics_smoke/`，使用 `SAM-with-analogy` 检索结果和本地启发式生成器。该 run 的生成验证通过率为 0.000，原因仍是启发式生成器不能代表正式 GPT-5.4 生成能力；但每条样本都形成了完整 planner -> retriever -> writer -> verifier 流程，并记录 4 条共享记忆、2 次 handoff、最大版本号 4、参与 agent 数 4。该结果说明多智能体协作轨迹和版本指标已经进入完整实验产物。下一步应构造真实冲突任务集，使 `resolve_conflict` 在 workflow 中被自动触发，再比较冲突裁决前后的答案质量和协作效率。

在此基础上，workflow 进一步加入自动冲突裁决：当 writer 生成答案未通过 verifier 检查时，系统会把 retriever handoff 和 writer handoff 作为候选记忆，由 verifier 写入一次 `agent_conflict_resolution` 裁决节点。5 条 HotpotQA 自动冲突 smoke 位于 `outputs/runs/agent_workflow_conflict_smoke/`，每条样本都记录 5 条共享记忆、2 次 handoff、1 次冲突裁决、最大版本号 5、参与 agent 数 4。该结果说明冲突裁决不再只是 coordinator 的手动接口，而是已经进入完整多智能体 workflow。后续需要把本地启发式生成器替换为 GPT-5.4，并构造包含多角色证据分歧的任务集，评估自动裁决是否能降低错误答案传播。

## 26. Embedding 正式实验请求量规划

为避免正式 embedding 实验直接消耗较多额度，系统新增 `scripts/plan_embedding_run.py`。该脚本只读取 SAM 统一数据文件和本地 SQLite embedding cache，不实例化在线 provider，也不发送网络请求。它复用当前 ingest 的文本构造方式，统计文档记忆节点和 query summary 节点分别需要多少 embedding，并根据 cache 命中情况估算还需要请求多少唯一文本和多少 batch。

HotpotQA 30 条样本的规划结果位于 `outputs/plans/hotpotqa_embedding_plan/`。在 `azure_openai_sdk`、`batch_size=16`、未指定 cache 的设置下，结果为：文档 embedding 文本数 300，summary embedding 文本数 30，唯一文本数 330，缓存命中数 0，预计需要请求文本数 330，预计 batch 数 21。该结果说明正式重跑 HotpotQA 30 条并不会只请求 300 个段落，还会额外请求 30 个查询上下文摘要节点。后续正式 300 条实验前，需要先开启 `SAM_EMBEDDING_CACHE_PATH`，并用该计划确认缓存命中和预计 batch 数，再决定是否扩大规模。

随后新增 `scripts/warm_embedding_cache.py`，用于按相同文本构造方式预热 embedding cache。该脚本会先生成预热前计划，只对缺失文本调用 provider，并在结束后重新统计缓存命中。使用 local provider 对 HotpotQA 30 条样本做 smoke，第一次预热写入 330 个文本，预热后缺失文本数为 0；第二次使用同一个 cache 重新运行时，本次写入文本数为 0，说明缓存复用逻辑有效。正式接入 `azure_openai_sdk` 时可以复用同一脚本，只需将 provider 和 env 文件切换为正式配置。

## 27. 图边质量审计

为进一步定位 `graph_noise`，系统新增 `scripts/audit_edge_quality.py` 和 `sam.edge_audit`。该审计读取 `cases.json` 中每个命中结果的 `candidate_paths`，按关系类型统计它们出现在支持证据路径和非支持证据路径中的次数，并计算噪声率。它不依赖 gold 图，只使用公开数据集中的 supporting evidence 标注和系统实际检索路径。

在 `outputs/runs/weak_relation_penalty_hotpotqa30/` 上运行审计后，SAM-full 的 30 条 HotpotQA run 中共有 89 个图路径命中，其中 22 个落在支持证据上，67 个落在非支持证据上，涉及 19 个图噪声 bad case。按关系类型看，`keyword_overlap` 出现 108 次，其中噪声 94 次，噪声率 0.870；`embedding_similarity` 出现 62 次，其中噪声 50 次，噪声率 0.806；`context_cooccurrence` 出现 6 次，噪声率 0.667。该结果说明下一阶段不能只继续增加图扩展，而应把弱关键词边和弱语义相似边的二跳权重继续下调，或者引入 GPT-5.4 RelationJudge 对这些边进行关系有效性判别。

随后 `PathReranker` 支持读取 `SAM_EDGE_QUALITY_AUDIT_PATH` 指向的 `edge_quality_audit.json`，将高噪声关系类型转化为 `relation_noise_penalty`，写入每个命中结果的 `score_breakdown`。在 `outputs/runs/edge_audit_penalty_hotpotqa30/` 的 30 条 smoke 中，SAM-full 证据召回率为 0.600，答案命中率为 0.733；相比原弱关系惩罚 run 的 0.617 和 0.667，证据召回略有下降，但答案命中率提升 0.066。进一步审计显示，图噪声路径数量没有下降，说明这一步主要改善排序而非建边质量。下一阶段仍需要在建边阶段引入 GPT-5.4 RelationJudge 或更严格实体链接，减少噪声边进入候选图。

## 28. 候选路径边质量字段透传

上一轮 edge audit 只能按关系类型统计噪声，例如 `keyword_overlap`、`embedding_similarity` 和 `shared_entity`，但无法继续区分同一种关系内部的质量差异。为支持更细粒度 bad case 分析，系统将 `GraphBuilder` 产生的边质量字段透传到 `Retriever` 的 `candidate_paths` 中，包括 `edge_quality`、`similarity`、`shared_entities` 和 `keyword_overlap`。这样每个检索命中不仅能说明“沿什么关系到达”，还能说明“这条边是否有实体支撑、语义相似度是多少、关键词重叠是什么”。

同时新增可选开关 `SAM_PENALIZE_UNSUPPORTED_KEYWORD_PATHS`。当该开关启用时，`PathReranker` 会对二跳 `keyword_overlap` 路径中缺少共享实体且边语义相似度较低的候选进行额外降权。该策略的目标是验证一个具体 bad case 假设：部分图噪声来自二跳弱关键词边，而不是所有关键词边都应被同等信任。

HotpotQA 30 条默认回归 run 位于 `outputs/runs/keyword_quality_signal_hotpotqa30/`。默认不启用上述额外惩罚，仅增加边质量字段透传。结果如下：

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.517 | 0.567 |
| SAM-full | 0.617 | 0.667 |
| SAM-no-graph | 0.517 | 0.567 |

审计结果显示，该 run 中 SAM-full 的图路径命中数为 89，噪声图路径命中数为 67，图噪声 bad case 数为 20。候选路径中已经可以看到边级字段，例如 `shared_entity` 路径会同时记录共享实体、边相似度和关键词重叠。该结果保留了 SAM-full 相比 Embedding Top-k 的收益：证据召回率提升 0.100，答案命中率提升 0.100。

随后对额外关键词二跳惩罚进行一次 smoke 验证，run 位于 `outputs/runs/unsupported_keyword_penalty_hotpotqa30/`。该策略下 SAM-full 证据召回率下降到 0.583，答案命中率下降到 0.633；图噪声 bad case 数没有下降，反而达到 21。该结果说明，单纯按“缺少共享实体且语义相似度低”惩罚二跳关键词路径过于粗糙，会错误压低一部分真实桥接证据。因此该策略保留为可控消融开关，默认关闭；后续应优先使用 GPT-5.4 RelationJudge 或实体类型约束在建边阶段过滤噪声，而不是只在排序阶段做静态惩罚。

## 29. RelationJudge 风险路由策略

GPT-5.4 RelationJudge 如果对所有候选边逐条判别，会在按需建图阶段产生大量模型调用。上一轮 30 条 HotpotQA smoke 的建边日志显示，候选边事件中 `shared_entity` 约 1460 次，`embedding_similarity` 约 984 次，`keyword_overlap` 约 1190 次。若全量调用模型，容易触发 QPM 限流，也不符合按需建图降低成本的设计目标。

因此 `GraphBuilder` 新增 `relation_judge_policy`，并在 `run_demo.py` 和 `run_end_to_end_experiment.py` 中提供 `--relation-judge-policy` 参数。当前支持三种策略：`risky`、`all` 和 `off`。默认 `risky` 只把高风险候选边送入 RelationJudge，主要包括 `keyword_overlap` 和 `embedding_similarity`；对于有明确共享实体支撑的 `shared_entity` 边，默认跳过模型判别并直接保留。`all` 用于严格实验，会对所有候选边调用模型；`off` 用于不调用模型的回归测试。

这样做的直接意义是把 GPT-5.4 用在最需要判断的地方。以 30 条 smoke 的建边事件粗略估算，`risky` 策略可以跳过约 40% 的强共享实体候选判别，同时仍保留对弱关键词边和偶然语义相似边的过滤能力。该策略已经通过单元测试验证：默认情况下强共享实体边不会触发模型判别，弱关键词边会触发判别并可被拒绝；隔离评测中也会保留同一 policy，避免不同方法的图构建条件不一致。

本轮同时运行 `outputs/runs/relation_policy_off_hotpotqa30_smoke/` 验证参数链路。在不启用模型判别的情况下，SAM-full 证据召回率仍为 0.617，答案命中率为 0.667，和默认主流程一致。下一步可以在小样本上使用 `--relation-judge cached_gpt54 --relation-judge-policy risky` 运行低额度 GPT-5.4 建边实验，并结合 `outputs/cache/relation_judge_cache.json` 缓存判别结果，逐步扩大到 30 条和 300 条。

## 30. RelationJudge 调用预算控制

在风险路由基础上，系统进一步加入关系判别调用预算。`BudgetedRelationJudge` 会包装在线 RelationJudge，并通过 `SAM_RELATION_JUDGE_MAX_CALLS` 限制本轮最多实际调用多少次模型。预算耗尽后，默认策略为 `skip`，即不再调用 GPT-5.4，并按“保留候选边”处理；也可以通过 `SAM_RELATION_JUDGE_BUDGET_EXHAUSTED=reject` 改为预算耗尽后拒绝候选边。

缓存和预算的组合顺序为：先查缓存，再消耗预算。也就是说，`cached_gpt54` 会使用 `CachedRelationJudge(BudgetedRelationJudge(ChatRelationJudge))` 结构，缓存命中的历史判别不会消耗新的 GPT-5.4 调用次数。这一点对于逐步扩大实验很重要：第一次小样本运行会积累关系判别缓存，后续同一候选边再次出现时可以直接复用结果。

`run_demo.py` 和 `run_end_to_end_experiment.py` 新增参数 `--relation-judge-max-calls`。例如低额度实验可以先运行：

```bash
conda run -n sam python scripts/run_demo.py \
  --env-file .env.local \
  --dataset-file data/processed/hotpotqa_midterm30_sam_sample.json \
  --run-name relation_judge_budget_hotpotqa30 \
  --relation-judge cached_gpt54 \
  --relation-judge-policy risky \
  --relation-judge-max-calls 20 \
  --methods embedding_topk,sam_full \
  --top-k 4 \
  --seed-k 1 \
  --hops 2
```

本轮已通过单元测试验证：预算上限为 1 时，第一次判别会调用底层模型，第二次判别会被预算拦截；在缓存模式下，同一候选边第二次命中缓存，不会再次消耗预算。无网络 smoke `outputs/runs/relation_budget_off_hotpotqa30_smoke/` 也验证了新增参数不会影响默认检索结果，SAM-full 证据召回率保持 0.617，答案命中率保持 0.667。

随后系统补充了关系判别使用统计产物。`run_demo.py` 和端到端 pipeline 会在每个 run 目录下写入 `relation_judge_usage.json`，记录 RelationJudge 是否启用、缓存路径、缓存大小、缓存命中/未命中、预算上限、实际调用次数和预算跳过次数。无网络 smoke `outputs/runs/relation_usage_output_hotpotqa30_smoke/` 已验证该文件会随运行产物生成；当关系判别关闭时，文件内容为 `enabled=false`，后续启用 `cached_gpt54` 后可用同一字段追踪真实 GPT-5.4 调用成本。

## 31. 在线 Embedding 调用超时保护

为继续推进正式 embedding 主实验，系统对 `azure_openai_sdk` provider 增加脚本级超时保护。旧版本虽然已经把 `SAM_AZURE_EMBEDDING_TIMEOUT` 传给 OpenAI SDK client，但单次 `embeddings.create` 没有外层 `asyncio.wait_for` 保护；如果公司网关或 SDK 内部连接阶段长时间无响应，实验脚本仍可能一直等待，导致 300 条主实验无法稳定失败、恢复或复跑。

本轮改造后，`AzureOpenAISDKEmbeddingProvider` 在 single 和 batch 两种输入模式下都会对 `embeddings.create` 设置同一个 timeout，并保留原有并发、维度、模型名和重试参数。单元测试 `test_azure_embedding_sdk_provider_times_out_hanging_request` 模拟网关挂起，验证 provider 能在低 timeout 设置下快速抛出 `TimeoutError`，不会无限阻塞。

随后使用本地 `.env.local` 做低额度真实 probe，命令临时设置 `SAM_AZURE_EMBEDDING_TIMEOUT=5`、`SAM_AZURE_EMBEDDING_MAX_RETRIES=1`，并调用 `scripts/check_embedding_provider.py --provider azure_openai_sdk --probe ...`。诊断结果显示配置项完整，但真实请求返回结构化 `TimeoutError`。这说明当前本地已具备正式 embedding 的配置读取、SDK 调用路径、超时失败和错误报告能力；但公司 embedding endpoint 在本次测试窗口内没有返回向量，因此 HotpotQA 300 条和 NovelQA 的正式在线 embedding 重跑仍需要在接口可用后继续执行。

该结果把问题边界收窄到在线 embedding 服务可用性，而不是 SAM 的配置读取或调用代码缺失。后续正式实验的推荐顺序是：先用 1 条 probe 确认 endpoint 返回 1024 维向量，再用 `plan_embedding_run.py` 估算请求量，再用 `warm_embedding_cache.py` 预热缓存，最后重跑 HotpotQA 300 条消融和 NovelQA 小样本实验。

## 32. GPT-5.4 RelationJudge 低预算链路验证

在关系判别预算控制完成后，系统运行了一个低预算 GPT-5.4 RelationJudge smoke，run 位于 `outputs/runs/relation_judge_gpt54_budget2_hotpotqa30_smoke/`。该实验使用 HotpotQA 30 条样本、local embedding、`cached_gpt54` 关系判别、`risky` 风险路由策略，并将 `SAM_RELATION_JUDGE_MAX_CALLS` 限制为 2。目的是验证 GPT-5.4 关系判别是否能进入按需建图链路，以及缓存和预算统计是否能真实记录调用成本。

本次运行生成了 `relation_judge_usage.json`。统计显示 RelationJudge 已启用，缓存路径为 `outputs/cache/relation_judge_cache.json`，缓存大小为 1089，缓存命中 4 次，缓存未命中 1089 次；预算上限为 2，实际调用 2 次，预算耗尽后跳过 1087 次。这说明风险路由、缓存封装、预算封装和 run 级使用统计已经形成闭环。

检索指标方面，Embedding Top-k 的证据召回率为 0.517，答案命中率为 0.567；SAM-full 的证据召回率为 0.567，答案命中率为 0.633。和不启用 RelationJudge 的 30 条 smoke 相比，SAM-full 证据召回率从 0.617 降至 0.567，答案命中率从 0.667 降至 0.633。进一步查看缓存可知，2 次真实 GPT-5.4 调用均返回 QPM 限流错误，其余候选边主要由预算耗尽策略处理。因此这个 run 的意义是验证在线关系判别链路、缓存和成本约束，而不是证明 RelationJudge 已改善检索质量。

下一步应在 QPM 可用或额度扩容后，把调用预算逐步提高到 20、100，再比较 `relation_judge_policy=risky` 与 `off` 的建边质量差异。只有当 GPT-5.4 能返回有效关系类型和置信度时，才适合把 RelationJudge 结果纳入 300 条主实验结论。

## 33. 官方 Baseline 独立 Embedding 配置与就绪审计

为推进 RAPTOR、Microsoft GraphRAG 和 HippoRAG 官方 baseline，系统进一步改造 `evaluation/official_baselines/` 适配层。此前官方 baseline env 默认 chat 和 embedding 使用同一个 API key、base url 和 api version；但当前本地实际配置中 GPT-5.4 与 `text-embedding-3-large` 分属不同公司网关。如果不支持分离配置，RAPTOR 和 GraphRAG 即使有官方代码，也无法正确调用 embedding 模型。

本轮改造后，官方 baseline 模板新增 `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_API_VERSION`、`EMBEDDING_MODEL` 和 `EMBEDDING_DIMENSIONS`。`run_raptor_official.py` 支持独立的 `RAPTOR_EMBEDDING_API_KEY`、`RAPTOR_EMBEDDING_AZURE_ENDPOINT`、`RAPTOR_EMBEDDING_API_VERSION` 和 `RAPTOR_EMBEDDING_DIMENSIONS`；`run_graphrag_official.py` 支持独立的 `GRAPHRAG_EMBEDDING_API_KEY`、`GRAPHRAG_EMBEDDING_API_BASE` 和 `GRAPHRAG_EMBEDDING_API_VERSION`。`test_company_api.py` 也可以分别测试 chat endpoint 和 embedding endpoint。

同时 `audit_official_baselines.py` 支持加载多个 ignored env 文件，并会把根目录 `.env.local` 中的 `SAM_AZURE_EMBEDDING_*` 自动映射为官方 baseline 所需变量。使用 `.env.local` 和 `evaluation/official_baselines/.env.local` 重新审计后，结果写入 `docs/official_baseline_audit.json`。当前 3 个官方 baseline 中，Microsoft GraphRAG 已达到 ready 状态；RAPTOR 的模型配置已完整，但官方导入检查超过 30 秒，仍为 partial；HippoRAG 的模型配置完整，但本机官方依赖缺失，仍为 partial。NovelQA demonstration 已导出为官方 prepared 数据，包含 120 个 documents 和 8 个 queries。

该结果说明官方 baseline 对接已经从“缺少 embedding 配置”推进到“GraphRAG 可进行 limit=1 小样本 smoke，RAPTOR/HippoRAG 需要修复官方依赖”。下一步应优先对 GraphRAG 运行 1 条 NovelQA smoke；如果 embedding endpoint 仍然超时，则记录为外部服务可用性问题，而不是 SAM 数据导出或官方 baseline 适配缺失。

随后使用 `test_company_api.py --timeout 8` 对公司网关做低额度 gate。该脚本现在支持 chat endpoint 与 embedding endpoint 分离，并对单次 HTTP 请求加硬超时。测试中 GPT-5.4 chat 请求成功返回 `OK`，说明 chat key、deployment 和 base url 可用；embedding 请求在 8 秒内未返回，外层 25 秒兜底有时仍会截断整个 gate。因此当前不继续启动 GraphRAG 官方 index，避免在 embedding 服务不稳定时产生不可控等待。后续只有当 embedding gate 能稳定返回向量维度后，才运行 GraphRAG `limit=1` smoke。

随后对 RAPTOR 官方导入超时做了定位。逐项导入检查显示，`umap`、`torch`、`transformers` 和 `sentence_transformers` 的冷启动导入本身就需要较长时间；RAPTOR 顶层 `raptor.__init__` 又会 eager import 多个重依赖模块。因此此前 30 秒审计阈值会把“冷启动较慢”误判为“官方依赖不可用”。将 `audit_official_baselines.py` 默认导入检查 timeout 调整为 75 秒后，RAPTOR import 检查通过。最新审计结果显示：RAPTOR 和 Microsoft GraphRAG 均达到 ready，HippoRAG 仍为 partial；prepared NovelQA demonstration 仍为 120 个 documents 和 8 个 queries。这样官方 baseline 的下一步已经明确为：在 embedding gate 成功后优先跑 RAPTOR 与 GraphRAG 的 `limit=1` smoke，HippoRAG 放到 Linux/CUDA 或单独依赖环境处理。

## 34. 结构路径类比复用实验

为进一步推进类比推理模块，系统将类比复用实验从“文本相似案例命中”扩展为“结构路径匹配”。新增 `AnalogyEngine.relation_pattern_for_case`，可以从某个历史案例的记忆图中抽取代表性关系路径，例如 `summary_parent -> shared_entity -> context_cooccurrence`。在 masked probe 实验中，probe 查询已知来源案例，因此系统会先从来源案例抽取关系路径模式，再用该模式约束 `AnalogyEngine.retrieve_cases`，检查当前问题是否能找回结构相似的历史案例。

同时 `analogy_reuse_results.json` 新增结构化指标：来源案例命中数、来源结构模式可用数、结构路径匹配数、平均 Top-1 类比分数和 bad case 分布。单个 top match 也会输出 `path_pattern_score`、`matched_relation_path`、`relation_path_count` 和 `longest_relation_path`，用于解释类比案例为什么被选中。

30 条 HotpotQA 结构类比复用实验位于 `outputs/runs/analogy_structural_reuse_hotpotqa30_v2/`，结果如下：

| 指标 | 数值 |
| --- | ---: |
| 查询数量 | 30 |
| 来源案例命中率 | 1.000 |
| 巩固案例命中率 | 0.900 |
| 支持证据重叠命中率 | 0.900 |
| 来源结构模式可用率 | 1.000 |
| 结构路径匹配率 | 1.000 |
| 平均 Top-1 类比分数 | 1.000 |

bad case 分布为：`success=27`，`source_case_without_consolidation=3`。这说明结构路径匹配本身可以稳定找回来源案例，但仍有 3 条样本虽然找到了正确来源案例，却没有命中可复用的巩固证据链。当前瓶颈因此从“类比触发是否能找回相似案例”转移到“历史案例是否完成了足够覆盖支持证据的记忆巩固”。下一步应优先改进 `MemoryConsolidator` 的支持证据覆盖和巩固条件，而不是继续只提高类比相似度分数。

## 35. 结构性记忆巩固改进

针对上一轮结构路径类比复用实验中的 3 个 `source_case_without_consolidation` bad case，本轮对 `MemoryConsolidator` 做了结构性巩固改造。旧逻辑只有在检索结果命中 gold supporting document 时才创建长期案例记忆；如果检索没有命中支持证据，即使该案例内部存在可复用的关系路径，后续类比模块也只能找回普通查询上下文，无法识别这是一个已经经历过检索激活的历史案例。

新逻辑将巩固结果拆成两类：第一类仍是 `feedback_support_hit`，只在命中支持证据时写入 `support_node_ids`；第二类是 `structural_activation`，在没有支持证据命中但存在实际激活节点时，写入 `evidence_node_ids`、`evidence_original_doc_ids` 和 `evidence_titles`。这一区分很重要：`support_node_ids` 继续只表示真实支持证据命中，`evidence_node_ids` 只表示系统当次检索实际激活过的结构证据，因此不会把结构性案例覆盖误写成证据召回提升。

改造后重跑 HotpotQA 30 条结构类比复用实验，结果位于 `outputs/runs/analogy_structural_consolidation_hotpotqa30/`：

| 指标 | 改造前 | 改造后 |
| --- | ---: | ---: |
| 查询数量 | 30 | 30 |
| 来源案例命中率 | 1.000 | 1.000 |
| 巩固案例命中率 | 0.900 | 1.000 |
| 支持证据重叠命中率 | 0.900 | 0.900 |
| 结构路径匹配率 | 1.000 | 1.000 |
| 巩固记忆数量 | 27 | 30 |

bad case 分布从 `success=27, source_case_without_consolidation=3` 变为 `success=27, no_support_overlap=3`。这说明改造后的系统已经能为所有来源案例建立可复用的长期案例记忆，但其中 3 条仍然缺少真实支持证据重叠。该结果把下一步问题边界进一步收窄：类比触发和案例巩固覆盖已基本打通，剩余瓶颈是初始检索、建边质量和图扩展排序没有把真正的 supporting paragraph 拉进激活子图。
