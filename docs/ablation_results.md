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
