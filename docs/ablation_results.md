# SAM 300 条消融实验记录

本文档记录当前阶段的 HotpotQA 300 条主实验。运行产物位于 `outputs/runs/ablation_hotpotqa_300/`，该目录不进入 Git 仓库。

## 1. 实验设置

- 数据集：HotpotQA dev distractor
- 样本类型：bridge-style 多跳问答
- 查询数量：300
- 候选文档节点数量：2992
- Gold 支持证据数量：600
- 默认参数：`top-k=4`，`seed-k=1`，`hops=2`
- 统一格式文件：`data/processed/hotpotqa_midterm300_sam_sample.json`

复现命令：

```bash
conda run -n sam python scripts/run_demo.py \
  --reset \
  --db outputs/runs/ablation_hotpotqa_300/sam.sqlite \
  --dataset hotpotqa \
  --dataset-file data/processed/hotpotqa_midterm300_sam_sample.json \
  --rebuild-dataset \
  --sample-size 300 \
  --max-scan 100000 \
  --run-name ablation_hotpotqa_300 \
  --methods embedding_topk,raptor_style,graphrag_style,hipporag_style,sam_full,sam_no_multipath,sam_no_memory_state,sam_no_graph,sam_static_graph \
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
| HippoRAG | 353 | 0.588 | 167 | 0.557 |
| SAM-full | 355 | 0.592 | 173 | 0.577 |
| SAM-no-multipath | 354 | 0.590 | 173 | 0.577 |
| SAM-no-memory-state | 352 | 0.587 | 173 | 0.577 |
| SAM-no-graph | 352 | 0.587 | 167 | 0.557 |
| SAM-static-graph | 356 | 0.593 | 173 | 0.577 |

## 3. SAM 消融结果

| 方法 | 证据命中数 | 证据召回率 | 答案命中率 | 平均路径长度 | 平均候选路径数 | 平均路径支持分 | 平均边记忆分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAM-full | 355 | 0.592 | 0.577 | 2.37 | 10.28 | 0.737 | 0.000 |
| SAM-no-multipath | 354 | 0.590 | 0.577 | 2.36 | 1.00 | 0.000 | 0.000 |
| SAM-no-memory-state | 352 | 0.587 | 0.577 | 2.36 | 10.30 | 0.742 | 0.000 |
| SAM-no-graph | 352 | 0.587 | 0.557 | 1.00 | 1.25 | 0.000 | 0.000 |
| SAM-static-graph | 356 | 0.593 | 0.577 | 2.37 | 10.39 | 0.741 | 0.555 |

## 4. 阶段结论

SAM-full 相比 Embedding Top-k 多命中 12 个支持证据，证据召回率从 0.572 提升到 0.592，答案命中率从 0.547 提升到 0.577。这个结果说明系统已经不是单纯的向量 top-k，而是可以通过图扩展补充一部分间接证据。

`sam_no_graph` 的平均路径长度为 1.00，答案命中率为 0.557，低于 SAM-full 的 0.577。该差异说明图扩展对最终答案上下文有贡献，尤其在 bridge-style 问题中，单个向量种子无法稳定覆盖完整证据链。

`sam_no_multipath` 与 SAM-full 的差距较小，说明当前多路径信号已经进入排序，但权重还不够强，或者候选图中多条有效路径的区分度不足。后续应重点优化路径支持分的归一化方式和 beam 搜索策略。

`sam_no_memory_state` 与 SAM-full 的证据召回率差距为 0.005，说明 usage、recency 和 edge activation 已经影响排序，但作用还比较温和。下一阶段应加入更明确的时间衰减函数、任务反馈强化和跨查询共激活边。

RAPTOR 在本轮实验中表现最好，证据召回率达到 0.635，说明摘要层级结构对当前 HotpotQA 候选集有效。SAM 后续可以吸收这一点，在动态记忆图中加入 summary memory node，把“摘要层级”和“动态图激活”结合起来。

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
