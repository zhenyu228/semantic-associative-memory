# 实验二：记忆重构与高层压缩有效性实验报告

## 实验目的

实验一表明，底层语义图扩展能够补充 embedding 检索漏掉的证据，但随着图边数量增加，证据召回很快进入平台期，单位边收益下降，并引入较高比例的噪声扩展节点。因此，实验二进一步验证：在相同底层检索和相同单次巩固记忆基础上，记忆重构与高层压缩是否能够减少长期记忆冗余，同时保留对底层证据的回溯能力。

本实验关注的问题不是“能否生成摘要”，而是“压缩后的高层记忆是否仍然可回溯、可复用、低冗余”。因此，实验二将 SAM 的高层压缩方法与不压缩、逐条保存、关键词聚类和 embedding 聚类进行对比。

## 与实验一的关系

实验一证明了无约束增加底层图边并不能持续提升效果。该结论说明 SAM 不能只依赖更密的低层图结构，而需要把被查询验证过的稳定语义路径重构为更高层的记忆单元。

实验二承接这一结论，固定底层检索与单次巩固记忆，只比较不同高层重构策略。这样可以避免把差异归因于 embedding 或底层检索，而是直接观察“如何压缩记忆”本身的效果。

## SAM 压缩框架

SAM 的压缩框架包含三层记忆结构。

第一层是底层证据图 `G0`。该层保存原始 `MemoryItem`，例如段落、论文摘要、章节片段或代码片段。底层节点保留原始文本、来源、位置、embedding 和上下文路径，保证后续所有高层记忆都可以回溯到底层证据。

第二层是单次巩固记忆 `G1`。当查询激活底层图并形成有效证据链后，系统会将本次查询、答案、命中证据和证据路径沉淀为 `consolidated_memory`。该层记录的是一次查询过程中被验证过的证据组合。

第三层是高层压缩记忆 `G2`。随着查询累积，系统会把语义相近、证据重叠、答案方向一致的单次巩固记忆重构为更少的高层记忆单元。高层记忆不是普通摘要，而是保留来源巩固记忆、底层证据和回溯路径的结构化压缩结果。

因此，SAM 的压缩不是：

```text
长文本 -> 短文本
```

而是：

```text
底层证据节点 + 语义边 + 查询路径 + 使用反馈
-> 高层压缩记忆 + 可回溯证据链
```

## 对照方法

本实验比较五种策略。

`no_reconstruction` 表示不做高层重构。该方法作为空白对照，用于说明如果没有高层压缩，系统没有形成可复用的高层记忆结构。

`flat_consolidated` 表示逐条保存单次巩固记忆。该方法可以保留全部证据，但不会减少记忆单元数量，也不会形成跨查询的共性结构。

`keyword_cluster` 表示按关键词聚合巩固记忆。该方法成本低，但容易受关键词抽取质量影响。

`embedding_cluster` 表示按巩固记忆 embedding 相似度进行聚类。该方法代表常见的语义聚类压缩基线。

`sam_hybrid_reconstruction` 表示 SAM 的混合重构方法。该方法综合语义相似、关键词重叠、证据重叠和答案一致性，将巩固记忆组织为高层压缩记忆。

## 评测指标

`compression_ratio` 表示压缩前巩固记忆数量除以压缩后高层记忆单元数量。数值越高，说明压缩越明显。

`retrieval_unit_reduction_rate` 表示压缩后优先检索的高层记忆单元减少比例。该指标越高，说明后续检索阶段需要比较的高层单元越少。

`support_trace_rate` 表示高层记忆能够回溯到多少标准支持证据。

`query_full_trace_rate` 表示一个查询所需的全部支持证据是否都能被高层记忆覆盖。该指标比单个证据命中更严格。

`trace_edge_count` 表示高层记忆回溯到底层证据所需的 trace 边数量。

`trace_edge_reduction_rate` 表示压缩后 trace 边数量相对原始巩固记忆证据边数量的减少比例。

`trace_noise_rate` 表示高层记忆回溯到非标准支持证据的比例。该指标用于衡量压缩后是否仍然携带大量弱相关证据。

`effective_trace_precision` 表示高层记忆回溯证据中属于标准支持证据的比例。

`quality_cost_score` 是阶段性综合指标，综合考虑支持证据回溯、查询完整回溯、答案一致性、证据覆盖、压缩率、构建耗时、冗余率和 trace 噪声。

## 实验结果

本实验在 HotpotQA、QASPER 和 LitSearch 三个数据集上运行。HotpotQA 使用 300 条查询作为主实验；QASPER 和 LitSearch 各使用 30 条查询作为论文问答与科研检索场景补充实验。

HotpotQA 使用 `hybrid_threshold=0.18`，QASPER 和 LitSearch 使用更保守的 `hybrid_threshold=0.34`。原因是不同数据集的记忆粒度不同：HotpotQA 的巩固记忆较分散，需要较低阈值形成适度压缩；QASPER 和 LitSearch 的文本来自论文段落或论文摘要，较低阈值容易造成过度合并，因此采用更保守阈值。

| 指标 | HotpotQA | QASPER | LitSearch |
|---|---:|---:|---:|
| 查询数量 | 300 | 30 | 30 |
| 巩固记忆数量 | 300 | 30 | 30 |
| SAM 高层单元数 | 223 | 18 | 29 |
| SAM 压缩率 | 1.345 | 1.667 | 1.034 |
| SAM 检索单元减少率 | 0.257 | 0.400 | 0.033 |
| SAM 支持证据回溯率 | 0.912 | 0.456 | 0.850 |
| SAM 查询完整回溯率 | 0.833 | 0.333 | 0.867 |
| SAM 平均暴露证据数 | 10.28 | 3.33 | 1.40 |
| SAM Query级Trace噪声率 | 0.260 | 0.416 | 0.067 |
| SAM 构建耗时ms | 432.259 | 11.357 | 14.061 |
| SAM 质量成本综合分 | 0.469 | 0.326 | 0.690 |

### HotpotQA 主实验

HotpotQA 上，`flat_consolidated` 不进行压缩，300 条巩固记忆仍保留为 300 个高层单元。该方法的支持证据回溯率为 0.912，查询完整回溯率为 0.833，Query 级 Trace 噪声率仅为 0.010，但没有减少记忆规模。

`keyword_cluster` 将 300 条记忆压缩为 194 个高层单元，压缩率为 1.546，检索单元减少率为 0.353。但它的平均暴露证据数从 1.85 增加到 5.98，Query 级 Trace 噪声率上升到 0.362，说明简单关键词聚合虽然便宜，但会把更多额外证据暴露给单个查询。

`embedding_cluster` 几乎没有形成压缩，只从 300 个单元减少到 299 个单元，且构建耗时达到 9702.983ms。该结果说明单纯 embedding 聚类在当前巩固记忆粒度下性价比较低。

`sam_hybrid_reconstruction` 将 300 条记忆压缩为 223 个高层单元，压缩率为 1.345，检索单元减少率为 0.257，同时保持与其他非空方法相同的支持证据回溯率 0.912 和查询完整回溯率 0.833。与关键词聚类相比，SAM 的压缩程度略低，但 Query 级 Trace 噪声率从 0.362 降低到 0.260，说明混合重构比简单关键词聚合更能控制单查询暴露噪声。与 embedding 聚类相比，SAM 的构建候选对从 44833 降到 2993，构建耗时从 9702.983ms 降到 432.259ms，体现出关键词倒排候选过滤的成本优势。

### QASPER 补充实验

QASPER 上，所有非空策略的支持证据回溯率均为 0.456，查询完整回溯率为 0.333。该数据集的整体回溯率偏低，说明前置检索和巩固阶段本身没有覆盖足够的论文证据，后续高层压缩无法凭空补回未进入巩固记忆的证据。

在压缩效果上，`sam_hybrid_reconstruction` 将 30 条巩固记忆压缩为 18 个高层单元，压缩率为 1.667，检索单元减少率为 0.400。它的 Query 级 Trace 噪声率为 0.416，略低于 embedding 聚类的 0.424，但高于关键词聚类的 0.326。该结果说明在 QASPER 论文长文场景下，当前 SAM 重构可以形成压缩，但仍需要进一步加强路径有效性评分，避免跨段落证据被过度合并。

### LitSearch 补充实验

LitSearch 上，`sam_hybrid_reconstruction` 保持了支持证据回溯率 0.850 和查询完整回溯率 0.867，同时 Query 级 Trace 噪声率为 0.067，与不压缩和 embedding 聚类相同。虽然该设置下压缩幅度较小，只从 30 个单元减少到 29 个单元，但质量成本综合分达到 0.690，高于 `flat_consolidated` 的 0.667、`embedding_cluster` 的 0.642 和 `keyword_cluster` 的 0.538。

该结果说明，在科研检索场景中，保守的 SAM 混合重构能够避免关键词聚类带来的明显噪声扩散，同时在成本上优于 embedding 聚类。

## 结果分析

实验二得到三个结论。

第一，高层压缩能够减少记忆单元数量，但压缩不是越强越好。早期固定较低阈值时，SAM 曾在 HotpotQA 上把 300 条巩固记忆压缩成极少数高层单元，压缩率很高，但单个 query 会暴露大量额外证据，Query 级 Trace 噪声率接近 1。加入 query 级噪声指标后，该问题可以被直接检测出来。

第二，简单关键词聚类具有较高压缩率和极低构建成本，但容易提高单查询暴露噪声。HotpotQA 中关键词聚类的检索单元减少率为 0.353，但 Query 级 Trace 噪声率达到 0.362，高于 SAM 的 0.260。LitSearch 中关键词聚类的 Query 级 Trace 噪声率为 0.392，也显著高于 SAM 的 0.067。

第三，SAM 当前的混合重构不是在所有指标上最优，但它揭示了一个更合理的方向：高层压缩需要同时考虑语义相似、关键词重叠、证据重叠、答案一致性和候选预算。相比纯 embedding 聚类，SAM 通过关键词倒排候选把 HotpotQA 的候选对从 44833 降到 2993，大幅降低构建成本；相比关键词聚类，SAM 在 HotpotQA 和 LitSearch 中能更好地控制 Query 级噪声。

因此，实验二不是简单证明“压缩一定提升所有指标”，而是证明了两点：一是压缩确实能减少记忆单元；二是压缩必须受控，否则会把实验一中的底层图噪声转移到高层记忆中。这为后续加入路径有效性评分和自适应阈值提供了直接依据。

## 可复现实验命令

HotpotQA 5 条 smoke：

```bash
/Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python scripts/run_insight_reconstruction_comparison.py \
  --dataset-file data/processed/hotpotqa_midterm30_sam_sample.json \
  --limit 5 \
  --embedding-provider local_hash \
  --top-k 4 \
  --seed-k 1 \
  --hops 1 \
  --run-name experiment2_smoke_local
```

HotpotQA 300 条主实验：

```bash
SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE=200 /Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python scripts/run_insight_reconstruction_comparison.py \
  --dataset-file data/processed/hotpotqa_midterm300_sam_sample.json \
  --limit 300 \
  --embedding-provider azure_openai_sdk \
  --embedding-cache \
  --embedding-cache-path outputs/runs/hotpotqa300_real_embedding_cache_warmup/embedding_cache.sqlite \
  --embedding-concurrency 20 \
  --top-k 5 \
  --seed-k 2 \
  --hops 1 \
  --embedding-threshold 0.82 \
  --hybrid-threshold 0.18 \
  --run-name experiment2_hotpotqa300_real_embedding
```

QASPER 30 条补充实验：

```bash
SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE=200 /Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python scripts/run_insight_reconstruction_comparison.py \
  --dataset-file data/processed/qasper_validation30_sam_sample.json \
  --limit 30 \
  --embedding-provider azure_openai_sdk \
  --embedding-cache \
  --embedding-cache-path outputs/graph_strategy_experiment_qasper30/embedding_cache.sqlite \
  --embedding-concurrency 20 \
  --top-k 5 \
  --seed-k 2 \
  --hops 1 \
  --embedding-threshold 0.82 \
  --hybrid-threshold 0.34 \
  --run-name experiment2_qasper30_hybrid_0_34
```

LitSearch 30 条补充实验：

```bash
SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE=200 /Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python scripts/run_insight_reconstruction_comparison.py \
  --dataset-file data/processed/litsearch_query30_sam_sample.json \
  --limit 30 \
  --embedding-provider azure_openai_sdk \
  --embedding-cache \
  --embedding-cache-path outputs/graph_strategy_experiment_litsearch30/embedding_cache.sqlite \
  --embedding-concurrency 20 \
  --top-k 5 \
  --seed-k 2 \
  --hops 1 \
  --embedding-threshold 0.82 \
  --hybrid-threshold 0.34 \
  --run-name experiment2_litsearch30_hybrid_0_34
```

## 阶段结论

实验二说明，SAM 已经从底层图扩展推进到高层记忆重构阶段。当前系统能够在保持证据回溯能力的前提下减少部分高层检索单元，并通过 query 级 trace 噪声指标检测过度压缩问题。

现阶段最重要的发现是：高层压缩必须受控。简单关键词聚类虽然便宜，但容易引入额外证据；单纯 embedding 聚类成本较高且压缩收益不稳定；SAM 混合重构在科研检索场景中表现较稳，在 HotpotQA 中相较关键词聚类能降低 query 级噪声，但仍需要进一步优化路径有效性评分和阈值自适应机制。

下一阶段应在当前实验二基础上继续推进两个方向。第一，引入路径有效性评分，使系统不仅根据记忆相似性合并，还根据证据链完整性和路径纯度决定是否压缩。第二，设计自适应压缩阈值，根据数据集粒度、平均证据数和 query 级噪声自动调整压缩强度，避免过度压缩。
