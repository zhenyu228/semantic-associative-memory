# 图密度与噪声实验

## 实验目的

本实验用于补充“为什么 SAM 需要受控联想和记忆重构”的证据。

此前 alpha 实验已经说明，不同建边信号组合会显著影响检索效果，但 alpha 实验主要改变的是语义相似与上下文路径邻近的权重，并没有直接改变图的密度。因此，本实验进一步固定建图策略为 `sam_context`，只扫描每个节点保留的边数和建边阈值，观察图变密后是否持续带来收益。

核心问题是：图边更多是否一定更好。

## 实验设置

实验流程如下：

1. 先用 embedding top-k 作为基础检索结果。
2. 从 top-k 中选择 seed 节点。
3. 在候选文档集合内按 `sam_context` 建图。
4. 沿图边扩展一跳。
5. 图扩展结果不替换 embedding top-k，只统计是否补回 embedding 漏掉的 gold evidence。

本实验扫描两个密度参数：

- `top_k_edges`：每个节点最多保留多少条出边。
- `threshold`：建边得分阈值。

使用真实 embedding 缓存运行，避免本地 hash embedding 对结论造成干扰。

## 指标解释

`edge_count` 表示当前配置下保留的图边数量。

`recall_gain` 表示图扩展后 evidence recall 相比 embedding baseline 的提升。

`rescue_precision` 表示图扩展出来的节点中，有多少是真正的 gold evidence。

`noise_expansion_rate` 表示图扩展出来但不是 gold evidence 的比例，计算方式为 `1 - rescue_precision`。

`recall_gain_per_100_edges` 表示每 100 条图边带来的 recall 增益。它用于观察边数增加后的边际收益是否下降。

## HotpotQA 300 条主实验

数据规模：300 个 query，2992 个候选段落，600 个 gold evidence。

建图候选空间：理论全量候选对为 8949072，按 query 候选集局部建图实际比较 26912 个候选对，占全量约 0.30%。这说明 SAM 的按需局部建图首先在候选边空间上进行了强约束。

| threshold | top_k_edges | 边数 | Recall 增益 | 图扩展 Precision | 噪声扩展率 | 每 100 边增益 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.18 | 1 | 2644 | 0.0150 | 0.1364 | 0.8636 | 0.000567 |
| 0.18 | 2 | 4975 | 0.0217 | 0.0855 | 0.9145 | 0.000436 |
| 0.18 | 4 | 8773 | 0.0217 | 0.0855 | 0.9145 | 0.000247 |
| 0.18 | 8 | 12995 | 0.0217 | 0.0855 | 0.9145 | 0.000167 |
| 0.18 | 16 | 13272 | 0.0217 | 0.0855 | 0.9145 | 0.000163 |
| 0.25 | 1 | 1837 | 0.0133 | 0.1569 | 0.8431 | 0.000726 |
| 0.25 | 2 | 3124 | 0.0183 | 0.1068 | 0.8932 | 0.000587 |
| 0.25 | 4 | 4805 | 0.0183 | 0.1068 | 0.8932 | 0.000382 |
| 0.25 | 8 | 6180 | 0.0183 | 0.1068 | 0.8932 | 0.000297 |
| 0.25 | 16 | 6210 | 0.0183 | 0.1068 | 0.8932 | 0.000295 |

结果显示，边数从 4975 增加到 13272 时，recall 增益仍保持在 0.0217，没有继续提升；但每 100 条边带来的增益从 0.000436 下降到 0.000163。同时，噪声扩展率维持在 0.91 左右，说明多数图扩展节点不是 gold evidence。

这说明在 HotpotQA 多跳问答中，图确实可以补回部分遗漏证据，但继续增加边数会快速降低边际收益，并引入大量无效扩展节点。

## QASPER 30 条补充实验

数据规模：30 个 query，523 个论文段落，73 个 gold evidence。

| threshold | top_k_edges | 边数 | Recall 增益 | 图扩展 Precision | 噪声扩展率 | 每 100 边增益 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 1 | 523 | 0.0411 | 0.1034 | 0.8966 | 0.007858 |
| 0.10 | 2 | 1046 | 0.0822 | 0.0870 | 0.9130 | 0.007858 |
| 0.10 | 4 | 2092 | 0.0822 | 0.0870 | 0.9130 | 0.003929 |
| 0.10 | 8 | 4184 | 0.0822 | 0.0870 | 0.9130 | 0.001964 |
| 0.10 | 16 | 8368 | 0.0822 | 0.0870 | 0.9130 | 0.000982 |

QASPER 上也出现了类似趋势：从 top-2 增加到 top-16 后，recall 增益不再增加，但每 100 条边带来的收益下降到原来的约八分之一。

## LitSearch 30 条补充实验

数据规模：30 个 query，630 个论文摘要，42 个 gold evidence。

| threshold | top_k_edges | 边数 | Recall 增益 | 图扩展 Precision | 噪声扩展率 | 每 100 边增益 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 1 | 630 | 0.0238 | 0.1667 | 0.8333 | 0.003779 |
| 0.10 | 2 | 1260 | 0.0476 | 0.1111 | 0.8889 | 0.003779 |
| 0.10 | 4 | 2520 | 0.0476 | 0.1111 | 0.8889 | 0.001890 |
| 0.10 | 8 | 5040 | 0.0476 | 0.1111 | 0.8889 | 0.000945 |
| 0.10 | 16 | 10080 | 0.0476 | 0.1111 | 0.8889 | 0.000472 |

LitSearch 上同样显示，top-2 已经达到最大 recall 增益，继续增加边数只会扩大图规模和噪声路径。

## 阶段结论

实验一的结论不是“图没有用”，而是：

1. 图扩展确实能够补回 embedding top-k 漏掉的一部分证据。
2. 图边数量增加后，recall 增益很快进入平台期。
3. 边数继续增加会显著降低单位边收益，并带来较高噪声扩展率。
4. 因此，SAM 不应追求全量建图或无限扩图，而应采用受控图联想。
5. 这为后续“记忆重构和高层压缩”提供了必要性：压缩不是为了省存储，而是为了压缩联想空间，减少弱相关路径对检索的干扰。

## 运行命令

HotpotQA 300 条：

```bash
SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE=200 /Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python scripts/run_graph_density_experiment.py \
  --dataset-file data/processed/hotpotqa_midterm300_sam_sample.json \
  --limit-queries 300 \
  --embedding-provider azure_openai_sdk \
  --embedding-concurrency 20 \
  --embedding-input-mode single \
  --embedding-cache \
  --embedding-cache-path outputs/runs/hotpotqa300_real_embedding_cache_warmup/embedding_cache.sqlite \
  --strategy sam_context \
  --top-k-edges-sweep 1,2,4,8,16 \
  --threshold-sweep 0.18,0.25 \
  --top-k 5 \
  --seed-k 2 \
  --hops 1 \
  --pair-scope query_candidates \
  --context-path-policy intrinsic \
  --output-dir outputs/graph_density_hotpotqa300_real_embedding
```

QASPER 30 条：

```bash
SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE=200 /Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python scripts/run_graph_density_experiment.py \
  --dataset-file data/processed/qasper_validation30_sam_sample.json \
  --limit-queries 30 \
  --embedding-provider azure_openai_sdk \
  --embedding-concurrency 20 \
  --embedding-input-mode single \
  --embedding-cache \
  --embedding-cache-path outputs/graph_strategy_experiment_qasper30/embedding_cache.sqlite \
  --strategy sam_context \
  --top-k-edges-sweep 1,2,4,8,16 \
  --threshold-sweep 0.10,0.18,0.25 \
  --top-k 5 \
  --seed-k 2 \
  --hops 1 \
  --pair-scope query_candidates \
  --context-path-policy metadata \
  --output-dir outputs/graph_density_qasper30_real_embedding
```

LitSearch 30 条：

```bash
SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE=200 /Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python scripts/run_graph_density_experiment.py \
  --dataset-file data/processed/litsearch_query30_sam_sample.json \
  --limit-queries 30 \
  --embedding-provider azure_openai_sdk \
  --embedding-concurrency 20 \
  --embedding-input-mode single \
  --embedding-cache \
  --embedding-cache-path outputs/graph_strategy_experiment_litsearch30/embedding_cache.sqlite \
  --strategy sam_context \
  --top-k-edges-sweep 1,2,4,8,16 \
  --threshold-sweep 0.10,0.18,0.25 \
  --top-k 5 \
  --seed-k 2 \
  --hops 1 \
  --pair-scope query_candidates \
  --context-path-policy intrinsic \
  --output-dir outputs/graph_density_litsearch30_real_embedding
```
