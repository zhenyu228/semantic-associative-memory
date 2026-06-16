# SAM 高层记忆重构对照实验设计

## 实验目标

第一部分现在不再只证明“系统能生成高层洞察记忆”，而是验证一个更可答辩的问题：

在相同底层检索结果和相同单次巩固记忆的前提下，SAM 的高层记忆重构是否比不重构、逐条保存、简单关键词聚类和单纯 embedding 聚类更有性价比。

这里的性价比同时包含四个方面：

1. 是否能压缩长期记忆，减少后续系统需要管理的记忆单元数量。
2. 是否仍然能回溯到底层证据，避免高层摘要脱离原始材料。
3. 是否能覆盖标准支持证据，尤其是一个问题所需的完整证据链。
4. 构建成本是否可控，避免为了生成高层记忆引入过高的额外开销。

## 实验变量控制

实验分为两个阶段。

第一阶段固定使用 `sam_full` 跑一遍检索和反馈巩固，生成同一批 `consolidated_memory`。这一阶段负责形成可复用的单次长期记忆。

第二阶段只改变高层记忆重构策略。所有对照方法都读取同一批 `consolidated_memory`，因此实验差异来自“如何重构高层记忆”，不是来自底层检索差异。

## 对照策略

`no_reconstruction`：不进行高层重构，只保留原始巩固记忆。这个对照用于说明如果没有重构，系统没有高层洞察和跨问题压缩。

`flat_consolidated`：每条巩固记忆单独作为一个高层单元。这个方法可以保留全部证据，但不会压缩记忆，也不会形成跨问题的共性结构。

`keyword_cluster`：按主要内容关键词聚合巩固记忆。这个方法成本低，但容易受关键词质量影响，适合检验“简单规则聚合是否足够”。

`embedding_cluster`：按巩固记忆向量相似度进行贪心聚类。这个方法是常见语义聚类基线，适合检验“只用语义相似是否足够”。

`sam_hybrid_reconstruction`：SAM 当前的高层重构策略。它把巩固记忆看成可重构对象，综合语义相似、关键词重叠、证据重叠和答案一致性形成记忆间连接，再从连接图中得到高层记忆分组。该方法强调高层洞察必须同时具备语义相关性和证据可追溯性。

## 评测指标

`compression_ratio` 表示巩固记忆数量除以重构后的记忆单元数量。数值越高，说明高层重构越能减少记忆冗余。

`support_trace_rate` 表示高层记忆覆盖标准支持证据节点的比例。它衡量重构后是否还能追溯到真实证据。

`query_full_trace_rate` 表示一个问题的全部支持证据都能被高层记忆覆盖的比例。它比单个证据命中更严格，适合多跳问答和科研证据链场景。

`answer_consistency` 表示同一高层记忆中来源问题答案是否一致。数值越高，说明聚合出的高层记忆内部越稳定。

`evidence_redundancy_rate` 表示同一证据被多个高层记忆重复覆盖的程度。数值越低，说明重构后的记忆结构越简洁。

`build_time_ms` 表示重构策略本身的构建耗时。

`quality_cost_score` 是阶段性综合指标，综合支持证据回溯、查询完整回溯、答案一致性、证据覆盖、压缩率、构建耗时和冗余率。它不是最终论文指标，而是用于中期阶段比较不同重构策略的质量成本平衡。

## 运行命令

小规模 smoke：

```bash
/Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python scripts/run_insight_reconstruction_comparison.py \
  --dataset-file data/processed/hotpotqa_midterm30_sam_sample.json \
  --limit 5 \
  --embedding-provider local_hash \
  --top-k 4 \
  --seed-k 1 \
  --hops 1 \
  --run-name insight_reconstruction_comparison_smoke
```

30 条正式对照实验：

```bash
/Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python scripts/run_insight_reconstruction_comparison.py \
  --dataset-file data/processed/hotpotqa_midterm30_sam_sample.json \
  --limit 30 \
  --embedding-provider local_hash \
  --top-k 4 \
  --seed-k 1 \
  --hops 1 \
  --embedding-threshold 0.82 \
  --hybrid-threshold 0.34 \
  --run-name insight_reconstruction_comparison_hotpotqa30
```

如果使用在线 embedding 并启用缓存：

```bash
/Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python scripts/run_insight_reconstruction_comparison.py \
  --env-file .env.local \
  --dataset-file data/processed/hotpotqa_midterm30_sam_sample.json \
  --limit 30 \
  --embedding-provider azure_openai_sdk \
  --embedding-cache \
  --embedding-cache-path outputs/runs/insight_reconstruction_comparison_hotpotqa30_azure/embedding_cache.sqlite \
  --embedding-concurrency 10 \
  --top-k 4 \
  --seed-k 1 \
  --hops 1 \
  --embedding-threshold 0.82 \
  --hybrid-threshold 0.34 \
  --run-name insight_reconstruction_comparison_hotpotqa30_azure
```

## 输出文件

每次运行会写入独立目录：

```text
outputs/runs/<run-name>/
```

核心文件包括：

- `config.json`：本次实验参数。
- `dataset_summary.json`：数据规模。
- `warmup_metrics.json`：前置 SAM 检索和巩固结果。
- `insight_memory_results.json`：原有高层洞察生成统计。
- `insight_memory_results.md`：原有高层洞察生成摘要。
- `insight_reconstruction_comparison.json`：本次新增的对照实验完整结果。
- `insight_reconstruction_comparison.md`：本次新增的对照实验可读报告。

## 当前结论表述方式

这组实验适合在中期中这样表述：

开题后，系统已从底层记忆节点和边的存储，推进到“单次巩固记忆”和“高层洞察记忆”的两级重构。为验证该设计的必要性，进一步设置了高层记忆重构对照实验，在相同底层检索结果和相同巩固记忆上比较不重构、逐条保存、关键词聚类、向量聚类和 SAM 混合重构。实验重点不只是看是否生成洞察，而是同时观察压缩率、证据回溯、查询完整证据链覆盖、答案一致性和构建耗时，从而评估高层记忆重构的质量成本平衡。

如果实验结果显示 `flat_consolidated` 的证据覆盖高但压缩率低，可以说明逐条保存虽然保真，但没有形成可复用高层结构。

如果 `keyword_cluster` 或 `embedding_cluster` 压缩较好但查询完整回溯不足，可以说明简单聚合容易损失证据链完整性。

如果 `sam_hybrid_reconstruction` 在压缩率和证据回溯之间取得更均衡结果，可以作为第一部分继续优化的主要依据。
