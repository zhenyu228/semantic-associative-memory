# SAM 高层洞察记忆重构实验

## 实验目的

本实验用于支撑中期第一部分：动态知识图谱记忆从“节点和边的存储”推进到“记忆重构和高层洞察生成”。

实验重点不是比较最终问答效果，而是验证系统是否已经形成可检查的三层记忆结构：

```text
底层证据节点 -> 单次巩固记忆 consolidated_memory -> 高层洞察记忆 insight_memory
```

其中，底层证据节点来自数据集文档；单次巩固记忆来自一次成功或结构性检索；高层洞察记忆由多个巩固记忆按有意义关键词聚类生成，并通过图边回溯到底层证据。

## 实验设置

数据集使用 `data/processed/hotpotqa_midterm30_sam_sample.json`。该文件来自 HotpotQA 小样本，包含 30 个问题和 300 个候选文档节点。每个问题一般有 2 个 supporting documents，因此总支持证据数量为 60。

运行命令：

```bash
conda run --no-capture-output -n sam python scripts/run_insight_memory_experiment.py \
  --dataset-file data/processed/hotpotqa_midterm30_sam_sample.json \
  --limit 30 \
  --embedding-provider local_hash \
  --top-k 4 \
  --seed-k 1 \
  --hops 1 \
  --run-name insight_memory_hotpotqa30_clustered_local
```

输出目录：

```text
outputs/runs/insight_memory_hotpotqa30_clustered_local/
```

核心产物：

- `insight_memory_results.json`
- `insight_memory_results.md`
- `warmup_metrics.json`
- `dataset_summary.json`
- `config.json`

## 指标说明

`consolidated_memory_count` 表示系统在 warmup 检索后沉淀出的单次巩固记忆数量。

`insight_memory_count` 表示系统从多个巩固记忆中重构出的高层洞察节点数量。

`insight_edge_count` 表示洞察节点和巩固记忆、底层证据之间形成的可解释图边数量。

`insight_evidence_coverage_rate` 表示高层洞察覆盖了多少已经进入巩固记忆的底层证据。

`support_trace_rate` 表示真实 supporting documents 中，有多少可以通过高层洞察节点回溯到。

`average_consolidated_per_insight` 表示每个洞察平均由多少个单次巩固记忆构成。

`average_evidence_per_insight` 表示每个洞察平均能回溯到多少个底层证据节点。

## 实验结果

本次 30 条 HotpotQA 实验结果如下：

- 查询数量：30
- 原始文档节点数：300
- 单次巩固记忆数：30
- 高层洞察记忆数：7
- 洞察关联边数：72
- 洞察可回溯证据数：22
- 巩固证据覆盖率：0.415
- 支持证据回溯率：0.267
- 平均每个洞察覆盖巩固记忆数：2.00
- 平均每个洞察覆盖底层证据数：3.14

实验说明系统已经不只是保存节点和边，而是能够把一次次检索产生的巩固记忆进一步重构为高层洞察节点。每个洞察节点都保留来源巩固记忆、共享关键词、底层证据集合和可回溯边。

## 典型洞察

实验中生成了多个主题聚类洞察，例如：

- `series` 主题洞察：覆盖 science fantasy、young adult series、Animorphs 等相关证据。
- `born` 主题洞察：覆盖人物出生时间、人物比较和职业身份相关证据。
- `david` 主题洞察：覆盖 David Beckham、The Class of '92、David Beckham Academy 等相关证据。
- `men/team` 主题洞察：覆盖篮球队、足球队和赛季相关证据。

这些洞察不是人工手写的标签，而是由多个 `consolidated_memory` 节点中的共享关键词和底层证据自动形成。

## 主要发现

第一，记忆重构机制已经跑通。系统能够从 30 次查询中生成 30 个单次巩固记忆，并进一步形成 7 个高层洞察节点。这说明动态图谱已经从“节点边存储”推进到“可复用中间记忆生成”。

第二，高层洞察具有可追溯性。7 个洞察节点共形成 72 条洞察相关边，其中 22 条为洞察到证据的回溯边。每个洞察都可以追踪到具体底层文档节点，避免高层摘要变成不可解释的黑盒。

第三，当前洞察覆盖还不充分。聚类后的高层洞察只覆盖 41.5% 的巩固证据和 26.7% 的真实支持证据。这说明系统已经具备记忆重构能力，但洞察生成策略仍然偏保守，后续需要优化聚类粒度、关键词过滤和跨证据合并策略。

第四，洞察粒度和覆盖率存在取舍。如果把所有巩固记忆粗暴合并成一个大洞察，覆盖率会更高，但解释性较差；当前采用按有意义关键词聚类，解释性更好，但覆盖率下降。这个结果为下一阶段优化提供了明确方向。

## 中期可表述结论

开题后，本阶段已经完成动态知识图谱记忆的底层存储、单次记忆巩固和高层洞察生成。实验表明，系统能够把一次检索得到的证据链沉淀为长期记忆，并进一步把多个长期记忆重构为可回溯的高层洞察节点。与此同时，实验也暴露出当前洞察生成覆盖率不足的问题，说明后续需要继续优化记忆重构策略，使高层洞察既保持解释性，又能覆盖更多关键证据。
