# 非 LLM 建图策略性价比实验设计

本实验用于回答一个核心问题：不同非生成式建图方法在效果、时间成本、图规模和计算量之间的性价比如何。实验不预设某个方法一定最好，而是用同一批数据、同一套 embedding、同一套 top-k 约束，客观比较多种建图公式。

为了让“效果”可以被客观评估，实验支持两类数据：

- HotpotQA：用于多跳问答证据召回，适合早期 smoke 和与历史结果对齐。
- SciFact：用于跨论文 scientific claim evidence retrieval。每个 claim 有 gold evidence abstract 和 rationale sentence，适合评估跨论文建图策略是否真的提升证据检索。

## 对比方法

实验比较以下非生成式建图策略：

1. `no_graph`：不建图，只保留无图检索基线。
2. `semantic_only`：只使用 MemoryItem 的 embedding 语义相似度。
3. `position_only`：只使用线性位置邻近度。
4. `cam_style`：使用 `semantic similarity + linear position proximity`，对应 CAM 的核心建边思想。
5. `context_path_only`：只使用上下文路径邻近度。
6. `sam_context`：使用 `semantic similarity + context path proximity`，这是当前 SAM 的候选主策略。

统一公式为：

```text
S(i, j) = alpha * Sim(i, j) + (1 - alpha) * Prox(i, j)
```

其中 `Sim(i, j)` 是 MemoryItem 向量相似度，`Prox(i, j)` 在 `cam_style` 中是线性位置邻近度，在 `sam_context` 中是上下文路径邻近度。

## 为什么使用 context path proximity

CAM 的位置邻近度适合线性长文，例如同一本书或同一篇文章中的相邻 chunk。SAM 希望支持更一般的知识对象，因此不直接使用全局样本顺序，而是允许 adapter 提供文档自身结构的 `context_path`。

示例：

```text
论文段落：["paper_1", "method", "paragraph_12"]
代码函数：["repo_1", "src/auth.py", "login_user"]
法律条款：["contract_1", "chapter_3", "clause_2"]
```

SAM 核心只计算两个路径的公共前缀比例，不需要知道这些路径来自论文、代码还是法律文档。

为了避免 HotpotQA 的候选集组织方式影响实验，正式命令默认使用 `--context-path-policy intrinsic`。该策略显式排除 `query_id`、`hotpotqa_id` 和 `original_doc_id`，只使用文档自身结构字段，例如 NovelQA 的 `book_id/chunk_index` 或已有数据中的 `source_id/section/title`。如果数据没有自然层级结构，`context_path` 方法不会被人为增强，这一点会反映在实验结果中。

## 成本指标

实验同时统计效果和成本：

- `evidence_recall`：支持证据召回率。
- `precision_at_k`：返回 top-k 结果中 gold evidence 的比例。
- `mrr`：第一个 gold evidence 的倒数排名，衡量证据是否排在前面。
- `ndcg_at_k`：考虑排序位置的证据检索质量。
- `graph_path_support_hits`：通过图路径命中的 gold evidence 数量。
- `graph_path_evidence_recall`：图路径命中的 gold evidence 占全部 gold evidence 的比例。
- `graph_rescue_rate`：已命中证据中有多少是通过图路径命中的。
- `edge_count`：实际生成边数量。
- `candidate_pair_count`：候选节点对数量。
- `theoretical_full_pair_count`：当前数据文件全局两两建边时的理论候选节点对数量。
- `candidate_pair_coverage`：实际比较候选节点对占理论全量候选节点对的比例。
- `average_edges_per_node`：平均每个节点关联边数。
- `edge_keep_rate`：保留边数量占候选节点对数量的比例。
- `build_time_seconds`：建图耗时。
- `retrieval_time_seconds`：检索评估耗时。
- `total_time_seconds`：建图耗时与检索耗时之和。
- `average_retrieval_time_ms`：平均每个 query 的检索耗时。
- `uses_llm`：是否使用大模型建图。本实验所有策略均为 `False`。
- `recall_per_100_edges`：单位边规模下的召回效率。
- `recall_per_second`：单位建图时间下的召回效率。
- `cost_index`：综合建图成本指数，由归一化边规模、归一化候选比较次数和归一化建图耗时加权得到。
- `cost_effectiveness_score`：综合性价比分，定义为 `Evidence Recall / (1 + cost_index)`。
- `recall_gain_vs_no_graph`：相对无图检索的召回增益。
- `gain_per_100_extra_edges`：相对无图检索，每新增 100 条边带来的召回增益。
- `gain_per_extra_second`：相对无图检索，每新增 1 秒建图耗时带来的召回增益。

正式实验中，文档和 query 都使用同一个 embedding provider 生成向量；`local_hash` 只用于 smoke test，不作为正式结论依据。
如果没有任何图策略在 `evidence_recall` 上超过 `no_graph`，报告会将推荐策略写为 `no_improving_graph_strategy`，避免把无增益或空图策略误报为推荐方法。

## SciFact 数据准备

SciFact 官方数据结构包含：

```text
corpus.jsonl
claims_train.jsonl
claims_dev.jsonl
claims_test.jsonl
```

其中 `corpus.jsonl` 是科学论文摘要语料，`claims_dev.jsonl` 中的 `evidence` 字段给出 gold evidence 文档和 rationale sentence。SAM 转换脚本会把每篇 evidence abstract 转成一个 MemoryItem，把 claim 转成 EvaluationQuery。候选文档由三部分组成：gold evidence docs、cited docs、基于词项重叠选出的 hard negatives。

下载并转换 SciFact dev 小样本：

```bash
conda run -n sam python scripts/prepare_scifact.py \
  --source data/raw/scifact \
  --download \
  --split dev \
  --sample-size 50 \
  --negative-docs-per-query 20 \
  --output data/processed/scifact_dev50_sam_sample.json
```

如果已经手动下载并解压 SciFact，则去掉 `--download`，把 `--source` 指向包含 `corpus.jsonl` 的目录。

## 运行方式

使用本地哈希 embedding 的 smoke 实验：

```bash
conda run -n sam python scripts/run_graph_strategy_experiment.py \
  --dataset-file data/processed/hotpotqa_sam_sample.json \
  --limit-queries 30 \
  --output-dir outputs/graph_strategy_experiment_smoke \
  --alpha-sweep 0,0.25,0.5,0.75,1
```

使用真实 embedding provider 的实验：

```bash
conda run -n sam python scripts/run_graph_strategy_experiment.py \
  --dataset-file data/processed/hotpotqa_midterm300_sam_sample.json \
  --limit-queries 300 \
  --embedding-provider azure_openai_sdk \
  --embedding-concurrency 20 \
  --embedding-input-mode single \
  --embedding-cache-path outputs/graph_strategy_experiment_hotpotqa300/embedding_cache.sqlite \
  --top-k 4 \
  --seed-k 1 \
  --hops 1 \
  --top-k-edges 4 \
  --threshold 0.18 \
  --alpha 0.55 \
  --context-path-policy intrinsic \
  --alpha-sweep 0,0.25,0.5,0.75,1 \
  --output-dir outputs/graph_strategy_experiment_hotpotqa300
```

使用 SciFact 跨论文证据检索的正式实验：

```bash
conda run -n sam python scripts/run_graph_strategy_experiment.py \
  --dataset-file data/processed/scifact_dev50_sam_sample.json \
  --limit-queries 50 \
  --embedding-provider azure_openai_sdk \
  --embedding-concurrency 20 \
  --embedding-input-mode single \
  --embedding-cache \
  --embedding-cache-path outputs/graph_strategy_experiment_scifact50/embedding_cache.sqlite \
  --top-k 5 \
  --seed-k 2 \
  --hops 1 \
  --pair-scope query_candidates \
  --top-k-edges 6 \
  --threshold 0.18 \
  --alpha 0.55 \
  --context-path-policy intrinsic \
  --alpha-sweep 0,0.25,0.5,0.75,1 \
  --output-dir outputs/graph_strategy_experiment_scifact50
```

`--pair-scope global` 表示在当前数据文件的所有 MemoryItem 之间做全局两两比较，适合小规模完整建图成本分析。`--pair-scope query_candidates` 表示只在每个 query 的候选文档集合内建边，并对重复节点对去重，适合 SciFact 这类跨论文检索实验。报告会同时输出实际候选对数、理论全量候选对数和候选覆盖率，因此可以直接看到成本节省。

`azure_openai_sdk` 底层使用异步请求。`--embedding-concurrency` 对应在线 embedding 的最大并发数；`--embedding-input-mode single` 表示每条文本单独发起一次异步 embedding 请求，和当前可用的公司网关调用方式一致。如果网关后续确认支持批量输入，可以切换为：

```bash
--embedding-input-mode batch --embedding-batch-size 16 --embedding-concurrency 10
```

输出文件：

```text
graph_strategy_results.json
graph_strategy_results.md
graph_strategy_alpha_sweep.json
graph_strategy_alpha_sweep.md
```

跑完正式实验后，使用审计脚本检查结果是否具备答辩可检查性：

```bash
conda run -n sam python scripts/audit_graph_strategy_report.py \
  --report outputs/graph_strategy_experiment_scifact50/graph_strategy_results.json \
  --expected-pair-scope query_candidates \
  --require-real-embedding
```

审计会检查 dataset 摘要、真实 embedding、context path 泄漏、策略完整性、效果指标、成本指标、性价比字段和 pair scope。审计失败时脚本返回非 0 退出码，避免把 smoke 或字段缺失的结果误当正式实验。

## 预期答辩表述

本实验不是比较复杂关系类型的堆叠，而是比较不同非生成式建边公式在效果、耗时和边规模之间的平衡。若实验结果支持 `sam_context`，可以表述为：

> 相比单纯语义相似、单纯位置邻近和 CAM-style 的线性位置邻近，SAM-style 的上下文路径邻近在保持公式简洁和无需大模型建图的前提下，更适合非线性的知识对象结构。它不依赖论文、代码或法律等领域特定关系类型，只要求 adapter 提供可解释的上下文路径，因此具有更好的迁移性和建图性价比。

若结果不支持 `sam_context`，则应如实表述为：在当前数据设置中，文档自身结构不足以形成稳定的上下文路径优势，后续需要调整 MemoryItem 粒度或 adapter 生成的上下文路径。SciFact 的作用是把这个判断放到更接近科研检索的跨论文场景中，而不是只依赖 HotpotQA 的题目候选集结构。
