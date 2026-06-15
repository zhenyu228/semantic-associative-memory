# 非 LLM 建图策略性价比实验设计

本实验用于回答一个核心问题：SAM 的语义联系建图是否需要复杂的领域规则或大模型逐边判别。当前结论假设是：主建图公式应保持简洁，使用 `semantic similarity + context path proximity`，并通过 top-k 与阈值控制边规模。

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

CAM 的位置邻近度适合线性长文，例如同一本书或同一篇文章中的相邻 chunk。SAM 希望支持更一般的知识对象，因此不直接使用线性位置，而是要求 adapter 提供通用 `context_path`。

示例：

```text
论文段落：["paper_1", "method", "paragraph_12"]
代码函数：["repo_1", "src/auth.py", "login_user"]
法律条款：["contract_1", "chapter_3", "clause_2"]
```

SAM 核心只计算两个路径的公共前缀比例，不需要知道这些路径来自论文、代码还是法律文档。

## 成本指标

实验同时统计效果和成本：

- `evidence_recall`：支持证据召回率。
- `edge_count`：实际生成边数量。
- `candidate_pair_count`：候选节点对数量。
- `average_edges_per_node`：平均每个节点关联边数。
- `build_time_seconds`：建图耗时。
- `uses_llm`：是否使用大模型建图。本实验所有策略均为 `False`。
- `recall_per_100_edges`：单位边规模下的召回效率。
- `recall_per_second`：单位建图时间下的召回效率。

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
  --output-dir outputs/graph_strategy_experiment_hotpotqa300
```

输出文件：

```text
graph_strategy_results.json
graph_strategy_results.md
graph_strategy_alpha_sweep.json
graph_strategy_alpha_sweep.md
```

## 预期答辩表述

本实验不是比较复杂关系类型的堆叠，而是比较不同非生成式建边公式在效果、耗时和边规模之间的平衡。若实验结果支持 `sam_context`，可以表述为：

> 相比单纯语义相似、单纯位置邻近和 CAM-style 的线性位置邻近，SAM-style 的上下文路径邻近在保持公式简洁和无需大模型建图的前提下，更适合非线性的知识对象结构。它不依赖论文、代码或法律等领域特定关系类型，只要求 adapter 提供可解释的上下文路径，因此具有更好的迁移性和建图性价比。
