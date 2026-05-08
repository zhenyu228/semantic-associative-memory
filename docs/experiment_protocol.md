# 中期展示实验复现协议

本文档记录当前中期阶段可展示实验的固定命令、数据规模、输出路径和主要结果。运行产物位于 `outputs/runs/`，该目录已被 `.gitignore` 忽略，不进入 Git 仓库。

## 1. 环境

所有命令均在项目根目录执行：

```bash
cd /Users/bytedance/Desktop/masterThesis/SAM
```

使用本地 conda 环境：

```bash
conda run -n sam python ...
```

## 2. HotpotQA 展示实验

### 2.1 数据集

- 数据集：HotpotQA dev distractor
- 样本类型：bridge-style 多跳问答
- 查询数量：8
- 候选文档节点数量：80
- Gold 支持证据数量：16
- 统一格式文件：`data/processed/hotpotqa_sam_sample.json`

### 2.2 复现命令

```bash
conda run -n sam python scripts/run_demo.py \
  --reset \
  --dataset hotpotqa \
  --run-name midterm_hotpotqa_final \
  --methods embedding_topk,raptor_style,graphrag_style,hipporag_style,sam \
  --top-k 4 \
  --seed-k 1 \
  --hops 2
```

### 2.3 输出路径

```text
outputs/runs/midterm_hotpotqa_final/
├── config.json
├── dataset_summary.json
├── hotpotqa_sample_manifest.json
├── metrics.json
├── metrics.md
├── cases.json
├── graphs/
│   ├── graph_view.html
│   ├── graph_artifact.json
│   ├── graph_mermaid.md
│   └── edge_creation_log.json
└── logs/
    └── run_summary.txt
```

### 2.4 主要结果

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.500 | 0.375 |
| RAPTOR-style | 0.688 | 0.625 |
| GraphRAG-style | 0.562 | 0.500 |
| HippoRAG-style | 0.562 | 0.500 |
| SAM 动态联想检索 | 0.625 | 0.625 |

总体指标：

- 纯向量命中支持证据数：8
- SAM 命中支持证据数：10
- SAM 新增有效证据数：2
- SAM 平均路径长度：2.41

说明：`*-style` baseline 是思想级对照，不是官方完整复现。官方 baseline 适配位于 `evaluation/official_baselines/`。

## 3. NovelQA Demonstration 展示实验

### 3.1 数据集

- 数据集：NovelQA demonstration
- 小说：Frankenstein
- 查询数量：8
- 候选 chunk 节点数量：120
- Gold 支持证据数量：18
- 统一格式文件：`data/processed/novelqa_demo_sam_sample.json`
- 原始文件：`data/raw/NovelQA.zip`

NovelQA demonstration 的价值在于验证系统可以处理长篇小说切块和长文本问答格式。当前 embedding 仍是轻量本地实现，因此 NovelQA 结果主要用于展示数据接入、图谱构建和可视化闭环，不作为最终效果结论。

### 3.2 复现命令

```bash
conda run -n sam python scripts/run_demo.py \
  --reset \
  --dataset novelqa \
  --dataset-file data/processed/novelqa_demo_sam_sample.json \
  --novelqa-source data/raw/NovelQA.zip \
  --novelqa-split demonstration \
  --run-name midterm_novelqa_demo_final \
  --methods embedding_topk,sam \
  --top-k 4 \
  --seed-k 1 \
  --hops 2
```

### 3.3 输出路径

```text
outputs/runs/midterm_novelqa_demo_final/
├── config.json
├── dataset_summary.json
├── novelqa_sample_manifest.json
├── metrics.json
├── metrics.md
├── cases.json
├── graphs/
│   ├── graph_view.html
│   ├── graph_artifact.json
│   ├── graph_mermaid.md
│   └── edge_creation_log.json
└── logs/
    └── run_summary.txt
```

### 3.4 主要结果

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.000 | 0.125 |
| SAM 动态联想检索 | 0.000 | 0.125 |

说明：NovelQA 当前结果较弱，主要原因是长篇小说 chunk 检索对 embedding 质量和切块策略更加敏感。该实验现阶段用于证明 NovelQA 已被接入统一数据格式，系统能够生成记忆节点、语义边、检索案例和可视化图谱。后续更换 Qwen3-Embedding、BGE 或 E5 后，再作为正式长文本实验。

## 4. 可检查产物

每个 run 至少检查以下文件：

```text
metrics.md                 指标表和案例分析
cases.json                 每条查询的各方法检索结果
graphs/graph_view.html     可交互图谱页面
graphs/graph_artifact.json 节点、边、查询和检索案例的结构化图数据
graphs/edge_creation_log.json 按需建边日志和 scorer 分数
logs/run_summary.txt       本次运行摘要
```

其中 `edge_creation_log.json` 是当前 P1 的核心可解释产物，可以看到每条边的：

- 起点和终点。
- 关系类型。
- 边权。
- 建边原因。
- 实体得分。
- 关键词得分。
- 语义相似得分。
- 建边阈值。

## 5. 当前结论边界

当前中期展示实验可以支持以下表述：

- 已完成 SAM 统一数据格式和公开数据集接入。
- 已完成基于 SQLite 的记忆节点、语义边和检索日志存储。
- 已完成按需建图、动态状态更新和联想检索原型。
- 已在 HotpotQA 小样本上观察到 SAM 相比纯向量检索新增命中支持证据。
- 已完成 NovelQA demonstration 的长文本数据接入和可视化闭环。

当前不应过度声称：

- 不声称当前 embedding 已达到最终论文效果。
- 不声称 `raptor_style`、`graphrag_style`、`hipporag_style` 是官方完整复现。
- 不将 NovelQA 当前分数作为最终方法结论。
