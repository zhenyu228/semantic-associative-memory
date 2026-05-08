# 中期考核阶段性进展素材

本文档用于整理硕士论文《基于语义联想机制的动态知识图谱记忆系统方法与实现》在开题后的阶段性进展，可作为中期考核“个人总结”中科研与论文进展部分的素材。

## 阶段目标

开题后，本阶段工作的核心目标是将论文技术路线从概念设计推进到可运行、可解释、可复现的系统原型。由于完整动态知识图谱记忆系统涉及数据集适配、记忆表示、按需建图、联想检索、实验评测和可视化等多个模块，当前阶段优先完成能够证明技术路线可行的最小闭环：

1. 建立清晰的工程结构和统一数据格式。
2. 接入真实公开数据集，而不是使用手工编造样例。
3. 实现本地记忆库、记忆节点、语义边和检索日志。
4. 实现按需建图和语义联想检索。
5. 输出可检查的实验指标、案例和图谱 HTML。
6. 形成后续论文方法章节和实验章节的基础材料。

## 已完成工作

### 1. 工程仓库与代码结构

已建立独立项目仓库 `SAM`，代码采用较清晰的分层结构：

```text
src/sam/              核心代码包
scripts/              数据处理与运行脚本
tests/                单元测试
docs/                 中期材料与系统设计文档
evaluation/           官方 baseline 适配
outputs/runs/         每次实验运行产物，已 gitignore
reports/              人工整理后的阶段报告材料
```

项目 README 使用中文说明研究背景、系统架构、运行方式、数据集、实验方法和产物路径；同时补充了 `docs/system_design.md`，系统说明 SAM 的模块设计、动态记忆机制和后续开发优先级。

### 2. SAM 统一数据格式

当前已实现 `sam-dataset-v1` 统一数据格式。外部数据集不直接进入检索系统，而是先通过专门脚本转换成统一格式：

- `scripts/prepare_hotpotqa.py`：处理 HotpotQA。
- `scripts/prepare_novelqa.py`：处理 NovelQA。
- `src/sam/dataset_format.py`：负责统一格式读写和摘要统计。

统一格式将数据集拆成：

- `documents`：待写入记忆系统的文档或 chunk。
- `queries`：评测问题、标准答案、候选文档、支持证据。
- `dataset_info`：数据集来源和说明。
- `processing`：转换脚本、采样参数和处理策略。

这样做避免系统只围绕 HotpotQA 的“标题 + 段落”格式写死，后续可以继续接入 MultiHop-RAG、MuSiQue、2WikiMultiHopQA 等数据集。

### 3. 真实公开数据集接入

当前已接入两个公开数据集方向：

**HotpotQA**

HotpotQA 是经典多跳问答数据集，提供候选 Wikipedia 段落、答案和 supporting facts。当前主实验使用 HotpotQA dev distractor 小样本，重点选择桥接型问题，用于验证跨文档证据联想能力。

**NovelQA**

NovelQA 面向长篇小说问答，适合后续验证长文本场景下的记忆系统。当前已支持读取本地 `NovelQA.zip` 或解压目录，并将小说正文按 chunk 切分成记忆文档。对于 demonstration 子集，系统可以读取 `Answer`、`Gold` 和 `Evidences`，用于小规模可评估实验。

### 4. 记忆节点、语义边与本地记忆库

已实现 SQLite 本地记忆库 `MemoryStore`，能够保存：

- `MemoryNode`：记忆节点。
- `MemoryEdge`：语义边。
- `retrieval_logs`：检索日志。

当前 `MemoryNode` 包含文本、摘要、关键词、标签、来源、创建时间、使用次数、最近访问时间、置信度和 embedding 等字段。

当前 `MemoryEdge` 包含起点、终点、关系类型、边权、建边原因、创建时间、更新时间、激活次数、最近激活时间和 metadata。

### 5. 动态记忆状态

已完成动态记忆 P0。每次检索后，系统会真实更新数据库中的记忆状态：

- 命中节点的 `usage_count` 增加。
- 命中节点写入 `last_accessed_at`。
- 检索路径上的边写入 `activation_count`。
- 检索路径上的边写入 `last_activated_at`。
- 检索日志写入 `dynamic_update`，记录本次更新了哪些节点、激活了哪些边。

这使“动态知识图谱记忆系统”不再只是概念描述，而是可以通过数据库、`cases.json` 和图谱 JSON 检查到的运行状态。

### 6. 按需建图与建边解释

已实现按需建图机制。系统不在写入阶段构建全量图，而是在查询触发后围绕被激活的种子节点补充关系边。

当前建边依据已拆分为三个 scorer：

- 实体 scorer：判断两个节点是否共享实体或专名。
- 关键词 scorer：判断关键词重叠程度。
- 语义 scorer：判断 embedding 相似度是否达到阈值。

每条边会保存：

- `relation_type`
- `weight`
- `reason`
- `score_breakdown`
- `shared_entities`
- `keyword_overlap`
- `similarity`
- thresholds

每次实验会输出：

```text
outputs/runs/<run_name>/graphs/edge_creation_log.json
```

该文件记录每条按需创建或更新的边，包括起点、终点、关系类型、边权、建边原因和 scorer 分数。这样可以直接解释“为什么这两个节点会被连起来”。

### 7. 多方法检索与评测

当前系统支持以下方法：

- `embedding_topk`：纯向量 top-k，作为最低基线。
- `raptor_style`：摘要树思想的轻量对照。
- `graphrag_style`：实体图局部检索思想的轻量对照。
- `hipporag_style`：图激活 / PPR 思想的轻量对照。
- `sam`：动态按需建图 + 语义联想检索。

需要说明的是，`*-style` 是思想级 baseline，不声称复现官方完整实现。官方 baseline 的适配代码已放在 `evaluation/official_baselines/`，后续具备合适 embedding 模型或 API 后，可以继续跑 RAPTOR、Microsoft GraphRAG 和 HippoRAG 官方实现。

当前评测指标包括：

- 支持证据召回率。
- 命中支持证据数量。
- 答案命中率。
- SAM 平均路径长度。
- 检索结果路径解释。

### 8. 可解释图谱可视化

已实现 HTML 图谱运行产物，路径示例：

```text
outputs/runs/<run_name>/graphs/graph_view.html
```

HTML 页面支持：

- 按样本切换。
- 同一问题下多方法纵向对比。
- 展示问题、标准答案、各方法检索答案状态。
- 点击节点查看完整 MemoryNode 信息。
- 点击边查看关系类型、边权、建边原因、激活次数和最近激活时间。
- 标注支持证据节点和方法最终 top-k 节点。

这部分用于回应“系统是否真的建图、图是否可检查、检索过程是否可解释”的问题。

## 初步实验结果

当前 HotpotQA 小样本实验来自真实 HotpotQA dev distractor 数据。该样本包含 8 个桥接型问题、80 个候选文档节点和 16 个 gold 支持证据文档。

一次 smoke run 的结果如下：

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.500 | 0.375 |
| RAPTOR-style | 0.688 | 0.625 |
| GraphRAG-style | 0.562 | 0.500 |
| HippoRAG-style | 0.562 | 0.500 |
| SAM 动态联想检索 | 0.625 | 0.625 |

在另一轮只对比 `embedding_topk` 和 `sam` 的可解释建边日志 smoke run 中，系统输出了：

```text
outputs/runs/p1_edge_log_smoke/metrics.json
outputs/runs/p1_edge_log_smoke/cases.json
outputs/runs/p1_edge_log_smoke/graphs/graph_view.html
outputs/runs/p1_edge_log_smoke/graphs/graph_artifact.json
outputs/runs/p1_edge_log_smoke/graphs/edge_creation_log.json
```

`edge_creation_log.json` 中可以看到如“共享实体：Shirley Temple”等真实建边原因，以及对应的实体得分、关键词得分、语义相似度和阈值。

## 阶段性结论

当前阶段已经完成了 SAM 的基础研究原型。系统能够从真实公开数据集读取样本，将文档写入本地记忆库，按需构建语义图谱，执行联想检索，并输出指标、案例和可交互图谱。

初步结果表明，纯向量检索可以作为高效入口，但在多跳问答中可能遗漏间接相关证据；SAM 通过种子节点激活和语义边扩展，能够补充部分支持证据，并提供可解释路径。同时，系统已经开始具备动态记忆特征：节点和边会随检索过程被更新，后续可以进一步引入记忆衰减、反馈强化和多路径激活机制。

目前的不足包括：

- Embedding 仍采用轻量本地实现，后续需要切换到更强的本地或在线 embedding 模型。
- 官方 RAPTOR、GraphRAG、HippoRAG baseline 尚未完整跑出正式分数。
- 当前样本规模较小，后续需要扩大数据量并进行更严格实验。
- SAM 的状态感知重排和多路径激活仍有提升空间。

## 后续安排

短期工作：

1. 固化 HotpotQA 与 NovelQA demonstration 的最终展示实验结果。
2. 继续完善 HTML 可视化，使动态状态变化更直观。
3. 将 `docs/system_design.md` 中的设计内容整理为论文方法章节初稿。
4. 形成 `docs/experiment_protocol.md`，记录数据集、样本量、指标、参数和复现实验命令。

中期后续工作：

1. 使用 Qwen3-Embedding、BGE 或 E5 等更强 embedding 模型替换当前轻量表示。
2. 扩大 HotpotQA 样本规模，并接入 MultiHop-RAG、MuSiQue、2WikiMultiHopQA。
3. 实现更完整的状态感知重排，包括使用频率、时间衰减、多路径激活和反馈强化。
4. 在条件允许时运行官方 RAPTOR、Microsoft GraphRAG 和 HippoRAG baseline。
5. 继续推进多智能体共享记忆接口和类比推理触发机制。
