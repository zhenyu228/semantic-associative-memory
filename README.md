# SAM：语义联想记忆系统原型

本仓库对应硕士论文《基于语义联想机制的动态知识图谱记忆系统方法与实现》。当前阶段的目标不是一次性完成完整论文系统，而是在中期考核前先做出一个可运行、可解释、可复现的最小原型，用真实代码支撑后续论文写作和答辩说明。

## 项目动机

传统 RAG 往往把文档切块后放入向量库，查询时按语义相似度取 top-k。这个方式简单有效，但在多跳问答、跨文档推理、长程科研阅读中容易漏掉证据链中的某一环：第一个文档和问题很像，第二个文档可能只和第一个文档有关，和原始问题并不直接相似。

本项目尝试把“记忆”表示为动态知识图谱中的节点和边：

- 记忆节点保存文本、摘要、关键词、来源、时间戳、使用次数、置信度和 embedding。
- 语义边保存两个记忆节点之间的关系、边权和建边原因。
- 检索时先用向量相似度找到种子节点，再沿语义边进行联想扩展。
- 建图采用按需策略：不是一开始全量建图，而是在节点被检索命中或具备明显语义关系时再补边。

这对应开题报告中的三条核心路线：动态演化的知识图谱记忆、基于语义关联路径的联想检索、多智能体共享记忆的底层支撑。

## 当前阶段已经实现的内容

当前版本完成了中期前可展示的第一版 MVP：

- Python 工程骨架：核心代码位于 `src/sam/`，脚本位于 `scripts/`，测试位于 `tests/`，中期材料位于 `docs/`。
- 本地记忆库：`MemoryStore` 使用 SQLite 保存记忆节点、语义边和检索日志。
- Embedding 抽象层：默认使用无需依赖的本地哈希 embedding，后续可通过环境变量切换到 OpenAI 兼容 embedding API。
- 按需建图：`GraphBuilder` 只围绕被激活的种子节点创建共享实体、关键词重叠、embedding 相似等语义边。
- 两阶段检索：`Retriever` 支持纯向量检索和“向量召回 + 图扩展”的联想检索。
- 评测脚本：`Evaluator` 对比纯向量检索和联想检索的证据召回率，并输出案例分析。
- 初步实验报告：已生成 `reports/experiment_results.md` 和 `reports/experiment_results.json`。

## 目录结构

```text
SAM/
├── src/sam/
│   ├── datasets.py      # 公开基准样本适配与内置兜底样本
│   ├── embedding.py     # embedding 抽象、本地哈希实现、OpenAI 兼容实现
│   ├── evaluator.py     # 实验评测与报告生成
│   ├── graph.py         # 按需建图逻辑
│   ├── models.py        # 记忆节点、语义边、检索结果等数据结构
│   ├── retriever.py     # 纯向量检索与联想图检索
│   ├── store.py         # SQLite 本地存储
│   └── text.py          # 分词、关键词、相似度等文本工具
├── scripts/
│   └── run_demo.py      # 端到端 demo 与实验入口
├── tests/
│   └── test_core.py     # 核心单元测试与集成测试
├── docs/
│   └── midterm_progress.md
├── reports/
│   ├── experiment_results.md
│   ├── experiment_results.json
│   └── dataset_references.json
└── pyproject.toml
```

## 快速运行

所有命令都基于本地 conda 环境 `sam`：

```bash
conda run -n sam python scripts/run_demo.py --reset
```

运行后会生成：

- `reports/experiment_results.md`：适合人工阅读的实验表格与案例。
- `reports/experiment_results.json`：方便后续画图或继续分析的结构化结果。
- `reports/dataset_references.json`：公开数据集来源说明。
- `data/sam_demo.sqlite`：本地运行数据库，已被 `.gitignore` 排除。

运行测试：

```bash
conda run -n sam python -m unittest discover -s tests -v
```

尝试下载公开数据集元信息：

```bash
conda run -n sam python scripts/run_demo.py --reset --try-download
```

如果网络不可用，脚本会自动回退到内置小样本，保证 demo 可以复现。

## 实验设计

当前实验使用公开多跳问答基准结构兼容的小样本，字段设计对齐 MultiHop-RAG、HotpotQA 和 MuSiQue：每个问题包含候选文档、答案、支持文档集合。这样做的原因是当前环境不能假设网络和第三方数据集包可用，但评测流程已经按公开基准的输入输出形式设计，后续替换成真实数据下载器时不需要改核心检索逻辑。

对比方法：

- Baseline：纯向量 top-k 检索。
- SAM：先取向量种子节点，再按需建语义边，并沿图扩展得到联想检索结果。

指标：

- 支持证据召回率。
- 命中支持证据数量。
- 联想检索新增有效证据数。
- 联想路径长度和路径解释。

当前一次运行结果如下：

| 指标 | 数值 |
| --- | ---: |
| 查询数量 | 3 |
| 数据集来源数 | 3 |
| 纯向量检索证据召回率 | 0.667 |
| 联想图检索证据召回率 | 1.000 |
| 纯向量命中支持证据数 | 4 |
| 联想检索命中支持证据数 | 6 |
| 联想检索新增有效证据数 | 2 |
| 联想检索平均路径长度 | 1.50 |

这说明在当前小样本中，纯向量检索会漏掉部分不直接匹配问题、但与种子证据有明确语义关系的支持文档；联想检索通过共享实体边补回了这些证据。

## 一个案例

问题：`Which city hosts the university where the researcher who introduced Graphiti-style temporal memory studied?`

纯向量检索找到了“Temporal memory researcher profile”，但第二个结果偏向“Temporal databases overview”，漏掉了真正回答城市所需的“Fudan University location”。

联想检索先命中研究者资料节点，再发现它和复旦大学位置节点共享实体 `Fudan University`，于是沿 `shared_entity` 边补回第二个支持证据，最终得到答案 `Shanghai`。

这个例子正好对应论文想解决的问题：很多关键记忆不是和查询直接相似，而是和已经被激活的记忆相连。

## OpenAI / GPT API 接入方式

当前默认使用本地哈希 embedding，保证无依赖可跑。如果要使用 OpenAI 兼容 embedding API，可以设置：

```bash
export OPENAI_API_KEY="你的 key"
export SAM_EMBEDDING_PROVIDER="openai"
export SAM_OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
conda run -n sam python scripts/run_demo.py --reset --embedding-provider openai
```

注意：API key 只从环境变量读取，绝不能写入仓库。

## 后续迭代方向

短期内优先做三件事：

- 接入真实 MultiHop-RAG、HotpotQA、MuSiQue 子集，替换当前内置兜底样本。
- 增强节点抽取能力，从完整文档中提取摘要、关键词、实体和关系。
- 扩展评测指标，加入支持事实召回、路径质量、噪声扩展率和检索耗时。

中期答辩前继续补齐：

- 多智能体共享记忆结构：全局洞察层、会话层、交互细节层。
- 类比推理触发器：检索结构相似的历史问题和解决路径。
- 更完整的对比实验：无记忆、普通 RAG、静态图谱、动态联想图谱。

