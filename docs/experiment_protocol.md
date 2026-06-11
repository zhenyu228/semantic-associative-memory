# 中期阶段实验记录

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

### 1.1 模型 provider 当前状态

Embedding provider 已按公司 Azure OpenAI SDK 调用方式接入，配置从本地 `.env.local` 读取，不进入 Git 仓库。当前验证命令如下：

```bash
SAM_AZURE_EMBEDDING_TIMEOUT=20 SAM_AZURE_EMBEDDING_MAX_RETRIES=1 \
conda run -n sam python scripts/check_embedding_provider.py \
  --env-file .env.local \
  --provider azure_openai_sdk \
  --probe "SAM embedding direct SDK probe" \
  --skip-preflight \
  --json
```

当前结果：2026-06-10 将 embedding endpoint 从旧的 `search-va.byteintl.net` 切换为 `aidp-i18ntt-sg.tiktok-row.net` 后，`azure_openai_sdk` probe 已成功返回 1024 维向量，L2 范数约为 1.000。该结果说明公司 embedding 模型链路已经可用。HotpotQA 1 条真实 embedding smoke 位于 `outputs/runs/hotpotqa1_incremental_cache_smoke/`，完整跑通检索、图谱、建图成本审计和 SQLite embedding cache。扩大到 HotpotQA 30 条时触发 qpm 429 限流，因此后续采用低并发、分批预热 cache 的方式逐步完成 30 条和 300 条正式实验。GPT-5.4 chat provider 已确认可用；端到端生成实验同样需要低并发、分批运行。

当前 embedding endpoint 模板：

```bash
export SAM_AZURE_EMBEDDING_ENDPOINT="https://aidp-i18ntt-sg.tiktok-row.net/gpt/openapi/online/v2/crawl"
export SAM_AZURE_EMBEDDING_API_VERSION="2023-07-01-preview"
export SAM_AZURE_EMBEDDING_MODEL="text-embedding-3-large"
export SAM_AZURE_EMBEDDING_DIMENSIONS="1024"
export SAM_AZURE_EMBEDDING_CONCURRENCY="1"
export SAM_AZURE_EMBEDDING_RATE_LIMIT_RETRIES="30"
export SAM_AZURE_EMBEDDING_RATE_LIMIT_SLEEP_SECONDS="5"
export SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE="1"
```

### 1.2 分批预热真实 embedding cache

在线 embedding 实验不要一次性请求全部文本。先使用 `scripts/warm_embedding_cache.py` 小批量预热，确认 cache 命中数持续增加后，再运行正式检索实验。

```bash
SAM_AZURE_EMBEDDING_CONCURRENCY=1 \
SAM_AZURE_EMBEDDING_RATE_LIMIT_SLEEP_SECONDS=5 \
SAM_AZURE_EMBEDDING_RATE_LIMIT_RETRIES=5 \
SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE=1 \
conda run -n sam python scripts/warm_embedding_cache.py \
  --env-file .env.local \
  --provider azure_openai_sdk \
  --dataset-file data/processed/hotpotqa_sam_sample.json \
  --cache-path outputs/runs/hotpotqa30_embedding_cache_warmup/embedding_cache.sqlite \
  --output-dir outputs/runs/hotpotqa30_embedding_cache_warmup \
  --max-texts 20 \
  --no-query-summaries \
  --json
```

`--max-texts` 控制本次最多请求多少条缺失文本；重复执行同一条命令会从已有 cache 继续补齐。当前真实 endpoint smoke `outputs/runs/hotpotqa30_embedding_cache_warmup_budgeted_v2/` 使用 `--max-texts 3`，预热前缺失 300 条，预热后 cache hit 为 3，缺失降为 297，证明分批预热和 namespace 统计已经可用。

2026-06-10 已继续补齐 HotpotQA 30 条样本文档与 query summary 的真实 embedding cache。最终 plan 显示唯一文本数 330、cache hit 330、cache miss 0，说明 30 条实验可以在不继续请求 embedding endpoint 的情况下复现。随后运行 `outputs/runs/hotpotqa30_real_embedding_smoke_v2/`，主要结果如下：

| 方法 | 证据命中数 | 证据召回率 | 答案命中数 | 答案命中率 |
| --- | ---: | ---: | ---: | ---: |
| Embedding Top-k | 52 | 0.867 | 26 | 0.867 |
| RAPTOR | 54 | 0.900 | 27 | 0.900 |
| GraphRAG | 46 | 0.767 | 20 | 0.667 |
| HippoRAG | 54 | 0.900 | 26 | 0.867 |
| SAM-full | 53 | 0.883 | 26 | 0.867 |
| SAM-no-graph | 52 | 0.867 | 26 | 0.867 |

该 run 同时输出了按需建图成本审计：300 个文档节点的全量建图理论边数为 44850，实际唯一新建无向节点对为 1164，占比 0.025953，估算节省比例为 0.974047。平均每个 query 新建无向节点对 38.8。这个结果可以用于回答“建图成本是否过高”的问题：当前实现只围绕检索激活上下文局部建图，没有对整个候选集合做全量两两建边。

2026-06-11 已完成 HotpotQA 300 条真实 embedding 主实验。正式 run 使用 `outputs/runs/hotpotqa300_real_embedding_cache_warmup/embedding_cache.sqlite`，预热范围包含文档、query、query summary 和 RAPTOR runtime summary；最终 6063 个唯一运行文本全部 cache hit。运行目录为 `outputs/runs/hotpotqa300_real_embedding_main_v4_hops1/`。

复现命令如下：

```bash
SAM_AZURE_EMBEDDING_CONCURRENCY=1 \
SAM_AZURE_EMBEDDING_RATE_LIMIT_SLEEP_SECONDS=5 \
SAM_AZURE_EMBEDDING_RATE_LIMIT_RETRIES=5 \
SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE=1 \
conda run -n sam python scripts/run_demo.py \
  --env-file .env.local \
  --reset \
  --db outputs/runs/hotpotqa300_real_embedding_main_v4_hops1/sam.sqlite \
  --dataset hotpotqa \
  --dataset-file data/processed/hotpotqa_midterm300_sam_sample.json \
  --query-limit 300 \
  --embedding-provider azure_openai_sdk \
  --embedding-cache-path outputs/runs/hotpotqa300_real_embedding_cache_warmup/embedding_cache.sqlite \
  --methods embedding_topk,raptor_style,graphrag_style,hipporag_style,sam_full,sam_no_graph \
  --top-k 4 \
  --seed-k 1 \
  --hops 1 \
  --run-name hotpotqa300_real_embedding_main_v4_hops1
```

主要结果如下：

| 方法 | 证据命中数 | 证据召回率 | 答案命中数 | 答案命中率 |
| --- | ---: | ---: | ---: | ---: |
| Embedding Top-k | 526 | 0.877 | 272 | 0.907 |
| RAPTOR | 534 | 0.890 | 273 | 0.910 |
| GraphRAG | 477 | 0.795 | 247 | 0.823 |
| HippoRAG | 529 | 0.882 | 271 | 0.903 |
| SAM-full | 534 | 0.890 | 272 | 0.907 |
| SAM-no-graph | 526 | 0.877 | 272 | 0.907 |

结论：在真实 embedding 下，SAM-full 相比 Embedding Top-k 和 SAM-no-graph 多命中 8 个支持证据，证据召回率从 0.877 提升到 0.890。二跳扩展版本曾在 `outputs/runs/hotpotqa300_real_embedding_main_v3/` 中测试，SAM-full 证据召回率为 0.860，说明当前二跳路径噪声较高。因此稳定主实验采用一跳联想，二跳作为后续 RelationJudge 和路径重排增强后的实验方向。

按需建图成本审计显示：2992 个文档节点对应的全量建图理论边数为 4474536，SAM 实际唯一新建无向节点对为 2347，占比 0.000525，估算节省比例为 0.999475，平均每个 query 新建无向节点对 7.823。该结果可直接支撑“动态按需建图能够控制建图成本”的答辩说明。

为避免在线 embedding endpoint 阻塞实验，系统新增本地 `sentence_transformers` provider。安装可选依赖后，可以使用本地 Qwen3-Embedding-0.6B、BGE 或 E5 路径运行：

```bash
conda run -n sam python -m pip install -e ".[local-embedding]"

export SAM_SENTENCE_TRANSFORMER_MODEL="/Users/bytedance/models/Qwen3-Embedding-0.6B"
export SAM_SENTENCE_TRANSFORMER_DEVICE="cpu"
export SAM_SENTENCE_TRANSFORMER_BATCH_SIZE="8"

conda run -n sam python scripts/check_embedding_provider.py \
  --provider sentence_transformers \
  --probe "SAM local embedding probe." \
  --json
```

正式重跑 HotpotQA 300 条和 NovelQA 时，将实验命令中的 `--embedding-provider` 改为 `sentence_transformers`，并开启 `--embedding-cache`。

本地 Qwen3-Embedding 准备状态可以用以下脚本检查，该脚本只检查依赖和模型目录，不会发送在线请求：

```bash
conda run -n sam python scripts/plan_local_embedding.py \
  --model-path /Users/bytedance/models/Qwen3-Embedding-0.6B \
  --json
```

2026-06-07 检查结果显示，当前 `sam` 环境缺少 `sentence-transformers`，且 `/Users/bytedance/models/Qwen3-Embedding-0.6B` 目录不存在。因此本地 Qwen3-Embedding 也尚不能作为正式实验 provider。

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
| RAPTOR | 0.688 | 0.625 |
| GraphRAG | 0.562 | 0.500 |
| HippoRAG | 0.562 | 0.500 |
| SAM 动态联想检索 | 0.625 | 0.625 |

总体指标：

- 纯向量命中支持证据数：8
- SAM 命中支持证据数：10
- SAM 新增有效证据数：2
- SAM 平均路径长度：2.41

说明：该实验是中期展示用的小样本实验，后续主实验以 `docs/ablation_results.md` 中的 300 条 HotpotQA 消融实验为准。

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

## 5. 运行时样本限制

已有 `data/processed/*.json` 数据集文件时，`--sample-size` 只在重建数据集时生效，不会限制当前运行评测的 query 数量。为了避免 smoke 实验误跑完整数据集，当前新增 `--query-limit` 参数：

```bash
conda run -n sam python scripts/run_demo.py \
  --dataset hotpotqa \
  --dataset-file data/processed/hotpotqa_midterm300_sam_sample.json \
  --query-limit 5 \
  --methods embedding_topk,sam_full \
  --embedding-provider local \
  --reset
```

该参数只影响本次运行，不改写数据集文件；脚本会同步过滤候选文档，因此日志中的 query 数和文档数就是实际评测规模。

## 6. 模型 Provider 状态

当前本地 `.env.local` 已配置 GPT-5.4 和 embedding provider，文件被 `.gitignore` 忽略，不进入仓库。Provider 诊断命令如下：

```bash
conda run -n sam python scripts/check_model_providers.py \
  --env-file .env.local \
  --embedding-provider azure_openai_sdk \
  --chat-provider heuristic \
  --embedding-probe "SAM embedding connectivity probe" \
  --require embedding \
  --json
```

当前诊断结果显示 embedding 配置完整，但本机到 embedding endpoint 的 TCP 预检超时，因此暂不能启动正式在线 embedding 主实验。脚本已经加入网络预检，避免 endpoint 不可达时长时间挂起。

GPT-5.4 chat provider 可用，低额度验证命令如下：

```bash
conda run -n sam python scripts/check_model_providers.py \
  --env-file .env.local \
  --embedding-provider local \
  --chat-provider azure_openai_sdk \
  --chat-probe "What is the result of 1+1?" \
  --chat-max-tokens 32 \
  --require chat \
  --json
```

历史验证结果中 chat probe 返回 `2`，说明 GPT-5.4 SDK 链路可用。当前已在 `outputs/runs/relation_judge_gpt54_querylimit5_smoke/` 完成低预算 RelationJudge smoke，证明 GPT-5.4 能参与关系级建边判别流程。如果遇到 qpm 429 限流，可以设置：

```bash
export SAM_AZURE_CHAT_MAX_RETRIES=3
export SAM_AZURE_CHAT_RETRY_BASE_SECONDS=2
```

## 7. GPT-5.4 检索-生成闭环

低额度端到端生成实验命令如下：

```bash
SAM_AZURE_CHAT_TIMEOUT=30 conda run -n sam python scripts/run_end_to_end_experiment.py \
  --env-file .env.local \
  --dataset-file data/processed/hotpotqa_midterm300_sam_sample.json \
  --run-name e2e_gpt54_generation_q3_grounded_v2 \
  --limit 3 \
  --embedding-provider local \
  --chat-provider azure_openai_sdk \
  --answer-judge rule \
  --retrieval-methods embedding_topk,sam_full \
  --generation-method sam_full \
  --top-k 4 \
  --seed-k 1 \
  --hops 2 \
  --max-context-chars 5000
```

当前生成评测采用 grounding gate：生成答案必须匹配标准答案，且检索上下文也覆盖答案线索，才计为命中。这样可以防止 GPT-5.4 凭外部知识答对但检索证据缺失的样本被误算为系统成功。

本轮 run 位于 `outputs/runs/e2e_gpt54_generation_q3_grounded_v2/`，3 条样本中 grounded 生成命中 1 条，命中率为 0.333。bad case 中包含一条 `ungrounded_generated_answer`，说明模型可以答对 `Chief of Protocol`，但检索上下文没有包含该证据，因此不计为闭环成功。

## 8. 当前结论

当前中期展示实验可以支持以下表述：

- 已完成 SAM 统一数据格式和公开数据集接入。
- 已完成基于 SQLite 的记忆节点、语义边和检索日志存储。
- 已完成按需建图、动态状态更新和联想检索原型。
- 已在 HotpotQA 小样本上观察到 SAM 相比纯向量检索新增命中支持证据。
- 已完成 NovelQA demonstration 的长文本数据接入和可视化闭环。

## 8. 连续记忆复用实验

该实验用于验证“状态与反馈驱动的记忆演化”是否能影响后续检索。实验分为 warmup 和 probe 两阶段：warmup 使用正常候选集，让 SAM 形成巩固记忆；probe 将每个问题的 gold 支持文档从候选集中移除，再观察系统能否通过历史巩固记忆把证据带回候选池。

真实 embedding run 位于 `outputs/runs/memory_reuse_hotpotqa30_real_embedding_v3/`。复现命令如下：

```bash
SAM_AZURE_EMBEDDING_CONCURRENCY=1 \
SAM_AZURE_EMBEDDING_RATE_LIMIT_SLEEP_SECONDS=5 \
SAM_AZURE_EMBEDDING_RATE_LIMIT_RETRIES=5 \
SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE=1 \
conda run -n sam python scripts/run_memory_reuse_experiment.py \
  --env-file .env.local \
  --dataset-file data/processed/hotpotqa_midterm300_sam_sample.json \
  --limit 30 \
  --embedding-provider azure_openai_sdk \
  --embedding-cache-path outputs/runs/hotpotqa300_real_embedding_cache_warmup/embedding_cache.sqlite \
  --hops 1 \
  --run-name memory_reuse_hotpotqa30_real_embedding_v3
```

主要结果如下：

| 方法 | 证据召回率 | 答案命中率 |
| --- | ---: | ---: |
| Embedding Top-k | 0.000 | 0.300 |
| SAM-full | 0.900 | 0.867 |
| SAM-no-memory-state | 0.900 | 0.867 |
| SAM-no-feedback | 0.900 | 0.867 |
| SAM-static-graph | 0.900 | 0.867 |

该 run 生成 `memory_events.md` 和 `feedback_edge_changes.md`。事件流记录 `support_hit` 108 条、`answer_hit` 48 条、`path_rejected` 131 条、`memory_consolidated` 60 条；边变化案例显示部分巩固边和共享实体边在 probe 后被增强。该结果可用于说明动态记忆能够在连续任务中被后续查询读取，并改变候选证据和边状态。

## 9. GPT-5.4 多智能体生成对照

低额度 q1 验证命令如下：

```bash
SAM_AZURE_CHAT_TIMEOUT=30 \
SAM_AZURE_CHAT_MAX_RETRIES=2 \
SAM_AZURE_CHAT_RETRY_BASE_SECONDS=5 \
conda run -n sam python scripts/run_agent_generation_experiment.py \
  --env-file .env.local \
  --cases-file outputs/runs/fair_ablation_hotpotqa_300/cases.json \
  --all-cases-file outputs/runs/fair_ablation_hotpotqa_300/cases.json \
  --method sam_full \
  --chat-provider azure_openai_sdk \
  --embedding-provider local \
  --limit 1 \
  --analogy-top-k 1 \
  --output-dir outputs/runs/agent_generation_gpt54_q1
```

当前运行结果位于 `outputs/runs/agent_generation_gpt54_q1/`。GPT-5.4 三个生成变体均遇到 qpm 429 限流，系统已将失败写入 `agent_generation_comparison.json` 和 `generation_bad_cases/generation_bad_cases.json`，bad case 类型为 `generation_error`。该结果说明实验入口和错误审计链路已经打通，但不作为方法效果结论。后续在限流恢复后，将同一命令的 `--limit` 逐步提高到 3、10，并比较 `baseline`、`shared_memory`、`shared_memory_with_analogy` 的 grounded answer hit rate。
