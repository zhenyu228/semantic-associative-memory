# 官方 baseline 评测目录

这个目录只放 SAM 对官方 baseline 的评测适配代码，不把 RAPTOR、GraphRAG、HippoRAG 的官方源码直接提交到本仓库。官方源码会下载到 `evaluation/external/`，实验结果会写到 `evaluation/runs/`，这两个目录都已经被 `.gitignore` 忽略。

## 官方代码来源

| 方法 | 官方仓库 | 说明 |
| --- | --- | --- |
| RAPTOR | https://github.com/parthsarthi03/raptor | 官方 `RetrievalAugmentation` 实现递归聚类、摘要树和 QA |
| Microsoft GraphRAG | https://github.com/microsoft/graphrag | 官方 CLI/包实现实体关系抽取、社区摘要、local/global/drift 查询 |
| HippoRAG | https://github.com/OSU-NLP-Group/HippoRAG | 官方 `HippoRAG` 类实现 KG、PPR 检索和 RAG QA |

## 1. 下载官方仓库

```bash
conda run -n sam python evaluation/official_baselines/fetch_official_repos.py
```

下载位置：

```text
evaluation/external/
├── raptor/
├── graphrag/
└── hipporag/
```

注意：这里只是下载官方代码。依赖安装仍按各官方 README 执行，因为 GraphRAG、RAPTOR、HippoRAG 依赖较重，而且通常需要 `OPENAI_API_KEY` 或本地模型服务。

## 1.1 当前本机依赖安装状态

本机已经创建隔离环境，避免污染 `sam` conda 环境：

```text
evaluation/.venvs/raptor
evaluation/.venvs/graphrag
evaluation/.venvs/hipporag
```

当前状态：

- RAPTOR：已安装依赖，并已验证 `from raptor import RetrievalAugmentation` 可导入。
- GraphRAG：已安装官方 `graphrag==3.0.9`，并已验证 CLI 可用。
- HippoRAG：暂不安装。官方依赖包含 `vllm==0.6.6.post1`，在当前 macOS arm64 环境会进入源码构建并卡在大依赖下载；官方 `requirements.txt` 还固定了 PyPI 上不存在的 `openai==1.91.1`。建议在 Linux/CUDA 环境单独安装。

如果需要重新安装：

```bash
conda run -n sam python -m venv evaluation/.venvs/raptor
evaluation/.venvs/raptor/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r evaluation/external/raptor/requirements.txt
evaluation/.venvs/raptor/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "huggingface-hub==0.25.2"

conda run -n sam python -m venv evaluation/.venvs/graphrag
evaluation/.venvs/graphrag/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple graphrag
```

## 1.2 API key 配置

复制模板到本地文件：

```bash
cp evaluation/official_baselines/env.template evaluation/official_baselines/.env.local
```

编辑 `.env.local`，填入真实 key：

```bash
export OPENAI_API_KEY="your-company-api-key"
export OPENAI_BASE_URL="https://your-company-openai-compatible-base-url/v1"

export RAPTOR_QA_MODEL="your-chat-model"
export RAPTOR_SUMMARY_MODEL="$RAPTOR_QA_MODEL"
export RAPTOR_EMBEDDING_MODEL="your-embedding-model"

export GRAPHRAG_API_KEY="$OPENAI_API_KEY"
export GRAPHRAG_API_BASE="$OPENAI_BASE_URL"
export GRAPHRAG_MODEL_PROVIDER="openai"
export GRAPHRAG_CHAT_MODEL="$RAPTOR_QA_MODEL"
export GRAPHRAG_EMBEDDING_MODEL="$RAPTOR_EMBEDDING_MODEL"
```

如果公司网关不是普通 `/v1/chat/completions`，而是 Azure-style deployment 路径，例如：

```text
{base_url}/openai/deployments/{deployment}/chat/completions?api-version=2024-02-01
```

则建议这样配置：

```bash
export GPT54_API_KEY="your-company-api-key"
export GPT54_BASE_URL="https://your-company-gateway.example.com/path"
export GPT54_API_VERSION="2024-02-01"
export GPT54_MODEL="your-chat-deployment"

export EMBEDDING_API_KEY="your-embedding-api-key"
export EMBEDDING_BASE_URL="https://your-embedding-gateway.example.com/path"
export EMBEDDING_API_VERSION="2023-07-01-preview"
export EMBEDDING_MODEL="text-embedding-3-large"
export EMBEDDING_DIMENSIONS="1024"

export OPENAI_API_KEY="$GPT54_API_KEY"
export RAPTOR_CLIENT_TYPE="azure"
export RAPTOR_AZURE_ENDPOINT="$GPT54_BASE_URL"
export RAPTOR_API_VERSION="$GPT54_API_VERSION"
export RAPTOR_QA_MODEL="$GPT54_MODEL"
export RAPTOR_SUMMARY_MODEL="$GPT54_MODEL"
export RAPTOR_EMBEDDING_MODEL="$EMBEDDING_MODEL"
export RAPTOR_EMBEDDING_API_KEY="$EMBEDDING_API_KEY"
export RAPTOR_EMBEDDING_AZURE_ENDPOINT="$EMBEDDING_BASE_URL"
export RAPTOR_EMBEDDING_API_VERSION="$EMBEDDING_API_VERSION"
export RAPTOR_EMBEDDING_DIMENSIONS="$EMBEDDING_DIMENSIONS"

export GRAPHRAG_API_KEY="$GPT54_API_KEY"
export GRAPHRAG_API_BASE="$GPT54_BASE_URL"
export GRAPHRAG_API_VERSION="$GPT54_API_VERSION"
export GRAPHRAG_MODEL_PROVIDER="azure"
export GRAPHRAG_CHAT_MODEL="$GPT54_MODEL"
export GRAPHRAG_CHAT_DEPLOYMENT="$GPT54_MODEL"
export GRAPHRAG_EMBEDDING_MODEL="$EMBEDDING_MODEL"
export GRAPHRAG_EMBEDDING_DEPLOYMENT="$EMBEDDING_MODEL"
export GRAPHRAG_EMBEDDING_API_KEY="$EMBEDDING_API_KEY"
export GRAPHRAG_EMBEDDING_API_BASE="$EMBEDDING_BASE_URL"
export GRAPHRAG_EMBEDDING_API_VERSION="$EMBEDDING_API_VERSION"
```

每次运行官方 baseline 前，在同一个终端执行：

```bash
source evaluation/official_baselines/.env.local
```

可以先做一次不泄露 key 的连通性测试：

```bash
source evaluation/official_baselines/.env.local
conda run -n sam python evaluation/official_baselines/test_company_api.py --timeout 8
```

如果暂时没有 embedding deployment，脚本会只测试 chat，然后提示 `embedding_ok=skipped`。RAPTOR 和 GraphRAG 的正式检索/建索引都需要 embedding 模型；只有 GPT-5.4 这类 chat 模型还不够。

说明：

- RAPTOR 普通模式读取 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`；Azure-style 模式由本仓库 runner 显式使用 Azure-style HTTP 请求。若 chat 和 embedding 不在同一个 endpoint，使用 `RAPTOR_EMBEDDING_API_KEY`、`RAPTOR_EMBEDDING_AZURE_ENDPOINT`、`RAPTOR_EMBEDDING_API_VERSION` 和 `RAPTOR_EMBEDDING_DIMENSIONS`。
- RAPTOR 需要 chat/summary 模型和 embedding 模型。没有 embedding deployment 时，不能完整构建 RAPTOR 摘要树。
- GraphRAG 官方配置读取 `GRAPHRAG_API_KEY`，本仓库 runner 会把 `GRAPHRAG_API_BASE`、`GRAPHRAG_API_VERSION`、`GRAPHRAG_CHAT_MODEL`、`GRAPHRAG_EMBEDDING_MODEL`、deployment 信息自动写入该次运行的 `settings.yaml`。若 embedding 使用独立网关，设置 `GRAPHRAG_EMBEDDING_API_KEY`、`GRAPHRAG_EMBEDDING_API_BASE` 和 `GRAPHRAG_EMBEDDING_API_VERSION`。
- `.env.local` 不会提交到 Git。
- `OPENAI_BASE_URL` / `GRAPHRAG_API_BASE` 通常需要带 `/v1`。如果公司网关地址已经内置 `/v1`，不要重复写成 `/v1/v1`。
- 如果公司网关要求额外 header、AK/SK 签名或非 OpenAI-compatible 路径，需要额外写适配层；只改 base url 不够。

## 2. 导出 SAM 数据为官方评测格式

以 NovelQA demonstration 为例：

```bash
conda run -n sam python evaluation/official_baselines/export_sam_for_official.py \
  --dataset-file data/processed/novelqa_demo_sam_sample.json \
  --dataset-name novelqa_demo
```

输出：

```text
evaluation/runs/novelqa_demo/prepared/
├── common/
│   ├── documents.json
│   └── queries.json
├── hipporag/
│   ├── novelqa_demo_corpus.json
│   └── novelqa_demo.json
├── graphrag/
│   ├── input/
│   └── questions.json
├── raptor/
│   ├── corpus.txt
│   └── queries.json
└── manifest.json
```

`common/` 是我们自己的统一评测输入；其他目录是给官方方法准备的格式。

## 2.1 审计官方 baseline 就绪状态

正式运行官方 baseline 前，先做本地审计。该脚本不会调用外部 API，也不会输出 API key 或 endpoint 明文，只检查官方仓库、隔离环境、导入/CLI、prepared 数据和模型配置变量是否齐全：

```bash
conda run -n sam python evaluation/official_baselines/audit_official_baselines.py \
  --env-file .env.local \
  --env-file evaluation/official_baselines/.env.local \
  --output-dir docs
```

输出：

```text
docs/official_baseline_audit.json
docs/official_baseline_audit.md
```

当前审计结果显示：RAPTOR、Microsoft GraphRAG 和 HippoRAG 官方仓库均已下载，NovelQA demonstration 已导出为 prepared 数据；GraphRAG 官方 CLI 可运行。审计脚本会把根目录 `.env.local` 中的 `SAM_AZURE_EMBEDDING_*` 自动映射到官方 baseline 所需的 embedding 变量。RAPTOR 导入检查在当前本机可能超过 30 秒，HippoRAG 官方依赖在 macOS arm64 环境未完整安装。

## 3. 运行官方 baseline

### RAPTOR

先按官方 README 安装依赖，然后运行：

```bash
source evaluation/official_baselines/.env.local
evaluation/.venvs/raptor/bin/python evaluation/official_baselines/run_raptor_official.py \
  --prepared-dir evaluation/runs/novelqa_demo/prepared \
  --client-type "$RAPTOR_CLIENT_TYPE" \
  --qa-model "$RAPTOR_QA_MODEL" \
  --summary-model "$RAPTOR_SUMMARY_MODEL" \
  --embedding-model "$RAPTOR_EMBEDDING_MODEL" \
  --limit 8
```

说明：RAPTOR 官方高层 API 主要返回答案文本，不稳定返回文档 id。因此脚本会计算答案命中率，文档命中只做诊断字段，不把它伪装成严格 evidence recall。

### GraphRAG

先安装官方包并准备 LLM 配置：

```bash
source evaluation/official_baselines/.env.local
```

然后运行：

```bash
evaluation/.venvs/graphrag/bin/python evaluation/official_baselines/run_graphrag_official.py \
  --prepared-dir evaluation/runs/novelqa_demo/prepared \
  --query-method local \
  --model-provider "$GRAPHRAG_MODEL_PROVIDER" \
  --api-base "$GRAPHRAG_API_BASE" \
  --api-version "$GRAPHRAG_API_VERSION" \
  --chat-model "$GRAPHRAG_CHAT_MODEL" \
  --embedding-model "$GRAPHRAG_EMBEDDING_MODEL" \
  --limit 8
```

说明：GraphRAG 官方 CLI 会创建自己的 `settings.yaml` 和索引目录；如果配置里没有模型/API key，官方索引会失败，这是官方方法本身的运行要求。

### HippoRAG

当前 Mac 本机暂不运行 HippoRAG 官方实现。推荐在 Linux/CUDA 环境按官方 README 安装，然后运行：

```bash
conda run -n sam python evaluation/official_baselines/run_hipporag_official.py \
  --prepared-dir evaluation/runs/novelqa_demo/prepared \
  --limit 8
```

说明：HippoRAG 官方实现通常需要 embedding 模型和 LLM 配置。默认参数参考官方 README：`gpt-4o-mini` 与 `nvidia/NV-Embed-v2`。

## 重要边界

- 这里不再使用 `src/sam/retriever.py` 里的 `*-style` baseline。
- `*-style` baseline 可以保留为快速 smoke test，但论文实验应以本目录的官方 baseline 结果为准。
- 官方方法通常需要额外依赖、API key 或 GPU，本目录只负责把 SAM 数据转换成它们能读的格式，并提供可复现运行入口。
