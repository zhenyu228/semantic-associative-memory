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
export OPENAI_API_KEY="sk-your-openai-api-key"
export GRAPHRAG_API_KEY="$OPENAI_API_KEY"
```

每次运行官方 baseline 前，在同一个终端执行：

```bash
source evaluation/official_baselines/.env.local
```

说明：

- RAPTOR 官方代码读取 `OPENAI_API_KEY`。
- GraphRAG 官方配置读取 `GRAPHRAG_API_KEY`。
- `.env.local` 不会提交到 Git。
- 如果使用 OpenAI-compatible 代理，GraphRAG 可以在它生成的 `settings.yaml` 中改模型 provider/base url；RAPTOR 官方代码当前主要按 OpenAI 官方 SDK 直连方式写，base url 支持不如 GraphRAG 清晰。

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

## 3. 运行官方 baseline

### RAPTOR

先按官方 README 安装依赖，然后运行：

```bash
source evaluation/official_baselines/.env.local
evaluation/.venvs/raptor/bin/python evaluation/official_baselines/run_raptor_official.py \
  --prepared-dir evaluation/runs/novelqa_demo/prepared \
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
