# 官方 baseline 就绪状态审计

- 方法数量：3
- Ready：0
- Partial：3
- Blocked：0
- 已导出 prepared 数据集：1

## 方法状态

| 方法 | 状态 | 官方代码 | 运行入口 | 配置状态 |
| --- | --- | --- | --- | --- |
| RAPTOR | partial | 已存在 | `evaluation/official_baselines/run_raptor_official.py` | 不完整 |
| Microsoft GraphRAG | partial | 已存在 | `evaluation/official_baselines/run_graphrag_official.py` | 不完整 |
| HippoRAG | partial | 已存在 | `evaluation/official_baselines/run_hipporag_official.py` | 完整 |

## 配置缺口

- RAPTOR：缺少变量：RAPTOR_EMBEDDING_MODEL。
- Microsoft GraphRAG：缺少变量：GRAPHRAG_EMBEDDING_MODEL, GRAPHRAG_EMBEDDING_DEPLOYMENT。
- HippoRAG：配置变量完整。

## 已导出数据集

- novelqa_demo：documents=120，queries=8，目录 `evaluation/runs/novelqa_demo/prepared`

## 下一步

- RAPTOR 需要补齐模型配置变量后再运行官方 runner。
- RAPTOR 需要先修复官方依赖导入或 CLI 可用性。
- Microsoft GraphRAG 需要补齐模型配置变量后再运行官方 runner。
- HippoRAG 需要先修复官方依赖导入或 CLI 可用性。
