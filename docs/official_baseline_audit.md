# 官方 baseline 就绪状态审计

- 方法数量：3
- Ready：2
- Partial：1
- Blocked：0
- 已导出 prepared 数据集：1

## 方法状态

| 方法 | 状态 | 官方代码 | 运行入口 | 配置状态 |
| --- | --- | --- | --- | --- |
| RAPTOR | ready | 已存在 | `evaluation/official_baselines/run_raptor_official.py` | 完整 |
| Microsoft GraphRAG | ready | 已存在 | `evaluation/official_baselines/run_graphrag_official.py` | 完整 |
| HippoRAG | partial | 已存在 | `evaluation/official_baselines/run_hipporag_official.py` | 完整 |

## 配置缺口

- RAPTOR：配置变量完整。
- Microsoft GraphRAG：配置变量完整。
- HippoRAG：配置变量完整。

## 已导出数据集

- novelqa_demo：documents=120，queries=8，目录 `evaluation/runs/novelqa_demo/prepared`

## 下一步

- RAPTOR 已具备本地运行条件，可选择小样本 limit=1 做 smoke。
- Microsoft GraphRAG 已具备本地运行条件，可选择小样本 limit=1 做 smoke。
- HippoRAG 需要先修复官方依赖导入或 CLI 可用性。
