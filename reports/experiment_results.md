# SAM 初步实验结果

## 总体指标

| 指标 | 数值 |
| --- | ---: |
| 数据集来源数 | 3 |
| 查询数量 | 3 |
| 纯向量检索证据召回率 | 0.667 |
| 联想图检索证据召回率 | 1.000 |
| 纯向量命中支持证据数 | 4 |
| 联想检索命中支持证据数 | 6 |
| 联想检索新增有效证据数 | 2 |
| 联想检索平均路径长度 | 1.50 |

## 案例分析

### mh_local_001 (multihop_rag)

- 问题：Which city hosts the university where the researcher who introduced Graphiti-style temporal memory studied?
- 答案：Shanghai
- 支持文档：mh_local_001_doc_a, mh_local_001_doc_b
- 纯向量命中支持证据数：1
- 联想检索命中支持证据数：2

| 模式 | 文档 | 是否支持证据 | 分数 | 路径 | 排序原因 |
| --- | --- | --- | ---: | --- | --- |
| 纯向量 | Temporal memory researcher profile | 是 | 0.5550 | mem_aba740843626fd60 | 向量相似度=0.521 |
| 纯向量 | Temporal databases overview | 否 | 0.2526 | mem_d48b9fb112886778 | 向量相似度=0.218 |
| 联想检索 | Temporal memory researcher profile | 是 | 0.3772 | mem_aba740843626fd60 | 向量种子节点 |
| 联想检索 | Fudan University location | 是 | 0.3536 | mem_aba740843626fd60 -> mem_6047c3838453ab12 | 向量种子节点 -> shared_entity(共享实体：Fudan University) |

### mh_local_002 (musique)

- 问题：What ability is evaluated by the benchmark associated with the dataset composed from single-hop questions?
- 答案：multi-hop reasoning
- 支持文档：mh_local_002_doc_a, mh_local_002_doc_b
- 纯向量命中支持证据数：2
- 联想检索命中支持证据数：2

| 模式 | 文档 | 是否支持证据 | 分数 | 路径 | 排序原因 |
| --- | --- | --- | ---: | --- | --- |
| 纯向量 | MuSiQue construction | 是 | 0.3017 | mem_b420db98c20a7e2b | 向量相似度=0.267 |
| 纯向量 | Multi-hop reasoning benchmark | 是 | 0.1977 | mem_dd692f6ef91b1aa6 | 向量相似度=0.163 |
| 联想检索 | Multi-hop reasoning benchmark | 是 | 0.3700 | mem_b420db98c20a7e2b -> mem_dd692f6ef91b1aa6 | 向量种子节点 -> shared_entity(共享实体：multi-hop reasoning) |
| 联想检索 | MuSiQue construction | 是 | 0.2201 | mem_b420db98c20a7e2b | 向量种子节点 |

### mh_local_003 (hotpotqa)

- 问题：What evidence-chain problem is addressed by the architecture inspired by the brain structure used in long-term memory?
- 答案：multi-hop retrieval
- 支持文档：mh_local_003_doc_a, mh_local_003_doc_b
- 纯向量命中支持证据数：1
- 联想检索命中支持证据数：2

| 模式 | 文档 | 是否支持证据 | 分数 | 路径 | 排序原因 |
| --- | --- | --- | ---: | --- | --- |
| 纯向量 | HippoRAG inspiration | 是 | 0.1835 | mem_66a43a641a57d401 | 向量相似度=0.149 |
| 纯向量 | Long context window | 否 | 0.0344 | mem_ee4e16e1c78ec5af | 向量相似度=0.000 |
| 联想检索 | Multi-hop retrieval challenge | 是 | 0.2136 | mem_66a43a641a57d401 -> mem_cb521638c5613657 | 向量种子节点 -> shared_entity(共享实体：multi-hop retrieval) |
| 联想检索 | HippoRAG inspiration | 是 | 0.1468 | mem_66a43a641a57d401 | 向量种子节点 |
