# SAM 通用对象图框架设计

本文档记录 SAM 从“问答数据集驱动”升级为“对象图驱动”的框架设计。科研论文是第一类落地场景，代码仓库、项目文档、法律文档等也可以作为同一框架下的知识对象接入。

## 1. 核心定义

SAM 当前新增的框架层由四个概念组成：

- `LocalEvidenceGraph`：一个知识对象的局部证据图，例如一篇论文或一个代码仓库。
- `LocalEvidenceUnit`：对象内部的证据单元，例如论文中的方法、实验结果、局限性，或代码仓库中的函数、测试、API。
- `BridgeEntity`：跨对象连接实体，例如论文中的方法、任务、数据集、指标，或代码中的符号、接口、模型。
- `GraphDelta`：一次对象接入或更新产生的增量，记录新增节点、更新节点、新增边和跨对象桥接边。

底层仍然使用 `MemoryNode` 和 `MemoryEdge`，新框架通过 `metadata.object_id`、`metadata.object_type`、`metadata.node_type` 和 `metadata.bridge_entities` 标记对象归属、节点类型和桥接实体。

## 2. 双层图结构

SAM 的对象图不是把所有内容混成一个全局图，而是拆成两层：

第一层是对象内局部图。每篇论文、每个代码仓库或每个文档集合都先独立建图，保留对象内部结构。例如论文内部可以包含章节、方法、数据集、指标、结果和引用；代码仓库内部可以包含目录、文件、类、函数、调用、API 和测试。

第二层是跨对象桥接图。不同对象之间不做全量两两相似度连接，而是通过标准化实体建立桥接边。例如两篇论文共同使用 `GraphRAG` 方法，或一个代码文件和一个测试文件共同指向 `login_user` 符号，就可以通过 `cross_object_entity_bridge` 连接。

## 3. 动态更新

当新对象进入系统时，SAM 不需要重建全局图。`ObjectGraphBuilder` 的流程是：

1. 为新对象创建或更新对象根节点。
2. 为对象内部证据单元创建或更新局部节点。
3. 创建对象根节点和局部节点之间的局部边。
4. 读取局部节点中的 `BridgeEntity`。
5. 只围绕这些新节点或更新节点查找已有实体索引。
6. 对命中的跨对象实体增量创建桥接边。

这个过程的输出是 `GraphDelta`。它可以用于审计本次更新到底新增了多少节点、更新了多少节点、创建了多少对象内边和跨对象桥接边。

## 4. 跨图联想检索

`CrossGraphRetriever` 的检索流程是：

1. 对用户问题进行初始向量和词项定位，找到种子证据节点。
2. 在种子所在对象内部沿局部图扩展。
3. 遇到跨对象实体桥时跳转到其他对象。
4. 进入目标对象内部继续查找证据。
5. 返回包含路径、关系类型、桥接实体和得分拆解的检索结果。

因此 SAM 的联想不再只是“相似 chunk 扩展”，而是：

```text
对象内证据定位 -> 跨对象实体桥跳转 -> 目标对象内证据补全
```

在科研论文场景中，这可以表达“论文 A 的方法实体连接到论文 B 的比较实验”；在代码仓库场景中，这可以表达“函数符号连接到调用方、测试和影响路径”。

## 5. 当前代码入口

核心实现位于：

- `src/sam/object_graph.py`
- `scripts/run_object_graph_demo.py`
- `tests/test_core.py` 中的对象图框架测试

可以运行下面命令查看 demo 产物：

```bash
conda run -n sam python scripts/run_object_graph_demo.py --reset
```

运行后会在 `outputs/object_graph_demo/` 下生成：

- `graph_deltas.json`
- `nodes.json`
- `edges.json`
- `retrieval_hits.json`

这些文件只用于框架结构检查，不属于正式评测结果。

## 6. 后续适配方向

下一步可以在不改变底层框架的前提下增加不同领域 adapter：

- `QasperObjectAdapter`：将 QASPER 论文转成 `LocalEvidenceGraph`。
- `PaperPdfObjectAdapter`：从 PDF 或 S2ORC 解析论文结构。
- `CodeRepositoryObjectAdapter`：从代码仓库解析文件、符号、调用和测试。

正式实验可以先聚焦科研论文场景；代码仓库场景保留为框架可扩展性说明。
