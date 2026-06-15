from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.embedding import LocalHashEmbeddingProvider
from sam.object_graph import (
    BridgeEntity,
    CrossGraphRetriever,
    LocalEvidenceGraph,
    LocalEvidenceUnit,
    ObjectGraphBuilder,
)
from sam.store import MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 SAM 通用对象图框架 demo，不执行评测")
    parser.add_argument("--db-path", default="outputs/object_graph_demo/memory.sqlite", help="SQLite 记忆库路径")
    parser.add_argument("--output-dir", default="outputs/object_graph_demo", help="demo 产物目录")
    parser.add_argument("--reset", action="store_true", help="运行前清空 demo 记忆库")
    parser.add_argument(
        "--query",
        default="How does GraphRAG relate to cross-paper memory and code impact analysis?",
        help="跨图检索示例问题",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(args.db_path)
    try:
        if args.reset:
            store.reset()
        embedding = LocalHashEmbeddingProvider()
        builder = ObjectGraphBuilder(store, embedding)
        deltas = [builder.ingest(graph) for graph in _demo_graphs()]
        hits = CrossGraphRetriever(store, embedding).retrieve(
            query=args.query,
            top_k=6,
            seed_k=2,
            hops=2,
        )
        _write_json(output_dir / "graph_deltas.json", [delta.to_dict() for delta in deltas])
        _write_json(output_dir / "nodes.json", [node.to_dict() for node in store.get_nodes()])
        _write_json(output_dir / "edges.json", [edge.to_dict() for edge in store.get_edges()])
        _write_json(output_dir / "retrieval_hits.json", [hit.to_dict() for hit in hits])
        print("对象图 demo 已完成")
        print(f"输出目录：{output_dir}")
        print(f"节点数：{len(store.get_nodes())}")
        print(f"边数：{len(store.get_edges())}")
        print("Top hits：")
        for rank, hit in enumerate(hits, start=1):
            print(
                f"{rank}. {hit.node.metadata.get('object_id')} / "
                f"{hit.node.metadata.get('node_type')} / "
                f"{hit.node.metadata.get('title')} / score={hit.score:.3f}"
            )
    finally:
        store.close()


def _demo_graphs() -> list[LocalEvidenceGraph]:
    return [
        LocalEvidenceGraph(
            object_id="paper_graphrag",
            object_type="paper",
            title="GraphRAG Paper",
            source="demo",
            units=[
                LocalEvidenceUnit(
                    id="method",
                    node_type="method",
                    title="GraphRAG entity graph retrieval",
                    text="GraphRAG organizes documents into an entity graph and retrieves local evidence for question answering.",
                    summary="GraphRAG 使用实体图组织文档并检索局部证据。",
                    keywords=["graphrag", "entity", "graph", "retrieval"],
                    entities=[
                        BridgeEntity("GraphRAG", "graphrag", "method"),
                        BridgeEntity("Entity graph", "entity_graph", "concept"),
                    ],
                ),
                LocalEvidenceUnit(
                    id="result",
                    node_type="result",
                    title="Graph-based evidence result",
                    text="The graph retrieval process improves evidence organization for multi-hop questions.",
                    summary="图检索过程改善多跳问题的证据组织。",
                    keywords=["evidence", "multi-hop", "retrieval"],
                    entities=[
                        BridgeEntity("Multi-hop retrieval", "multi_hop_retrieval", "task"),
                    ],
                ),
            ],
        ),
        LocalEvidenceGraph(
            object_id="paper_sam",
            object_type="paper",
            title="SAM Cross-Graph Memory",
            source="demo",
            units=[
                LocalEvidenceUnit(
                    id="framework",
                    node_type="claim",
                    title="Local graph and entity bridge memory",
                    text="SAM keeps a local evidence graph for each object and connects objects through bridge entities.",
                    summary="SAM 为每个对象保留局部证据图，并通过实体桥连接对象。",
                    keywords=["sam", "local", "graph", "entity", "bridge"],
                    entities=[
                        BridgeEntity("GraphRAG", "graphrag", "method"),
                        BridgeEntity("Entity graph", "entity_graph", "concept"),
                        BridgeEntity("Cross-graph memory", "cross_graph_memory", "concept"),
                    ],
                )
            ],
        ),
        LocalEvidenceGraph(
            object_id="repo_codegraph",
            object_type="code_repository",
            title="CodeGraph Repository",
            source="demo",
            units=[
                LocalEvidenceUnit(
                    id="impact",
                    node_type="capability",
                    title="Impact analysis over code symbols",
                    text="CodeGraph indexes symbols, callers, callees and impact paths to answer codebase questions with fewer file reads.",
                    summary="CodeGraph 索引符号、调用关系和影响路径，减少代码库探索成本。",
                    keywords=["codegraph", "symbol", "impact", "call", "analysis"],
                    entities=[
                        BridgeEntity("Impact analysis", "impact_analysis", "capability"),
                        BridgeEntity("Knowledge graph", "knowledge_graph", "concept"),
                    ],
                )
            ],
        ),
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
