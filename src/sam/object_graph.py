from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from sam.embedding import EmbeddingProvider
from sam.models import MemoryEdge, MemoryNode, RetrievalHit, utc_now_iso
from sam.store import MemoryStore
from sam.text import cosine_similarity, extract_keywords


@dataclass(frozen=True, slots=True)
class BridgeEntity:
    """跨对象桥接实体，例如论文方法、数据集，或代码符号。"""

    name: str
    canonical_name: str
    entity_type: str
    aliases: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.entity_type.strip().lower(), normalize_entity_name(self.canonical_name))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "canonical_name": normalize_entity_name(self.canonical_name),
            "entity_type": self.entity_type.strip().lower(),
            "aliases": list(self.aliases),
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class LocalEvidenceUnit:
    """对象内部的证据单元，可对应论文段落、方法、实验结果、函数或测试。"""

    id: str
    node_type: str
    title: str
    text: str
    summary: str
    keywords: list[str]
    entities: list[BridgeEntity] = field(default_factory=list)
    confidence: float = 0.82
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LocalEvidenceGraph:
    """一个知识对象的局部证据图。"""

    object_id: str
    object_type: str
    title: str
    source: str
    units: list[LocalEvidenceUnit]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class GraphDelta:
    """一次对象接入或更新产生的图谱增量。"""

    object_id: str
    object_type: str
    added_node_ids: list[str] = field(default_factory=list)
    updated_node_ids: list[str] = field(default_factory=list)
    added_edge_keys: list[tuple[str, str, str]] = field(default_factory=list)
    updated_edge_keys: list[tuple[str, str, str]] = field(default_factory=list)
    added_bridge_edge_keys: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def added_node_count(self) -> int:
        return len(self.added_node_ids)

    @property
    def updated_node_count(self) -> int:
        return len(self.updated_node_ids)

    @property
    def added_edge_count(self) -> int:
        return len(self.added_edge_keys)

    @property
    def updated_edge_count(self) -> int:
        return len(self.updated_edge_keys)

    @property
    def added_bridge_edge_count(self) -> int:
        return len(self.added_bridge_edge_keys)

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "added_node_ids": self.added_node_ids,
            "updated_node_ids": self.updated_node_ids,
            "added_edge_keys": [list(key) for key in self.added_edge_keys],
            "updated_edge_keys": [list(key) for key in self.updated_edge_keys],
            "added_bridge_edge_keys": [list(key) for key in self.added_bridge_edge_keys],
        }


class ObjectGraphBuilder:
    """将领域对象写入 SAM 的通用对象图层。"""

    def __init__(self, store: MemoryStore, embedding_provider: EmbeddingProvider) -> None:
        self.store = store
        self.embedding_provider = embedding_provider

    def ingest(self, graph: LocalEvidenceGraph) -> GraphDelta:
        """写入或更新一个对象的局部图，并增量更新跨对象实体桥。"""

        delta = GraphDelta(object_id=graph.object_id, object_type=graph.object_type)
        nodes = self._nodes_for_graph(graph)
        existing_nodes = {
            node.id: self.store.get_node(node.id)
            for node in nodes
        }
        for node in nodes:
            if existing_nodes[node.id] is None:
                delta.added_node_ids.append(node.id)
            else:
                delta.updated_node_ids.append(node.id)
        self.store.upsert_nodes(nodes)

        local_edges = self._local_edges(graph, nodes)
        self._upsert_edges_with_delta(local_edges, delta)

        bridge_edges = EntityBridgeIndex(self.store).bridge_edges_for_nodes(
            [
                node
                for node in nodes
                if node.metadata.get("node_type") != "object_root"
            ]
        )
        self._upsert_edges_with_delta(bridge_edges, delta, bridge=True)
        return delta

    def _nodes_for_graph(self, graph: LocalEvidenceGraph) -> list[MemoryNode]:
        now = utc_now_iso()
        root_id = object_root_node_id(graph.object_id)
        root_metadata = {
            **graph.metadata,
            "object_id": graph.object_id,
            "object_type": graph.object_type,
            "node_type": "object_root",
            "title": graph.title,
            "bridge_entities": [],
        }
        root = MemoryNode(
            id=root_id,
            text=graph.title,
            summary=f"{graph.object_type} 对象：{graph.title}",
            keywords=extract_keywords(graph.title, limit=12),
            tags=["object_graph", graph.object_type, "object_root"],
            source=graph.source,
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.9,
            embedding=self.embedding_provider.embed(graph.title),
            metadata=root_metadata,
        )
        unit_nodes = [
            self._unit_node(graph, unit, now)
            for unit in graph.units
        ]
        return [root, *unit_nodes]

    def _unit_node(
        self,
        graph: LocalEvidenceGraph,
        unit: LocalEvidenceUnit,
        created_at: str,
    ) -> MemoryNode:
        node_id = object_unit_node_id(graph.object_id, unit.id)
        bridge_entities = [entity.to_dict() for entity in unit.entities]
        text_for_embedding = "\n".join([unit.title, unit.summary, unit.text])
        metadata = {
            **unit.metadata,
            "object_id": graph.object_id,
            "object_type": graph.object_type,
            "node_type": unit.node_type,
            "title": unit.title,
            "root_node_id": object_root_node_id(graph.object_id),
            "local_unit_id": unit.id,
            "bridge_entities": bridge_entities,
            "entities": [
                str(entity["canonical_name"])
                for entity in bridge_entities
            ],
        }
        return MemoryNode(
            id=node_id,
            text=unit.text,
            summary=unit.summary,
            keywords=unit.keywords or extract_keywords(text_for_embedding, limit=12),
            tags=["object_graph", graph.object_type, unit.node_type],
            source=graph.source,
            created_at=created_at,
            last_accessed_at=None,
            usage_count=0,
            confidence=unit.confidence,
            embedding=self.embedding_provider.embed(text_for_embedding),
            metadata=metadata,
        )

    def _local_edges(
        self,
        graph: LocalEvidenceGraph,
        nodes: list[MemoryNode],
    ) -> list[MemoryEdge]:
        now = utc_now_iso()
        root_id = object_root_node_id(graph.object_id)
        edges: list[MemoryEdge] = []
        for node in nodes:
            if node.id == root_id:
                continue
            edges.append(
                MemoryEdge(
                    source_id=root_id,
                    target_id=node.id,
                    relation_type="object_contains",
                    weight=0.64,
                    reason="对象根节点包含该局部证据单元",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                    metadata={
                        "edge_scope": "local",
                        "object_id": graph.object_id,
                        "object_type": graph.object_type,
                        "source_role": "object_root",
                        "target_node_type": node.metadata.get("node_type"),
                    },
                )
            )
            edges.append(
                MemoryEdge(
                    source_id=node.id,
                    target_id=root_id,
                    relation_type="contained_by_object",
                    weight=0.48,
                    reason="局部证据单元归属于该知识对象",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                    metadata={
                        "edge_scope": "local",
                        "object_id": graph.object_id,
                        "object_type": graph.object_type,
                        "source_node_type": node.metadata.get("node_type"),
                        "target_role": "object_root",
                    },
                )
            )
        return edges

    def _upsert_edges_with_delta(
        self,
        edges: Iterable[MemoryEdge],
        delta: GraphDelta,
        bridge: bool = False,
    ) -> None:
        edge_list = list(edges)
        if not edge_list:
            return
        for edge in edge_list:
            if self.store.get_edge(*edge.key) is None:
                delta.added_edge_keys.append(edge.key)
                if bridge:
                    delta.added_bridge_edge_keys.append(edge.key)
            else:
                delta.updated_edge_keys.append(edge.key)
        self.store.upsert_edges(edge_list)


class EntityBridgeIndex:
    """从 MemoryStore 中构建跨对象实体倒排索引。"""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def bridge_edges_for_nodes(self, nodes: list[MemoryNode]) -> list[MemoryEdge]:
        all_nodes = self.store.get_nodes()
        by_entity = self._nodes_by_entity(all_nodes)
        edges: dict[tuple[str, str, str], MemoryEdge] = {}
        for node in nodes:
            for entity in bridge_entities_for_node(node):
                for other in by_entity.get(entity_key(entity), []):
                    if other.id == node.id:
                        continue
                    if other.metadata.get("object_id") == node.metadata.get("object_id"):
                        continue
                    forward, reverse = self._bridge_edge_pair(node, other, entity)
                    edges[forward.key] = forward
                    edges[reverse.key] = reverse
        return list(edges.values())

    def connected_nodes(
        self,
        node: MemoryNode,
        entity_types: set[str] | None = None,
    ) -> list[MemoryNode]:
        all_nodes = self.store.get_nodes()
        by_entity = self._nodes_by_entity(all_nodes)
        connected: dict[str, MemoryNode] = {}
        for entity in bridge_entities_for_node(node):
            if entity_types and str(entity.get("entity_type")) not in entity_types:
                continue
            for other in by_entity.get(entity_key(entity), []):
                if other.id != node.id and other.metadata.get("object_id") != node.metadata.get("object_id"):
                    connected[other.id] = other
        return list(connected.values())

    def _nodes_by_entity(
        self,
        nodes: list[MemoryNode],
    ) -> dict[tuple[str, str], list[MemoryNode]]:
        by_entity: dict[tuple[str, str], list[MemoryNode]] = {}
        for node in nodes:
            if node.metadata.get("node_type") == "object_root":
                continue
            for entity in bridge_entities_for_node(node):
                by_entity.setdefault(entity_key(entity), []).append(node)
        return by_entity

    def _bridge_edge_pair(
        self,
        left: MemoryNode,
        right: MemoryNode,
        entity: dict[str, object],
    ) -> tuple[MemoryEdge, MemoryEdge]:
        now = utc_now_iso()
        metadata = {
            "edge_scope": "cross_object",
            "bridge_entity": entity,
            "source_object_id": left.metadata.get("object_id"),
            "target_object_id": right.metadata.get("object_id"),
            "source_object_type": left.metadata.get("object_type"),
            "target_object_type": right.metadata.get("object_type"),
        }
        reason = (
            "跨对象实体桥："
            f"{entity.get('entity_type')}={entity.get('canonical_name')}"
        )
        forward = MemoryEdge(
            source_id=left.id,
            target_id=right.id,
            relation_type="cross_object_entity_bridge",
            weight=0.72,
            reason=reason,
            created_at=now,
            updated_at=now,
            activation_count=0,
            last_activated_at=None,
            metadata=metadata,
        )
        reverse = MemoryEdge(
            source_id=right.id,
            target_id=left.id,
            relation_type="cross_object_entity_bridge",
            weight=0.72,
            reason=reason,
            created_at=now,
            updated_at=now,
            activation_count=0,
            last_activated_at=None,
            metadata={
                **metadata,
                "source_object_id": right.metadata.get("object_id"),
                "target_object_id": left.metadata.get("object_id"),
                "source_object_type": right.metadata.get("object_type"),
                "target_object_type": left.metadata.get("object_type"),
            },
        )
        return forward, reverse


class CrossGraphRetriever:
    """对象内定位与跨对象实体桥扩展结合的检索器。"""

    def __init__(self, store: MemoryStore, embedding_provider: EmbeddingProvider) -> None:
        self.store = store
        self.embedding_provider = embedding_provider

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        seed_k: int = 2,
        hops: int = 2,
        object_types: set[str] | None = None,
    ) -> list[RetrievalHit]:
        query_embedding = self.embedding_provider.embed(query)
        candidates = [
            node
            for node in self.store.get_nodes()
            if node.metadata.get("node_type") != "object_root"
            and (object_types is None or str(node.metadata.get("object_type")) in object_types)
        ]
        seed_hits = self._seed_hits(query, query_embedding, candidates, max(seed_k, 1))
        return self._expand(query, query_embedding, candidates, seed_hits, top_k, hops)

    def _seed_hits(
        self,
        query: str,
        query_embedding: list[float],
        candidates: list[MemoryNode],
        seed_k: int,
    ) -> list[RetrievalHit]:
        query_terms = set(extract_keywords(query, limit=24))
        hits: list[RetrievalHit] = []
        for node in candidates:
            similarity = cosine_similarity(query_embedding, node.embedding)
            lexical = _node_lexical_score(query_terms, node)
            score = 0.62 * similarity + 0.38 * lexical
            hits.append(
                RetrievalHit(
                    node=node,
                    score=score,
                    similarity_score=similarity,
                    graph_score=0.0,
                    usage_score=0.0,
                    confidence_score=node.confidence * 0.03,
                    path=[node.id],
                    reason=f"跨图初始定位：语义相似={similarity:.3f}，实体/词项命中={lexical:.3f}",
                    metadata={
                        "path_relation_types": [],
                        "bridge_entities": [],
                        "score_breakdown": {
                            "similarity": round(similarity, 4),
                            "lexical": round(lexical, 4),
                        },
                    },
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:seed_k]

    def _expand(
        self,
        query: str,
        query_embedding: list[float],
        candidates: list[MemoryNode],
        seed_hits: list[RetrievalHit],
        top_k: int,
        hops: int,
    ) -> list[RetrievalHit]:
        candidate_ids = {node.id for node in candidates}
        nodes_by_id = {node.id: node for node in candidates}
        best_paths: dict[str, tuple[list[str], list[str], float, list[dict[str, object]]]] = {}
        queue: deque[tuple[str, list[str], list[str], float, int, list[dict[str, object]]]] = deque()
        for hit in seed_hits:
            best_paths[hit.node.id] = ([hit.node.id], [], 0.0, [])
            queue.append((hit.node.id, [hit.node.id], [], 0.0, 0, []))

        while queue:
            current_id, path, relation_types, graph_score, depth, bridge_entities = queue.popleft()
            if depth >= hops:
                continue
            for edge in self.store.get_edges_for([current_id]):
                next_id = edge.target_id if edge.source_id == current_id else edge.source_id
                if next_id not in candidate_ids or next_id in path:
                    continue
                next_path = [*path, next_id]
                next_relation_types = [*relation_types, edge.relation_type]
                next_bridge_entities = list(bridge_entities)
                if edge.relation_type == "cross_object_entity_bridge":
                    entity = edge.metadata.get("bridge_entity")
                    if isinstance(entity, dict):
                        next_bridge_entities.append(entity)
                next_graph_score = graph_score + edge.weight / max(1, depth + 1)
                previous = best_paths.get(next_id)
                if previous is None or next_graph_score > previous[2]:
                    best_paths[next_id] = (
                        next_path,
                        next_relation_types,
                        next_graph_score,
                        next_bridge_entities,
                    )
                    queue.append(
                        (
                            next_id,
                            next_path,
                            next_relation_types,
                            next_graph_score,
                            depth + 1,
                            next_bridge_entities,
                        )
                    )

        hits: list[RetrievalHit] = []
        for node_id, (path, relation_types, graph_score, bridge_entities) in best_paths.items():
            node = nodes_by_id[node_id]
            similarity = cosine_similarity(query_embedding, node.embedding)
            cross_object_bonus = 0.12 if "cross_object_entity_bridge" in relation_types else 0.0
            score = 0.52 * similarity + 0.36 * graph_score + cross_object_bonus + node.confidence * 0.03
            hits.append(
                RetrievalHit(
                    node=node,
                    score=score,
                    similarity_score=similarity,
                    graph_score=graph_score,
                    usage_score=min(0.1, node.usage_count * 0.02),
                    confidence_score=node.confidence * 0.03,
                    path=path,
                    reason=_cross_graph_reason(relation_types, bridge_entities),
                    metadata={
                        "path_relation_types": relation_types,
                        "bridge_entities": bridge_entities,
                        "object_id": node.metadata.get("object_id"),
                        "object_type": node.metadata.get("object_type"),
                        "score_breakdown": {
                            "similarity": round(similarity, 4),
                            "graph_score": round(graph_score, 4),
                            "cross_object_bonus": round(cross_object_bonus, 4),
                        },
                    },
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        self.store.increment_usage([hit.node.id for hit in hits[:top_k]])
        self.store.activate_edges(_edge_keys_from_paths(self.store, hits[:top_k]))
        self.store.log_retrieval(
            query=query,
            mode="sam_cross_graph",
            hits=hits[:top_k],
            metadata={"top_k": top_k, "seed_k": len(seed_hits), "hops": hops},
        )
        return hits[:top_k]


def object_root_node_id(object_id: str) -> str:
    return f"{object_id}::root"


def object_unit_node_id(object_id: str, unit_id: str) -> str:
    return f"{object_id}::{unit_id}"


def normalize_entity_name(value: str) -> str:
    return "_".join(extract_keywords(value.replace("_", " "), limit=16)) or value.strip().lower()


def bridge_entities_for_node(node: MemoryNode) -> list[dict[str, object]]:
    entities = node.metadata.get("bridge_entities", [])
    if not isinstance(entities, list):
        return []
    normalized: list[dict[str, object]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        canonical_name = normalize_entity_name(str(entity.get("canonical_name") or entity.get("name") or ""))
        entity_type = str(entity.get("entity_type") or "concept").strip().lower()
        if not canonical_name:
            continue
        normalized.append(
            {
                **entity,
                "canonical_name": canonical_name,
                "entity_type": entity_type,
            }
        )
    return normalized


def entity_key(entity: dict[str, object]) -> tuple[str, str]:
    return (
        str(entity.get("entity_type") or "concept").strip().lower(),
        normalize_entity_name(str(entity.get("canonical_name") or entity.get("name") or "")),
    )


def _node_lexical_score(query_terms: set[str], node: MemoryNode) -> float:
    if not query_terms:
        return 0.0
    entity_terms = {
        token
        for entity in bridge_entities_for_node(node)
        for token in extract_keywords(str(entity.get("canonical_name", "")), limit=8)
    }
    node_terms = (
        set(node.keywords)
        | set(extract_keywords(str(node.metadata.get("title", "")), limit=12))
        | set(extract_keywords(node.summary, limit=12))
        | entity_terms
    )
    return len(query_terms & node_terms) / max(1, len(query_terms))


def _cross_graph_reason(
    relation_types: list[str],
    bridge_entities: list[dict[str, object]],
) -> str:
    if not relation_types:
        return "跨图检索：初始对象内证据节点"
    if bridge_entities:
        entity = bridge_entities[-1]
        return (
            "跨图联想：沿对象内边和跨对象实体桥扩展，"
            f"桥接实体={entity.get('entity_type')}:{entity.get('canonical_name')}"
        )
    return "跨图联想：沿对象内局部证据图扩展"


def _edge_keys_from_paths(
    store: MemoryStore,
    hits: list[RetrievalHit],
) -> list[tuple[str, str, str]]:
    edge_keys: list[tuple[str, str, str]] = []
    for hit in hits:
        for left, right in zip(hit.path, hit.path[1:], strict=False):
            candidates = [
                edge
                for edge in store.get_edges_for([left])
                if edge.source_id == left and edge.target_id == right
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda edge: edge.weight, reverse=True)
            edge_keys.append(candidates[0].key)
    return edge_keys
