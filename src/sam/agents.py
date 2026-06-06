from __future__ import annotations

from dataclasses import dataclass

from sam.embedding import EmbeddingProvider
from sam.models import MemoryNode, utc_now_iso
from sam.store import MemoryStore
from sam.text import cosine_similarity, extract_keywords, stable_id


MEMORY_LAYERS = {"global_insight", "session", "interaction"}


@dataclass(slots=True)
class AgentMemoryRecord:
    """多智能体共享记忆写入结果。"""

    node_id: str
    agent_id: str
    layer: str
    session_id: str | None
    target_agent_id: str | None = None
    task_id: str | None = None


class SharedMemoryCoordinator:
    """多智能体共享记忆的最小接口。

    开题报告规划了全局洞察层、会话层和交互细节层。当前实现先把三层
    映射到 MemoryNode.metadata，并提供统一写入和跨层查询能力。
    """

    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider

    def write_memory(
        self,
        *,
        agent_id: str,
        text: str,
        layer: str,
        session_id: str | None = None,
        source: str = "agent",
        confidence: float = 0.8,
        tags: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> AgentMemoryRecord:
        if layer not in MEMORY_LAYERS:
            raise ValueError(f"未知记忆层级：{layer}")
        extra_metadata = metadata or {}
        task_id = (
            str(extra_metadata["task_id"])
            if extra_metadata.get("task_id") is not None
            else None
        )
        memory_version = int(
            extra_metadata.get("memory_version")
            or self._next_memory_version(session_id=session_id, task_id=task_id)
        )
        keywords = extract_keywords(text, limit=8)
        node_id = stable_id(
            "agent_mem",
            (
                f"{agent_id}:{layer}:{session_id or ''}:"
                f"{extra_metadata.get('target_agent_id', '')}:{memory_version}:{text}"
            ),
        )
        now = utc_now_iso()
        node = MemoryNode(
            id=node_id,
            text=text,
            summary=text[:160],
            keywords=keywords,
            tags=[*(tags or []), "agent_memory", layer, agent_id],
            source=source,
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=confidence,
            embedding=self.embedding_provider.embed(text),
            metadata={
                "node_type": "agent_memory",
                "agent_id": agent_id,
                "memory_layer": layer,
                "session_id": session_id,
                "memory_version": memory_version,
                **extra_metadata,
            },
        )
        self.store.upsert_node(node)
        return AgentMemoryRecord(
            node_id=node_id,
            agent_id=agent_id,
            layer=layer,
            session_id=session_id,
            target_agent_id=(
                str(extra_metadata["target_agent_id"])
                if extra_metadata.get("target_agent_id") is not None
                else None
            ),
            task_id=(
                task_id
            ),
        )

    def write_handoff(
        self,
        *,
        source_agent_id: str,
        target_agent_id: str,
        text: str,
        session_id: str,
        task_id: str | None = None,
        confidence: float = 0.84,
    ) -> AgentMemoryRecord:
        """记录一个智能体传递给另一个智能体的中间结论。"""

        return self.write_memory(
            agent_id=source_agent_id,
            layer="session",
            text=text,
            session_id=session_id,
            confidence=confidence,
            tags=["handoff", f"from:{source_agent_id}", f"to:{target_agent_id}"],
            metadata={
                "source_agent_id": source_agent_id,
                "target_agent_id": target_agent_id,
                "task_id": task_id,
                "handoff": True,
            },
        )

    def query_memory(
        self,
        query: str,
        *,
        top_k: int = 5,
        layers: set[str] | None = None,
        session_id: str | None = None,
        include_other_sessions: bool = True,
        agent_id: str | None = None,
    ) -> list[MemoryNode]:
        query_embedding = self.embedding_provider.embed(query)
        allowed_layers = layers or MEMORY_LAYERS
        candidates = []
        for node in self.store.get_nodes():
            if node.metadata.get("node_type") != "agent_memory":
                continue
            if node.metadata.get("memory_layer") not in allowed_layers:
                continue
            node_session_id = node.metadata.get("session_id")
            if session_id and not include_other_sessions and node_session_id != session_id:
                continue
            target_agent_id = node.metadata.get("target_agent_id")
            if agent_id and target_agent_id and target_agent_id != agent_id:
                continue
            candidates.append(node)

        candidates.sort(
            key=lambda node: (
                cosine_similarity(query_embedding, node.embedding)
                + min(0.12, node.usage_count * 0.02)
                + node.confidence * 0.03
            ),
            reverse=True,
        )
        hits = candidates[:top_k]
        self.store.increment_usage([node.id for node in hits])
        return hits

    def resolve_conflict(
        self,
        *,
        resolver_agent_id: str,
        session_id: str,
        task_id: str,
        topic: str,
        candidate_node_ids: list[str],
    ) -> AgentMemoryRecord:
        """将同一任务下的冲突交接结论裁决为一个版本化共享记忆。"""

        candidates = self.store.get_nodes(candidate_node_ids)
        if len(candidates) < 2:
            raise ValueError("冲突裁决至少需要两个候选记忆节点")
        candidates.sort(
            key=lambda node: (
                node.confidence,
                int(node.metadata.get("memory_version", 0)),
            ),
            reverse=True,
        )
        selected = candidates[0]
        rejected = candidates[1:]
        resolution_text = (
            f"冲突主题 {topic} 已由 {resolver_agent_id} 裁决。"
            f"采纳 {selected.metadata.get('agent_id')} 的版本：{selected.text}"
        )
        record = self.write_memory(
            agent_id=resolver_agent_id,
            layer="session",
            session_id=session_id,
            text=resolution_text,
            confidence=min(0.98, max(0.7, selected.confidence + 0.04)),
            tags=["conflict_resolution", f"topic:{topic}"],
            metadata={
                "node_type": "agent_conflict_resolution",
                "task_id": task_id,
                "topic": topic,
                "resolver_agent_id": resolver_agent_id,
                "selected_node_id": selected.id,
                "rejected_node_ids": [node.id for node in rejected],
                "candidate_node_ids": [node.id for node in candidates],
                "conflict_status": "resolved",
            },
        )
        self._mark_conflict_candidates(
            selected=selected,
            rejected=rejected,
            resolution_node_id=record.node_id,
            topic=topic,
        )
        return record

    def collaboration_metrics(
        self,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, object]:
        """统计共享记忆协作过程中的版本、交接和冲突裁决情况。"""

        nodes = [
            node for node in self.store.get_nodes()
            if node.metadata.get("node_type") in {"agent_memory", "agent_conflict_resolution"}
        ]
        if session_id is not None:
            nodes = [
                node for node in nodes
                if node.metadata.get("session_id") == session_id
            ]
        if task_id is not None:
            nodes = [
                node for node in nodes
                if node.metadata.get("task_id") == task_id
            ]
        agents = {
            str(node.metadata.get("agent_id") or node.metadata.get("resolver_agent_id"))
            for node in nodes
            if node.metadata.get("agent_id") or node.metadata.get("resolver_agent_id")
        }
        versions = [
            int(node.metadata.get("memory_version", 0))
            for node in nodes
            if node.metadata.get("memory_version") is not None
        ]
        return {
            "memory_count": len(nodes),
            "handoff_count": sum(1 for node in nodes if node.metadata.get("handoff") is True),
            "conflict_resolution_count": sum(
                1
                for node in nodes
                if node.metadata.get("node_type") == "agent_conflict_resolution"
            ),
            "selected_conflict_count": sum(
                1 for node in nodes if node.metadata.get("conflict_status") == "selected"
            ),
            "rejected_conflict_count": sum(
                1 for node in nodes if node.metadata.get("conflict_status") == "rejected"
            ),
            "max_memory_version": max(versions) if versions else 0,
            "participating_agent_count": len(agents),
            "participating_agents": sorted(agents),
        }

    def _next_memory_version(
        self,
        *,
        session_id: str | None,
        task_id: str | None,
    ) -> int:
        versions: list[int] = []
        for node in self.store.get_nodes():
            if node.metadata.get("node_type") not in {"agent_memory", "agent_conflict_resolution"}:
                continue
            if session_id is not None and node.metadata.get("session_id") != session_id:
                continue
            if task_id is not None and node.metadata.get("task_id") != task_id:
                continue
            version = node.metadata.get("memory_version")
            if version is not None:
                versions.append(int(version))
        return max(versions, default=0) + 1

    def _mark_conflict_candidates(
        self,
        *,
        selected: MemoryNode,
        rejected: list[MemoryNode],
        resolution_node_id: str,
        topic: str,
    ) -> None:
        now = utc_now_iso()
        updated_nodes: list[MemoryNode] = []
        for node, status in [(selected, "selected"), *((item, "rejected") for item in rejected)]:
            metadata = {
                **node.metadata,
                "conflict_status": status,
                "conflict_topic": topic,
                "resolved_by_node_id": resolution_node_id,
            }
            updated_nodes.append(
                MemoryNode(
                    id=node.id,
                    text=node.text,
                    summary=node.summary,
                    keywords=node.keywords,
                    tags=node.tags,
                    source=node.source,
                    created_at=node.created_at,
                    last_accessed_at=node.last_accessed_at,
                    usage_count=node.usage_count,
                    confidence=node.confidence,
                    embedding=node.embedding,
                    metadata=metadata,
                )
            )
        self.store.upsert_nodes(updated_nodes)
