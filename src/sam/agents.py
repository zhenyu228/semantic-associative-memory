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
    ) -> AgentMemoryRecord:
        if layer not in MEMORY_LAYERS:
            raise ValueError(f"未知记忆层级：{layer}")
        keywords = extract_keywords(text, limit=8)
        node_id = stable_id(
            "agent_mem",
            f"{agent_id}:{layer}:{session_id or ''}:{text}",
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
            },
        )
        self.store.upsert_node(node)
        return AgentMemoryRecord(
            node_id=node_id,
            agent_id=agent_id,
            layer=layer,
            session_id=session_id,
        )

    def query_memory(
        self,
        query: str,
        *,
        top_k: int = 5,
        layers: set[str] | None = None,
        session_id: str | None = None,
        include_other_sessions: bool = True,
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
