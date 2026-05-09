from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from sam.models import MemoryEdge, MemoryEvent, MemoryNode, RetrievalHit, utc_now_iso


class MemoryStore:
    """SQLite 本地记忆库，负责节点、边和检索日志的持久化。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_nodes (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                summary TEXT NOT NULL,
                keywords TEXT NOT NULL,
                tags TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT,
                usage_count INTEGER NOT NULL,
                confidence REAL NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                activation_count INTEGER NOT NULL DEFAULT 0,
                last_activated_at TEXT,
                metadata TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, relation_type)
            );

            CREATE TABLE IF NOT EXISTS retrieval_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL,
                hits TEXT NOT NULL,
                metadata TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                query_id TEXT,
                query TEXT NOT NULL,
                mode TEXT NOT NULL,
                node_id TEXT,
                edge_key TEXT,
                path TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL
            );
            """
        )
        self._ensure_column("memory_nodes", "last_accessed_at", "TEXT")
        self._ensure_column("memory_edges", "activation_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("memory_edges", "last_activated_at", "TEXT")
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            str(row["name"])
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def reset(self) -> None:
        self.connection.executescript(
            """
            DELETE FROM retrieval_logs;
            DELETE FROM memory_events;
            DELETE FROM memory_edges;
            DELETE FROM memory_nodes;
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def upsert_node(self, node: MemoryNode) -> None:
        self.connection.execute(
            _UPSERT_NODE_SQL,
            _node_params(node),
        )
        self.connection.commit()

    def upsert_nodes(self, nodes: Iterable[MemoryNode]) -> None:
        """批量写入节点，避免大规模实验时每条节点单独提交。"""

        self.connection.executemany(_UPSERT_NODE_SQL, [_node_params(node) for node in nodes])
        self.connection.commit()

    def get_node(self, node_id: str) -> MemoryNode | None:
        row = self.connection.execute(
            "SELECT * FROM memory_nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        return self._row_to_node(row) if row else None

    def get_nodes(self, ids: Iterable[str] | None = None) -> list[MemoryNode]:
        if ids is None:
            rows = self.connection.execute("SELECT * FROM memory_nodes").fetchall()
            return [self._row_to_node(row) for row in rows]
        node_ids = list(ids)
        if not node_ids:
            return []
        placeholders = ",".join("?" for _ in node_ids)
        rows = self.connection.execute(
            f"SELECT * FROM memory_nodes WHERE id IN ({placeholders})",
            node_ids,
        ).fetchall()
        nodes = {row["id"]: self._row_to_node(row) for row in rows}
        return [nodes[node_id] for node_id in node_ids if node_id in nodes]

    def increment_usage(self, node_ids: Iterable[str], accessed_at: str | None = None) -> None:
        accessed_at = accessed_at or utc_now_iso()
        for node_id in node_ids:
            self.connection.execute(
                """
                UPDATE memory_nodes
                SET usage_count = usage_count + 1,
                    last_accessed_at = ?
                WHERE id = ?
                """,
                (accessed_at, node_id),
            )
        self.connection.commit()

    def upsert_edge(self, edge: MemoryEdge) -> None:
        self.connection.execute(
            _UPSERT_EDGE_SQL,
            _edge_params(edge),
        )
        self.connection.commit()

    def upsert_edges(self, edges: Iterable[MemoryEdge]) -> None:
        """批量写入语义边，服务于中等规模实验。"""

        self.connection.executemany(_UPSERT_EDGE_SQL, [_edge_params(edge) for edge in edges])
        self.connection.commit()

    def activate_edges(
        self,
        edge_keys: Iterable[tuple[str, str, str]],
        activated_at: str | None = None,
    ) -> None:
        """记录本次检索真正走过的边。"""

        activated_at = activated_at or utc_now_iso()
        for source_id, target_id, relation_type in edge_keys:
            self.connection.execute(
                """
                UPDATE memory_edges
                SET activation_count = activation_count + 1,
                    last_activated_at = ?,
                    updated_at = ?,
                    weight = min(1.0, weight + 0.015)
                WHERE source_id = ? AND target_id = ? AND relation_type = ?
                """,
                (activated_at, activated_at, source_id, target_id, relation_type),
            )
        self.connection.commit()

    def adjust_edges(
        self,
        edge_keys: Iterable[tuple[str, str, str]],
        delta: float,
        updated_at: str | None = None,
    ) -> None:
        """根据反馈强化或抑制边权。"""

        updated_at = updated_at or utc_now_iso()
        for source_id, target_id, relation_type in edge_keys:
            self.connection.execute(
                """
                UPDATE memory_edges
                SET weight = min(1.0, max(0.01, weight + ?)),
                    updated_at = ?
                WHERE source_id = ? AND target_id = ? AND relation_type = ?
                """,
                (delta, updated_at, source_id, target_id, relation_type),
            )
        self.connection.commit()

    def get_edges_for(self, node_ids: Iterable[str]) -> list[MemoryEdge]:
        ids = list(node_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(
            f"""
            SELECT * FROM memory_edges
            WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
            """,
            [*ids, *ids],
        ).fetchall()
        return [self._row_to_edge(row) for row in rows]

    def get_edge(self, source_id: str, target_id: str, relation_type: str) -> MemoryEdge | None:
        row = self.connection.execute(
            """
            SELECT * FROM memory_edges
            WHERE source_id = ? AND target_id = ? AND relation_type = ?
            """,
            (source_id, target_id, relation_type),
        ).fetchone()
        return self._row_to_edge(row) if row else None

    def get_edges(self) -> list[MemoryEdge]:
        rows = self.connection.execute("SELECT * FROM memory_edges").fetchall()
        return [self._row_to_edge(row) for row in rows]

    def get_retrieval_logs(self, limit: int | None = None) -> list[dict[str, object]]:
        sql = "SELECT * FROM retrieval_logs ORDER BY id DESC"
        params: tuple[object, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self.connection.execute(sql, params).fetchall()
        return [
            {
                "id": int(row["id"]),
                "query": row["query"],
                "mode": row["mode"],
                "created_at": row["created_at"],
                "hits": json.loads(row["hits"]),
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    def log_memory_events(self, events: Iterable[MemoryEvent]) -> None:
        event_list = list(events)
        if not event_list:
            return
        self.connection.executemany(
            """
            INSERT INTO memory_events (
                event_type, query_id, query, mode, node_id, edge_key,
                path, score, created_at, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_memory_event_params(event) for event in event_list],
        )
        self.connection.commit()

    def get_memory_events(
        self,
        limit: int | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, object]]:
        sql = "SELECT * FROM memory_events"
        params: list[object] = []
        if event_type:
            sql += " WHERE event_type = ?"
            params.append(event_type)
        sql += " ORDER BY id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.connection.execute(sql, params).fetchall()
        return [
            {
                "id": int(row["id"]),
                "event_type": row["event_type"],
                "query_id": row["query_id"],
                "query": row["query"],
                "mode": row["mode"],
                "node_id": row["node_id"],
                "edge_key": json.loads(row["edge_key"]) if row["edge_key"] else None,
                "path": json.loads(row["path"]),
                "score": float(row["score"]),
                "created_at": row["created_at"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    def log_retrieval(
        self,
        query: str,
        mode: str,
        hits: list[RetrievalHit],
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO retrieval_logs (query, mode, created_at, hits, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                query,
                mode,
                utc_now_iso(),
                json.dumps([_retrieval_log_hit(hit) for hit in hits], ensure_ascii=False),
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        self.connection.commit()

    def _row_to_node(self, row: sqlite3.Row) -> MemoryNode:
        return MemoryNode(
            id=row["id"],
            text=row["text"],
            summary=row["summary"],
            keywords=json.loads(row["keywords"]),
            tags=json.loads(row["tags"]),
            source=row["source"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            usage_count=int(row["usage_count"]),
            confidence=float(row["confidence"]),
            embedding=[float(value) for value in json.loads(row["embedding"])],
            metadata=json.loads(row["metadata"]),
        )

    def _row_to_edge(self, row: sqlite3.Row) -> MemoryEdge:
        return MemoryEdge(
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=row["relation_type"],
            weight=float(row["weight"]),
            reason=row["reason"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            activation_count=int(row["activation_count"]),
            last_activated_at=row["last_activated_at"],
            metadata=json.loads(row["metadata"]),
        )


_UPSERT_NODE_SQL = """
INSERT INTO memory_nodes (
    id, text, summary, keywords, tags, source, created_at,
    last_accessed_at, usage_count, confidence, embedding, metadata
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    text=excluded.text,
    summary=excluded.summary,
    keywords=excluded.keywords,
    tags=excluded.tags,
    source=excluded.source,
    last_accessed_at=COALESCE(memory_nodes.last_accessed_at, excluded.last_accessed_at),
    confidence=excluded.confidence,
    embedding=excluded.embedding,
    metadata=excluded.metadata
"""


_UPSERT_EDGE_SQL = """
INSERT INTO memory_edges (
    source_id, target_id, relation_type, weight, reason,
    created_at, updated_at, activation_count, last_activated_at, metadata
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
    weight=max(memory_edges.weight, excluded.weight),
    reason=excluded.reason,
    updated_at=excluded.updated_at,
    activation_count=memory_edges.activation_count,
    last_activated_at=memory_edges.last_activated_at,
    metadata=excluded.metadata
"""


def _node_params(node: MemoryNode) -> tuple[object, ...]:
    return (
        node.id,
        node.text,
        node.summary,
        json.dumps(node.keywords, ensure_ascii=False),
        json.dumps(node.tags, ensure_ascii=False),
        node.source,
        node.created_at,
        node.last_accessed_at,
        node.usage_count,
        node.confidence,
        json.dumps(node.embedding),
        json.dumps(node.metadata, ensure_ascii=False),
    )


def _edge_params(edge: MemoryEdge) -> tuple[object, ...]:
    return (
        edge.source_id,
        edge.target_id,
        edge.relation_type,
        edge.weight,
        edge.reason,
        edge.created_at,
        edge.updated_at,
        edge.activation_count,
        edge.last_activated_at,
        json.dumps(edge.metadata, ensure_ascii=False),
    )


def _memory_event_params(event: MemoryEvent) -> tuple[object, ...]:
    return (
        event.event_type,
        event.query_id,
        event.query,
        event.mode,
        event.node_id,
        json.dumps(event.edge_key, ensure_ascii=False) if event.edge_key else None,
        json.dumps(event.path, ensure_ascii=False),
        event.score,
        event.created_at,
        json.dumps(event.metadata, ensure_ascii=False),
    )


def _retrieval_log_hit(hit: RetrievalHit) -> dict[str, object]:
    return {
        "node_id": hit.node.id,
        "original_doc_id": hit.node.metadata.get("original_doc_id"),
        "title": hit.node.metadata.get("title"),
        "score": round(hit.score, 4),
        "similarity_score": round(hit.similarity_score, 4),
        "graph_score": round(hit.graph_score, 4),
        "usage_score": round(hit.usage_score, 4),
        "confidence_score": round(hit.confidence_score, 4),
        "path": hit.path,
        "reason": hit.reason,
        "metadata": hit.metadata,
    }
