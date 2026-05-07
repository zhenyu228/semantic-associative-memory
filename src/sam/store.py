from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from sam.models import MemoryEdge, MemoryNode, RetrievalHit, utc_now_iso


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
            """
        )
        self.connection.commit()

    def reset(self) -> None:
        self.connection.executescript(
            """
            DELETE FROM retrieval_logs;
            DELETE FROM memory_edges;
            DELETE FROM memory_nodes;
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def upsert_node(self, node: MemoryNode) -> None:
        self.connection.execute(
            """
            INSERT INTO memory_nodes (
                id, text, summary, keywords, tags, source, created_at,
                usage_count, confidence, embedding, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                text=excluded.text,
                summary=excluded.summary,
                keywords=excluded.keywords,
                tags=excluded.tags,
                source=excluded.source,
                confidence=excluded.confidence,
                embedding=excluded.embedding,
                metadata=excluded.metadata
            """,
            (
                node.id,
                node.text,
                node.summary,
                json.dumps(node.keywords, ensure_ascii=False),
                json.dumps(node.tags, ensure_ascii=False),
                node.source,
                node.created_at,
                node.usage_count,
                node.confidence,
                json.dumps(node.embedding),
                json.dumps(node.metadata, ensure_ascii=False),
            ),
        )
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

    def increment_usage(self, node_ids: Iterable[str]) -> None:
        for node_id in node_ids:
            self.connection.execute(
                "UPDATE memory_nodes SET usage_count = usage_count + 1 WHERE id = ?",
                (node_id,),
            )
        self.connection.commit()

    def upsert_edge(self, edge: MemoryEdge) -> None:
        self.connection.execute(
            """
            INSERT INTO memory_edges (
                source_id, target_id, relation_type, weight, reason,
                created_at, updated_at, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                weight=max(memory_edges.weight, excluded.weight),
                reason=excluded.reason,
                updated_at=excluded.updated_at,
                metadata=excluded.metadata
            """,
            (
                edge.source_id,
                edge.target_id,
                edge.relation_type,
                edge.weight,
                edge.reason,
                edge.created_at,
                edge.updated_at,
                json.dumps(edge.metadata, ensure_ascii=False),
            ),
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

    def get_edges(self) -> list[MemoryEdge]:
        rows = self.connection.execute("SELECT * FROM memory_edges").fetchall()
        return [self._row_to_edge(row) for row in rows]

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
                json.dumps([hit.to_dict() for hit in hits], ensure_ascii=False),
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
            metadata=json.loads(row["metadata"]),
        )
