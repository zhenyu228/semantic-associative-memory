from __future__ import annotations

from sam.models import EvaluationQuery, MemoryEvent, RetrievalHit, utc_now_iso
from sam.store import MemoryStore


class FeedbackUpdater:
    """根据评测反馈更新动态记忆事件和边权。"""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def apply(
        self,
        *,
        query: EvaluationQuery,
        mode: str,
        hits: list[RetrievalHit],
        support_node_ids: set[str],
        answer_status: str,
    ) -> None:
        created_at = utc_now_iso()
        events: list[MemoryEvent] = []
        positive_edge_keys: list[tuple[str, str, str]] = []
        negative_edge_keys: list[tuple[str, str, str]] = []

        for rank, hit in enumerate(hits, start=1):
            edge_keys = self._edge_keys_from_path(hit.path)
            is_support = hit.node.id in support_node_ids
            if is_support:
                positive_edge_keys.extend(edge_keys)
                events.append(
                    MemoryEvent(
                        event_type="support_hit",
                        query_id=query.id,
                        query=query.question,
                        mode=mode,
                        node_id=hit.node.id,
                        path=[str(node_id) for node_id in hit.path],
                        score=hit.score,
                        created_at=created_at,
                        metadata={
                            "rank": rank,
                            "answer": query.answer,
                            "feedback_delta": 0.025,
                        },
                    )
                )
            elif len(hit.path) > 1:
                negative_edge_keys.extend(edge_keys)
                events.append(
                    MemoryEvent(
                        event_type="path_rejected",
                        query_id=query.id,
                        query=query.question,
                        mode=mode,
                        node_id=hit.node.id,
                        path=[str(node_id) for node_id in hit.path],
                        score=hit.score,
                        created_at=created_at,
                        metadata={
                            "rank": rank,
                            "feedback_delta": -0.004,
                            "reason": "expanded path did not hit supporting evidence",
                        },
                    )
                )

        if answer_status in {"found_in_retrieved_context", "matched_option"}:
            events.append(
                MemoryEvent(
                    event_type="answer_hit",
                    query_id=query.id,
                    query=query.question,
                    mode=mode,
                    created_at=created_at,
                    score=1.0,
                    metadata={"answer": query.answer, "answer_status": answer_status},
                )
            )

        self.store.adjust_edges(positive_edge_keys, delta=0.025, updated_at=created_at)
        self.store.adjust_edges(negative_edge_keys, delta=-0.004, updated_at=created_at)
        self.store.log_memory_events(events)

    def _edge_keys_from_path(self, path: list[str]) -> list[tuple[str, str, str]]:
        path_pairs = list(zip(path, path[1:], strict=False))
        if not path_pairs:
            return []
        involved_node_ids = {node_id for pair in path_pairs for node_id in pair}
        best_by_pair: dict[tuple[str, str], tuple[str, str, str, float]] = {}
        for edge in self.store.get_edges_for(involved_node_ids):
            pair = (edge.source_id, edge.target_id)
            if pair not in path_pairs:
                continue
            previous = best_by_pair.get(pair)
            if previous is None or edge.weight > previous[3]:
                best_by_pair[pair] = (edge.source_id, edge.target_id, edge.relation_type, edge.weight)
        return [
            (source_id, target_id, relation_type)
            for source_id, target_id, relation_type, _ in best_by_pair.values()
        ]
