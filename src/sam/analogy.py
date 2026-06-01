from __future__ import annotations

from dataclasses import dataclass, field

from sam.embedding import EmbeddingProvider
from sam.graph import GraphBuilder
from sam.models import MemoryNode
from sam.store import MemoryStore
from sam.text import cosine_similarity, extract_keywords


@dataclass(slots=True)
class AnalogyMatch:
    """类比检索命中的历史案例。"""

    case_id: str
    score: float
    matched_nodes: list[MemoryNode]
    shared_keywords: list[str]
    relation_types: list[str]
    prompt_hint: str
    metadata: dict[str, object] = field(default_factory=dict)


class AnalogyEngine:
    """基于记忆图的类比推理触发器。

    当前版本先实现“可运行的案例式类比检索”：把同一 query_id 下的候选
    记忆视为一个历史案例，综合查询语义、关键词重叠和图关系类型匹配来排序。
    后续可以把 relation_types 扩展成更严格的子图匹配或路径同构算法。
    """

    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider,
        graph_builder: GraphBuilder,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.graph_builder = graph_builder

    def retrieve_cases(
        self,
        query: str,
        *,
        top_k: int = 3,
        exclude_case_id: str | None = None,
    ) -> list[AnalogyMatch]:
        query_embedding = self.embedding_provider.embed(query)
        query_keywords = set(extract_keywords(query, limit=10))
        candidates = [
            node for node in self.store.get_nodes()
            if node.metadata.get("node_type") != "query_summary"
        ]
        if not candidates:
            return []

        seed_nodes = sorted(
            candidates,
            key=lambda node: cosine_similarity(query_embedding, node.embedding),
            reverse=True,
        )[: max(1, min(3, len(candidates)))]
        self.graph_builder.build_edges_on_demand(seed_nodes, candidates)
        relation_types_by_case = self._relation_types_by_case()

        matches: list[AnalogyMatch] = []
        for case_id, nodes in self._group_nodes_by_case(candidates).items():
            if exclude_case_id and case_id == exclude_case_id:
                continue
            score, shared_keywords = self._score_case(
                query_embedding=query_embedding,
                query_keywords=query_keywords,
                nodes=nodes,
                relation_types=relation_types_by_case.get(case_id, []),
            )
            if score <= 0.0:
                continue
            matched_nodes = sorted(
                nodes,
                key=lambda node: cosine_similarity(query_embedding, node.embedding),
                reverse=True,
            )[:3]
            matches.append(
                AnalogyMatch(
                    case_id=case_id,
                    score=score,
                    matched_nodes=matched_nodes,
                    shared_keywords=shared_keywords,
                    relation_types=relation_types_by_case.get(case_id, []),
                    prompt_hint=_build_prompt_hint(case_id, matched_nodes, shared_keywords),
                    metadata={
                        "node_count": len(nodes),
                        "best_node_id": matched_nodes[0].id if matched_nodes else None,
                    },
                )
            )
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:top_k]

    def _group_nodes_by_case(self, nodes: list[MemoryNode]) -> dict[str, list[MemoryNode]]:
        groups: dict[str, list[MemoryNode]] = {}
        for node in nodes:
            case_id = str(
                node.metadata.get("query_id")
                or node.metadata.get("case_id")
                or node.metadata.get("source_id")
                or node.source
            )
            groups.setdefault(case_id, []).append(node)
        return groups

    def _relation_types_by_case(self) -> dict[str, list[str]]:
        node_to_case = {
            node.id: str(
                node.metadata.get("query_id")
                or node.metadata.get("case_id")
                or node.metadata.get("source_id")
                or node.source
            )
            for node in self.store.get_nodes()
        }
        relations: dict[str, set[str]] = {}
        for edge in self.store.get_edges():
            source_case = node_to_case.get(edge.source_id)
            target_case = node_to_case.get(edge.target_id)
            if source_case and source_case == target_case:
                relations.setdefault(source_case, set()).add(edge.relation_type)
        return {
            case_id: sorted(case_relations)
            for case_id, case_relations in relations.items()
        }

    def _score_case(
        self,
        *,
        query_embedding: list[float],
        query_keywords: set[str],
        nodes: list[MemoryNode],
        relation_types: list[str],
    ) -> tuple[float, list[str]]:
        best_similarity = max(
            cosine_similarity(query_embedding, node.embedding)
            for node in nodes
        )
        case_keywords = {
            keyword
            for node in nodes
            for keyword in [*node.keywords, *extract_keywords(node.summary, limit=6)]
        }
        shared_keywords = sorted(query_keywords & case_keywords)
        keyword_score = min(1.0, len(shared_keywords) / 4.0)
        relation_score = min(0.2, len(relation_types) * 0.04)
        score = 0.72 * best_similarity + 0.2 * keyword_score + relation_score
        return score, shared_keywords


def _build_prompt_hint(
    case_id: str,
    matched_nodes: list[MemoryNode],
    shared_keywords: list[str],
) -> str:
    titles = [
        str(node.metadata.get("title") or node.summary or node.id)
        for node in matched_nodes
    ]
    keyword_text = "、".join(shared_keywords[:5]) if shared_keywords else "语义结构"
    title_text = "；".join(titles[:3]) if titles else case_id
    return (
        f"当前问题可类比历史案例 {case_id}。该案例的相关记忆包括：{title_text}。"
        f"共同线索为：{keyword_text}。可以参考该案例中的证据连接方式组织当前推理。"
    )
