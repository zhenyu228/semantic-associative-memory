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
        relation_pattern: list[str] | None = None,
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
        relation_paths_by_case = self._relation_paths_by_case()

        matches: list[AnalogyMatch] = []
        for case_id, nodes in self._group_nodes_by_case(candidates).items():
            if exclude_case_id and case_id == exclude_case_id:
                continue
            score, shared_keywords, path_pattern_score, matched_relation_path = self._score_case(
                query_embedding=query_embedding,
                query_keywords=query_keywords,
                nodes=nodes,
                relation_types=relation_types_by_case.get(case_id, []),
                relation_paths=relation_paths_by_case.get(case_id, []),
                relation_pattern=relation_pattern,
            )
            if score <= 0.0:
                continue
            matched_nodes = sorted(
                nodes,
                key=lambda node: cosine_similarity(query_embedding, node.embedding),
                reverse=True,
            )[:3]
            consolidated_metadata = _consolidated_case_metadata(nodes)
            relation_path_signatures = _unique_relation_paths(
                relation_paths_by_case.get(case_id, [])
            )
            matches.append(
                AnalogyMatch(
                    case_id=case_id,
                    score=score,
                    matched_nodes=matched_nodes,
                    shared_keywords=shared_keywords,
                    relation_types=relation_types_by_case.get(case_id, []),
                    prompt_hint=_build_prompt_hint(
                        case_id,
                        matched_nodes,
                        shared_keywords,
                        matched_relation_path=matched_relation_path,
                    ),
                    metadata={
                        "node_count": len(nodes),
                        "best_node_id": matched_nodes[0].id if matched_nodes else None,
                        "path_pattern_score": round(path_pattern_score, 4),
                        "matched_relation_path": matched_relation_path,
                        "relation_path_count": len(relation_path_signatures),
                        "longest_relation_path": _longest_relation_path(relation_path_signatures),
                        **consolidated_metadata,
                    },
                )
            )
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:top_k]

    def relation_pattern_for_case(self, case_id: str, *, max_length: int = 3) -> list[str]:
        """返回历史案例中最能代表结构的关系路径。"""

        relation_paths = self._relation_paths_by_case().get(case_id, [])
        normalized_paths = _unique_relation_paths(
            [
                path[:max_length]
                for path in relation_paths
                if path
            ]
        )
        if not normalized_paths:
            return []
        normalized_paths.sort(
            key=lambda path: (
                len(path),
                len(set(path)),
                _relation_path_strength(path),
                tuple(path),
            ),
            reverse=True,
        )
        return normalized_paths[0]

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

    def _relation_paths_by_case(self) -> dict[str, list[list[str]]]:
        node_to_case = {
            node.id: str(
                node.metadata.get("query_id")
                or node.metadata.get("case_id")
                or node.metadata.get("source_id")
                or node.source
            )
            for node in self.store.get_nodes()
        }
        adjacency: dict[str, list[tuple[str, str]]] = {}
        for edge in self.store.get_edges():
            source_case = node_to_case.get(edge.source_id)
            target_case = node_to_case.get(edge.target_id)
            if not source_case or source_case != target_case:
                continue
            adjacency.setdefault(edge.source_id, []).append((edge.target_id, edge.relation_type))

        paths_by_case: dict[str, list[list[str]]] = {}
        for node_id, case_id in node_to_case.items():
            for relation_path in self._walk_relation_paths(
                adjacency=adjacency,
                start_node_id=node_id,
                visited={node_id},
                relation_path=[],
                max_depth=3,
            ):
                paths_by_case.setdefault(case_id, []).append(relation_path)
        return paths_by_case

    def _walk_relation_paths(
        self,
        *,
        adjacency: dict[str, list[tuple[str, str]]],
        start_node_id: str,
        visited: set[str],
        relation_path: list[str],
        max_depth: int,
    ) -> list[list[str]]:
        if len(relation_path) >= max_depth:
            return [relation_path]
        paths: list[list[str]] = []
        for target_id, relation_type in adjacency.get(start_node_id, []):
            if target_id in visited:
                continue
            next_path = [*relation_path, relation_type]
            paths.append(next_path)
            paths.extend(
                self._walk_relation_paths(
                    adjacency=adjacency,
                    start_node_id=target_id,
                    visited={*visited, target_id},
                    relation_path=next_path,
                    max_depth=max_depth,
                )
            )
        return paths

    def _score_case(
        self,
        *,
        query_embedding: list[float],
        query_keywords: set[str],
        nodes: list[MemoryNode],
        relation_types: list[str],
        relation_paths: list[list[str]],
        relation_pattern: list[str] | None,
    ) -> tuple[float, list[str], float, list[str]]:
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
        path_pattern_score, matched_relation_path = _match_relation_pattern(
            relation_pattern,
            relation_paths,
        )
        score = (
            0.62 * best_similarity
            + 0.18 * keyword_score
            + relation_score
            + 0.22 * path_pattern_score
        )
        return score, shared_keywords, path_pattern_score, matched_relation_path


def _build_prompt_hint(
    case_id: str,
    matched_nodes: list[MemoryNode],
    shared_keywords: list[str],
    matched_relation_path: list[str] | None = None,
) -> str:
    titles = [
        str(node.metadata.get("title") or node.summary or node.id)
        for node in matched_nodes
    ]
    keyword_text = "、".join(shared_keywords[:5]) if shared_keywords else "语义结构"
    title_text = "；".join(titles[:3]) if titles else case_id
    path_text = (
        f"该案例还匹配关系路径：{' -> '.join(matched_relation_path)}。"
        if matched_relation_path
        else ""
    )
    return (
        f"当前问题可类比历史案例 {case_id}。该案例的相关记忆包括：{title_text}。"
        f"共同线索为：{keyword_text}。{path_text}"
        "可以参考该案例中的证据连接方式组织当前推理。"
    )


def _consolidated_case_metadata(nodes: list[MemoryNode]) -> dict[str, object]:
    consolidated_nodes = [
        node for node in nodes
        if node.metadata.get("node_type") == "consolidated_memory"
    ]
    if not consolidated_nodes:
        return {
            "is_consolidated_case": False,
            "case_answer": None,
            "support_node_ids": [],
            "support_titles": [],
            "evidence_node_ids": [],
            "evidence_original_doc_ids": [],
            "evidence_titles": [],
        }
    consolidated_nodes.sort(key=lambda node: node.confidence, reverse=True)
    primary = consolidated_nodes[0]
    support_node_ids = [
        str(node_id)
        for node_id in primary.metadata.get("support_node_ids", [])
    ]
    evidence_node_ids = [
        str(node_id)
        for node_id in primary.metadata.get("evidence_node_ids", [])
    ]
    return {
        "is_consolidated_case": True,
        "case_answer": primary.metadata.get("answer"),
        "support_node_ids": support_node_ids,
        "support_original_doc_ids": [
            str(node.metadata.get("original_doc_id"))
            for node in nodes
            if node.id in set(support_node_ids)
            and node.metadata.get("original_doc_id")
        ],
        "support_titles": [
            str(title)
            for title in primary.metadata.get("support_titles", [])
        ],
        "evidence_node_ids": evidence_node_ids,
        "evidence_original_doc_ids": [
            str(original_doc_id)
            for original_doc_id in primary.metadata.get("evidence_original_doc_ids", [])
        ],
        "evidence_titles": [
            str(title)
            for title in primary.metadata.get("evidence_titles", [])
        ],
        "consolidation_source": primary.metadata.get("consolidation_source"),
        "consolidated_node_id": primary.id,
        "consolidated_confidence": primary.confidence,
    }


def _match_relation_pattern(
    relation_pattern: list[str] | None,
    relation_paths: list[list[str]],
) -> tuple[float, list[str]]:
    if not relation_pattern or not relation_paths:
        return 0.0, []

    best_score = 0.0
    best_path: list[str] = []
    for path in relation_paths:
        score = _ordered_relation_overlap(relation_pattern, path)
        if score > best_score:
            best_score = score
            best_path = path
        if best_score == 1.0:
            break
    return best_score, best_path


def _ordered_relation_overlap(pattern: list[str], path: list[str]) -> float:
    if not pattern:
        return 0.0
    if len(path) >= len(pattern):
        for start in range(0, len(path) - len(pattern) + 1):
            if path[start:start + len(pattern)] == pattern:
                return 1.0

    matched = 0
    path_index = 0
    for expected in pattern:
        while path_index < len(path) and path[path_index] != expected:
            path_index += 1
        if path_index >= len(path):
            continue
        matched += 1
        path_index += 1
    return matched / len(pattern)


def _unique_relation_paths(paths: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for path in paths:
        key = tuple(path)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(list(path))
    return unique


def _longest_relation_path(paths: list[list[str]]) -> list[str]:
    if not paths:
        return []
    return max(paths, key=lambda path: (len(path), len(set(path)), tuple(path)))


def _relation_path_strength(path: list[str]) -> float:
    strength_by_type = {
        "shared_entity": 1.0,
        "context_cooccurrence": 0.75,
        "summary_parent": 0.65,
        "summary_child": 0.65,
        "keyword_overlap": 0.55,
        "embedding_similarity": 0.4,
        "analogy_case_reuse": 0.35,
    }
    if not path:
        return 0.0
    return sum(
        strength_by_type.get(relation_type, 0.5)
        for relation_type in path
    ) / len(path)
