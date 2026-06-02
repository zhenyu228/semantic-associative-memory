from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sam.badcase import BadCaseAnalyzer, write_bad_case_reports
from sam.datasets import build_query_summary_nodes, documents_to_nodes
from sam.embedding import EmbeddingProvider
from sam.consolidation import MemoryConsolidator
from sam.graph import GraphBuilder
from sam.models import DatasetDocument, EvaluationQuery, MemoryNode, RetrievalHit
from sam.query_planner import QueryPlanner
from sam.retriever import RETRIEVAL_METHOD_NAMES, Retriever
from sam.store import MemoryStore
from sam.feedback import FeedbackUpdater


DEFAULT_EVALUATION_METHODS = [
    "embedding_topk",
    "raptor_style",
    "graphrag_style",
    "hipporag_style",
    "sam",
]


@dataclass(slots=True)
class ExperimentResult:
    dataset_count: int
    query_count: int
    document_count: int
    total_supporting_evidence: int
    vector_recall: float
    associative_recall: float
    vector_support_hits: int
    associative_support_hits: int
    associative_gain: int
    average_path_length: float
    method_metrics: dict[str, dict[str, object]]
    ablation_metrics: dict[str, dict[str, object]]
    cases: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_count": self.dataset_count,
            "query_count": self.query_count,
            "document_count": self.document_count,
            "total_supporting_evidence": self.total_supporting_evidence,
            "vector_recall": self.vector_recall,
            "associative_recall": self.associative_recall,
            "vector_support_hits": self.vector_support_hits,
            "associative_support_hits": self.associative_support_hits,
            "associative_gain": self.associative_gain,
            "average_path_length": self.average_path_length,
            "method_metrics": self.method_metrics,
            "ablation_metrics": self.ablation_metrics,
            "cases": self.cases,
        }


class Evaluator:
    """评估纯向量检索和联想图检索的证据召回差异。"""

    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider,
        graph_builder: GraphBuilder,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.graph_builder = graph_builder

    def ingest(self, documents: list[DatasetDocument]) -> list[MemoryNode]:
        nodes = documents_to_nodes(documents, self.embedding_provider)
        summary_nodes = build_query_summary_nodes(nodes, self.embedding_provider)
        self.store.upsert_nodes([*nodes, *summary_nodes])
        self.graph_builder.bootstrap_context_edges(nodes)
        self.graph_builder.bootstrap_summary_edges(summary_nodes)
        return [*nodes, *summary_nodes]

    def evaluate(
        self,
        queries: list[EvaluationQuery],
        top_k: int = 2,
        seed_k: int = 1,
        hops: int = 2,
        methods: list[str] | None = None,
        use_retrieval_query: bool = False,
        query_planner: QueryPlanner | None = None,
    ) -> ExperimentResult:
        active_methods = methods or DEFAULT_EVALUATION_METHODS
        total_support = 0
        method_support_hits = {method: 0 for method in active_methods}
        method_answer_hits = {method: 0 for method in active_methods}
        method_path_lengths = {method: [] for method in active_methods}
        method_candidate_path_counts = {method: [] for method in active_methods}
        method_path_support_scores = {method: [] for method in active_methods}
        method_edge_memory_scores = {method: [] for method in active_methods}
        cases: list[dict[str, object]] = []

        original_to_node = {
            node.metadata["original_doc_id"]: node.id
            for node in self.store.get_nodes()
            if "original_doc_id" in node.metadata
        }
        summary_nodes_by_query = {
            str(node.metadata["query_id"]): node.id
            for node in self.store.get_nodes()
            if node.metadata.get("node_type") == "query_summary" and "query_id" in node.metadata
        }
        query_contexts: list[dict[str, object]] = []
        cases_by_query: dict[str, dict[str, object]] = {}
        for query in queries:
            query_plan = query_planner.plan(query) if query_planner else None
            support_node_ids = {
                original_to_node[doc_id]
                for doc_id in query.supporting_doc_ids
                if doc_id in original_to_node
            }
            candidate_node_ids = [
                original_to_node[doc_id]
                for doc_id in query.candidate_doc_ids
                if doc_id in original_to_node
            ]
            summary_node_id = summary_nodes_by_query.get(query.id)
            sam_candidate_node_ids = (
                [*candidate_node_ids, summary_node_id]
                if summary_node_id
                else candidate_node_ids
            )
            total_support += len(support_node_ids)
            query_contexts.append(
                {
                    "query": query,
                    "retrieval_query": (
                        query_plan.retrieval_query
                        if query_plan
                        else _retrieval_query(query)
                        if use_retrieval_query
                        else query.question
                    ),
                    "query_plan": query_plan.to_dict() if query_plan else None,
                    "support_node_ids": support_node_ids,
                    "candidate_node_ids": candidate_node_ids,
                    "sam_candidate_node_ids": sam_candidate_node_ids,
                }
            )
            cases_by_query[query.id] = {
                "query_id": query.id,
                "dataset": query.dataset,
                "question": query.question,
                "answer": query.answer,
                "query_metadata": query.metadata,
                "query_plan": query_plan.to_dict() if query_plan else None,
                "supporting_doc_ids": query.supporting_doc_ids,
                "methods": {},
                "final_answers": {},
                "support_hits_by_method": {},
            }

        display_method = _display_method(active_methods)
        baseline_temp_dir = tempfile.TemporaryDirectory()
        baseline_store = MemoryStore(Path(baseline_temp_dir.name) / "baseline.sqlite")
        self.store.connection.backup(baseline_store.connection)
        try:
            for method in active_methods:
                temp_dir = tempfile.TemporaryDirectory()
                method_store = MemoryStore(Path(temp_dir.name) / f"{method}.sqlite")
                baseline_store.connection.backup(method_store.connection)
                method_graph_builder = GraphBuilder(
                    method_store,
                    relation_judge=self.graph_builder.relation_judge,
                )
                retriever = Retriever(method_store, self.embedding_provider, method_graph_builder)
                try:
                    self._evaluate_method(
                        method=method,
                        method_store=method_store,
                        retriever=retriever,
                        query_contexts=query_contexts,
                        cases_by_query=cases_by_query,
                        method_support_hits=method_support_hits,
                        method_answer_hits=method_answer_hits,
                        method_path_lengths=method_path_lengths,
                        method_candidate_path_counts=method_candidate_path_counts,
                        method_path_support_scores=method_path_support_scores,
                        method_edge_memory_scores=method_edge_memory_scores,
                        top_k=top_k,
                        seed_k=seed_k,
                        hops=hops,
                    )
                    if method == display_method:
                        method_store.connection.backup(self.store.connection)
                        self.graph_builder.edge_creation_log = method_graph_builder.edge_creation_log
                finally:
                    method_store.close()
                    temp_dir.cleanup()
        finally:
            baseline_store.close()
            baseline_temp_dir.cleanup()

        cases: list[dict[str, object]] = []
        for context in query_contexts:
            query = context["query"]
            assert isinstance(query, EvaluationQuery)
            case_payload = cases_by_query[query.id]
            serialized_methods = case_payload["methods"]
            final_answers = case_payload["final_answers"]
            support_hits_by_method = case_payload["support_hits_by_method"]
            assert isinstance(serialized_methods, dict)
            assert isinstance(final_answers, dict)
            assert isinstance(support_hits_by_method, dict)
            vector_hits = serialized_methods.get("embedding_topk", [])
            sam_hits = serialized_methods.get("sam", serialized_methods.get("sam_full", []))
            vector_case_hits = support_hits_by_method.get("embedding_topk", 0)
            sam_case_hits = support_hits_by_method.get("sam", support_hits_by_method.get("sam_full", 0))
            case_payload.update(
                {
                    "vector": vector_hits,
                    "associative": sam_hits,
                    "vector_final_answer": final_answers.get("embedding_topk", {}),
                    "associative_final_answer": final_answers.get("sam", final_answers.get("sam_full", {})),
                    "vector_support_hits": vector_case_hits,
                    "associative_support_hits": sam_case_hits,
                    "gain": sam_case_hits - vector_case_hits,
                }
            )
            cases.append(case_payload)

        vector_support_hits = method_support_hits.get("embedding_topk", 0)
        associative_support_hits = method_support_hits.get("sam", method_support_hits.get("sam_full", 0))
        vector_recall = vector_support_hits / total_support if total_support else 0.0
        associative_recall = associative_support_hits / total_support if total_support else 0.0
        associative_lengths = method_path_lengths.get("sam") or method_path_lengths.get("sam_full") or []
        avg_path_length = _average(associative_lengths)
        method_metrics = {
            method: {
                "display_name": RETRIEVAL_METHOD_NAMES.get(method, method),
                "support_hits": method_support_hits[method],
                "evidence_recall": method_support_hits[method] / total_support if total_support else 0.0,
                "answer_hit_count": method_answer_hits[method],
                "answer_hit_rate": method_answer_hits[method] / len(queries) if queries else 0.0,
                "average_path_length": _average(method_path_lengths[method]),
                "average_candidate_path_count": _average(method_candidate_path_counts[method]),
                "average_path_support_score": _average(method_path_support_scores[method]),
                "average_edge_memory_score": _average(method_edge_memory_scores[method]),
            }
            for method in active_methods
        }
        ablation_metrics = {
            method: metric
            for method, metric in method_metrics.items()
            if method.startswith("sam_") or method in {"sam"}
        }
        return ExperimentResult(
            dataset_count=len({query.dataset for query in queries}),
            query_count=len(queries),
            document_count=len(self.store.get_nodes()),
            total_supporting_evidence=total_support,
            vector_recall=vector_recall,
            associative_recall=associative_recall,
            vector_support_hits=vector_support_hits,
            associative_support_hits=associative_support_hits,
            associative_gain=associative_support_hits - vector_support_hits,
            average_path_length=avg_path_length,
            method_metrics=method_metrics,
            ablation_metrics=ablation_metrics,
            cases=cases,
        )

    def _evaluate_method(
        self,
        method: str,
        method_store: MemoryStore,
        retriever: Retriever,
        query_contexts: list[dict[str, object]],
        cases_by_query: dict[str, dict[str, object]],
        method_support_hits: dict[str, int],
        method_answer_hits: dict[str, int],
        method_path_lengths: dict[str, list[int]],
        method_candidate_path_counts: dict[str, list[int]],
        method_path_support_scores: dict[str, list[float]],
        method_edge_memory_scores: dict[str, list[float]],
        top_k: int,
        seed_k: int,
        hops: int,
    ) -> None:
        for context in query_contexts:
            query = context["query"]
            retrieval_query = context["retrieval_query"]
            support_node_ids = context["support_node_ids"]
            candidate_ids = (
                context["sam_candidate_node_ids"]
                if method.startswith("sam")
                else context["candidate_node_ids"]
            )
            assert isinstance(query, EvaluationQuery)
            assert isinstance(retrieval_query, str)
            assert isinstance(support_node_ids, set)
            assert isinstance(candidate_ids, list)
            candidate_ids = self._candidate_ids_for_method(method_store, method, candidate_ids)

            hits = retriever.retrieve(
                query=retrieval_query,
                mode=method,
                top_k=top_k,
                seed_k=seed_k,
                hops=hops,
                candidate_doc_ids=candidate_ids,
            )
            hit_ids = {hit.node.id for hit in hits}
            case_support_hits = len(hit_ids & support_node_ids)
            extracted_answer = self._extract_answer(query.answer, hits, query.metadata)
            if _feedback_enabled(method):
                FeedbackUpdater(method_store).apply(
                    query=query,
                    mode=method,
                    hits=hits,
                    support_node_ids=support_node_ids,
                    answer_status=str(extracted_answer["status"]),
                )
                MemoryConsolidator(method_store, self.embedding_provider).consolidate_query(
                    query=query,
                    mode=method,
                    hits=hits,
                    support_node_ids=support_node_ids,
                    answer_status=str(extracted_answer["status"]),
                )
            method_support_hits[method] += case_support_hits
            if extracted_answer["status"] in {
                "found_in_retrieved_context",
                "matched_option",
                "answer_terms_covered",
            }:
                method_answer_hits[method] += 1
            method_path_lengths[method].extend(len(hit.path) for hit in hits)
            method_candidate_path_counts[method].extend(
                int(hit.metadata.get("candidate_path_count", 1)) for hit in hits
            )
            method_path_support_scores[method].extend(
                float(hit.metadata.get("path_support_score", 0.0)) for hit in hits
            )
            method_edge_memory_scores[method].extend(
                float(hit.metadata.get("edge_memory_score", 0.0)) for hit in hits
            )
            case_payload = cases_by_query[query.id]
            case_payload["methods"][method] = self._serialize_hits(hits, support_node_ids)
            case_payload["final_answers"][method] = extracted_answer
            case_payload["support_hits_by_method"][method] = case_support_hits

    def _candidate_ids_for_method(
        self,
        store: MemoryStore,
        method: str,
        base_candidate_ids: list[str],
    ) -> list[str]:
        candidate_ids = list(base_candidate_ids)
        if not method.startswith("sam"):
            return candidate_ids
        for node in store.get_nodes():
            if node.metadata.get("node_type") != "consolidated_memory":
                continue
            candidate_ids.append(node.id)
            candidate_ids.extend(
                str(support_id)
                for support_id in node.metadata.get("support_node_ids", [])
            )
        return list(dict.fromkeys(candidate_ids))

    def write_reports(self, result: ExperimentResult, report_dir: str | Path) -> tuple[Path, Path]:
        output_dir = Path(report_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "metrics.json"
        markdown_path = output_dir / "metrics.md"
        json_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "cases.json").write_text(
            json.dumps(result.cases, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "ablation_metrics.json").write_text(
            json.dumps(result.ablation_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "ablation_metrics.md").write_text(
            self._ablation_to_markdown(result),
            encoding="utf-8",
        )
        bad_cases = BadCaseAnalyzer().analyze(result.cases)
        write_bad_case_reports(bad_cases, output_dir)
        memory_events = self.store.get_memory_events()
        (output_dir / "memory_events.json").write_text(
            json.dumps(memory_events, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "memory_events.md").write_text(
            _memory_events_to_markdown(memory_events),
            encoding="utf-8",
        )
        markdown_path.write_text(self._to_markdown(result), encoding="utf-8")
        return json_path, markdown_path

    def _serialize_hits(
        self,
        hits: list[RetrievalHit],
        support_node_ids: set[str],
    ) -> list[dict[str, object]]:
        return [
            {
                "node_id": hit.node.id,
                "original_doc_id": hit.node.metadata.get("original_doc_id"),
                "title": hit.node.metadata.get("title"),
                "is_supporting": hit.node.id in support_node_ids,
                "score": round(hit.score, 4),
                "similarity_score": round(hit.similarity_score, 4),
                "graph_score": round(hit.graph_score, 4),
                "usage_score": round(hit.usage_score, 4),
                "confidence_score": round(hit.confidence_score, 4),
                "usage_count": hit.node.usage_count,
                "last_accessed_at": hit.node.last_accessed_at,
                "path": hit.path,
                "reason": hit.reason,
                "score_breakdown": hit.metadata.get("score_breakdown", {}),
                "path_support_score": hit.metadata.get("path_support_score", 0.0),
                "edge_memory_score": hit.metadata.get("edge_memory_score", 0.0),
                "recency_score": hit.metadata.get("recency_score", 0.0),
                "candidate_path_count": hit.metadata.get("candidate_path_count", 1),
                "candidate_paths": hit.metadata.get("candidate_paths", []),
                "adaptive_anchor_count": hit.metadata.get("adaptive_anchor_count"),
                "adaptive_anchor_reason": hit.metadata.get("adaptive_anchor_reason"),
                "text": hit.node.text,
            }
            for hit in hits
        ]

    def _extract_answer(
        self,
        gold_answer: str,
        hits: list[RetrievalHit],
        query_metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """当前阶段的轻量答案抽取。

        这里还没有接 LLM 生成答案，因此先做“检索文本是否覆盖标准答案”的抽取检测：
        如果 top-k 文档里包含标准答案字符串，就认为该方法具备生成该答案的证据基础。
        """

        query_metadata = query_metadata or {}
        options = query_metadata.get("options")
        if isinstance(options, dict) and gold_answer in options:
            gold_text = str(options[gold_answer])
        else:
            gold_text = gold_answer
        if not str(gold_answer).strip() and not str(gold_text).strip():
            return {
                "answer": "数据集中未提供标准答案",
                "status": "answer_not_available",
                "evidence_node_id": None,
                "evidence_title": None,
                "note": "当前样本没有可评估的标准答案，通常需要使用 NovelQA 官方评测或人工答案文件。",
            }
        normalized_gold = gold_answer.lower()
        normalized_gold_text = gold_text.lower()
        for hit in hits:
            haystack = f"{hit.node.metadata.get('title', '')}\n{hit.node.text}".lower()
            if normalized_gold in haystack or normalized_gold_text in haystack:
                return {
                    "answer": gold_text,
                    "status": "matched_option" if isinstance(options, dict) else "found_in_retrieved_context",
                    "evidence_node_id": hit.node.id,
                    "evidence_title": hit.node.metadata.get("title", hit.node.id),
                    "note": "当前阶段未接 LLM，此处表示 top-k 检索文本中包含标准答案字符串。",
                }
            coverage = _answer_term_coverage(gold_text, haystack)
            if coverage >= 0.5:
                return {
                    "answer": gold_text,
                    "status": "answer_terms_covered",
                    "evidence_node_id": hit.node.id,
                    "evidence_title": hit.node.metadata.get("title", hit.node.id),
                    "term_coverage": round(coverage, 4),
                    "note": "当前阶段未接 LLM，此处表示 top-k 检索文本覆盖了标准答案中的关键实体或内容词。",
                }
        return {
            "answer": "未在检索文本中找到答案",
            "status": "not_found_in_retrieved_context",
            "evidence_node_id": None,
            "evidence_title": None,
            "note": "当前阶段未接 LLM，此处表示 top-k 检索文本中没有覆盖标准答案字符串。",
        }

    def _to_markdown(self, result: ExperimentResult) -> str:
        lines = [
            "# SAM 初步实验结果",
            "",
            "## 总体指标",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
            f"| 数据集来源数 | {result.dataset_count} |",
            f"| 查询数量 | {result.query_count} |",
            f"| 候选文档节点数量 | {result.document_count} |",
            f"| Gold 支持证据数量 | {result.total_supporting_evidence} |",
            f"| 纯向量检索证据召回率 | {result.vector_recall:.3f} |",
            f"| 联想图检索证据召回率 | {result.associative_recall:.3f} |",
            f"| 纯向量命中支持证据数 | {result.vector_support_hits} |",
            f"| 联想检索命中支持证据数 | {result.associative_support_hits} |",
            f"| 联想检索新增有效证据数 | {result.associative_gain} |",
            f"| 联想检索平均路径长度 | {result.average_path_length:.2f} |",
            "",
            "## 方法对比",
            "",
            "| 方法 | 证据命中数 | 证据召回率 | 答案命中数 | 答案命中率 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for metric in result.method_metrics.values():
            recall = metric["evidence_recall"]
            recall_text = "N/A" if recall is None else f"{float(recall):.3f}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(metric["display_name"]),
                        str(metric["support_hits"]),
                        recall_text,
                        str(metric["answer_hit_count"]),
                        f"{float(metric['answer_hit_rate']):.3f}",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
            "## 案例分析",
            "",
            ]
        )
        for case in result.cases:
            lines.extend(
                [
                    f"### {case['query_id']} ({case['dataset']})",
                    "",
                    f"- 问题：{case['question']}",
                    f"- 答案：{case['answer']}",
                    f"- 支持文档：{', '.join(case['supporting_doc_ids'])}",
                    f"- 纯向量命中支持证据数：{case['vector_support_hits']}",
                    f"- 联想检索命中支持证据数：{case['associative_support_hits']}",
                    "",
                    "| 模式 | 文档 | 是否支持证据 | 分数 | 路径 | 排序原因 |",
                    "| --- | --- | --- | ---: | --- | --- |",
                ]
            )
            for method, hits in case["methods"].items():
                for hit in hits:
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                RETRIEVAL_METHOD_NAMES.get(method, method),
                                str(hit["title"]),
                                "是" if hit["is_supporting"] else "否",
                                f"{hit['score']:.4f}",
                                " -> ".join(hit["path"]),
                                str(hit["reason"]).replace("|", "/"),
                            ]
                        )
                        + " |"
                    )
            lines.append("")
        return "\n".join(lines)

    def _ablation_to_markdown(self, result: ExperimentResult) -> str:
        lines = [
            "# SAM 消融实验结果",
            "",
            "| 方法 | 证据命中数 | 证据召回率 | 答案命中率 | 平均路径长度 | 平均候选路径数 | 平均路径支持分 | 平均边记忆分 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for metric in result.ablation_metrics.values():
            recall = metric["evidence_recall"]
            recall_text = "N/A" if recall is None else f"{float(recall):.3f}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(metric["display_name"]),
                        str(metric["support_hits"]),
                        recall_text,
                        f"{float(metric['answer_hit_rate']):.3f}",
                        f"{float(metric['average_path_length']):.2f}",
                        f"{float(metric['average_candidate_path_count']):.2f}",
                        f"{float(metric['average_path_support_score']):.3f}",
                        f"{float(metric['average_edge_memory_score']):.3f}",
                    ]
                )
                + " |"
            )
        lines.append("")
        return "\n".join(lines)


def _average(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _retrieval_query(query: EvaluationQuery) -> str:
    value = query.metadata.get("retrieval_query")
    if isinstance(value, str) and value.strip():
        return value
    return query.question


def _answer_term_coverage(gold_answer: str, haystack: str) -> float:
    terms = _answer_terms(gold_answer)
    if len(terms) < 2:
        return 0.0
    matched = sum(1 for term in terms if term in haystack)
    return matched / len(terms)


def _answer_terms(text: str) -> list[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
    stopwords = {
        "and",
        "or",
        "the",
        "a",
        "an",
        "of",
        "in",
        "to",
        "is",
        "are",
        "was",
        "were",
        "with",
        "from",
        "when",
        "after",
        "before",
        "but",
        "his",
        "her",
        "him",
        "she",
        "he",
        "it",
        "this",
        "that",
    }
    terms = [
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in stopwords
    ]
    return list(dict.fromkeys(terms))


def _display_method(methods: list[str]) -> str | None:
    for method in ["sam_full", "sam"]:
        if method in methods:
            return method
    for method in methods:
        if method.startswith("sam"):
            return method
    return methods[0] if methods else None


def _feedback_enabled(method: str) -> bool:
    return method in {"sam", "sam_full", "sam_with_summary"}


def _memory_events_to_markdown(events: list[dict[str, object]]) -> str:
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type", "unknown"))
        counts[event_type] = counts.get(event_type, 0) + 1
    lines = [
        "# SAM 记忆事件摘要",
        "",
        "| 事件类型 | 数量 |",
        "| --- | ---: |",
    ]
    for event_type, count in sorted(counts.items()):
        lines.append(f"| {event_type} | {count} |")
    lines.append("")
    lines.extend(
        [
            "## 最近事件",
            "",
            "| 时间 | 类型 | 方法 | 节点 | 分数 |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for event in events[:20]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(event.get("created_at", "")),
                    str(event.get("event_type", "")),
                    str(event.get("mode", "")),
                    str(event.get("node_id") or ""),
                    f"{float(event.get('score', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)
