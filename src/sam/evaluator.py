from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sam.datasets import documents_to_nodes
from sam.embedding import EmbeddingProvider
from sam.graph import GraphBuilder
from sam.models import DatasetDocument, EvaluationQuery, MemoryNode, RetrievalHit
from sam.retriever import Retriever
from sam.store import MemoryStore


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
        for node in nodes:
            self.store.upsert_node(node)
        self.graph_builder.bootstrap_context_edges(nodes)
        return nodes

    def evaluate(
        self,
        queries: list[EvaluationQuery],
        top_k: int = 2,
        seed_k: int = 1,
        hops: int = 2,
    ) -> ExperimentResult:
        retriever = Retriever(self.store, self.embedding_provider, self.graph_builder)
        total_support = 0
        vector_support_hits = 0
        associative_support_hits = 0
        path_lengths: list[int] = []
        cases: list[dict[str, object]] = []

        original_to_node = {
            node.metadata["original_doc_id"]: node.id
            for node in self.store.get_nodes()
            if "original_doc_id" in node.metadata
        }
        for query in queries:
            support_node_ids = {original_to_node[doc_id] for doc_id in query.supporting_doc_ids}
            candidate_node_ids = [original_to_node[doc_id] for doc_id in query.candidate_doc_ids]

            vector_hits = retriever.retrieve(
                query=query.question,
                mode="vector",
                top_k=top_k,
                candidate_doc_ids=candidate_node_ids,
            )
            associative_hits = retriever.retrieve(
                query=query.question,
                mode="associative",
                top_k=top_k,
                seed_k=seed_k,
                hops=hops,
                candidate_doc_ids=candidate_node_ids,
            )
            vector_hit_ids = {hit.node.id for hit in vector_hits}
            associative_hit_ids = {hit.node.id for hit in associative_hits}
            vector_case_hits = len(vector_hit_ids & support_node_ids)
            associative_case_hits = len(associative_hit_ids & support_node_ids)
            total_support += len(support_node_ids)
            vector_support_hits += vector_case_hits
            associative_support_hits += associative_case_hits
            path_lengths.extend(len(hit.path) for hit in associative_hits)
            cases.append(
                {
                    "query_id": query.id,
                    "dataset": query.dataset,
                    "question": query.question,
                    "answer": query.answer,
                    "supporting_doc_ids": query.supporting_doc_ids,
                    "vector": self._serialize_hits(vector_hits, support_node_ids),
                    "associative": self._serialize_hits(associative_hits, support_node_ids),
                    "vector_final_answer": self._extract_answer(query.answer, vector_hits),
                    "associative_final_answer": self._extract_answer(query.answer, associative_hits),
                    "vector_support_hits": vector_case_hits,
                    "associative_support_hits": associative_case_hits,
                    "gain": associative_case_hits - vector_case_hits,
                }
            )

        vector_recall = vector_support_hits / total_support if total_support else 0.0
        associative_recall = associative_support_hits / total_support if total_support else 0.0
        average_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 0.0
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
            average_path_length=average_path_length,
            cases=cases,
        )

    def write_reports(self, result: ExperimentResult, report_dir: str | Path) -> tuple[Path, Path]:
        output_dir = Path(report_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "experiment_results.json"
        markdown_path = output_dir / "experiment_results.md"
        json_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
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
                "path": hit.path,
                "reason": hit.reason,
            }
            for hit in hits
        ]

    def _extract_answer(self, gold_answer: str, hits: list[RetrievalHit]) -> dict[str, object]:
        """当前阶段的轻量答案抽取。

        这里还没有接 LLM 生成答案，因此先做“检索文本是否覆盖标准答案”的抽取检测：
        如果 top-k 文档里包含标准答案字符串，就认为该方法具备生成该答案的证据基础。
        """

        normalized_gold = gold_answer.lower()
        for hit in hits:
            haystack = f"{hit.node.metadata.get('title', '')}\n{hit.node.text}".lower()
            if normalized_gold in haystack:
                return {
                    "answer": gold_answer,
                    "status": "found_in_retrieved_context",
                    "evidence_node_id": hit.node.id,
                    "evidence_title": hit.node.metadata.get("title", hit.node.id),
                    "note": "当前阶段未接 LLM，此处表示 top-k 检索文本中包含标准答案字符串。",
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
            "## 案例分析",
            "",
        ]
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
            for mode in ["vector", "associative"]:
                for hit in case[mode]:
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                "纯向量" if mode == "vector" else "联想检索",
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
