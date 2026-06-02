from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import json
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.datasets import load_builtin_benchmark_sample, load_novelqa_sample
from sam.dataset_format import load_sam_dataset, save_sam_dataset, summarize_sam_dataset
from sam.agent_workflow import MultiAgentResearchWorkflow, write_agent_workflow_reports
from sam.agent_reuse_experiment import (
    compare_agent_generation_variants,
    run_agent_memory_reuse_probe,
    write_agent_generation_comparison_reports,
    write_agent_memory_reuse_reports,
)
from sam.agents import SharedMemoryCoordinator
from sam.analogy import AnalogyEngine
from sam.analogy_experiment import run_analogy_reuse_probe
from sam.badcase import (
    BadCaseAnalyzer,
    GenerationBadCaseAnalyzer,
    write_bad_case_reports,
    write_generation_bad_case_reports,
)
from sam.answer_judge import AnswerJudgment, RuleBasedAnswerJudge
from sam.consolidation import MemoryConsolidator
from sam.embedding import (
    AzureOpenAIEmbeddingProvider,
    CachedEmbeddingProvider,
    LocalHashEmbeddingProvider,
    create_embedding_provider,
)
from sam.evaluator import Evaluator
from sam.generation import (
    CaseAnalogyHintBuilder,
    ContextAnswerGenerator,
    compare_generation_variants,
    generate_answers_for_cases,
    write_generation_comparison_reports,
    write_generation_reports,
)
from sam.graph import GraphBuilder
from sam.llm import ChatClient, HeuristicChatClient
from sam.models import DatasetDocument, EvaluationQuery, MemoryEdge, MemoryNode, RetrievalHit, utc_now_iso
from sam.query_planner import ChatQueryPlanner, HeuristicQueryPlanner, QueryPlan
from sam.reranker import PathReranker
from sam.reranker_experiment import (
    run_reranker_profile_comparison,
    write_reranker_profile_reports,
)
from sam.relation_judge import RelationJudgment
from sam.retriever import Retriever
from sam.reuse_experiment import build_masked_queries, summarize_memory_reuse
from sam.store import MemoryStore
from scripts.run_demo import _nodes_for_graph_export


class SamCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.sqlite"
        self.store = MemoryStore(self.db_path)
        self.embedding = LocalHashEmbeddingProvider()
        self.graph = GraphBuilder(self.store)
        self.evaluator = Evaluator(self.store, self.embedding, self.graph)
        documents, self.queries = load_builtin_benchmark_sample()
        self.nodes = self.evaluator.ingest(documents)

    def tearDown(self) -> None:
        self.store.close()
        self.temp_dir.cleanup()

    def test_nodes_are_persisted(self) -> None:
        nodes = self.store.get_nodes()
        self.assertEqual(len(nodes), len(self.nodes))
        self.assertTrue(nodes[0].embedding)

    def test_query_summary_nodes_are_created(self) -> None:
        summary_nodes = [
            node
            for node in self.store.get_nodes()
            if node.metadata.get("node_type") == "query_summary"
        ]
        self.assertEqual(len(summary_nodes), len(self.queries))
        self.assertTrue(summary_nodes[0].metadata.get("child_node_ids"))
        summary_edges = [
            edge
            for edge in self.store.get_edges()
            if edge.relation_type in {"summary_parent", "summary_child"}
        ]
        self.assertTrue(summary_edges)
        self.assertIn("score_breakdown", summary_edges[0].metadata)

    def test_edges_are_created_on_demand(self) -> None:
        seed = self.store.get_nodes([self.nodes[0].id])
        edges = self.graph.build_edges_on_demand(seed)
        self.assertTrue(edges)
        self.assertTrue(any(edge.reason for edge in edges))
        self.assertTrue(any("score_breakdown" in edge.metadata for edge in edges))
        self.assertTrue(self.graph.edge_creation_log)
        self.assertIn("score_breakdown", self.graph.edge_creation_log[0])

    def test_low_information_keyword_overlap_does_not_create_edge(self) -> None:
        self.store.reset()
        now = utc_now_iso()
        left = MemoryNode(
            id="low_info_left",
            text="system report alpha",
            summary="system report alpha",
            keywords=["system", "report"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[1.0, 0.0, 0.0],
            metadata={},
        )
        right = MemoryNode(
            id="low_info_right",
            text="system report beta",
            summary="system report beta",
            keywords=["system", "report"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[0.0, 1.0, 0.0],
            metadata={},
        )
        self.store.upsert_nodes([left, right])

        edges = self.graph.build_edges_on_demand([left], [left, right])

        self.assertEqual(edges, [])
        score = self.graph._score_candidate_edge(left, right)
        self.assertEqual(score.relation_type, None)
        self.assertEqual(score.score_breakdown["edge_quality"], "low_information_keyword_overlap")

    def test_novel_chunk_structural_keywords_do_not_create_edge(self) -> None:
        self.store.reset()
        now = utc_now_iso()
        left = MemoryNode(
            id="novel_noise_left",
            text="Frankenstein chunk had his letter and she was there.",
            summary="Frankenstein chunk had his letter.",
            keywords=["frankenstein", "chunk", "had", "his", "she"],
            tags=["novelqa"],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[1.0, 0.0, 0.0],
            metadata={"book_id": "Frankenstein"},
        )
        right = MemoryNode(
            id="novel_noise_right",
            text="Frankenstein chunk had his memory and she replied.",
            summary="Frankenstein chunk had his memory.",
            keywords=["frankenstein", "chunk", "had", "his", "she"],
            tags=["novelqa"],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[0.0, 1.0, 0.0],
            metadata={"book_id": "Frankenstein"},
        )
        self.store.upsert_nodes([left, right])

        edges = self.graph.build_edges_on_demand([left], [left, right])
        score = self.graph._score_candidate_edge(left, right)

        self.assertEqual(edges, [])
        self.assertEqual(score.relation_type, None)
        self.assertEqual(score.score_breakdown["edge_quality"], "low_information_keyword_overlap")
        self.assertEqual(score.score_breakdown["keyword_overlap"], [])

    def test_relation_judge_can_reject_noisy_candidate_edge(self) -> None:
        class RejectingRelationJudge:
            def judge(
                self,
                seed: MemoryNode,
                other: MemoryNode,
                score_breakdown: dict[str, object],
            ) -> RelationJudgment:
                return RelationJudgment(
                    should_link=False,
                    relation_type="unrelated",
                    confidence=0.92,
                    reason="两个段落主题不同，共享词不足以构成语义关系",
                )

        self.store.reset()
        now = utc_now_iso()
        left = MemoryNode(
            id="judge_left",
            text="Alpha bridge evidence focuses on a film award.",
            summary="Alpha bridge evidence focuses on a film award.",
            keywords=["alpha", "bridge"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[1.0, 0.0, 0.0],
            metadata={},
        )
        right = MemoryNode(
            id="judge_right",
            text="Alpha bridge evidence focuses on a sports roster.",
            summary="Alpha bridge evidence focuses on a sports roster.",
            keywords=["alpha", "bridge"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[1.0, 0.0, 0.0],
            metadata={},
        )
        self.store.upsert_nodes([left, right])
        graph = GraphBuilder(self.store, relation_judge=RejectingRelationJudge())

        edges = graph.build_edges_on_demand([left], [left, right])
        score = graph._score_candidate_edge(left, right)

        self.assertEqual(edges, [])
        self.assertEqual(score.relation_type, None)
        self.assertEqual(score.score_breakdown["relation_judge"]["should_link"], False)
        self.assertEqual(score.score_breakdown["relation_judge"]["relation_type"], "unrelated")

    def test_edge_creation_log_is_written(self) -> None:
        seed = self.store.get_nodes([self.nodes[0].id])
        self.graph.build_edges_on_demand(seed)
        output_path = Path(self.temp_dir.name) / "edge_creation_log.json"
        self.graph.write_edge_creation_log(output_path)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertTrue(payload)
        self.assertIn("relation_type", payload[0])
        self.assertIn("score_breakdown", payload[0])

    def test_vector_and_associative_retrieval_return_hits(self) -> None:
        query = self.queries[0]
        retriever = Retriever(self.store, self.embedding, self.graph)
        candidate_ids = [
            node.id
            for node in self.store.get_nodes()
            if node.metadata.get("original_doc_id") in query.candidate_doc_ids
        ]
        vector_hits = retriever.retrieve(query.question, "vector", top_k=2, candidate_doc_ids=candidate_ids)
        associative_hits = retriever.retrieve(
            query.question,
            "associative",
            top_k=2,
            seed_k=1,
            hops=2,
            candidate_doc_ids=candidate_ids,
        )
        self.assertEqual(len(vector_hits), 2)
        self.assertEqual(len(associative_hits), 2)
        self.assertTrue(any(len(hit.path) > 1 for hit in associative_hits))
        self.assertTrue(all(hit.node.metadata.get("node_type") != "query_summary" for hit in associative_hits))
        self.assertTrue(any("score_breakdown" in hit.metadata for hit in associative_hits))
        self.assertTrue(any(hit.metadata.get("candidate_path_count", 0) >= 1 for hit in associative_hits))

    def test_retrieval_updates_dynamic_memory_state(self) -> None:
        query = self.queries[0]
        retriever = Retriever(self.store, self.embedding, self.graph)
        candidate_ids = [
            node.id
            for node in self.store.get_nodes()
            if node.metadata.get("original_doc_id") in query.candidate_doc_ids
        ]
        hits = retriever.retrieve(
            query.question,
            "associative",
            top_k=3,
            seed_k=1,
            hops=2,
            candidate_doc_ids=candidate_ids,
        )
        updated_nodes = self.store.get_nodes([hit.node.id for hit in hits])
        self.assertTrue(all(node.usage_count >= 1 for node in updated_nodes))
        self.assertTrue(all(node.last_accessed_at for node in updated_nodes))
        activated_edges = [edge for edge in self.store.get_edges() if edge.activation_count > 0]
        self.assertTrue(activated_edges)
        self.assertTrue(all(edge.last_activated_at for edge in activated_edges))
        logs = self.store.get_retrieval_logs(limit=1)
        self.assertEqual(logs[0]["mode"], "sam")
        self.assertIn("dynamic_update", logs[0]["metadata"])
        self.assertTrue(logs[0]["metadata"]["dynamic_update"]["updated_node_ids"])
        events = self.store.get_memory_events(limit=20)
        event_types = {event["event_type"] for event in events}
        self.assertIn("node_retrieved", event_types)
        self.assertIn("edge_traversed", event_types)

    def test_repeated_retrieval_uses_memory_state_in_scoring(self) -> None:
        query = self.queries[0]
        retriever = Retriever(self.store, self.embedding, self.graph)
        candidate_ids = [
            node.id
            for node in self.store.get_nodes()
            if node.metadata.get("original_doc_id") in query.candidate_doc_ids
        ]
        retriever.retrieve(
            query.question,
            "associative",
            top_k=3,
            seed_k=1,
            hops=2,
            candidate_doc_ids=candidate_ids,
        )
        second_hits = retriever.retrieve(
            query.question,
            "associative",
            top_k=3,
            seed_k=1,
            hops=2,
            candidate_doc_ids=candidate_ids,
        )
        self.assertTrue(
            any(hit.metadata.get("edge_memory_score", 0.0) > 0 for hit in second_hits)
        )
        self.assertTrue(
            any(hit.metadata.get("recency_score", 0.0) > 0 for hit in second_hits)
        )

    def test_sam_ablation_modes_return_expected_shapes(self) -> None:
        query = self.queries[0]
        retriever = Retriever(self.store, self.embedding, self.graph)
        candidate_ids = [
            node.id
            for node in self.store.get_nodes()
            if node.metadata.get("original_doc_id") in query.candidate_doc_ids
        ]
        for mode in [
            "sam_full",
            "sam_no_multipath",
            "sam_no_memory_state",
            "sam_no_graph",
            "sam_static_graph",
            "sam_no_summary",
            "sam_with_summary",
            "sam_no_feedback",
            "sam_vector_anchor",
            "sam_adaptive_anchor",
        ]:
            hits = retriever.retrieve(
                query.question,
                mode,
                top_k=3,
                seed_k=1,
                hops=2,
                candidate_doc_ids=candidate_ids,
            )
            self.assertEqual(len(hits), 3, mode)
            self.assertTrue(all(hit.metadata.get("score_breakdown") for hit in hits), mode)

        no_graph_hits = retriever.retrieve(
            query.question,
            "sam_no_graph",
            top_k=3,
            seed_k=1,
            hops=2,
            candidate_doc_ids=candidate_ids,
        )
        self.assertTrue(all(len(hit.path) == 1 for hit in no_graph_hits))

        no_multipath_hits = retriever.retrieve(
            query.question,
            "sam_no_multipath",
            top_k=3,
            seed_k=1,
            hops=2,
            candidate_doc_ids=candidate_ids,
        )
        self.assertTrue(all(hit.metadata.get("candidate_path_count") == 1 for hit in no_multipath_hits))

        no_memory_hits = retriever.retrieve(
            query.question,
            "sam_no_memory_state",
            top_k=3,
            seed_k=1,
            hops=2,
            candidate_doc_ids=candidate_ids,
        )
        for hit in no_memory_hits:
            breakdown = hit.metadata["score_breakdown"]
            self.assertNotIn("usage_component", breakdown)
            self.assertNotIn("recency_component", breakdown)
            self.assertNotIn("edge_memory_component", breakdown)

        no_summary_hits = retriever.retrieve(
            query.question,
            "sam_no_summary",
            top_k=3,
            seed_k=1,
            hops=2,
            candidate_doc_ids=[
                *candidate_ids,
                *[
                    node.id
                    for node in self.store.get_nodes()
                    if node.metadata.get("node_type") == "query_summary"
                    and node.metadata.get("query_id") == query.id
                ],
            ],
        )
        self.assertTrue(
            all("summary_" not in " ".join(hit.path) for hit in no_summary_hits)
        )

    def test_sam_adaptive_anchor_keeps_more_vectors_when_paths_are_weak(self) -> None:
        query = self.queries[0]
        retriever = Retriever(self.store, self.embedding, self.graph)
        candidate_ids = [
            node.id
            for node in self.store.get_nodes()
            if node.metadata.get("original_doc_id") in query.candidate_doc_ids
        ]
        vector_hits = retriever.retrieve(
            query.question,
            "embedding_topk",
            top_k=3,
            candidate_doc_ids=candidate_ids,
        )
        adaptive_hits = retriever.retrieve(
            query.question,
            "sam_adaptive_anchor",
            top_k=3,
            seed_k=1,
            hops=0,
            candidate_doc_ids=candidate_ids,
        )

        self.assertEqual(
            [hit.node.id for hit in adaptive_hits[:2]],
            [hit.node.id for hit in vector_hits[:2]],
        )
        self.assertTrue(
            all(hit.metadata.get("adaptive_anchor_count") == 2 for hit in adaptive_hits)
        )
        self.assertTrue(
            all(hit.metadata.get("adaptive_anchor_reason") == "weak_graph_paths" for hit in adaptive_hits)
        )

    def test_path_reranker_profiles_change_score_weights(self) -> None:
        node = self.nodes[0]
        signals = [
            {
                "path": ["seed", node.id],
                "graph_score": 0.8,
                "depth": 1,
                "edge_activation_count": 2,
            }
        ]

        balanced = PathReranker(profile="balanced").score(
            similarity=0.4,
            graph_score=0.8,
            signals=signals,
            node=node,
            use_multipath=True,
            use_memory_state=True,
        )
        graph_heavy = PathReranker(profile="graph_heavy").score(
            similarity=0.4,
            graph_score=0.8,
            signals=signals,
            node=node,
            use_multipath=True,
            use_memory_state=True,
        )

        self.assertEqual(balanced.profile, "balanced")
        self.assertEqual(graph_heavy.profile, "graph_heavy")
        self.assertGreater(
            graph_heavy.breakdown["graph_component"],
            balanced.breakdown["graph_component"],
        )
        self.assertLess(
            graph_heavy.breakdown["similarity_component"],
            balanced.breakdown["similarity_component"],
        )

    def test_path_reranker_default_profile_is_semantic_heavy(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            reranker = PathReranker.from_env()

        self.assertEqual(reranker.profile, "semantic_heavy")

    def test_path_reranker_penalizes_over_dense_candidate_paths(self) -> None:
        node = self.nodes[0]
        sparse_signals = [
            {
                "path": ["seed", node.id],
                "graph_score": 0.7,
                "depth": 1,
                "edge_activation_count": 0,
            }
        ]
        dense_signals = [
            {
                "path": [f"seed_{index}", node.id],
                "graph_score": 0.7,
                "depth": 1,
                "edge_activation_count": 0,
            }
            for index in range(24)
        ]

        reranker = PathReranker(profile="semantic_heavy")
        sparse_score = reranker.score(
            similarity=0.5,
            graph_score=0.7,
            signals=sparse_signals,
            node=node,
            use_multipath=True,
            use_memory_state=True,
        )
        dense_score = reranker.score(
            similarity=0.5,
            graph_score=0.7,
            signals=dense_signals,
            node=node,
            use_multipath=True,
            use_memory_state=True,
        )

        self.assertIn("path_noise_penalty", dense_score.breakdown)
        self.assertGreater(dense_score.path_noise_penalty, 0.0)
        self.assertLess(dense_score.total, sparse_score.total)

    def test_retriever_reads_reranker_profile_from_environment(self) -> None:
        query = self.queries[0]
        candidate_ids = [
            node.id
            for node in self.store.get_nodes()
            if node.metadata.get("original_doc_id") in query.candidate_doc_ids
        ]

        with patch.dict("os.environ", {"SAM_RERANKER_PROFILE": "graph_heavy"}, clear=False):
            retriever = Retriever(self.store, self.embedding, self.graph)
            hits = retriever.retrieve(
                query.question,
                "sam_full",
                top_k=2,
                seed_k=1,
                hops=2,
                candidate_doc_ids=candidate_ids,
            )

        self.assertTrue(hits)
        self.assertTrue(
            all(hit.metadata.get("reranker_profile") == "graph_heavy" for hit in hits)
        )

    def test_evaluator_serializes_reranker_profile_for_bad_case_analysis(self) -> None:
        with patch.dict("os.environ", {"SAM_RERANKER_PROFILE": "memory_heavy"}, clear=False):
            evaluator = Evaluator(self.store, self.embedding, self.graph)
            result = evaluator.evaluate(
                self.queries[:1],
                top_k=2,
                seed_k=1,
                hops=2,
                methods=["sam_full"],
            )

        sam_hits = result.cases[0]["methods"]["sam_full"]
        self.assertTrue(sam_hits)
        self.assertTrue(
            all(hit["reranker_profile"] == "memory_heavy" for hit in sam_hits)
        )

    def test_reranker_profile_comparison_reports_metrics_and_bad_cases(self) -> None:
        documents, queries = load_builtin_benchmark_sample()

        comparison = run_reranker_profile_comparison(
            documents=documents,
            queries=queries[:2],
            embedding_provider=self.embedding,
            profiles=["balanced", "graph_heavy"],
            top_k=2,
            seed_k=1,
            hops=2,
        )

        self.assertEqual(comparison["query_count"], 2)
        self.assertEqual(set(comparison["profiles"]), {"balanced", "graph_heavy"})
        self.assertIn(comparison["best_profile"], {"balanced", "graph_heavy"})
        for profile in ["balanced", "graph_heavy"]:
            profile_result = comparison["profile_results"][profile]
            self.assertIn("metrics", profile_result)
            self.assertIn("bad_case_summary", profile_result)
            self.assertIn("evidence_recall", profile_result["metrics"])
            self.assertIn("category_counts", profile_result["bad_case_summary"])

        output_dir = Path(self.temp_dir.name) / "reranker_profiles"
        json_path, markdown_path = write_reranker_profile_reports(comparison, output_dir)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["best_profile"], comparison["best_profile"])
        self.assertIn("PathReranker Profile 对比实验", markdown_path.read_text(encoding="utf-8"))

    def test_sam_static_graph_does_not_update_dynamic_state(self) -> None:
        query = self.queries[0]
        retriever = Retriever(self.store, self.embedding, self.graph)
        candidate_ids = [
            node.id
            for node in self.store.get_nodes()
            if node.metadata.get("original_doc_id") in query.candidate_doc_ids
        ]
        before_usage = {
            node.id: node.usage_count
            for node in self.store.get_nodes(candidate_ids)
        }
        before_activations = {
            edge.key: edge.activation_count
            for edge in self.store.get_edges()
        }
        retriever.retrieve(
            query.question,
            "sam_static_graph",
            top_k=3,
            seed_k=1,
            hops=2,
            candidate_doc_ids=candidate_ids,
        )
        after_usage = {
            node.id: node.usage_count
            for node in self.store.get_nodes(candidate_ids)
        }
        after_activations = {
            edge.key: edge.activation_count
            for edge in self.store.get_edges()
        }
        self.assertEqual(before_usage, after_usage)
        for key, activation_count in before_activations.items():
            self.assertEqual(activation_count, after_activations.get(key, activation_count))

    def test_evaluation_produces_gain(self) -> None:
        result = self.evaluator.evaluate(self.queries, top_k=2, seed_k=1, hops=2)
        self.assertGreaterEqual(result.associative_recall, result.vector_recall)
        self.assertGreater(result.average_path_length, 1.0)

    def test_evaluator_can_use_retrieval_query_metadata_without_changing_question(self) -> None:
        self.store.reset()
        now_documents = [
            DatasetDocument(
                id="support-doc",
                dataset="unit",
                title="Support",
                text="alpha beta gamma",
                source="unit-test",
                tags=[],
                keywords=["alpha", "beta", "gamma"],
            ),
            DatasetDocument(
                id="distractor-doc",
                dataset="unit",
                title="Distractor",
                text="Which document ordinary wording",
                source="unit-test",
                tags=[],
                keywords=["which", "document"],
            ),
        ]
        query = EvaluationQuery(
            id="retrieval-query-case",
            dataset="unit",
            question="Which document?",
            answer="alpha",
            supporting_doc_ids=["support-doc"],
            candidate_doc_ids=["support-doc", "distractor-doc"],
            metadata={"retrieval_query": "alpha beta gamma"},
        )
        evaluator = Evaluator(self.store, self.embedding, GraphBuilder(self.store))
        evaluator.ingest(now_documents)

        result = evaluator.evaluate(
            [query],
            top_k=1,
            methods=["embedding_topk"],
            use_retrieval_query=True,
        )

        self.assertEqual(result.method_metrics["embedding_topk"]["support_hits"], 1)
        self.assertEqual(result.cases[0]["question"], "Which document?")
        self.assertEqual(result.cases[0]["query_metadata"]["retrieval_query"], "alpha beta gamma")

    def test_heuristic_query_planner_uses_question_metadata_without_all_options(self) -> None:
        query = EvaluationQuery(
            id="novelqa-query-plan",
            dataset="novelqa",
            question="Why does Alice leave the tea party?",
            answer="A",
            supporting_doc_ids=[],
            candidate_doc_ids=[],
            metadata={
                "aspect": "plot",
                "complexity": "causal",
                "options": {
                    "A": "Because the conversation becomes frustrating.",
                    "B": "Because the Mad Hatter asks her to sing.",
                    "C": "Because the Queen arrives with soldiers.",
                },
            },
        )

        plan = HeuristicQueryPlanner().plan(query)

        self.assertIn("Alice", plan.retrieval_query)
        self.assertIn("plot", plan.retrieval_query)
        self.assertIn("causal", plan.retrieval_query)
        self.assertNotIn("Mad Hatter", plan.retrieval_query)
        self.assertNotIn("Queen arrives", plan.retrieval_query)
        self.assertEqual(plan.metadata["planner"], "heuristic")

    def test_chat_query_planner_parses_structured_plan(self) -> None:
        class PlanningChatClient(ChatClient):
            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                return json.dumps(
                    {
                        "retrieval_query": "Alice tea party frustration cause",
                        "keywords": ["alice", "tea", "party"],
                        "entities": ["Alice"],
                        "reason": "定位角色和事件原因",
                    },
                    ensure_ascii=False,
                )

        query = EvaluationQuery(
            id="chat-query-plan",
            dataset="novelqa",
            question="Why does Alice leave the tea party?",
            answer="A",
            supporting_doc_ids=[],
            candidate_doc_ids=[],
            metadata={"aspect": "plot", "complexity": "causal"},
        )

        plan = ChatQueryPlanner(PlanningChatClient()).plan(query)

        self.assertEqual(plan.retrieval_query, "Alice tea party frustration cause")
        self.assertEqual(plan.entities, ["Alice"])
        self.assertEqual(plan.metadata["planner"], "chat")

    def test_evaluator_can_use_query_planner_and_records_plan(self) -> None:
        class FixedQueryPlanner:
            def plan(self, query: EvaluationQuery) -> QueryPlan:
                return QueryPlan(
                    retrieval_query="alpha beta gamma",
                    keywords=["alpha", "beta", "gamma"],
                    entities=["alpha"],
                    reason="单元测试固定查询规划",
                    metadata={"planner": "fixed"},
                )

        self.store.reset()
        documents = [
            DatasetDocument(
                id="support-doc",
                dataset="unit",
                title="Support",
                text="alpha beta gamma",
                source="unit-test",
                tags=[],
                keywords=["alpha", "beta", "gamma"],
            ),
            DatasetDocument(
                id="distractor-doc",
                dataset="unit",
                title="Distractor",
                text="Which document ordinary wording",
                source="unit-test",
                tags=[],
                keywords=["which", "document"],
            ),
        ]
        query = EvaluationQuery(
            id="query-planner-case",
            dataset="unit",
            question="Which document?",
            answer="alpha",
            supporting_doc_ids=["support-doc"],
            candidate_doc_ids=["support-doc", "distractor-doc"],
        )
        evaluator = Evaluator(self.store, self.embedding, GraphBuilder(self.store))
        evaluator.ingest(documents)

        result = evaluator.evaluate(
            [query],
            top_k=1,
            methods=["embedding_topk"],
            query_planner=FixedQueryPlanner(),
        )

        self.assertEqual(result.method_metrics["embedding_topk"]["support_hits"], 1)
        self.assertEqual(result.cases[0]["question"], "Which document?")
        self.assertEqual(result.cases[0]["query_plan"]["retrieval_query"], "alpha beta gamma")
        self.assertEqual(result.cases[0]["query_plan"]["metadata"]["planner"], "fixed")

    def test_evaluator_extracts_long_answer_by_key_terms(self) -> None:
        now = utc_now_iso()
        node = MemoryNode(
            id="novel-answer-node",
            text="Felix lived with old De Lacey and his daughter Agatha in the cottage.",
            summary="Felix lived with De Lacey and Agatha.",
            keywords=["felix", "de", "lacey", "agatha"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[1.0, 0.0],
            metadata={"title": "Novel answer chunk"},
        )
        hit = RetrievalHit(
            node=node,
            score=1.0,
            similarity_score=1.0,
            graph_score=0.0,
            usage_score=0.0,
            confidence_score=0.0,
            path=[node.id],
            reason="测试命中",
        )

        answer = self.evaluator._extract_answer(
            "Felix, De Lacey, Agatha, (Safie)",
            [hit],
            {},
        )

        self.assertEqual(answer["status"], "answer_terms_covered")
        self.assertGreaterEqual(answer["term_coverage"], 0.5)
        self.assertIn("Felix", answer["answer"])

    def test_evaluation_isolates_method_state(self) -> None:
        first = self.evaluator.evaluate(
            self.queries,
            top_k=2,
            seed_k=1,
            hops=2,
            methods=["embedding_topk", "sam_full", "sam_no_graph"],
        )
        self.store.reset()
        documents, self.queries = load_builtin_benchmark_sample()
        self.evaluator.ingest(documents)
        second = self.evaluator.evaluate(
            self.queries,
            top_k=2,
            seed_k=1,
            hops=2,
            methods=["sam_no_graph", "sam_full", "embedding_topk"],
        )
        self.assertEqual(
            first.method_metrics["sam_full"]["support_hits"],
            second.method_metrics["sam_full"]["support_hits"],
        )
        self.assertEqual(
            first.method_metrics["sam_no_graph"]["answer_hit_count"],
            second.method_metrics["sam_no_graph"]["answer_hit_count"],
        )

    def test_evaluator_preserves_relation_judge_in_isolated_method_runs(self) -> None:
        class CountingRelationJudge:
            def __init__(self) -> None:
                self.calls = 0

            def judge(
                self,
                seed: MemoryNode,
                other: MemoryNode,
                score_breakdown: dict[str, object],
            ) -> RelationJudgment:
                self.calls += 1
                return RelationJudgment(
                    should_link=False,
                    relation_type="unrelated",
                    confidence=0.9,
                    reason="测试用关系判别器拒绝所有候选边",
                )

        judge = CountingRelationJudge()
        evaluator = Evaluator(
            self.store,
            self.embedding,
            GraphBuilder(self.store, relation_judge=judge),
        )

        evaluator.evaluate(
            self.queries[:1],
            top_k=2,
            seed_k=1,
            hops=1,
            methods=["sam_full"],
        )

        self.assertGreater(judge.calls, 0)

    def test_feedback_events_are_written(self) -> None:
        self.evaluator.evaluate(
            self.queries,
            top_k=3,
            seed_k=1,
            hops=2,
            methods=["sam_full"],
        )
        events = self.store.get_memory_events(limit=200)
        event_types = {event["event_type"] for event in events}
        self.assertIn("support_hit", event_types)
        self.assertTrue({"answer_hit", "path_rejected"} & event_types)

    def test_successful_retrieval_consolidates_support_memory(self) -> None:
        self.evaluator.evaluate(
            self.queries[:1],
            top_k=3,
            seed_k=1,
            hops=2,
            methods=["sam_full"],
        )

        consolidated_nodes = [
            node
            for node in self.store.get_nodes()
            if node.metadata.get("node_type") == "consolidated_memory"
        ]
        self.assertTrue(consolidated_nodes)
        consolidated = consolidated_nodes[0]
        self.assertEqual(consolidated.metadata.get("query_id"), self.queries[0].id)
        self.assertIn("consolidated_memory", consolidated.tags)
        self.assertGreater(consolidated.confidence, 0.7)
        self.assertIn(self.queries[0].answer, consolidated.text)

        edges = self.store.get_edges_for([consolidated.id])
        self.assertTrue(
            any(edge.relation_type == "consolidates_support" for edge in edges)
        )
        support_nodes = [
            node
            for node in self.store.get_nodes()
            if consolidated.id in node.metadata.get("consolidated_by", [])
        ]
        self.assertTrue(support_nodes)
        event_types = {
            event["event_type"]
            for event in self.store.get_memory_events(limit=200)
        }
        self.assertIn("memory_consolidated", event_types)

    def test_graph_export_nodes_include_consolidated_memory(self) -> None:
        self.evaluator.evaluate(
            self.queries[:1],
            top_k=3,
            seed_k=1,
            hops=2,
            methods=["sam_full"],
        )

        export_nodes = _nodes_for_graph_export(self.store)

        self.assertTrue(
            any(node.metadata.get("node_type") == "consolidated_memory" for node in export_nodes)
        )

    def test_consolidated_memory_is_intermediate_not_final_hit(self) -> None:
        self.evaluator.evaluate(
            self.queries[:1],
            top_k=3,
            seed_k=1,
            hops=2,
            methods=["sam_full"],
        )
        consolidated = next(
            node
            for node in self.store.get_nodes()
            if node.metadata.get("node_type") == "consolidated_memory"
        )
        query = self.queries[0]
        candidate_ids = [
            node.id
            for node in self.store.get_nodes()
            if node.metadata.get("original_doc_id") in query.candidate_doc_ids
        ]
        candidate_ids.append(consolidated.id)
        retriever = Retriever(self.store, self.embedding, self.graph)

        hits = retriever.retrieve(
            query=f"{query.question} {query.answer}",
            mode="sam_full",
            top_k=3,
            seed_k=1,
            hops=2,
            candidate_doc_ids=candidate_ids,
        )

        self.assertFalse(
            any(hit.node.metadata.get("node_type") == "consolidated_memory" for hit in hits)
        )
        self.assertTrue(any(consolidated.id in hit.path for hit in hits))

    def test_sam_candidate_pool_reuses_existing_consolidated_memory(self) -> None:
        self.evaluator.evaluate(
            self.queries[:1],
            top_k=3,
            seed_k=1,
            hops=2,
            methods=["sam_full"],
        )
        consolidated_ids = {
            node.id
            for node in self.store.get_nodes()
            if node.metadata.get("node_type") == "consolidated_memory"
        }
        consolidated_support_ids = {
            str(support_id)
            for node in self.store.get_nodes()
            if node.metadata.get("node_type") == "consolidated_memory"
            for support_id in node.metadata.get("support_node_ids", [])
        }
        support_original_ids = set(self.queries[0].supporting_doc_ids)
        base_candidate_ids = [
            node.id
            for node in self.store.get_nodes()
            if node.metadata.get("original_doc_id") in self.queries[0].candidate_doc_ids
            and node.metadata.get("original_doc_id") not in support_original_ids
        ][:2]
        self.assertTrue(consolidated_support_ids)
        self.assertFalse(consolidated_support_ids & set(base_candidate_ids))

        sam_candidates = self.evaluator._candidate_ids_for_method(
            self.store,
            "sam_full",
            base_candidate_ids,
        )
        vector_candidates = self.evaluator._candidate_ids_for_method(
            self.store,
            "embedding_topk",
            base_candidate_ids,
        )

        self.assertTrue(consolidated_ids & set(sam_candidates))
        self.assertTrue(consolidated_support_ids & set(sam_candidates))
        self.assertFalse(consolidated_ids & set(vector_candidates))
        self.assertFalse(consolidated_support_ids & set(vector_candidates))

    def test_memory_reuse_experiment_masks_gold_support(self) -> None:
        masked = build_masked_queries(self.queries[:1])

        self.assertEqual(masked[0].supporting_doc_ids, self.queries[0].supporting_doc_ids)
        self.assertFalse(set(masked[0].supporting_doc_ids) & set(masked[0].candidate_doc_ids))
        self.assertEqual(masked[0].metadata["reuse_probe"], True)

    def test_memory_reuse_summary_reports_gain(self) -> None:
        baseline = {"support_hits": 0, "evidence_recall": 0.0}
        sam = {"support_hits": 2, "evidence_recall": 1.0}

        summary = summarize_memory_reuse(
            warmup_consolidated_count=1,
            warmup_consolidation_edge_count=2,
            baseline_metric=baseline,
            sam_metric=sam,
        )

        self.assertEqual(summary["support_hit_gain"], 2)
        self.assertEqual(summary["evidence_recall_gain"], 1.0)

    def test_no_feedback_mode_skips_feedback_events(self) -> None:
        self.evaluator.evaluate(
            self.queries,
            top_k=3,
            seed_k=1,
            hops=2,
            methods=["sam_no_feedback"],
        )
        events = self.store.get_memory_events(limit=200)
        event_types = {event["event_type"] for event in events}
        self.assertIn("node_retrieved", event_types)
        self.assertNotIn("support_hit", event_types)
        self.assertNotIn("answer_hit", event_types)
        self.assertNotIn("path_rejected", event_types)

    def test_azure_embedding_provider_uses_env_config(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "test-key",
                "SAM_AZURE_EMBEDDING_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
                "SAM_AZURE_EMBEDDING_API_VERSION": "2023-07-01-preview",
                "SAM_AZURE_EMBEDDING_MODEL": "text-embedding-3-large",
                "SAM_AZURE_EMBEDDING_DIMENSIONS": "1024",
            },
            clear=False,
        ):
            provider = create_embedding_provider("azure_openai")
            self.assertIsInstance(provider, AzureOpenAIEmbeddingProvider)
            assert isinstance(provider, AzureOpenAIEmbeddingProvider)
            self.assertEqual(provider.dimensions, 1024)
            self.assertIn("/openai/deployments/text-embedding-3-large/embeddings", provider.request_url)
            self.assertIn("api-version=2023-07-01-preview", provider.request_url)

    def test_azure_embedding_provider_batches_requests_with_model_and_dimensions(self) -> None:
        class FakeResponse:
            def __init__(self, payload: dict[str, object]) -> None:
                inputs = payload["input"]
                assert isinstance(inputs, list)
                self.body = json.dumps(
                    {
                        "data": [
                            {"embedding": [float(index), float(len(text))]}
                            for index, text in enumerate(inputs)
                        ]
                    }
                ).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def read(self) -> bytes:
                return self.body

        requests: list[dict[str, object]] = []

        def fake_urlopen(request, timeout=60):
            payload = json.loads(request.data.decode("utf-8"))
            requests.append(payload)
            return FakeResponse(payload)

        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "test-key",
                "SAM_AZURE_EMBEDDING_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
                "SAM_AZURE_EMBEDDING_API_VERSION": "2023-07-01-preview",
                "SAM_AZURE_EMBEDDING_MODEL": "text-embedding-3-large",
                "SAM_AZURE_EMBEDDING_DIMENSIONS": "1024",
                "SAM_AZURE_EMBEDDING_BATCH_SIZE": "2",
                "SAM_AZURE_EMBEDDING_CONCURRENCY": "1",
            },
            clear=False,
        ), patch("urllib.request.urlopen", fake_urlopen):
            provider = AzureOpenAIEmbeddingProvider()
            embeddings = provider.embed_many(["alpha", "beta", "gamma"])

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["input"], ["alpha", "beta"])
        self.assertEqual(requests[1]["input"], ["gamma"])
        self.assertEqual(requests[0]["model"], "text-embedding-3-large")
        self.assertEqual(requests[0]["dimensions"], 1024)
        self.assertEqual(embeddings, [[0.0, 5.0], [1.0, 4.0], [0.0, 5.0]])

    def test_cached_embedding_provider_reuses_vectors(self) -> None:
        class CountingEmbeddingProvider(LocalHashEmbeddingProvider):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            @property
            def cache_namespace(self) -> str:
                return "counting-local"

            def embed(self, text: str) -> list[float]:
                self.calls += 1
                return super().embed(text)

        inner = CountingEmbeddingProvider()
        provider = CachedEmbeddingProvider(inner, Path(self.temp_dir.name) / "embedding_cache.sqlite")
        first = provider.embed_many(["alpha", "beta", "alpha"])
        second = provider.embed_many(["alpha", "beta"])
        provider.close()
        self.assertEqual(first[0], first[2])
        self.assertEqual(first[:2], second)
        self.assertEqual(inner.calls, 2)

    def test_analogy_engine_returns_case_hints(self) -> None:
        engine = AnalogyEngine(self.store, self.embedding, self.graph)
        matches = engine.retrieve_cases(
            "Which university location can help connect graph memory research to a city?",
            top_k=2,
        )
        self.assertTrue(matches)
        self.assertTrue(matches[0].case_id)
        self.assertTrue(matches[0].matched_nodes)
        self.assertIn("当前问题可类比历史案例", matches[0].prompt_hint)

    def test_analogy_engine_prefers_matching_relation_path(self) -> None:
        self.store.reset()
        now = utc_now_iso()

        def node(case_id: str, suffix: str, text: str) -> MemoryNode:
            return MemoryNode(
                id=f"{case_id}_{suffix}",
                text=text,
                summary=text,
                keywords=text.lower().split()[:8],
                tags=["case"],
                source="unit-test",
                created_at=now,
                last_accessed_at=None,
                usage_count=0,
                confidence=0.8,
                embedding=self.embedding.embed(text),
                metadata={"query_id": case_id, "title": f"{case_id}-{suffix}"},
            )

        matching_nodes = [
            node("case_path_match", "seed", "bridge evidence activates shared entity memory"),
            node("case_path_match", "middle", "shared entity leads to keyword bridge"),
            node("case_path_match", "answer", "keyword bridge supports final answer"),
        ]
        mismatch_nodes = [
            node("case_path_mismatch", "seed", "bridge evidence starts another case"),
            node("case_path_mismatch", "middle", "unrelated context cooccurrence appears"),
            node("case_path_mismatch", "answer", "semantic similarity gives weak answer"),
        ]
        self.store.upsert_nodes([*matching_nodes, *mismatch_nodes])
        self.store.upsert_edges(
            [
                MemoryEdge(
                    source_id="case_path_match_seed",
                    target_id="case_path_match_middle",
                    relation_type="shared_entity",
                    weight=0.8,
                    reason="测试路径第一跳",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                ),
                MemoryEdge(
                    source_id="case_path_match_middle",
                    target_id="case_path_match_answer",
                    relation_type="keyword_overlap",
                    weight=0.7,
                    reason="测试路径第二跳",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                ),
                MemoryEdge(
                    source_id="case_path_mismatch_seed",
                    target_id="case_path_mismatch_middle",
                    relation_type="context_cooccurrence",
                    weight=0.8,
                    reason="不匹配路径第一跳",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                ),
                MemoryEdge(
                    source_id="case_path_mismatch_middle",
                    target_id="case_path_mismatch_answer",
                    relation_type="embedding_similarity",
                    weight=0.7,
                    reason="不匹配路径第二跳",
                    created_at=now,
                    updated_at=now,
                    activation_count=0,
                    last_activated_at=None,
                ),
            ]
        )

        engine = AnalogyEngine(self.store, self.embedding, self.graph)
        matches = engine.retrieve_cases(
            "bridge evidence should use a shared entity and then a keyword bridge",
            top_k=2,
            relation_pattern=["shared_entity", "keyword_overlap"],
        )

        self.assertEqual(matches[0].case_id, "case_path_match")
        self.assertGreater(matches[0].metadata["path_pattern_score"], 0.0)
        self.assertEqual(
            matches[0].metadata["matched_relation_path"],
            ["shared_entity", "keyword_overlap"],
        )
        self.assertIn("关系路径", matches[0].prompt_hint)

    def test_analogy_engine_exposes_consolidated_case_metadata(self) -> None:
        self.evaluator.evaluate(
            self.queries[:1],
            top_k=3,
            seed_k=1,
            hops=2,
            methods=["sam_full"],
        )
        engine = AnalogyEngine(self.store, self.embedding, self.graph)

        matches = engine.retrieve_cases(
            f"Use previous evidence pattern to answer: {self.queries[0].question}",
            top_k=1,
        )

        self.assertTrue(matches)
        self.assertEqual(matches[0].case_id, self.queries[0].id)
        self.assertEqual(matches[0].metadata["is_consolidated_case"], True)
        self.assertEqual(matches[0].metadata["case_answer"], self.queries[0].answer)
        self.assertTrue(matches[0].metadata["support_node_ids"])

    def test_analogy_reuse_probe_hits_consolidated_source_case(self) -> None:
        self.evaluator.evaluate(
            self.queries[:1],
            top_k=3,
            seed_k=1,
            hops=2,
            methods=["sam_full"],
        )
        masked = build_masked_queries(self.queries[:1])
        engine = AnalogyEngine(self.store, self.embedding, self.graph)

        result = run_analogy_reuse_probe(engine, masked, top_k=1)

        self.assertEqual(result["query_count"], 1)
        self.assertEqual(result["consolidated_case_hit_count"], 1)
        self.assertEqual(result["support_overlap_hit_count"], 1)
        self.assertTrue(result["cases"][0]["top_match"]["is_consolidated_case"])

    def test_shared_memory_coordinator_writes_layered_agent_memory(self) -> None:
        coordinator = SharedMemoryCoordinator(self.store, self.embedding)
        coordinator.write_memory(
            agent_id="planner",
            layer="global_insight",
            text="跨文档问答需要先锁定种子证据，再沿语义边寻找桥接证据。",
            session_id="s1",
        )
        coordinator.write_memory(
            agent_id="writer",
            layer="interaction",
            text="本轮回答应引用 HotpotQA 的 bridge-style 证据链。",
            session_id="s1",
        )
        hits = coordinator.query_memory(
            "如何寻找跨文档桥接证据？",
            layers={"global_insight", "interaction"},
            session_id="s1",
            include_other_sessions=False,
        )
        self.assertEqual(len(hits), 2)
        self.assertTrue({hit.metadata["agent_id"] for hit in hits} >= {"planner", "writer"})
        self.assertTrue(all(hit.usage_count == 0 for hit in hits))
        updated = self.store.get_nodes([hit.id for hit in hits])
        self.assertTrue(all(node.usage_count >= 1 for node in updated))

    def test_shared_memory_coordinator_filters_agent_handoffs(self) -> None:
        coordinator = SharedMemoryCoordinator(self.store, self.embedding)
        coordinator.write_handoff(
            source_agent_id="planner",
            target_agent_id="writer",
            text="写作智能体需要使用 bridge evidence 的两跳证据链组织答案。",
            session_id="s2",
            task_id="task-bridge",
        )
        coordinator.write_handoff(
            source_agent_id="planner",
            target_agent_id="verifier",
            text="验证智能体需要检查答案是否覆盖 supporting facts。",
            session_id="s2",
            task_id="task-bridge",
        )

        writer_hits = coordinator.query_memory(
            "如何组织两跳证据链答案？",
            layers={"session"},
            session_id="s2",
            agent_id="writer",
        )

        self.assertEqual(len(writer_hits), 1)
        self.assertEqual(writer_hits[0].metadata["target_agent_id"], "writer")
        self.assertEqual(writer_hits[0].metadata["source_agent_id"], "planner")
        self.assertEqual(writer_hits[0].metadata["task_id"], "task-bridge")

    def test_multi_agent_workflow_uses_shared_handoffs(self) -> None:
        class EvidenceAwareChatClient(ChatClient):
            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                prompt = "\n".join(str(message.get("content", "")) for message in messages)
                if "writer handoff" in prompt:
                    return "author"
                return "证据不足"

        case = {
            "query_id": "workflow_case",
            "question": "Which answer is identified by the writer handoff?",
            "answer": "author",
            "support_hits_by_method": {"sam_full": 1},
            "final_answers": {"sam_full": {"status": "found_in_retrieved_context"}},
            "methods": {
                "sam_full": [
                    {
                        "title": "Workflow evidence",
                        "text": "The writer handoff says the answer is author.",
                        "reason": "向量种子节点 -> shared_entity",
                        "candidate_paths": [{"relation_type": "shared_entity"}],
                        "is_supporting": True,
                    }
                ]
            },
        }
        coordinator = SharedMemoryCoordinator(self.store, self.embedding)
        workflow = MultiAgentResearchWorkflow(
            coordinator=coordinator,
            generator=ContextAnswerGenerator(EvidenceAwareChatClient()),
            method="sam_full",
        )

        result = workflow.run_case(case)

        self.assertEqual(result["query_id"], "workflow_case")
        self.assertEqual(
            [step["agent_id"] for step in result["agent_steps"]],
            ["planner", "retriever", "writer", "verifier"],
        )
        self.assertTrue(result["shared_memory_node_ids"])
        self.assertTrue(result["writer_memory"])
        self.assertTrue(result["verifier"]["answer_hit"])
        self.assertEqual(result["verifier"]["status"], "passed")
        output_dir = Path(self.temp_dir.name) / "agent_workflow"
        json_path, markdown_path = write_agent_workflow_reports([result], output_dir)
        self.assertTrue(json_path.exists())
        self.assertTrue(markdown_path.exists())

    def test_agent_memory_reuse_probe_reports_cross_agent_reuse(self) -> None:
        class EvidenceAwareChatClient(ChatClient):
            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                prompt = "\n".join(str(message.get("content", "")) for message in messages)
                if "writer handoff" in prompt:
                    return "author"
                return "证据不足"

        cases = [
            {
                "query_id": "agent_reuse_case",
                "question": "Which answer is carried by the shared memory handoff?",
                "answer": "author",
                "support_hits_by_method": {"embedding_topk": 0, "sam_no_feedback": 1},
                "final_answers": {"sam_no_feedback": {"status": "found_in_retrieved_context"}},
                "methods": {
                    "embedding_topk": [],
                    "sam_no_feedback": [
                        {
                            "title": "Shared memory evidence",
                            "text": "The writer handoff says the answer is author.",
                            "reason": "巩固记忆 -> 证据节点",
                            "candidate_paths": [{"relation_type": "consolidates_support"}],
                            "is_supporting": True,
                        }
                    ],
                },
            }
        ]
        coordinator = SharedMemoryCoordinator(self.store, self.embedding)
        workflow = MultiAgentResearchWorkflow(
            coordinator=coordinator,
            generator=ContextAnswerGenerator(EvidenceAwareChatClient()),
            method="sam_no_feedback",
        )

        result = run_agent_memory_reuse_probe(
            cases,
            workflow=workflow,
            method="sam_no_feedback",
            baseline_method="embedding_topk",
        )

        self.assertEqual(result["summary"]["query_count"], 1)
        self.assertEqual(result["summary"]["support_gain_count"], 1)
        self.assertEqual(result["summary"]["writer_handoff_used_count"], 1)
        self.assertEqual(result["summary"]["verifier_handoff_used_count"], 1)
        self.assertEqual(result["summary"]["multi_agent_reuse_success_count"], 1)
        self.assertTrue(result["cases"][0]["writer_used_retriever_handoff"])
        self.assertTrue(result["cases"][0]["verifier_used_writer_handoff"])

        output_dir = Path(self.temp_dir.name) / "agent_memory_reuse"
        json_path, markdown_path = write_agent_memory_reuse_reports(result, output_dir)
        self.assertTrue(json_path.exists())
        self.assertTrue(markdown_path.exists())

    def test_agent_generation_variants_compare_shared_memory_and_analogy(self) -> None:
        class VariantAwareChatClient(ChatClient):
            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                prompt = "\n".join(str(message.get("content", "")) for message in messages)
                if "历史案例" in prompt and "old_agent_case" in prompt:
                    return "author"
                if "retriever handoff" in prompt:
                    return "author"
                return "证据不足"

        old_case = {
            "query_id": "old_agent_case",
            "question": "Which bridge evidence identifies a director?",
            "answer": "director",
            "support_hits_by_method": {"sam_full": 1},
            "final_answers": {"sam_full": {"status": "found_in_retrieved_context"}},
            "methods": {
                "sam_full": [
                    {
                        "title": "Old shared relation",
                        "text": "The old bridge evidence identifies a director.",
                        "reason": "向量种子节点 -> shared_entity",
                        "candidate_paths": [{"relation_type": "shared_entity"}],
                    }
                ]
            },
        }
        new_case = {
            "query_id": "new_agent_case",
            "question": "Which bridge evidence identifies an author?",
            "answer": "author",
            "support_hits_by_method": {"sam_full": 1},
            "final_answers": {"sam_full": {"status": "found_in_retrieved_context"}},
            "methods": {
                "sam_full": [
                    {
                        "title": "New shared relation",
                        "text": "The current evidence needs another role.",
                        "reason": "向量种子节点 -> shared_entity",
                        "candidate_paths": [{"relation_type": "shared_entity"}],
                    }
                ]
            },
        }
        chat_client = VariantAwareChatClient()
        generator = ContextAnswerGenerator(chat_client)
        workflow = MultiAgentResearchWorkflow(
            coordinator=SharedMemoryCoordinator(self.store, self.embedding),
            generator=ContextAnswerGenerator(chat_client),
            method="sam_full",
        )

        comparison = compare_agent_generation_variants(
            [new_case],
            all_cases=[old_case, new_case],
            workflow=workflow,
            generator=generator,
            method="sam_full",
            analogy_top_k=1,
        )

        self.assertEqual(comparison["query_count"], 1)
        self.assertEqual(comparison["variants"]["baseline"]["answer_hit_count"], 0)
        self.assertEqual(comparison["variants"]["shared_memory"]["answer_hit_count"], 1)
        self.assertEqual(comparison["variants"]["shared_memory_with_analogy"]["answer_hit_count"], 1)
        self.assertEqual(comparison["delta"]["shared_memory_vs_baseline_answer_hits"], 1)
        self.assertEqual(comparison["case_deltas"][0]["shared_memory_status"], "improved")
        self.assertTrue(comparison["answers"]["shared_memory_with_analogy"][0]["metadata"]["analogy_hints"])

        output_dir = Path(self.temp_dir.name) / "agent_generation"
        json_path, markdown_path = write_agent_generation_comparison_reports(comparison, output_dir)
        self.assertTrue(json_path.exists())
        self.assertTrue(markdown_path.exists())

    def test_generation_and_badcase_reports_are_written(self) -> None:
        result = self.evaluator.evaluate(
            self.queries,
            top_k=2,
            seed_k=1,
            hops=2,
            methods=["embedding_topk", "sam_full"],
        )
        report_dir = Path(self.temp_dir.name) / "reports"
        self.evaluator.write_reports(result, report_dir)
        self.assertTrue((report_dir / "bad_cases.json").exists())
        self.assertTrue((report_dir / "bad_cases.md").exists())
        first_case = result.cases[0]
        self.assertIn("text", first_case["methods"]["sam_full"][0])

        generator = ContextAnswerGenerator(HeuristicChatClient())
        generated = generator.generate_for_case(first_case, method="sam_full")
        self.assertEqual(generated.query_id, first_case["query_id"])
        self.assertTrue(generated.generated_answer)
        json_path, markdown_path = write_generation_reports([generated], report_dir)
        self.assertTrue(json_path.exists())
        self.assertTrue(markdown_path.exists())
        self.assertTrue((report_dir / "generation_bad_cases.json").exists())
        self.assertTrue((report_dir / "generation_bad_cases.md").exists())

    def test_rule_based_answer_judge_accepts_key_term_coverage(self) -> None:
        judgment = RuleBasedAnswerJudge().judge(
            question="Who lived in the cottage?",
            gold_answer="Felix, De Lacey, Agatha, Safie",
            generated_answer="The retrieved context identifies Felix, De Lacey, and Agatha.",
        )

        self.assertTrue(judgment.answer_hit)
        self.assertEqual(judgment.status, "key_terms_covered")
        self.assertGreaterEqual(judgment.score, 0.5)

    def test_context_answer_generator_uses_injected_answer_judge(self) -> None:
        class FixedChatClient(ChatClient):
            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                return "The answer is semantically correct but does not repeat the exact string."

        class AcceptingJudge:
            def judge(self, question: str, gold_answer: str, generated_answer: str) -> AnswerJudgment:
                return AnswerJudgment(
                    answer_hit=True,
                    status="llm_equivalent",
                    score=0.91,
                    reason="测试用 judge 判断语义等价",
                    metadata={"judge": "fixed"},
                )

        generator = ContextAnswerGenerator(FixedChatClient(), answer_judge=AcceptingJudge())
        answer = generator.generate_from_hits(
            query_id="judge-case",
            method="sam_full",
            question="What is the answer?",
            gold_answer="gold string",
            hits=[
                {
                    "title": "Evidence",
                    "text": "Evidence text.",
                    "path": ["n1"],
                }
            ],
        )

        self.assertTrue(answer.answer_hit)
        self.assertEqual(answer.metadata["answer_judgment"]["status"], "llm_equivalent")
        self.assertEqual(answer.metadata["answer_judgment"]["metadata"]["judge"], "fixed")

    def test_heuristic_chat_client_does_not_return_system_prompt(self) -> None:
        answer = HeuristicChatClient().complete(
            [
                {
                    "role": "system",
                    "content": "你是一个严格基于检索证据回答问题的研究助手。",
                },
                {
                    "role": "user",
                    "content": "问题：What is the answer?\n\n上下文：\n[1] Evidence\nNo answer here.\n\n请输出最终答案。",
                },
            ]
        )

        self.assertNotIn("严格基于检索证据", answer)
        self.assertEqual(answer, "证据不足")

    def test_generation_can_use_case_analogy_hints(self) -> None:
        cases = [
            {
                "query_id": "old_bridge_case",
                "question": "Which bridge evidence connects a film to its director?",
                "answer": "director",
                "support_hits_by_method": {"sam_full": 2},
                "final_answers": {"sam_full": {"status": "found_in_retrieved_context"}},
                "methods": {
                    "sam_full": [
                        {
                            "title": "Old evidence",
                            "text": "The film evidence connects to the director.",
                            "reason": "向量种子节点 -> shared_entity -> keyword_overlap",
                            "candidate_paths": [
                                {"relation_type": "shared_entity"},
                                {"relation_type": "keyword_overlap"},
                            ],
                        }
                    ]
                },
            },
            {
                "query_id": "new_bridge_case",
                "question": "Which bridge evidence connects a novel to its author?",
                "answer": "author",
                "support_hits_by_method": {"sam_full": 1},
                "final_answers": {"sam_full": {"status": "found_in_retrieved_context"}},
                "methods": {
                    "sam_full": [
                        {
                            "title": "New evidence",
                            "text": "The novel evidence identifies the author.",
                            "reason": "向量种子节点 -> shared_entity",
                            "candidate_paths": [
                                {"relation_type": "shared_entity"},
                            ],
                        }
                    ]
                },
            },
        ]

        hint_builder = CaseAnalogyHintBuilder(cases, method="sam_full")
        hints = hint_builder.hints_for(cases[1], top_k=1)
        self.assertEqual(len(hints), 1)
        self.assertIn("old_bridge_case", hints[0])
        self.assertIn("关系路径", hints[0])

        generator = ContextAnswerGenerator(HeuristicChatClient())
        answers = generate_answers_for_cases(
            [cases[1]],
            generator,
            method="sam_full",
            analogy_hint_builder=hint_builder,
        )
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0].metadata["analogy_hints"], hints)

    def test_generation_comparison_reports_analogy_delta(self) -> None:
        class AnalogyAwareChatClient(ChatClient):
            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                prompt = "\n".join(str(message.get("content", "")) for message in messages)
                if "历史案例" in prompt:
                    return "author"
                return "证据不足"

        cases = [
            {
                "query_id": "old_bridge_case",
                "question": "Which bridge evidence connects a film to its director?",
                "answer": "director",
                "support_hits_by_method": {"sam_full": 2},
                "final_answers": {"sam_full": {"status": "found_in_retrieved_context"}},
                "methods": {
                    "sam_full": [
                        {
                            "title": "Old evidence",
                            "text": "The film evidence connects to the director.",
                            "reason": "向量种子节点 -> shared_entity",
                            "candidate_paths": [{"relation_type": "shared_entity"}],
                        }
                    ]
                },
            },
            {
                "query_id": "new_bridge_case",
                "question": "Which bridge evidence connects a novel to its author?",
                "answer": "author",
                "support_hits_by_method": {"sam_full": 1},
                "final_answers": {"sam_full": {"status": "found_in_retrieved_context"}},
                "methods": {
                    "sam_full": [
                        {
                            "title": "New evidence",
                            "text": "The novel evidence identifies the author.",
                            "reason": "向量种子节点 -> shared_entity",
                            "candidate_paths": [{"relation_type": "shared_entity"}],
                        }
                    ]
                },
            },
        ]

        comparison = compare_generation_variants(
            [cases[1]],
            all_cases=cases,
            generator=ContextAnswerGenerator(AnalogyAwareChatClient()),
            method="sam_full",
            analogy_top_k=1,
        )

        self.assertEqual(comparison["variants"]["baseline"]["answer_hit_rate"], 0.0)
        self.assertEqual(comparison["variants"]["with_analogy"]["answer_hit_rate"], 1.0)
        self.assertEqual(comparison["delta"]["answer_hit_rate"], 1.0)
        self.assertEqual(comparison["case_deltas"][0]["status"], "improved")
        output_dir = Path(self.temp_dir.name) / "generation_comparison"
        json_path, markdown_path = write_generation_comparison_reports(comparison, output_dir)
        self.assertTrue(json_path.exists())
        self.assertTrue(markdown_path.exists())

    def test_badcase_analyzer_classifies_missing_support(self) -> None:
        cases = [
            {
                "query_id": "q1",
                "question": "question",
                "answer": "answer",
                "supporting_doc_ids": ["d1", "d2"],
                "vector_support_hits": 2,
                "support_hits_by_method": {"sam_full": 1},
                "final_answers": {"sam_full": {"status": "not_found_in_retrieved_context"}},
                "methods": {
                    "sam_full": [
                        {"is_supporting": False, "path": ["a", "b"]},
                        {"is_supporting": True, "path": ["c"]},
                    ]
                },
            }
        ]
        bad_cases = BadCaseAnalyzer().analyze(cases, method="sam_full")
        self.assertEqual(len(bad_cases), 1)
        self.assertIn("missing_support_evidence", bad_cases[0].categories)
        self.assertIn("worse_than_vector", bad_cases[0].categories)
        output_dir = Path(self.temp_dir.name) / "badcase"
        json_path, markdown_path = write_bad_case_reports(bad_cases, output_dir)
        self.assertTrue(json_path.exists())
        self.assertTrue(markdown_path.exists())

    def test_generation_badcase_analyzer_uses_answer_judgment(self) -> None:
        answers = [
            {
                "query_id": "gen-q1",
                "method": "sam_full",
                "question": "Who lived in the cottage?",
                "gold_answer": "Felix, De Lacey, Agatha, Safie",
                "generated_answer": "The answer is unclear.",
                "answer_hit": False,
                "context_titles": ["Novel chunk 1"],
                "metadata": {
                    "answer_judgment": {
                        "answer_hit": False,
                        "status": "not_matched",
                        "score": 0.2,
                        "reason": "关键内容词覆盖不足",
                        "metadata": {"judge": "rule"},
                    }
                },
            }
        ]

        bad_cases = GenerationBadCaseAnalyzer().analyze(answers)

        self.assertEqual(len(bad_cases), 1)
        self.assertIn("generated_answer_not_equivalent", bad_cases[0].categories)
        self.assertIn("answer_judgment", bad_cases[0].metadata)
        output_dir = Path(self.temp_dir.name) / "generation_badcase"
        json_path, markdown_path = write_generation_bad_case_reports(bad_cases, output_dir)
        self.assertTrue(json_path.exists())
        self.assertIn("生成 Bad Case 分析", markdown_path.read_text(encoding="utf-8"))

    def test_sam_dataset_format_round_trip(self) -> None:
        documents, queries = load_builtin_benchmark_sample()
        output_path = Path(self.temp_dir.name) / "sample_sam_dataset.json"
        save_sam_dataset(
            output_path,
            documents=documents,
            queries=queries,
            dataset_info={"name": "unit-test"},
            processing={"source_script": "tests/test_core.py"},
        )
        loaded_documents, loaded_queries, payload = load_sam_dataset(output_path)
        summary = summarize_sam_dataset(output_path)
        self.assertEqual(payload["schema_version"], "sam-dataset-v1")
        self.assertEqual(len(loaded_documents), len(documents))
        self.assertEqual(len(loaded_queries), len(queries))
        self.assertEqual(summary["query_count"], len(queries))
        self.assertEqual(loaded_queries[0].metadata, queries[0].metadata)

    def test_novelqa_adapter_reads_local_directory(self) -> None:
        source_root = Path(self.temp_dir.name) / "NovelQA"
        (source_root / "Books" / "PublicDomain").mkdir(parents=True)
        (source_root / "Data" / "PublicDomain").mkdir(parents=True)
        (source_root / "Books" / "PublicDomain" / "B00.txt").write_text(
            "Alice met the White Rabbit near the river. " * 80,
            encoding="utf-8",
        )
        (source_root / "Data" / "PublicDomain" / "B00.json").write_text(
            json.dumps(
                {
                    "Q0001": {
                        "QID": "Q0001",
                        "Aspect": "plot",
                        "Complexity": "mh",
                        "Question": "Who did Alice meet?",
                        "Options": {"A": "White Rabbit", "B": "Mad Hatter"},
                        "Answer": "A",
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        documents, queries, manifest = load_novelqa_sample(
            source_root,
            sample_size=1,
            max_books=1,
            chunk_chars=300,
            chunk_overlap=20,
            max_chunks_per_book=3,
        )
        self.assertEqual(len(queries), 1)
        self.assertEqual(len(documents), 3)
        self.assertEqual(queries[0].metadata["options"]["A"], "White Rabbit")
        self.assertIn("Alice", queries[0].metadata["retrieval_query"])
        self.assertIn("plot", queries[0].metadata["retrieval_query"])
        self.assertNotIn("Mad Hatter", queries[0].metadata["retrieval_query"])
        self.assertEqual(manifest["selected_books"][0]["book_id"], "B00")

    def test_novelqa_demonstration_maps_evidence_to_chunks(self) -> None:
        source_root = Path(self.temp_dir.name) / "NovelQA"
        (source_root / "Demonstration").mkdir(parents=True)
        (source_root / "Demonstration" / "Frankenstein.txt").write_text(
            "Victor studies natural philosophy. The creature speaks with Victor near the mountain. "
            "Elizabeth waits for news from Geneva.",
            encoding="utf-8",
        )
        (source_root / "Demonstration" / "Frankenstein.json").write_text(
            json.dumps(
                [
                    {
                        "QID": "Q0147",
                        "Aspect": "plot",
                        "Complexity": "mh",
                        "Question": "Who speaks with Victor?",
                        "Answer": "The creature",
                        "Gold": "A",
                        "Options": {"A": "The creature", "B": "Elizabeth"},
                        "Evidences": [
                            {
                                "EID": "E0001",
                                "Evidence": "The creature speaks with Victor near the mountain.",
                            }
                        ],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        documents, queries, manifest = load_novelqa_sample(
            source_root,
            sample_size=1,
            max_books=1,
            chunk_chars=80,
            chunk_overlap=10,
            max_chunks_per_book=3,
            split="demonstration",
        )
        self.assertEqual(len(documents), 2)
        self.assertEqual(queries[0].answer, "The creature")
        self.assertEqual(len(queries[0].supporting_doc_ids), 1)
        self.assertEqual(manifest["split"], "demonstration")


if __name__ == "__main__":
    unittest.main()
