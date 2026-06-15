from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import asyncio
import hashlib
import os
import sys
import json
import sqlite3
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.datasets import documents_to_nodes, load_builtin_benchmark_sample, load_novelqa_sample, load_scifact_sample
from sam.dataset_format import load_sam_dataset, save_sam_dataset, summarize_sam_dataset
from sam import agent_workflow
from sam.agent_workflow import MultiAgentResearchWorkflow, write_agent_workflow_reports
from sam.agent_reuse_experiment import (
    compare_agent_generation_variants,
    run_agent_memory_reuse_probe,
    write_agent_generation_comparison_reports,
    write_agent_memory_reuse_reports,
)
from sam.agents import SharedMemoryCoordinator
from sam.analogy import AnalogyEngine
from sam.analogy_experiment import run_analogy_reuse_probe, write_analogy_reuse_reports
from sam.badcase import (
    BadCaseAnalyzer,
    GenerationBadCaseAnalyzer,
    write_bad_case_reports,
    write_generation_bad_case_reports,
)
from sam.answer_judge import AnswerJudgment, RuleBasedAnswerJudge
from sam.consolidation import MemoryConsolidator
from sam.embedding import (
    AzureOpenAISDKEmbeddingProvider,
    AzureOpenAIEmbeddingProvider,
    CachedEmbeddingProvider,
    EmbeddingProvider,
    LocalHashEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    create_embedding_provider,
    inspect_embedding_provider_config,
    preflight_embedding_endpoint,
)
from sam.embedding_plan import build_embedding_run_plan, warm_embedding_cache
from sam.edge_audit import audit_edge_quality, write_edge_quality_audit
from sam.env import apply_provider_env_aliases, load_default_env_file, load_env_file
from sam.evaluator import Evaluator, _feedback_enabled
from sam.experiment_audit import audit_run_directory, write_experiment_audit
from sam.generation import (
    CaseAnalogyHintBuilder,
    ContextAnswerGenerator,
    compare_generation_variants,
    generate_answers_for_cases,
    write_generation_comparison_reports,
    write_generation_reports,
)
from sam.graph_cost_audit import audit_graph_build_cost, write_graph_build_cost_audit
from sam.graph import GraphBuilder
from sam.graph_strategy_experiment import (
    GraphStrategyConfig,
    GraphStrategyExperiment,
    context_path_proximity,
    progress_iter,
    run_alpha_sweep,
    position_proximity,
    write_graph_strategy_report,
)
from sam.llm import (
    AzureOpenAIChatClient,
    AzureOpenAISDKChatClient,
    ChatClient,
    HeuristicChatClient,
    create_chat_client,
    inspect_chat_provider_config,
)
from sam.models import DatasetDocument, EvaluationQuery, MemoryEdge, MemoryNode, RetrievalHit, utc_now_iso
from sam.object_graph import (
    BridgeEntity,
    CrossGraphRetriever,
    LocalEvidenceGraph,
    LocalEvidenceUnit,
    ObjectGraphBuilder,
)
from sam.opening_audit import build_opening_plan_audit, write_opening_plan_audit
from sam.query_planner import ChatQueryPlanner, HeuristicQueryPlanner, QueryPlan
from sam.pipeline_experiment import run_retrieval_generation_pipeline
from sam.reranker import PathReranker
from sam.reranker_experiment import (
    run_reranker_profile_comparison,
    write_reranker_profile_reports,
)
from sam.relation_judge import (
    BudgetedRelationJudge,
    CachedRelationJudge,
    ChatRelationJudge,
    RelationJudgment,
    _relation_cache_key,
    create_relation_judge,
    relation_judge_stats,
)
from sam.retriever import Retriever
from sam.reuse_experiment import (
    build_masked_queries,
    memory_reuse_candidate_ids,
    snapshot_edges,
    summarize_memory_reuse,
    write_memory_reuse_event_reports,
    write_memory_reuse_reports,
)
from sam.store import MemoryStore
from scripts import run_graph_strategy_experiment as graph_strategy_script
from scripts.run_demo import _nodes_for_graph_export
from scripts.check_embedding_provider import build_embedding_status
from scripts.check_model_providers import build_provider_status
from scripts.create_env_template import write_env_template
from scripts.plan_local_embedding import build_local_embedding_plan, write_local_embedding_plan
from scripts.run_provider_smoke_experiment import run_provider_smoke_experiment


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

    def test_graph_strategy_script_loads_env_before_explicit_provider(self) -> None:
        order: list[str] = []

        class DummyProvider(EmbeddingProvider):
            def embed(self, text: str) -> list[float]:
                return [0.0]

        with patch.object(graph_strategy_script, "load_default_env_file", side_effect=lambda: order.append("load")):
            with patch.object(
                graph_strategy_script,
                "create_embedding_provider",
                side_effect=lambda name: order.append(f"create:{name}") or DummyProvider(),
            ):
                provider = graph_strategy_script._create_embedding_provider("azure_openai_sdk")

        self.assertEqual(order, ["load", "create:azure_openai_sdk"])
        self.assertIsInstance(provider, DummyProvider)

    def test_graph_strategy_script_overrides_embedding_parallel_env_before_provider(self) -> None:
        order: list[str] = []

        class DummyProvider(EmbeddingProvider):
            def embed(self, text: str) -> list[float]:
                return [0.0]

        def fake_load_env() -> None:
            order.append("load")
            os.environ["SAM_AZURE_EMBEDDING_CONCURRENCY"] = "1"
            os.environ["SAM_OPENAI_EMBEDDING_CONCURRENCY"] = "1"
            os.environ["SAM_AZURE_EMBEDDING_BATCH_SIZE"] = "4"
            os.environ["SAM_OPENAI_EMBEDDING_BATCH_SIZE"] = "4"
            os.environ["SAM_AZURE_EMBEDDING_INPUT_MODE"] = "batch"

        def fake_create_provider(name: str) -> EmbeddingProvider:
            order.append(f"create:{name}")
            self.assertEqual(os.environ["SAM_AZURE_EMBEDDING_CONCURRENCY"], "12")
            self.assertEqual(os.environ["SAM_OPENAI_EMBEDDING_CONCURRENCY"], "12")
            self.assertEqual(os.environ["SAM_AZURE_EMBEDDING_BATCH_SIZE"], "16")
            self.assertEqual(os.environ["SAM_OPENAI_EMBEDDING_BATCH_SIZE"], "16")
            self.assertEqual(os.environ["SAM_AZURE_EMBEDDING_INPUT_MODE"], "single")
            return DummyProvider()

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(graph_strategy_script, "load_default_env_file", side_effect=fake_load_env):
                with patch.object(
                    graph_strategy_script,
                    "create_embedding_provider",
                    side_effect=fake_create_provider,
                ):
                    provider = graph_strategy_script._create_embedding_provider(
                        "azure_openai_sdk",
                        embedding_concurrency=12,
                        embedding_batch_size=16,
                        embedding_input_mode="single",
                    )

        self.assertEqual(order, ["load", "create:azure_openai_sdk"])
        self.assertIsInstance(provider, DummyProvider)

    def test_progress_iter_uses_tqdm_factory_when_enabled(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_tqdm(iterable, total=None, desc=None):
            calls.append({"total": total, "desc": desc})
            return iterable

        values = list(progress_iter([1, 2], total=2, desc="测试进度", progress_factory=fake_tqdm))

        self.assertEqual(values, [1, 2])
        self.assertEqual(calls, [{"total": 2, "desc": "测试进度"}])

    def test_progress_iter_can_be_disabled(self) -> None:
        def fake_tqdm(iterable, total=None, desc=None):
            raise AssertionError("禁用进度时不应调用 tqdm")

        values = list(progress_iter([1, 2], total=2, desc="测试进度", enabled=False, progress_factory=fake_tqdm))

        self.assertEqual(values, [1, 2])

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

    def test_graph_build_cost_audit_compares_local_edges_to_full_graph(self) -> None:
        edge_log = [
            {"action": "created", "relation_type": "shared_entity", "source_id": "a", "target_id": "b"},
            {"action": "created", "relation_type": "shared_entity", "source_id": "b", "target_id": "a"},
            {"action": "created", "relation_type": "keyword_overlap", "source_id": "a", "target_id": "c"},
            {"action": "updated", "relation_type": "shared_entity", "source_id": "b", "target_id": "c"},
        ]

        audit = audit_graph_build_cost(
            edge_log=edge_log,
            document_count=10,
            query_count=2,
        )

        self.assertEqual(audit["summary"]["created_edge_log_count"], 3)
        self.assertEqual(audit["summary"]["touched_edge_log_count"], 4)
        self.assertEqual(audit["summary"]["unique_created_directed_edge_count"], 3)
        self.assertEqual(audit["summary"]["unique_created_undirected_pair_count"], 2)
        self.assertEqual(audit["summary"]["theoretical_full_edge_count"], 45)
        self.assertEqual(audit["summary"]["average_created_undirected_pairs_per_query"], 1.0)
        self.assertLess(audit["summary"]["unique_created_pair_to_full_ratio"], 0.05)
        self.assertEqual(audit["relation_type_counts"]["shared_entity"], 3)

        output_dir = Path(self.temp_dir.name) / "graph_cost_audit"
        json_path, markdown_path = write_graph_build_cost_audit(audit, output_dir)
        self.assertTrue(json_path.exists())
        self.assertIn("全量建图理论边数", markdown_path.read_text(encoding="utf-8"))

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

    def test_object_graph_builder_creates_local_evidence_graph(self) -> None:
        self.store.reset()
        graph = LocalEvidenceGraph(
            object_id="paper_raptor",
            object_type="paper",
            title="RAPTOR",
            source="unit-test",
            units=[
                LocalEvidenceUnit(
                    id="method",
                    node_type="method",
                    title="Recursive tree retrieval",
                    text="RAPTOR builds a recursive summary tree for long-context retrieval.",
                    summary="RAPTOR 使用递归摘要树进行长上下文检索。",
                    keywords=["raptor", "summary", "tree", "retrieval"],
                    entities=[
                        BridgeEntity(
                            name="RAPTOR",
                            canonical_name="raptor",
                            entity_type="method",
                        ),
                        BridgeEntity(
                            name="Long-context QA",
                            canonical_name="long_context_qa",
                            entity_type="task",
                        ),
                    ],
                ),
                LocalEvidenceUnit(
                    id="result",
                    node_type="result",
                    title="Evidence recall result",
                    text="The method improves retrieval on long documents.",
                    summary="该方法提升长文检索效果。",
                    keywords=["evidence", "recall", "long", "documents"],
                    entities=[
                        BridgeEntity(
                            name="Long-context QA",
                            canonical_name="long_context_qa",
                            entity_type="task",
                        )
                    ],
                ),
            ],
        )

        delta = ObjectGraphBuilder(self.store, self.embedding).ingest(graph)

        self.assertEqual(delta.object_id, "paper_raptor")
        self.assertEqual(delta.added_node_count, 3)
        self.assertGreaterEqual(delta.added_edge_count, 4)
        root = self.store.get_node("paper_raptor::root")
        method = self.store.get_node("paper_raptor::method")
        self.assertIsNotNone(root)
        self.assertIsNotNone(method)
        self.assertEqual(method.metadata["object_id"], "paper_raptor")
        self.assertEqual(method.metadata["object_type"], "paper")
        self.assertEqual(method.metadata["node_type"], "method")
        self.assertEqual(method.metadata["bridge_entities"][0]["canonical_name"], "raptor")
        local_edges = [
            edge
            for edge in self.store.get_edges()
            if edge.metadata.get("edge_scope") == "local"
        ]
        self.assertTrue(local_edges)

    def test_object_graph_builder_connects_objects_through_entity_bridges(self) -> None:
        self.store.reset()
        builder = ObjectGraphBuilder(self.store, self.embedding)
        builder.ingest(
            LocalEvidenceGraph(
                object_id="paper_a",
                object_type="paper",
                title="Paper A",
                source="unit-test",
                units=[
                    LocalEvidenceUnit(
                        id="method",
                        node_type="method",
                        title="GraphRAG method",
                        text="GraphRAG retrieves evidence with an entity graph.",
                        summary="GraphRAG 使用实体图检索证据。",
                        keywords=["graphrag", "entity", "graph"],
                        entities=[
                            BridgeEntity(
                                name="GraphRAG",
                                canonical_name="graphrag",
                                entity_type="method",
                            )
                        ],
                    )
                ],
            )
        )
        delta = builder.ingest(
            LocalEvidenceGraph(
                object_id="paper_b",
                object_type="paper",
                title="Paper B",
                source="unit-test",
                units=[
                    LocalEvidenceUnit(
                        id="comparison",
                        node_type="claim",
                        title="GraphRAG comparison",
                        text="This paper compares GraphRAG with hierarchical memory methods.",
                        summary="该论文比较 GraphRAG 与层次记忆方法。",
                        keywords=["graphrag", "comparison", "memory"],
                        entities=[
                            BridgeEntity(
                                name="GraphRAG",
                                canonical_name="graphrag",
                                entity_type="method",
                            )
                        ],
                    )
                ],
            )
        )

        self.assertEqual(delta.added_bridge_edge_count, 2)
        bridge_edges = [
            edge
            for edge in self.store.get_edges()
            if edge.relation_type == "cross_object_entity_bridge"
        ]
        self.assertEqual(len(bridge_edges), 2)
        self.assertEqual(bridge_edges[0].metadata["bridge_entity"]["canonical_name"], "graphrag")
        self.assertNotEqual(
            self.store.get_node(bridge_edges[0].source_id).metadata["object_id"],
            self.store.get_node(bridge_edges[0].target_id).metadata["object_id"],
        )

    def test_object_graph_incremental_delta_only_reports_changed_object(self) -> None:
        self.store.reset()
        builder = ObjectGraphBuilder(self.store, self.embedding)
        first = builder.ingest(
            LocalEvidenceGraph(
                object_id="repo_auth",
                object_type="code_repository",
                title="Auth Repository",
                source="unit-test",
                units=[
                    LocalEvidenceUnit(
                        id="login",
                        node_type="function",
                        title="login_user",
                        text="login_user validates credentials and creates a session.",
                        summary="登录函数校验凭证并创建会话。",
                        keywords=["login", "session", "credentials"],
                        entities=[
                            BridgeEntity(
                                name="login_user",
                                canonical_name="login_user",
                                entity_type="symbol",
                            )
                        ],
                    )
                ],
            )
        )
        second = builder.ingest(
            LocalEvidenceGraph(
                object_id="repo_auth",
                object_type="code_repository",
                title="Auth Repository",
                source="unit-test",
                units=[
                    LocalEvidenceUnit(
                        id="login",
                        node_type="function",
                        title="login_user",
                        text="login_user validates credentials, creates a session, and emits audit events.",
                        summary="登录函数新增审计事件。",
                        keywords=["login", "session", "credentials", "audit"],
                        entities=[
                            BridgeEntity(
                                name="login_user",
                                canonical_name="login_user",
                                entity_type="symbol",
                            )
                        ],
                    ),
                    LocalEvidenceUnit(
                        id="test_login",
                        node_type="test",
                        title="test_login_user",
                        text="test_login_user covers credential validation.",
                        summary="登录测试覆盖凭证校验。",
                        keywords=["login", "test", "credentials"],
                        entities=[
                            BridgeEntity(
                                name="login_user",
                                canonical_name="login_user",
                                entity_type="symbol",
                            )
                        ],
                    ),
                ],
            )
        )

        self.assertEqual(first.added_node_count, 2)
        self.assertIn("repo_auth::test_login", second.added_node_ids)
        self.assertIn("repo_auth::login", second.updated_node_ids)
        self.assertNotIn("repo_auth::root", second.added_node_ids)
        self.assertEqual(second.object_id, "repo_auth")

    def test_cross_graph_retriever_returns_bridge_path_between_objects(self) -> None:
        self.store.reset()
        builder = ObjectGraphBuilder(self.store, self.embedding)
        builder.ingest(
            LocalEvidenceGraph(
                object_id="paper_graphrag",
                object_type="paper",
                title="GraphRAG Paper",
                source="unit-test",
                units=[
                    LocalEvidenceUnit(
                        id="method",
                        node_type="method",
                        title="GraphRAG local search",
                        text="GraphRAG uses community and entity graph search for question answering.",
                        summary="GraphRAG 使用社区与实体图搜索。",
                        keywords=["graphrag", "entity", "graph", "search"],
                        entities=[
                            BridgeEntity(
                                name="GraphRAG",
                                canonical_name="graphrag",
                                entity_type="method",
                            )
                        ],
                    )
                ],
            )
        )
        builder.ingest(
            LocalEvidenceGraph(
                object_id="paper_sam",
                object_type="paper",
                title="SAM Paper",
                source="unit-test",
                units=[
                    LocalEvidenceUnit(
                        id="related",
                        node_type="claim",
                        title="GraphRAG comparison in SAM",
                        text="SAM compares its cross-paper memory with GraphRAG on entity bridge retrieval.",
                        summary="SAM 将跨论文记忆与 GraphRAG 对比。",
                        keywords=["sam", "graphrag", "entity", "bridge"],
                        entities=[
                            BridgeEntity(
                                name="GraphRAG",
                                canonical_name="graphrag",
                                entity_type="method",
                            )
                        ],
                    )
                ],
            )
        )

        hits = CrossGraphRetriever(self.store, self.embedding).retrieve(
            query="How does GraphRAG use community entity graph search?",
            top_k=4,
            seed_k=1,
            hops=2,
        )

        self.assertTrue(hits)
        self.assertTrue(
            any(
                hit.node.metadata.get("object_id") == "paper_sam"
                and "cross_object_entity_bridge" in hit.metadata.get("path_relation_types", [])
                for hit in hits
            )
        )
        bridged = [
            hit
            for hit in hits
            if "cross_object_entity_bridge" in hit.metadata.get("path_relation_types", [])
        ][0]
        self.assertGreaterEqual(len(bridged.path), 2)
        self.assertEqual(
            bridged.metadata["bridge_entities"][0]["canonical_name"],
            "graphrag",
        )

    def test_context_path_proximity_generalizes_linear_position(self) -> None:
        same_section_left = ["paper_1", "method", "paragraph_1"]
        same_section_right = ["paper_1", "method", "paragraph_2"]
        same_object_other_section = ["paper_1", "experiment", "paragraph_8"]
        different_object = ["paper_2", "method", "paragraph_1"]

        self.assertGreater(
            context_path_proximity(same_section_left, same_section_right),
            context_path_proximity(same_section_left, same_object_other_section),
        )
        self.assertGreater(
            context_path_proximity(same_section_left, same_object_other_section),
            context_path_proximity(same_section_left, different_object),
        )
        self.assertGreater(
            position_proximity(3, 4),
            position_proximity(3, 12),
        )

    def test_graph_strategy_experiment_compares_non_llm_builders(self) -> None:
        nodes = self._strategy_nodes()
        experiment = GraphStrategyExperiment(
            nodes=nodes,
            queries=[],
            alpha=0.55,
            top_k_edges=2,
            threshold=0.08,
        )

        results = experiment.compare_build_strategies(
            ["no_graph", "semantic_only", "position_only", "cam_style", "context_path_only", "sam_context"]
        )

        self.assertEqual(results["no_graph"].edge_count, 0)
        self.assertEqual(results["semantic_only"].strategy, "semantic_only")
        self.assertGreater(results["semantic_only"].candidate_pair_count, 0)
        self.assertGreater(results["sam_context"].edge_count, 0)
        self.assertGreaterEqual(
            results["sam_context"].average_edge_score,
            results["context_path_only"].average_edge_score,
        )
        sample_edge = results["sam_context"].edges[0]
        self.assertIn("semantic_similarity", sample_edge.metadata["score_breakdown"])
        self.assertIn("context_path_proximity", sample_edge.metadata["score_breakdown"])
        self.assertFalse(sample_edge.metadata["uses_llm"])

    def test_graph_strategy_experiment_reports_cost_effectiveness(self) -> None:
        nodes = self._strategy_nodes()
        queries = [
            EvaluationQuery(
                id="q1",
                dataset="unit",
                question="Which evidence explains graph retrieval for long context?",
                answer="Graph retrieval improves long context evidence.",
                supporting_doc_ids=["doc_b"],
                candidate_doc_ids=["doc_a", "doc_b", "doc_c"],
            )
        ]
        experiment = GraphStrategyExperiment(
            nodes=nodes,
            queries=queries,
            alpha=0.55,
            top_k_edges=2,
            threshold=0.08,
        )

        report = experiment.run(
            strategies=["no_graph", "semantic_only", "cam_style", "sam_context"],
            top_k=2,
            seed_k=1,
            hops=1,
        )

        self.assertEqual(report["config"]["alpha"], 0.55)
        self.assertIn("sam_context", report["strategies"])
        self.assertIn("evidence_recall", report["strategies"]["sam_context"]["metrics"])
        self.assertIn("precision_at_k", report["strategies"]["sam_context"]["metrics"])
        self.assertIn("mrr", report["strategies"]["sam_context"]["metrics"])
        self.assertIn("ndcg_at_k", report["strategies"]["sam_context"]["metrics"])
        self.assertIn("graph_path_support_hits", report["strategies"]["sam_context"]["metrics"])
        self.assertIn("graph_rescue_rate", report["strategies"]["sam_context"]["metrics"])
        self.assertIn("build_time_seconds", report["strategies"]["sam_context"]["cost"])
        self.assertIn("cost_effectiveness", report["strategies"]["sam_context"])
        self.assertIn("retrieval_time_seconds", report["strategies"]["sam_context"]["cost"])
        self.assertIn("total_time_seconds", report["strategies"]["sam_context"]["cost"])
        self.assertIn("average_retrieval_time_ms", report["strategies"]["sam_context"]["cost"])
        self.assertIn("edge_keep_rate", report["strategies"]["sam_context"]["cost"])
        self.assertIn("build_pairs_per_second", report["strategies"]["sam_context"]["cost"])
        cost_effectiveness = report["strategies"]["sam_context"]["cost_effectiveness"]
        self.assertIn("cost_index", cost_effectiveness)
        self.assertIn("cost_effectiveness_score", cost_effectiveness)
        self.assertIn("normalized_edge_cost", cost_effectiveness)
        self.assertIn("normalized_candidate_pair_cost", cost_effectiveness)
        self.assertIn("normalized_build_time_cost", cost_effectiveness)
        self.assertIn("recall_gain_vs_no_graph", cost_effectiveness)
        self.assertIn("gain_per_100_extra_edges", cost_effectiveness)
        self.assertIn("gain_per_extra_second", cost_effectiveness)
        self.assertIn("recommended_strategy", report["summary"])
        self.assertIn("best_recall_strategy", report["summary"])
        self.assertIn("best_cost_effectiveness_strategy", report["summary"])
        self.assertIn("best_balanced_strategy", report["summary"])
        self.assertIn("ranking", report["summary"])
        self.assertGreaterEqual(cost_effectiveness["cost_effectiveness_score"], 0.0)
        self.assertIn("recommended_strategy", report["summary"])
        self.assertFalse(report["strategies"]["sam_context"]["cost"]["uses_llm"])

    def test_graph_strategy_experiment_uses_supplied_query_embeddings(self) -> None:
        nodes = self._strategy_nodes()
        query = EvaluationQuery(
            id="q1",
            dataset="unit",
            question="Which document is the unrelated table extraction baseline?",
            answer="Unrelated preprocessing baseline.",
            supporting_doc_ids=["doc_c"],
            candidate_doc_ids=["doc_a", "doc_b", "doc_c"],
        )

        report = GraphStrategyExperiment(
            nodes=nodes,
            queries=[query],
            query_embeddings={"q1": nodes[2].embedding},
        ).run(
            strategies=["no_graph"],
            top_k=1,
            seed_k=1,
            hops=1,
        )

        case = report["strategies"]["no_graph"]["cases"][0]
        self.assertEqual(case["hit_node_ids"], ["node_c"])
        self.assertEqual(report["strategies"]["no_graph"]["metrics"]["support_hits"], 1)

    def test_graph_strategy_markdown_report_includes_full_cost_effectiveness(self) -> None:
        nodes = self._strategy_nodes()
        queries = [
            EvaluationQuery(
                id="q1",
                dataset="unit",
                question="Which evidence explains graph retrieval for long context?",
                answer="Graph retrieval improves long context evidence.",
                supporting_doc_ids=["doc_b"],
                candidate_doc_ids=["doc_a", "doc_b", "doc_c"],
            )
        ]
        report = GraphStrategyExperiment(
            nodes=nodes,
            queries=queries,
            alpha=0.55,
            top_k_edges=2,
            threshold=0.08,
        ).run(
            strategies=["no_graph", "semantic_only", "cam_style", "sam_context"],
            top_k=2,
            seed_k=1,
            hops=1,
        )

        _json_path, markdown_path = write_graph_strategy_report(report, Path(self.temp_dir.name) / "strategy_report")
        markdown = markdown_path.read_text(encoding="utf-8")

        self.assertIn("综合性价比分", markdown)
        self.assertIn("成本指数", markdown)
        self.assertIn("Recall/s", markdown)
        self.assertIn("相对 no_graph 召回增益", markdown)
        self.assertIn("平均路径长度", markdown)
        self.assertIn("平均扩展节点数", markdown)
        self.assertIn("Precision@k", markdown)
        self.assertIn("MRR", markdown)
        self.assertIn("nDCG@k", markdown)
        self.assertIn("图路径命中", markdown)
        self.assertIn("检索耗时", markdown)
        self.assertIn("总耗时", markdown)
        self.assertIn("保边率", markdown)

    def test_graph_strategy_summary_does_not_recommend_graph_without_gain(self) -> None:
        nodes = self._strategy_nodes()
        queries = [
            EvaluationQuery(
                id="q1",
                dataset="unit",
                question="Which evidence explains graph retrieval for long context?",
                answer="Graph retrieval improves long context evidence.",
                supporting_doc_ids=["doc_b"],
                candidate_doc_ids=["doc_a", "doc_b", "doc_c"],
            )
        ]
        report = GraphStrategyExperiment(
            nodes=nodes,
            queries=queries,
            query_embeddings={"q1": nodes[1].embedding},
            alpha=0.55,
            top_k_edges=2,
            threshold=0.08,
        ).run(
            strategies=["no_graph", "semantic_only", "sam_context"],
            top_k=1,
            seed_k=1,
            hops=1,
        )

        self.assertEqual(report["strategies"]["no_graph"]["metrics"]["evidence_recall"], 1.0)
        self.assertEqual(report["summary"]["recommended_strategy"], "no_improving_graph_strategy")

    def test_graph_strategy_script_intrinsic_context_path_excludes_query_id(self) -> None:
        node = MemoryNode(
            id="mem_hotpotqa_case_doc_1",
            text="Alpha evidence",
            summary="Alpha evidence",
            keywords=["alpha"],
            tags=[],
            source="unit",
            created_at=utc_now_iso(),
            last_accessed_at=None,
            usage_count=0,
            confidence=0.9,
            embedding=[1.0, 0.0],
            metadata={
                "dataset": "hotpotqa_real",
                "query_id": "hotpotqa_case",
                "hotpotqa_id": "case",
                "original_doc_id": "hotpotqa_case_doc_1",
                "title": "Alpha Evidence",
                "paragraph_index": 3,
            },
        )

        audit = graph_strategy_script._attach_context_metadata([node], policy="intrinsic")

        path = node.metadata["context_path"]
        self.assertEqual(path, ["title:alpha_evidence"])
        self.assertTrue(audit["is_leak_safe"])
        self.assertEqual(audit["context_paths_containing_query_ids"], 0)
        self.assertNotIn("hotpotqa_case", path)
        self.assertNotIn("case", path)

    def test_graph_strategy_script_intrinsic_context_path_excludes_original_doc_id_source(self) -> None:
        node = MemoryNode(
            id="mem_scifact_doc_1001",
            text="Scientific abstract evidence",
            summary="Scientific abstract evidence",
            keywords=["scientific"],
            tags=[],
            source="unit",
            created_at=utc_now_iso(),
            last_accessed_at=None,
            usage_count=0,
            confidence=0.9,
            embedding=[1.0, 0.0],
            metadata={
                "dataset": "scifact",
                "original_doc_id": "scifact_doc_1001",
                "source_id": "scifact_doc_1001",
                "section": "abstract",
                "title": "Cancer Immunotherapy Response",
            },
        )

        audit = graph_strategy_script._attach_context_metadata([node], policy="intrinsic")

        path = node.metadata["context_path"]
        self.assertEqual(path, ["section:abstract", "title:cancer_immunotherapy_response"])
        self.assertTrue(audit["is_leak_safe"])
        self.assertEqual(audit["context_paths_containing_query_ids"], 0)
        self.assertNotIn("scifact_doc_1001", "/".join(path))

    def test_no_graph_strategy_uses_embedding_top_k_not_seed_k_only(self) -> None:
        nodes = self._strategy_nodes()
        queries = [
            EvaluationQuery(
                id="q1",
                dataset="unit",
                question="Which evidence explains graph retrieval for long context?",
                answer="Graph retrieval improves long context evidence.",
                supporting_doc_ids=["doc_b"],
                candidate_doc_ids=["doc_a", "doc_b", "doc_c"],
            )
        ]

        report = GraphStrategyExperiment(nodes=nodes, queries=queries).run(
            strategies=["no_graph"],
            top_k=2,
            seed_k=1,
            hops=1,
        )

        hits = report["strategies"]["no_graph"]["cases"][0]["hits"]
        self.assertEqual(len(hits), 2)
        self.assertTrue(all(len(hit["path"]) == 1 for hit in hits))

    def test_alpha_sweep_compares_sam_context_weights(self) -> None:
        sweep = run_alpha_sweep(
            nodes=self._strategy_nodes(),
            queries=[
                EvaluationQuery(
                    id="q1",
                    dataset="unit",
                    question="Which evidence explains graph retrieval for long context?",
                    answer="Graph retrieval improves long context evidence.",
                    supporting_doc_ids=["doc_b"],
                    candidate_doc_ids=["doc_a", "doc_b", "doc_c"],
                )
            ],
            alphas=[0.0, 0.5, 1.0],
            top_k_edges=2,
            threshold=0.08,
            top_k=2,
            seed_k=1,
            hops=1,
        )

        self.assertEqual([row["alpha"] for row in sweep["rows"]], [0.0, 0.5, 1.0])
        self.assertIn("best_alpha", sweep)
        self.assertIn("selection_rule", sweep)

    def _strategy_nodes(self) -> list[MemoryNode]:
        now = utc_now_iso()
        return [
            MemoryNode(
                id="node_a",
                text="Graph retrieval introduces semantic memory for long context question answering.",
                summary="Graph retrieval semantic memory.",
                keywords=["graph", "retrieval", "semantic", "memory"],
                tags=["strategy_test"],
                source="unit-test",
                created_at=now,
                last_accessed_at=None,
                usage_count=0,
                confidence=0.9,
                embedding=self.embedding.embed("Graph retrieval semantic memory"),
                metadata={
                    "original_doc_id": "doc_a",
                    "title": "Graph retrieval intro",
                    "context_path": ["paper_1", "introduction", "paragraph_1"],
                    "position": 1,
                },
            ),
            MemoryNode(
                id="node_b",
                text="The graph retrieval method improves evidence organization for long context tasks.",
                summary="Graph retrieval improves evidence organization.",
                keywords=["graph", "retrieval", "evidence", "long"],
                tags=["strategy_test"],
                source="unit-test",
                created_at=now,
                last_accessed_at=None,
                usage_count=0,
                confidence=0.9,
                embedding=self.embedding.embed("Graph retrieval improves evidence organization"),
                metadata={
                    "original_doc_id": "doc_b",
                    "title": "Graph retrieval result",
                    "context_path": ["paper_1", "method", "paragraph_2"],
                    "position": 2,
                },
            ),
            MemoryNode(
                id="node_c",
                text="A separate baseline discusses table extraction and unrelated preprocessing.",
                summary="Unrelated preprocessing baseline.",
                keywords=["table", "extraction", "baseline"],
                tags=["strategy_test"],
                source="unit-test",
                created_at=now,
                last_accessed_at=None,
                usage_count=0,
                confidence=0.9,
                embedding=self.embedding.embed("table extraction unrelated preprocessing"),
                metadata={
                    "original_doc_id": "doc_c",
                    "title": "Unrelated baseline",
                    "context_path": ["paper_2", "experiment", "paragraph_1"],
                    "position": 20,
                },
            ),
        ]

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

    def test_graph_builder_preserves_relation_type_when_relation_budget_is_exhausted(self) -> None:
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
                    confidence=0.9,
                    reason="预算未耗尽时会拒绝候选边",
                )

        self.store.reset()
        now = utc_now_iso()
        left = MemoryNode(
            id="budget_left",
            text="Alpha bridge evidence focuses on a film award.",
            summary="Alpha bridge evidence focuses on a film award.",
            keywords=["alpha", "bridge", "film"],
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
            id="budget_right",
            text="Alpha bridge evidence explains the same film award.",
            summary="Alpha bridge evidence explains the same film award.",
            keywords=["alpha", "bridge", "award"],
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
        judge = BudgetedRelationJudge(RejectingRelationJudge(), max_calls=0, on_exhausted="skip")
        graph = GraphBuilder(self.store, relation_judge=judge, relation_judge_policy="all")

        edges = graph.build_edges_on_demand([left], [left, right])
        score = graph._score_candidate_edge(left, right)

        self.assertTrue(edges)
        self.assertEqual(score.relation_type, "keyword_overlap")
        self.assertEqual(score.score_breakdown["relation_judge"]["relation_type"], "keyword_overlap")
        self.assertIn("预算已耗尽", score.score_breakdown["relation_judge"]["reason"])

    def test_graph_builder_uses_relation_judge_only_for_risky_edges_by_default(self) -> None:
        class RecordingRelationJudge:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def judge(
                self,
                seed: MemoryNode,
                other: MemoryNode,
                score_breakdown: dict[str, object],
            ) -> RelationJudgment:
                self.calls.append(str(score_breakdown.get("relation_type_hint")))
                return RelationJudgment(
                    should_link=False,
                    relation_type="unrelated",
                    confidence=0.9,
                    reason="测试中拒绝高风险边",
                )

        self.store.reset()
        now = utc_now_iso()
        seed = MemoryNode(
            id="risk_seed",
            text="Bridge Person appears in Alpha Film.",
            summary="Bridge Person appears in Alpha Film.",
            keywords=["bridge", "alpha"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[1.0, 0.0, 0.0],
            metadata={"entities": ["Bridge Person"]},
        )
        strong = MemoryNode(
            id="risk_strong",
            text="Bridge Person later received a notable award.",
            summary="Bridge Person later received a notable award.",
            keywords=["award", "person"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[0.9, 0.1, 0.0],
            metadata={"entities": ["Bridge Person"]},
        )
        risky = MemoryNode(
            id="risk_keyword",
            text="Alpha bridge wording appears in an unrelated sports note.",
            summary="Alpha bridge wording appears in an unrelated sports note.",
            keywords=["bridge", "alpha"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[1.0, 0.0, 0.0],
            metadata={"entities": []},
        )
        self.store.upsert_nodes([seed, strong, risky])
        judge = RecordingRelationJudge()
        graph = GraphBuilder(self.store, relation_judge=judge)

        edges = graph.build_edges_on_demand([seed], [seed, strong, risky])
        edge_keys = {edge.key for edge in edges}

        self.assertEqual(judge.calls, ["keyword_overlap"])
        self.assertIn(("risk_seed", "risk_strong", "shared_entity"), edge_keys)
        self.assertNotIn(("risk_seed", "risk_keyword", "keyword_overlap"), edge_keys)

    def test_graph_builder_relation_judge_policy_all_judges_strong_edges(self) -> None:
        class RecordingRelationJudge:
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
                    should_link=True,
                    relation_type="shared_entity",
                    confidence=0.9,
                    reason="测试判别通过",
                )

        self.store.reset()
        now = utc_now_iso()
        left = MemoryNode(
            id="policy_left",
            text="Bridge Person appears in Alpha Film.",
            summary="Bridge Person appears in Alpha Film.",
            keywords=["bridge"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[1.0, 0.0, 0.0],
            metadata={"entities": ["Bridge Person"]},
        )
        right = MemoryNode(
            id="policy_right",
            text="Bridge Person received an award.",
            summary="Bridge Person received an award.",
            keywords=["award"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[0.9, 0.1, 0.0],
            metadata={"entities": ["Bridge Person"]},
        )
        self.store.upsert_nodes([left, right])
        judge = RecordingRelationJudge()
        graph = GraphBuilder(self.store, relation_judge=judge, relation_judge_policy="all")

        graph.build_edges_on_demand([left], [left, right])

        self.assertEqual(judge.calls, 1)

    def test_graph_builder_relation_judge_policy_off_skips_all_judgments(self) -> None:
        class FailingRelationJudge:
            def judge(
                self,
                seed: MemoryNode,
                other: MemoryNode,
                score_breakdown: dict[str, object],
            ) -> RelationJudgment:
                raise AssertionError("policy=off 不应调用关系判别器")

        self.store.reset()
        now = utc_now_iso()
        left = MemoryNode(
            id="policy_off_left",
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
            id="policy_off_right",
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
        graph = GraphBuilder(
            self.store,
            relation_judge=FailingRelationJudge(),
            relation_judge_policy="off",
        )

        edges = graph.build_edges_on_demand([left], [left, right])

        self.assertTrue(edges)

    def test_cached_relation_judge_reuses_previous_judgment(self) -> None:
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
                    should_link=True,
                    relation_type="shared_entity",
                    confidence=0.8,
                    reason="测试关系判别结果",
                )

        base_judge = CountingRelationJudge()
        cache_path = Path(self.temp_dir.name) / "relation_cache.json"
        judge = CachedRelationJudge(base_judge, cache_path=cache_path)
        seed = self.nodes[0]
        other = self.nodes[1]
        score_breakdown = {"relation_type_hint": "shared_entity", "similarity": 0.5}

        first = judge.judge(seed, other, score_breakdown)
        second = judge.judge(seed, other, score_breakdown)
        reloaded = CachedRelationJudge(base_judge, cache_path=cache_path)
        third = reloaded.judge(seed, other, score_breakdown)

        self.assertEqual(base_judge.calls, 1)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.to_dict(), third.to_dict())
        self.assertEqual(judge.cache_hits, 1)
        self.assertEqual(judge.cache_misses, 1)
        self.assertEqual(reloaded.cache_hits, 1)
        self.assertEqual(reloaded.cache_misses, 0)
        self.assertTrue(cache_path.exists())

    def test_budgeted_relation_judge_skips_after_max_calls(self) -> None:
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
                    reason="测试判别器拒绝候选边",
                )

        base_judge = CountingRelationJudge()
        judge = BudgetedRelationJudge(base_judge, max_calls=1, on_exhausted="skip")
        seed = self.nodes[0]
        other = self.nodes[1]

        first = judge.judge(seed, other, {"relation_type_hint": "keyword_overlap"})
        second = judge.judge(seed, other, {"relation_type_hint": "keyword_overlap"})

        self.assertEqual(base_judge.calls, 1)
        self.assertEqual(judge.calls_made, 1)
        self.assertEqual(judge.skipped_count, 1)
        self.assertEqual(first.should_link, False)
        self.assertEqual(second.should_link, True)
        self.assertEqual(second.relation_type, "keyword_overlap")
        self.assertEqual(second.confidence, 1.0)

    def test_cached_relation_judge_normalizes_stale_budget_exhausted_skip(self) -> None:
        cache_path = Path(self.temp_dir.name) / "relation_cache.json"
        seed = self.nodes[0]
        other = self.nodes[1]
        score_breakdown = {"relation_type_hint": "embedding_similarity", "similarity": 0.2}
        stale_key = _relation_cache_key(seed, other, score_breakdown)
        cache_path.write_text(
            json.dumps(
                {
                    stale_key: {
                        "should_link": True,
                        "relation_type": "budget_exhausted",
                        "confidence": 0.0,
                        "reason": "旧缓存：预算耗尽后保留候选边",
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        class FailingRelationJudge:
            def judge(
                self,
                seed: MemoryNode,
                other: MemoryNode,
                score_breakdown: dict[str, object],
            ) -> RelationJudgment:
                raise AssertionError("命中旧缓存时不应重新调用底层判别器")

        judge = CachedRelationJudge(FailingRelationJudge(), cache_path=cache_path)
        judgment = judge.judge(seed, other, score_breakdown)

        self.assertTrue(judgment.should_link)
        self.assertEqual(judgment.relation_type, "embedding_similarity")
        self.assertEqual(judgment.confidence, 1.0)

    def test_relation_judge_stats_reports_cache_and_budget(self) -> None:
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
                    should_link=True,
                    relation_type="same_topic",
                    confidence=0.9,
                    reason="测试判别通过",
                )

        cache_path = Path(self.temp_dir.name) / "relation_cache.json"
        judge = CachedRelationJudge(
            BudgetedRelationJudge(CountingRelationJudge(), max_calls=1),
            cache_path=cache_path,
        )
        seed = self.nodes[0]
        other = self.nodes[1]
        payload = {"relation_type_hint": "keyword_overlap", "similarity": 0.5}

        judge.judge(seed, other, payload)
        judge.judge(seed, other, payload)

        stats = relation_judge_stats(judge)

        self.assertEqual(stats["type"], "CachedRelationJudge")
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["base"]["type"], "BudgetedRelationJudge")
        self.assertEqual(stats["base"]["calls_made"], 1)
        self.assertEqual(stats["base"]["skipped_count"], 0)

    def test_create_relation_judge_wraps_cached_gpt54_with_budget_inside_cache(self) -> None:
        cache_path = Path(self.temp_dir.name) / "relation_cache.json"

        class FakeChatClient(ChatClient):
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                self.calls += 1
                return '{"should_link": true, "relation_type": "same_topic", "confidence": 0.9, "reason": "测试"}'

        fake_client = FakeChatClient()

        with patch.dict(
            "os.environ",
            {
                "SAM_RELATION_JUDGE_CACHE_PATH": str(cache_path),
                "SAM_RELATION_JUDGE_MAX_CALLS": "1",
                "SAM_RELATION_JUDGE_BUDGET_EXHAUSTED": "skip",
            },
            clear=True,
        ), patch("sam.relation_judge.create_chat_client", return_value=fake_client):
            judge = create_relation_judge("cached_gpt54")

        self.assertIsInstance(judge, CachedRelationJudge)
        self.assertIsInstance(judge.base_judge, BudgetedRelationJudge)

        seed = self.nodes[0]
        other = self.nodes[1]
        payload = {"relation_type_hint": "keyword_overlap", "similarity": 0.5}
        judge.judge(seed, other, payload)
        judge.judge(seed, other, payload)

        self.assertEqual(fake_client.calls, 1)
        self.assertEqual(judge.base_judge.calls_made, 1)

    def test_create_relation_judge_supports_cached_gpt54(self) -> None:
        cache_path = Path(self.temp_dir.name) / "relation_cache.json"
        captured: dict[str, object] = {}

        class FakeChatClient(ChatClient):
            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                return '{"should_link": true, "relation_type": "same_topic", "confidence": 0.9, "reason": "测试"}'

        def fake_create_chat_client(name: str | None = None) -> ChatClient:
            captured["provider"] = name
            return FakeChatClient()

        with patch.dict(
            "os.environ",
            {
                "SAM_RELATION_JUDGE_CHAT_PROVIDER": "azure_openai_sdk",
                "SAM_RELATION_JUDGE_CACHE_PATH": str(cache_path),
                "SAM_RELATION_JUDGE_MIN_CONFIDENCE": "0.7",
                "SAM_RELATION_JUDGE_FAIL_OPEN": "0",
            },
            clear=True,
        ), patch("sam.relation_judge.create_chat_client", side_effect=fake_create_chat_client):
            judge = create_relation_judge("cached_gpt54")

        self.assertEqual(captured["provider"], "azure_openai_sdk")
        self.assertIsInstance(judge, CachedRelationJudge)
        self.assertEqual(judge.cache_path, cache_path)
        self.assertIsInstance(judge.base_judge, ChatRelationJudge)
        self.assertEqual(judge.base_judge.min_confidence, 0.7)
        self.assertEqual(judge.base_judge.fail_open, False)

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
        candidate_paths = [
            path
            for hit in associative_hits
            for path in hit.metadata.get("candidate_paths", [])
            if path.get("relation_type")
        ]
        self.assertTrue(any("edge_quality" in path for path in candidate_paths))

    def test_sam_initial_activation_uses_lexical_signal_without_changing_embedding_baseline(self) -> None:
        self.store.reset()
        now = utc_now_iso()
        support = MemoryNode(
            id="lexical_support",
            text="rare bridge target evidence appears in this paragraph",
            summary="rare bridge target evidence",
            keywords=["rare", "bridge", "target", "evidence"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[0.70, 0.714],
            metadata={"title": "Rare Bridge Target", "entities": ["Rare Bridge"]},
        )
        distractor = MemoryNode(
            id="semantic_distractor",
            text="ordinary background text without the exact bridge clue",
            summary="ordinary background text",
            keywords=["ordinary", "background"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[0.72, 0.694],
            metadata={"title": "Ordinary Background", "entities": ["Other"]},
        )
        self.store.upsert_nodes([support, distractor])

        class QueryEmbeddingProvider(LocalHashEmbeddingProvider):
            def embed(self, text: str) -> list[float]:
                return [1.0, 0.0]

        retriever = Retriever(self.store, QueryEmbeddingProvider(), GraphBuilder(self.store))
        embedding_hit = retriever.retrieve(
            "rare bridge target",
            "embedding_topk",
            top_k=1,
            candidate_doc_ids=["lexical_support", "semantic_distractor"],
        )[0]
        sam_hit = retriever.retrieve(
            "rare bridge target",
            "sam_with_lexical_activation",
            top_k=1,
            seed_k=1,
            hops=0,
            candidate_doc_ids=["lexical_support", "semantic_distractor"],
        )[0]

        self.assertEqual(embedding_hit.node.id, "semantic_distractor")
        self.assertEqual(sam_hit.node.id, "lexical_support")
        self.assertIn("initial_lexical_activation_score", sam_hit.metadata["score_breakdown"])
        self.assertNotIn("lexical_activation", sam_hit.metadata["score_breakdown"])

    def test_feedback_policy_is_defined_for_sam_variants(self) -> None:
        self.assertTrue(_feedback_enabled("sam_full"))
        self.assertTrue(_feedback_enabled("sam_with_lexical_activation"))
        self.assertFalse(_feedback_enabled("sam_no_feedback"))
        self.assertFalse(_feedback_enabled("sam_static_graph"))

    def test_consolidated_memory_candidates_are_only_added_for_memory_reuse_modes(self) -> None:
        now = utc_now_iso()
        support = MemoryNode(
            id="prior_support",
            text="prior support evidence",
            summary="prior support",
            keywords=["prior", "support"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=self.embedding.embed("prior support"),
            metadata={"title": "Prior Support"},
        )
        consolidated = MemoryNode(
            id="prior_consolidated",
            text="prior consolidated memory",
            summary="prior consolidated",
            keywords=["prior", "consolidated"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=self.embedding.embed("prior consolidated"),
            metadata={
                "node_type": "consolidated_memory",
                "support_node_ids": ["prior_support"],
            },
        )
        self.store.upsert_nodes([support, consolidated])

        sam_candidates = self.evaluator._candidate_ids_for_method(
            self.store,
            "sam_full",
            ["base_candidate"],
        )
        analogy_candidates = self.evaluator._candidate_ids_for_method(
            self.store,
            "sam_with_analogy",
            ["base_candidate"],
        )

        self.assertEqual(sam_candidates, ["base_candidate"])
        self.assertIn("prior_consolidated", analogy_candidates)
        self.assertIn("prior_support", analogy_candidates)

    def test_associative_retrieval_builds_edges_for_expanded_bridge_nodes(self) -> None:
        self.store.reset()
        now = utc_now_iso()
        seed = MemoryNode(
            id="seed_doc",
            text="The query starts from Alpha Film and mentions the actor Bridge Person.",
            summary="Alpha Film mentions Bridge Person.",
            keywords=["alpha", "film", "bridge"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[1.0, 0.0, 0.0],
            metadata={"title": "Alpha Film", "entities": ["Bridge Person"]},
        )
        bridge = MemoryNode(
            id="bridge_doc",
            text="Bridge Person later served as the connector to Target Office.",
            summary="Bridge Person connects to Target Office.",
            keywords=["bridge", "person", "target"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[0.5, 0.5, 0.0],
            metadata={"title": "Bridge Person", "entities": ["Bridge Person", "Target Office"]},
        )
        answer = MemoryNode(
            id="answer_doc",
            text="Target Office is the final answer.",
            summary="Target Office is the answer.",
            keywords=["target", "office", "answer"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[-1.0, 0.0, 0.0],
            metadata={"title": "Target Office", "entities": ["Target Office"]},
        )
        distractor = MemoryNode(
            id="distractor_doc",
            text="Unrelated background text.",
            summary="Unrelated background.",
            keywords=["unrelated"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[0.05, -1.0, 0.0],
            metadata={"title": "Distractor", "entities": ["Other"]},
        )
        distractor_two = MemoryNode(
            id="distractor_two_doc",
            text="Another unrelated background text.",
            summary="Another unrelated background.",
            keywords=["another"],
            tags=[],
            source="unit-test",
            created_at=now,
            last_accessed_at=None,
            usage_count=0,
            confidence=0.8,
            embedding=[0.04, -1.0, 0.0],
            metadata={"title": "Distractor Two", "entities": ["Other Two"]},
        )
        self.store.upsert_nodes([seed, bridge, answer, distractor, distractor_two])

        class FixedEmbeddingProvider(LocalHashEmbeddingProvider):
            def embed(self, text: str) -> list[float]:
                return [1.0, 0.0, 0.0]

        retriever = Retriever(self.store, FixedEmbeddingProvider(), GraphBuilder(self.store))
        hits = retriever.retrieve(
            "Alpha Film Bridge Person Target Office",
            "sam_full",
            top_k=3,
            seed_k=1,
            hops=2,
        )

        answer_hit = next(hit for hit in hits if hit.node.id == "answer_doc")
        self.assertEqual(answer_hit.path, ["seed_doc", "bridge_doc", "answer_doc"])
        self.assertIn("shared_entity", answer_hit.reason)

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

    def test_path_reranker_penalizes_weak_second_hop_paths(self) -> None:
        node = self.nodes[0]
        strong_signals = [
            {
                "path": ["seed", "bridge", node.id],
                "graph_score": 0.9,
                "depth": 2,
                "edge_activation_count": 0,
                "relation_type": "shared_entity",
            }
        ]
        weak_signals = [
            {
                "path": ["seed", "bridge", node.id],
                "graph_score": 0.9,
                "depth": 2,
                "edge_activation_count": 0,
                "relation_type": "embedding_similarity",
            }
        ]

        reranker = PathReranker(profile="semantic_heavy")
        strong_score = reranker.score(
            similarity=0.3,
            graph_score=0.9,
            signals=strong_signals,
            node=node,
            use_multipath=True,
            use_memory_state=False,
        )
        weak_score = reranker.score(
            similarity=0.3,
            graph_score=0.9,
            signals=weak_signals,
            node=node,
            use_multipath=True,
            use_memory_state=False,
        )

        self.assertIn("weak_relation_penalty", weak_score.breakdown)
        self.assertNotIn("weak_relation_penalty", strong_score.breakdown)
        self.assertLess(weak_score.total, strong_score.total)

    def test_path_reranker_penalizes_unsupported_keyword_second_hop_paths(self) -> None:
        node = self.nodes[0]
        supported_keyword_signals = [
            {
                "path": ["seed", "bridge", node.id],
                "graph_score": 0.75,
                "depth": 2,
                "edge_activation_count": 0,
                "relation_type": "keyword_overlap",
                "shared_entities": ["Bridge Entity"],
                "similarity": 0.36,
                "edge_quality": "normal",
            }
        ]
        unsupported_keyword_signals = [
            {
                "path": ["seed", "bridge", node.id],
                "graph_score": 0.75,
                "depth": 2,
                "edge_activation_count": 0,
                "relation_type": "keyword_overlap",
                "shared_entities": [],
                "similarity": 0.04,
                "edge_quality": "normal",
            }
        ]

        reranker = PathReranker(
            profile="semantic_heavy",
            penalize_unsupported_keyword_paths=True,
        )
        supported_score = reranker.score(
            similarity=0.3,
            graph_score=0.75,
            signals=supported_keyword_signals,
            node=node,
            use_multipath=True,
            use_memory_state=False,
        )
        unsupported_score = reranker.score(
            similarity=0.3,
            graph_score=0.75,
            signals=unsupported_keyword_signals,
            node=node,
            use_multipath=True,
            use_memory_state=False,
        )

        self.assertIn("weak_relation_penalty", unsupported_score.breakdown)
        self.assertGreater(unsupported_score.weak_relation_penalty, supported_score.weak_relation_penalty)
        self.assertLess(unsupported_score.total, supported_score.total)

    def test_path_reranker_disables_unsupported_keyword_penalty_by_default(self) -> None:
        node = self.nodes[0]
        signals = [
            {
                "path": ["seed", "bridge", node.id],
                "graph_score": 0.75,
                "depth": 2,
                "edge_activation_count": 0,
                "relation_type": "keyword_overlap",
                "shared_entities": [],
                "similarity": 0.04,
                "edge_quality": "normal",
            }
        ]

        default_score = PathReranker(profile="semantic_heavy").score(
            similarity=0.3,
            graph_score=0.75,
            signals=signals,
            node=node,
            use_multipath=True,
            use_memory_state=False,
        )
        enabled_score = PathReranker(
            profile="semantic_heavy",
            penalize_unsupported_keyword_paths=True,
        ).score(
            similarity=0.3,
            graph_score=0.75,
            signals=signals,
            node=node,
            use_multipath=True,
            use_memory_state=False,
        )

        self.assertLess(default_score.weak_relation_penalty, enabled_score.weak_relation_penalty)

    def test_path_reranker_uses_edge_audit_noise_rates(self) -> None:
        node = self.nodes[0]
        signals = [
            {
                "path": ["seed", node.id],
                "graph_score": 0.8,
                "depth": 1,
                "edge_activation_count": 0,
                "relation_type": "keyword_overlap",
            }
        ]

        plain_score = PathReranker(profile="semantic_heavy").score(
            similarity=0.5,
            graph_score=0.8,
            signals=signals,
            node=node,
            use_multipath=True,
            use_memory_state=False,
        )
        audited_score = PathReranker(
            profile="semantic_heavy",
            relation_noise_rates={"keyword_overlap": 0.87},
        ).score(
            similarity=0.5,
            graph_score=0.8,
            signals=signals,
            node=node,
            use_multipath=True,
            use_memory_state=False,
        )

        self.assertIn("relation_noise_penalty", audited_score.breakdown)
        self.assertGreater(audited_score.relation_noise_penalty, 0.0)
        self.assertLess(audited_score.total, plain_score.total)

    def test_path_reranker_loads_edge_audit_from_environment(self) -> None:
        audit_path = Path(self.temp_dir.name) / "edge_quality_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "relation_stats": {
                        "keyword_overlap": {"noise_rate": 0.88},
                        "shared_entity": {"noise_rate": 0.1},
                    }
                }
            ),
            encoding="utf-8",
        )

        with patch.dict("os.environ", {"SAM_EDGE_QUALITY_AUDIT_PATH": str(audit_path)}, clear=False):
            reranker = PathReranker.from_env()

        self.assertEqual(reranker.relation_noise_rates["keyword_overlap"], 0.88)

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
            GraphBuilder(self.store, relation_judge=judge, relation_judge_policy="all"),
        )

        evaluator.evaluate(
            self.queries[:1],
            top_k=2,
            seed_k=1,
            hops=1,
            methods=["sam_full"],
        )

        self.assertGreater(judge.calls, 0)

    def test_evaluator_preserves_relation_judge_policy_in_isolated_method_runs(self) -> None:
        class FailingRelationJudge:
            def judge(
                self,
                seed: MemoryNode,
                other: MemoryNode,
                score_breakdown: dict[str, object],
            ) -> RelationJudgment:
                raise AssertionError("policy=off 应在隔离评测中继续跳过关系判别器")

        evaluator = Evaluator(
            self.store,
            self.embedding,
            GraphBuilder(
                self.store,
                relation_judge=FailingRelationJudge(),
                relation_judge_policy="off",
            ),
        )

        result = evaluator.evaluate(
            self.queries[:1],
            top_k=2,
            seed_k=1,
            hops=1,
            methods=["sam_full"],
        )

        self.assertIn("sam_full", result.method_metrics)

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

    def test_consolidated_memory_uses_evidence_centroid_without_provider_call(self) -> None:
        class FailingEmbeddingProvider(EmbeddingProvider):
            @property
            def cache_namespace(self) -> str:
                return "failing"

            def embed(self, text: str) -> list[float]:
                raise AssertionError("巩固记忆不应重新请求 embedding provider")

        query = self.queries[0]
        support_node = next(
            node
            for node in self.store.get_nodes()
            if node.metadata.get("original_doc_id") in query.supporting_doc_ids
        )
        hit = RetrievalHit(
            node=support_node,
            score=0.8,
            similarity_score=0.8,
            graph_score=0.0,
            usage_score=0.0,
            confidence_score=support_node.confidence,
            path=[support_node.id],
            reason="单元测试支持证据",
            metadata={},
        )

        record = MemoryConsolidator(
            self.store,
            FailingEmbeddingProvider(),
        ).consolidate_query(
            query=query,
            mode="sam_full",
            hits=[hit],
            support_node_ids={support_node.id},
            answer_status="found_in_retrieved_context",
        )

        self.assertIsNotNone(record)
        assert record is not None
        consolidated = self.store.get_node(record.node_id)
        self.assertIsNotNone(consolidated)
        assert consolidated is not None
        self.assertEqual(consolidated.metadata["embedding_source"], "evidence_centroid")
        self.assertEqual(len(consolidated.embedding), len(support_node.embedding))

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
        analogy_candidates = self.evaluator._candidate_ids_for_method(
            self.store,
            "sam_with_analogy",
            base_candidate_ids,
        )
        vector_candidates = self.evaluator._candidate_ids_for_method(
            self.store,
            "embedding_topk",
            base_candidate_ids,
        )

        self.assertFalse(consolidated_ids & set(sam_candidates))
        self.assertFalse(consolidated_support_ids & set(sam_candidates))
        self.assertTrue(consolidated_ids & set(analogy_candidates))
        self.assertTrue(consolidated_support_ids & set(analogy_candidates))
        self.assertFalse(consolidated_ids & set(vector_candidates))
        self.assertFalse(consolidated_support_ids & set(vector_candidates))

    def test_memory_reuse_experiment_masks_gold_support(self) -> None:
        masked = build_masked_queries(self.queries[:1])

        self.assertEqual(masked[0].supporting_doc_ids, self.queries[0].supporting_doc_ids)
        self.assertFalse(set(masked[0].supporting_doc_ids) & set(masked[0].candidate_doc_ids))
        self.assertEqual(masked[0].metadata["reuse_probe"], True)

    def test_memory_reuse_candidates_expose_history_only_to_sam_methods(self) -> None:
        self.evaluator.evaluate(
            self.queries[:1],
            top_k=3,
            seed_k=1,
            hops=1,
            methods=["sam_full"],
        )
        masked = build_masked_queries(self.queries[:1])[0]
        support_node_ids = {
            node.id
            for node in self.store.get_nodes()
            if node.metadata.get("original_doc_id") in masked.supporting_doc_ids
        }
        base_candidate_ids = [
            node.id
            for node in self.store.get_nodes()
            if node.metadata.get("original_doc_id") in masked.candidate_doc_ids
        ]
        self.assertFalse(support_node_ids & set(base_candidate_ids))

        baseline_candidates = memory_reuse_candidate_ids(
            store=self.store,
            query=masked,
            method="embedding_topk",
            base_candidate_ids=base_candidate_ids,
        )
        sam_candidates = memory_reuse_candidate_ids(
            store=self.store,
            query=masked,
            method="sam_full",
            base_candidate_ids=base_candidate_ids,
        )

        self.assertFalse(support_node_ids & set(baseline_candidates))
        self.assertTrue(support_node_ids & set(sam_candidates))

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

    def test_memory_reuse_reports_include_method_table_and_edge_changes(self) -> None:
        output_dir = Path(self.temp_dir.name) / "memory_reuse_report"
        summary = {
            "warmup_consolidated_count": 1,
            "warmup_consolidation_edge_count": 2,
            "baseline_support_hits": 0,
            "sam_support_hits": 2,
            "support_hit_gain": 2,
            "baseline_evidence_recall": 0.0,
            "sam_evidence_recall": 1.0,
            "evidence_recall_gain": 1.0,
        }
        probe_metrics = {
            "method_metrics": {
                "embedding_topk": {
                    "display_name": "Embedding Top-k",
                    "support_hits": 0,
                    "evidence_recall": 0.0,
                    "answer_hit_rate": 0.0,
                    "average_path_length": 1.0,
                    "average_edge_memory_score": 0.0,
                },
                "sam_full": {
                    "display_name": "SAM-full",
                    "support_hits": 2,
                    "evidence_recall": 1.0,
                    "answer_hit_rate": 1.0,
                    "average_path_length": 2.0,
                    "average_edge_memory_score": 0.1,
                },
            }
        }

        _json_path, markdown_path = write_memory_reuse_reports(
            output_dir=output_dir,
            summary=summary,
            warmup_metrics={},
            probe_metrics=probe_metrics,
            probe_cases=[],
        )

        self.assertIn("Probe 方法对比", markdown_path.read_text(encoding="utf-8"))
        self.assertIn("SAM-full", markdown_path.read_text(encoding="utf-8"))

        self.graph.build_edges_on_demand(self.nodes[:1], self.nodes)
        before = snapshot_edges(self.store.get_edges())
        edge = self.store.get_edges()[0]
        self.store.activate_edges([(edge.source_id, edge.target_id, edge.relation_type)])
        _events_json, events_md, changes_json, changes_md = write_memory_reuse_event_reports(
            output_dir=output_dir,
            events=self.store.get_memory_events(limit=100),
            edges_after=self.store.get_edges(),
            edges_before=before,
        )

        self.assertTrue(changes_json.exists())
        self.assertIn("连续记忆复用反馈边变化", changes_md.read_text(encoding="utf-8"))
        self.assertIn("连续记忆复用事件流", events_md.read_text(encoding="utf-8"))

    def test_edge_quality_audit_counts_support_and_noise_relations(self) -> None:
        cases = [
            {
                "query_id": "edge_case",
                "supporting_doc_ids": ["gold_a", "gold_b"],
                "support_hits_by_method": {"sam_full": 1},
                "methods": {
                    "sam_full": [
                        {
                            "node_id": "support",
                            "is_supporting": True,
                            "path": ["seed", "support"],
                            "candidate_paths": [
                                {"relation_type": "shared_entity", "graph_score": 0.8}
                            ],
                        },
                        {
                            "node_id": "noise",
                            "is_supporting": False,
                            "path": ["seed", "middle", "noise"],
                            "candidate_paths": [
                                {"relation_type": "embedding_similarity", "graph_score": 0.7},
                                {"relation_type": "keyword_overlap", "graph_score": 0.4},
                            ],
                        },
                    ]
                },
            }
        ]

        audit = audit_edge_quality(cases, method="sam_full")
        output_dir = Path(self.temp_dir.name) / "edge_audit"
        json_path, markdown_path = write_edge_quality_audit(audit, output_dir)

        relation_stats = audit["relation_stats"]
        self.assertEqual(relation_stats["shared_entity"]["support_count"], 1)
        self.assertEqual(relation_stats["embedding_similarity"]["noise_count"], 1)
        self.assertEqual(relation_stats["embedding_similarity"]["noise_rate"], 1.0)
        self.assertEqual(audit["summary"]["graph_noise_case_count"], 1)
        self.assertTrue(json_path.exists())
        self.assertIn("边质量审计", markdown_path.read_text(encoding="utf-8"))

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

    def test_azure_embedding_provider_can_use_full_request_url(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "test-key",
                "SAM_AZURE_EMBEDDING_URL": "https://example.test/custom/embeddings",
                "SAM_AZURE_EMBEDDING_MODEL": "text-embedding-3-large",
            },
            clear=True,
        ):
            provider = AzureOpenAIEmbeddingProvider()

        self.assertEqual(provider.request_url, "https://example.test/custom/embeddings")

    def test_embedding_config_diagnostic_does_not_expose_secret_values(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "test-secret-value",
                "SAM_AZURE_EMBEDDING_URL": "https://example.test/custom/embeddings",
                "SAM_AZURE_EMBEDDING_DIMENSIONS": "1024",
            },
            clear=True,
        ):
            status = inspect_embedding_provider_config("azure_openai")

        rendered = json.dumps(status, ensure_ascii=False)
        self.assertTrue(status["ready"])
        self.assertNotIn("test-secret-value", rendered)
        self.assertNotIn("https://example.test/custom/embeddings", rendered)
        self.assertIn("SAM_AZURE_EMBEDDING_DIMENSIONS", status["configured_optional"])

    def test_embedding_config_diagnostic_requires_endpoint_or_full_url(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "test-key",
            },
            clear=True,
        ):
            status = inspect_embedding_provider_config("azure_openai")

        self.assertFalse(status["ready"])
        self.assertEqual(
            status["required_any_missing"],
            [["SAM_AZURE_EMBEDDING_ENDPOINT", "SAM_AZURE_EMBEDDING_URL"]],
        )

    def test_embedding_network_preflight_does_not_expose_endpoint_or_key(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "test-secret-value",
                "SAM_AZURE_EMBEDDING_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
            },
            clear=True,
        ), patch("sam.embedding.socket.create_connection", side_effect=TimeoutError("timed out")):
            status = preflight_embedding_endpoint("azure_openai_sdk", timeout=0.01)

        rendered = json.dumps(status, ensure_ascii=False)
        self.assertTrue(status["checked"])
        self.assertFalse(status["ok"])
        self.assertEqual(status["error_type"], "TimeoutError")
        self.assertNotIn("example.test", rendered)
        self.assertNotIn("test-secret-value", rendered)

    def test_azure_chat_client_can_use_full_request_url(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_CHAT_API_KEY": "test-key",
                "SAM_AZURE_CHAT_URL": "https://example.test/custom/chat/completions",
                "SAM_AZURE_CHAT_MODEL": "gpt-5.4-2026-03-05",
            },
            clear=True,
        ):
            client = AzureOpenAIChatClient()

        self.assertEqual(client.request_url, "https://example.test/custom/chat/completions")

    def test_chat_config_diagnostic_does_not_expose_secret_values(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_CHAT_API_KEY": "test-chat-secret",
                "SAM_AZURE_CHAT_URL": "https://example.test/custom/chat/completions",
                "SAM_AZURE_CHAT_MODEL": "gpt-5.4-2026-03-05",
            },
            clear=True,
        ):
            status = inspect_chat_provider_config("azure_openai")

        rendered = json.dumps(status, ensure_ascii=False)
        self.assertTrue(status["ready"])
        self.assertNotIn("test-chat-secret", rendered)
        self.assertNotIn("https://example.test/custom/chat/completions", rendered)
        self.assertIn("SAM_AZURE_CHAT_MODEL", status["configured_optional"])

    def test_chat_config_diagnostic_requires_endpoint_or_full_url(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_CHAT_API_KEY": "test-key",
            },
            clear=True,
        ):
            status = inspect_chat_provider_config("azure_openai")

        self.assertFalse(status["ready"])
        self.assertEqual(
            status["required_any_missing"],
            [["SAM_AZURE_CHAT_ENDPOINT", "SAM_AZURE_CHAT_URL"]],
        )

    def test_azure_chat_client_sends_model_messages_and_max_tokens(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {"choices": [{"message": {"content": "OK"}}]}
                ).encode("utf-8")

        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=120):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["headers"] = dict(request.header_items())
            return FakeResponse()

        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_CHAT_API_KEY": "test-key",
                "SAM_AZURE_CHAT_URL": "https://example.test/custom/chat/completions",
                "SAM_AZURE_CHAT_MODEL": "gpt-5.4-2026-03-05",
            },
            clear=True,
        ), patch("urllib.request.urlopen", fake_urlopen):
            answer = AzureOpenAIChatClient().complete(
                [{"role": "user", "content": "What is 1+1?"}],
                max_tokens=32,
            )

        self.assertEqual(answer, "OK")
        self.assertEqual(captured["url"], "https://example.test/custom/chat/completions")
        payload = captured["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(payload["model"], "gpt-5.4-2026-03-05")
        self.assertEqual(payload["max_tokens"], 32)
        self.assertEqual(payload["stream"], False)
        self.assertEqual(payload["messages"][0]["content"], "What is 1+1?")

    def test_azure_chat_sdk_client_uses_openai_sdk(self) -> None:
        captured: dict[str, object] = {}

        class FakeMessage:
            content = "OK"

        class FakeChoice:
            message = FakeMessage()

        class FakeChatResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                captured["payload"] = kwargs
                return FakeChatResponse()

        class FakeChat:
            def __init__(self) -> None:
                self.completions = FakeCompletions()

        class FakeAzureOpenAI:
            def __init__(self, **kwargs) -> None:
                captured["client"] = kwargs
                self.chat = FakeChat()

        fake_openai = type("FakeOpenAI", (), {"AzureOpenAI": FakeAzureOpenAI})()

        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_CHAT_API_KEY": "test-key",
                "SAM_AZURE_CHAT_ENDPOINT": "https://example.test/api/modelhub/online/v2/crawl",
                "SAM_AZURE_CHAT_API_VERSION": "2024-02-01",
                "SAM_AZURE_CHAT_MODEL": "gpt-5.4-2026-03-05",
                "SAM_AZURE_CHAT_TIMEOUT": "42",
            },
            clear=True,
        ), patch.dict("sys.modules", {"openai": fake_openai}):
            answer = AzureOpenAISDKChatClient().complete(
                [{"role": "user", "content": [{"type": "text", "text": "What is 1+1?"}]}],
                max_tokens=32,
            )

        self.assertEqual(answer, "OK")
        self.assertEqual(captured["client"]["azure_endpoint"], "https://example.test/api/modelhub/online/v2/crawl")
        self.assertEqual(captured["client"]["api_version"], "2024-02-01")
        self.assertEqual(captured["client"]["api_key"], "test-key")
        self.assertEqual(captured["client"]["timeout"], 42.0)
        self.assertEqual(captured["payload"]["model"], "gpt-5.4-2026-03-05")
        self.assertEqual(captured["payload"]["max_tokens"], 32)
        self.assertFalse(captured["payload"]["stream"])
        self.assertEqual(captured["payload"]["messages"][0]["content"][0]["text"], "What is 1+1?")

    def test_azure_chat_sdk_client_retries_rate_limit_errors(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeRateLimitError(RuntimeError):
            status_code = 429

        class FakeMessage:
            content = "OK after retry"

        class FakeChoice:
            message = FakeMessage()

        class FakeChatResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise FakeRateLimitError("qpm limit")
                return FakeChatResponse()

        class FakeChat:
            def __init__(self) -> None:
                self.completions = FakeCompletions()

        class FakeAzureOpenAI:
            def __init__(self, **kwargs) -> None:
                self.chat = FakeChat()

        fake_openai = type("FakeOpenAI", (), {"AzureOpenAI": FakeAzureOpenAI})()

        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_CHAT_API_KEY": "test-key",
                "SAM_AZURE_CHAT_ENDPOINT": "https://example.test/api/modelhub/online/v2/crawl",
                "SAM_AZURE_CHAT_MAX_RETRIES": "2",
                "SAM_AZURE_CHAT_RETRY_BASE_SECONDS": "0",
            },
            clear=True,
        ), patch.dict("sys.modules", {"openai": fake_openai}):
            answer = AzureOpenAISDKChatClient().complete(
                [{"role": "user", "content": "What is 1+1?"}],
                max_tokens=32,
            )

        self.assertEqual(answer, "OK after retry")
        self.assertEqual(len(calls), 2)

    def test_azure_chat_sdk_client_uses_dedicated_rate_limit_retry_budget(self) -> None:
        calls: list[dict[str, object]] = []
        sleeps: list[float] = []

        class FakeRateLimitError(RuntimeError):
            status_code = 429

        class FakeMessage:
            content = "OK after dedicated retry"

        class FakeChoice:
            message = FakeMessage()

        class FakeChatResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                if len(calls) < 3:
                    raise FakeRateLimitError("qpm limit")
                return FakeChatResponse()

        class FakeChat:
            def __init__(self) -> None:
                self.completions = FakeCompletions()

        class FakeAzureOpenAI:
            def __init__(self, **kwargs) -> None:
                self.chat = FakeChat()

        fake_openai = type("FakeOpenAI", (), {"AzureOpenAI": FakeAzureOpenAI})()

        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_CHAT_API_KEY": "test-key",
                "SAM_AZURE_CHAT_ENDPOINT": "https://example.test/api/modelhub/online/v2/crawl",
                "SAM_AZURE_CHAT_MAX_RETRIES": "1",
                "SAM_AZURE_CHAT_RATE_LIMIT_RETRIES": "3",
                "SAM_AZURE_CHAT_RETRY_BASE_SECONDS": "0",
                "SAM_AZURE_CHAT_RATE_LIMIT_SLEEP_SECONDS": "7",
            },
            clear=True,
        ), patch.dict("sys.modules", {"openai": fake_openai}), patch(
            "sam.llm.time.sleep",
            side_effect=lambda seconds: sleeps.append(seconds),
        ):
            answer = AzureOpenAISDKChatClient().complete(
                [{"role": "user", "content": "What is 1+1?"}],
                max_tokens=32,
            )

        self.assertEqual(answer, "OK after dedicated retry")
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [7.0, 7.0])

    def test_azure_chat_sdk_client_throttles_between_requests(self) -> None:
        calls: list[dict[str, object]] = []
        sleeps: list[float] = []

        class FakeMessage:
            content = "OK"

        class FakeChoice:
            message = FakeMessage()

        class FakeChatResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return FakeChatResponse()

        class FakeChat:
            def __init__(self) -> None:
                self.completions = FakeCompletions()

        class FakeAzureOpenAI:
            def __init__(self, **kwargs) -> None:
                self.chat = FakeChat()

        fake_openai = type("FakeOpenAI", (), {"AzureOpenAI": FakeAzureOpenAI})()

        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_CHAT_API_KEY": "test-key",
                "SAM_AZURE_CHAT_ENDPOINT": "https://example.test/api/modelhub/online/v2/crawl",
                "SAM_AZURE_CHAT_MIN_INTERVAL_SECONDS": "5",
            },
            clear=True,
        ), patch.dict("sys.modules", {"openai": fake_openai}), patch(
            "sam.llm.time.monotonic",
            side_effect=[100.0, 101.0, 105.0],
        ), patch(
            "sam.llm.time.sleep",
            side_effect=lambda seconds: sleeps.append(seconds),
        ):
            client = AzureOpenAISDKChatClient()
            client.complete([{"role": "user", "content": "first"}])
            client.complete([{"role": "user", "content": "second"}])

        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [4.0])

    def test_create_chat_client_supports_azure_openai_sdk(self) -> None:
        fake_openai = type("FakeOpenAI", (), {"AzureOpenAI": staticmethod(lambda **kwargs: object())})()
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_CHAT_API_KEY": "test-key",
                "SAM_AZURE_CHAT_ENDPOINT": "https://example.test/api/modelhub/online/v2/crawl",
            },
            clear=True,
        ), patch.dict("sys.modules", {"openai": fake_openai}):
            client = create_chat_client("azure_openai_sdk")

        self.assertIsInstance(client, AzureOpenAISDKChatClient)

    def test_chat_sdk_config_diagnostic_reports_missing_openai_package(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_CHAT_API_KEY": "test-key",
                "SAM_AZURE_CHAT_ENDPOINT": "https://example.test/api/modelhub/online/v2/crawl",
                "SAM_AZURE_CHAT_MAX_RETRIES": "4",
                "SAM_AZURE_CHAT_RETRY_BASE_SECONDS": "1",
            },
            clear=True,
        ), patch("importlib.util.find_spec", return_value=None):
            status = inspect_chat_provider_config("azure_openai_sdk")

        self.assertFalse(status["ready"])
        self.assertIn("openai", status["missing_packages"])
        self.assertIn("python -m pip install", status["install_hint"])
        self.assertIn("SAM_AZURE_CHAT_MAX_RETRIES", status["configured_optional"])

    def test_combined_provider_diagnostic_supports_local_probes(self) -> None:
        status = build_provider_status(
            embedding_provider="local",
            chat_provider="heuristic",
            embedding_probe="SAM combined diagnostic",
            chat_probe="The answer is ready.",
        )

        self.assertTrue(status["ready"])
        embedding = status["embedding"]
        chat = status["chat"]
        assert isinstance(embedding, dict)
        assert isinstance(chat, dict)
        self.assertEqual(embedding["probe"]["dimension"], 256)
        self.assertEqual(chat["probe"]["answer_preview"], "ready")
        self.assertEqual(embedding["alias_sources"], {})
        self.assertEqual(chat["alias_sources"], {})

    def test_combined_provider_diagnostic_does_not_expose_secret_values(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "embedding-secret",
                "SAM_AZURE_EMBEDDING_URL": "https://example.test/custom/embeddings",
                "SAM_AZURE_CHAT_API_KEY": "chat-secret",
                "SAM_AZURE_CHAT_URL": "https://example.test/custom/chat/completions",
            },
            clear=True,
        ):
            status = build_provider_status(
                embedding_provider="azure_openai",
                chat_provider="azure_openai",
            )

        rendered = json.dumps(status, ensure_ascii=False)
        self.assertTrue(status["ready"])
        self.assertNotIn("embedding-secret", rendered)
        self.assertNotIn("chat-secret", rendered)
        self.assertNotIn("https://example.test/custom/embeddings", rendered)
        self.assertNotIn("https://example.test/custom/chat/completions", rendered)

    def test_combined_provider_diagnostic_reports_probe_errors_without_traceback(self) -> None:
        class FailingChatClient(ChatClient):
            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                raise RuntimeError("HTTP 429 from https://example.test/custom/chat/completions")

        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_CHAT_API_KEY": "chat-secret",
                "SAM_AZURE_CHAT_URL": "https://example.test/custom/chat/completions",
            },
            clear=True,
        ), patch("scripts.check_model_providers.create_chat_client", return_value=FailingChatClient()):
            status = build_provider_status(
                embedding_provider="local",
                chat_provider="azure_openai",
                chat_probe="What is 1+1?",
                required_providers="chat",
            )

        rendered = json.dumps(status, ensure_ascii=False)
        self.assertFalse(status["ready"])
        self.assertEqual(status["chat"]["probe_error"]["type"], "RuntimeError")
        self.assertIn("HTTP 429", status["chat"]["probe_error"]["message"])
        self.assertNotIn("https://example.test/custom/chat/completions", rendered)
        self.assertNotIn("chat-secret", rendered)

    def test_combined_provider_diagnostic_skips_embedding_probe_after_preflight_failure(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "embedding-secret",
                "SAM_AZURE_EMBEDDING_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
            },
            clear=True,
        ), patch(
            "scripts.check_model_providers.preflight_embedding_endpoint",
            return_value={
                "checked": True,
                "ok": False,
                "error_type": "TimeoutError",
                "message": "embedding endpoint TCP preflight failed",
            },
        ), patch("scripts.check_model_providers.create_embedding_provider") as create_provider:
            status = build_provider_status(
                embedding_provider="azure_openai_sdk",
                chat_provider="heuristic",
                embedding_probe="probe text",
                required_providers="embedding",
            )

        rendered = json.dumps(status, ensure_ascii=False)
        create_provider.assert_not_called()
        self.assertFalse(status["ready"])
        self.assertEqual(status["embedding"]["probe_error"]["type"], "TimeoutError")
        self.assertNotIn("example.test", rendered)
        self.assertNotIn("embedding-secret", rendered)

    def test_combined_provider_diagnostic_can_skip_embedding_preflight(self) -> None:
        class FixedEmbeddingProvider(LocalHashEmbeddingProvider):
            def embed(self, text: str) -> list[float]:
                return [3.0, 4.0]

        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "embedding-secret",
                "SAM_AZURE_EMBEDDING_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
            },
            clear=True,
        ), patch(
            "scripts.check_model_providers.preflight_embedding_endpoint",
            side_effect=AssertionError("跳过预检时不应该调用 TCP preflight"),
        ), patch(
            "scripts.check_model_providers.create_embedding_provider",
            return_value=FixedEmbeddingProvider(),
        ) as create_provider:
            status = build_provider_status(
                embedding_provider="azure_openai_sdk",
                chat_provider="heuristic",
                embedding_probe="probe text",
                required_providers="embedding",
                skip_embedding_preflight=True,
            )

        rendered = json.dumps(status, ensure_ascii=False)
        create_provider.assert_called_once()
        self.assertTrue(status["ready"])
        self.assertEqual(status["embedding"]["probe"]["dimension"], 2)
        self.assertEqual(status["embedding"]["probe"]["l2_norm"], 5.0)
        self.assertEqual(status["embedding"]["network_preflight"]["reason"], "skipped_by_user")
        self.assertNotIn("example.test", rendered)
        self.assertNotIn("embedding-secret", rendered)

    def test_embedding_provider_diagnostic_reports_probe_errors_without_traceback(self) -> None:
        class FailingEmbeddingProvider(LocalHashEmbeddingProvider):
            def embed(self, text: str) -> list[float]:
                raise RuntimeError("HTTP 429 from https://example.test/custom/embeddings")

        with patch(
            "scripts.check_embedding_provider.create_embedding_provider",
            return_value=FailingEmbeddingProvider(),
        ):
            status = build_embedding_status(provider_name="local", probe="probe text")

        rendered = json.dumps(status, ensure_ascii=False)
        self.assertFalse(status["ready"])
        self.assertEqual(status["probe_error"]["type"], "RuntimeError")
        self.assertIn("HTTP 429", status["probe_error"]["message"])
        self.assertNotIn("https://example.test/custom/embeddings", rendered)

    def test_embedding_provider_diagnostic_can_skip_preflight(self) -> None:
        class FixedEmbeddingProvider(LocalHashEmbeddingProvider):
            def embed(self, text: str) -> list[float]:
                return [1.0, 2.0, 2.0]

        with patch(
            "scripts.check_embedding_provider.preflight_embedding_endpoint",
            side_effect=AssertionError("跳过预检时不应该调用 TCP preflight"),
        ), patch(
            "scripts.check_embedding_provider.create_embedding_provider",
            return_value=FixedEmbeddingProvider(),
        ):
            status = build_embedding_status(
                provider_name="local",
                probe="probe text",
                skip_preflight=True,
            )

        self.assertTrue(status["ready"])
        self.assertEqual(status["probe"]["dimension"], 3)
        self.assertEqual(status["probe"]["l2_norm"], 3.0)
        self.assertEqual(status["network_preflight"]["reason"], "skipped_by_user")

    def test_combined_provider_diagnostic_can_require_only_embedding(self) -> None:
        status = build_provider_status(
            embedding_provider="local",
            chat_provider="azure_openai",
            required_providers="embedding",
        )

        self.assertTrue(status["ready"])
        self.assertTrue(status["embedding"]["ready"])
        self.assertFalse(status["chat"]["ready"])

    def test_provider_smoke_experiment_writes_provider_and_pipeline_reports(self) -> None:
        output_dir = Path(self.temp_dir.name) / "provider_smoke"
        dataset_path = Path(self.temp_dir.name) / "sample_dataset.json"
        documents, queries = load_builtin_benchmark_sample()
        save_sam_dataset(
            path=dataset_path,
            documents=documents,
            queries=queries,
            dataset_info={"name": "builtin-test"},
            processing={"source_script": "unit-test"},
        )

        summary = run_provider_smoke_experiment(
            dataset_file=dataset_path,
            output_dir=output_dir,
            limit=1,
            embedding_provider_name="local",
            chat_provider_name="heuristic",
            answer_judge_name="rule",
            query_planner_name="heuristic",
            relation_judge_name="disabled",
        )

        self.assertTrue(summary["provider_status"]["ready"])
        self.assertEqual(summary["pipeline"]["query_count"], 1)
        self.assertIn("audit", summary)
        self.assertTrue((output_dir / "provider_status.json").exists())
        self.assertTrue((output_dir / "pipeline_summary.json").exists())
        self.assertTrue((output_dir / "experiment_audit.json").exists())
        self.assertTrue((output_dir / "smoke_summary.md").exists())

    def test_provider_smoke_experiment_writes_summary_when_provider_gate_fails(self) -> None:
        output_dir = Path(self.temp_dir.name) / "provider_smoke_failed"
        dataset_path = Path(self.temp_dir.name) / "sample_dataset.json"
        documents, queries = load_builtin_benchmark_sample()
        save_sam_dataset(
            path=dataset_path,
            documents=documents,
            queries=queries,
            dataset_info={"name": "builtin-test"},
            processing={"source_script": "unit-test"},
        )

        summary = run_provider_smoke_experiment(
            dataset_file=dataset_path,
            output_dir=output_dir,
            limit=1,
            embedding_provider_name="local",
            chat_provider_name="azure_openai",
            answer_judge_name="rule",
            query_planner_name="disabled",
            relation_judge_name="disabled",
            required_providers="both",
        )

        self.assertFalse(summary["provider_status"]["ready"])
        self.assertIsNone(summary["pipeline"])
        self.assertTrue((output_dir / "provider_status.json").exists())
        self.assertTrue((output_dir / "smoke_summary.json").exists())
        self.assertTrue((output_dir / "smoke_summary.md").exists())
        self.assertFalse((output_dir / "pipeline_summary.json").exists())

    def test_experiment_audit_identifies_weak_graph_gain_and_generation_failure(self) -> None:
        run_dir = Path(self.temp_dir.name) / "audit_run"
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "method_metrics": {
                        "embedding_topk": {
                            "display_name": "Embedding Top-k",
                            "evidence_recall": 0.50,
                            "answer_hit_rate": 0.40,
                        },
                        "sam_full": {
                            "display_name": "SAM-full",
                            "evidence_recall": 0.50,
                            "answer_hit_rate": 0.40,
                        },
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (run_dir / "cases.json").write_text(
            json.dumps(
                [
                    {
                        "query_id": "q1",
                        "question": "question",
                        "answer": "answer",
                        "supporting_doc_ids": ["d1", "d2"],
                        "support_hits_by_method": {"embedding_topk": 1, "sam_full": 1},
                        "vector_support_hits": 1,
                        "methods": {"sam_full": [{"is_supporting": False, "path": ["a", "b"]}]},
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (run_dir / "generated_answers.json").write_text(
            json.dumps(
                [
                    {
                        "query_id": "q1",
                        "question": "question",
                        "gold_answer": "answer",
                        "method": "sam_full",
                        "generated_answer": "证据不足",
                        "answer_hit": False,
                        "context_titles": ["doc"],
                        "metadata": {"answer_judgment": {"status": "not_matched"}},
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        audit = audit_run_directory(run_dir, primary_method="sam_full")
        json_path, md_path = write_experiment_audit(audit, run_dir)

        bottleneck_types = {item["type"] for item in audit["bottlenecks"]}
        self.assertIn("weak_graph_gain", bottleneck_types)
        self.assertIn("generation_failure", bottleneck_types)
        self.assertIn("missing_support_evidence", audit["bad_case_summary"]["retrieval_categories"])
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())

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

    def test_azure_embedding_sdk_provider_uses_async_client_with_dimensions(self) -> None:
        captured: dict[str, object] = {}
        payloads: list[dict[str, object]] = []

        class FakeEmbeddingResponse:
            def __init__(self, texts: list[str]) -> None:
                self.data = [
                    type("EmbeddingItem", (), {"embedding": [float(index), float(len(text))]})()
                    for index, text in enumerate(texts)
                ]

        class FakeEmbeddings:
            async def create(self, **kwargs):
                payloads.append(kwargs)
                input_value = kwargs["input"]
                texts = [input_value] if isinstance(input_value, str) else list(input_value)
                return FakeEmbeddingResponse(texts)

        class FakeAsyncAzureOpenAI:
            def __init__(self, **kwargs) -> None:
                captured["client"] = kwargs
                self.embeddings = FakeEmbeddings()

        fake_openai = type("FakeOpenAI", (), {"AsyncAzureOpenAI": FakeAsyncAzureOpenAI})()

        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "test-key",
                "SAM_AZURE_EMBEDDING_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
                "SAM_AZURE_EMBEDDING_API_VERSION": "2023-07-01-preview",
                "SAM_AZURE_EMBEDDING_MODEL": "text-embedding-3-large",
                "SAM_AZURE_EMBEDDING_DIMENSIONS": "1024",
                "SAM_AZURE_EMBEDDING_BATCH_SIZE": "4",
                "SAM_AZURE_EMBEDDING_CONCURRENCY": "2",
                "SAM_AZURE_EMBEDDING_TIMEOUT": "180",
            },
            clear=False,
        ), patch.dict("sys.modules", {"openai": fake_openai}):
            provider = AzureOpenAISDKEmbeddingProvider()
            embeddings = provider.embed_many(["alpha", "beta"])

        self.assertEqual(captured["client"]["azure_endpoint"], "https://example.test/gpt/openapi/online/v2/crawl")
        self.assertEqual(captured["client"]["api_version"], "2023-07-01-preview")
        self.assertEqual(captured["client"]["api_key"], "test-key")
        self.assertEqual(captured["client"]["timeout"], 180.0)
        self.assertEqual([payload["input"] for payload in payloads], ["alpha", "beta"])
        self.assertTrue(all(payload["model"] == "text-embedding-3-large" for payload in payloads))
        self.assertTrue(all(payload["dimensions"] == 1024 for payload in payloads))
        self.assertEqual(embeddings, [[0.0, 5.0], [0.0, 4.0]])

    def test_azure_embedding_sdk_provider_can_use_batch_input_mode(self) -> None:
        payloads: list[dict[str, object]] = []

        class FakeEmbeddingResponse:
            def __init__(self, texts: list[str]) -> None:
                self.data = [
                    type("EmbeddingItem", (), {"embedding": [float(index), float(len(text))]})()
                    for index, text in enumerate(texts)
                ]

        class FakeEmbeddings:
            async def create(self, **kwargs):
                payloads.append(kwargs)
                return FakeEmbeddingResponse(list(kwargs["input"]))

        class FakeAsyncAzureOpenAI:
            def __init__(self, **kwargs) -> None:
                self.embeddings = FakeEmbeddings()

        fake_openai = type("FakeOpenAI", (), {"AsyncAzureOpenAI": FakeAsyncAzureOpenAI})()

        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "test-key",
                "SAM_AZURE_EMBEDDING_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
                "SAM_AZURE_EMBEDDING_INPUT_MODE": "batch",
                "SAM_AZURE_EMBEDDING_BATCH_SIZE": "4",
            },
            clear=True,
        ), patch.dict("sys.modules", {"openai": fake_openai}):
            provider = AzureOpenAISDKEmbeddingProvider()
            embeddings = provider.embed_many(["alpha", "beta"])

        self.assertEqual([payload["input"] for payload in payloads], [["alpha", "beta"]])
        self.assertEqual(embeddings, [[0.0, 5.0], [1.0, 4.0]])

    def test_azure_embedding_sdk_provider_retries_qpm_limit(self) -> None:
        class FakeEmbeddingResponse:
            data = [type("EmbeddingItem", (), {"embedding": [0.3, 0.4]})()]

        class FakeEmbeddings:
            def __init__(self) -> None:
                self.calls = 0

            async def create(self, **_kwargs):
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError("qpm limit, retry later")
                return FakeEmbeddingResponse()

        provider = AzureOpenAISDKEmbeddingProvider.__new__(AzureOpenAISDKEmbeddingProvider)
        provider.client = type("FakeClient", (), {"embeddings": FakeEmbeddings()})()
        provider.model = "text-embedding-3-large"
        provider.dimensions = 1024
        provider.max_retries = 1
        provider.rate_limit_retries = 3
        provider.retry_base_seconds = 0.0
        provider.rate_limit_sleep_seconds = 0.0
        provider.request_timeout = 5.0

        vector = asyncio.run(provider._embed_one_async("retry text"))

        self.assertEqual(vector, [0.3, 0.4])
        self.assertEqual(provider.client.embeddings.calls, 3)

    def test_azure_embedding_sdk_provider_times_out_hanging_request(self) -> None:
        class FakeEmbeddings:
            async def create(self, **kwargs):
                await asyncio.sleep(1.0)
                raise AssertionError("请求应该先被 wait_for 超时中断")

        class FakeAsyncAzureOpenAI:
            def __init__(self, **kwargs) -> None:
                self.embeddings = FakeEmbeddings()

        fake_openai = type("FakeOpenAI", (), {"AsyncAzureOpenAI": FakeAsyncAzureOpenAI})()

        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "test-key",
                "SAM_AZURE_EMBEDDING_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
                "SAM_AZURE_EMBEDDING_TIMEOUT": "0.01",
                "SAM_AZURE_EMBEDDING_MAX_RETRIES": "1",
            },
            clear=True,
        ), patch.dict("sys.modules", {"openai": fake_openai}):
            provider = AzureOpenAISDKEmbeddingProvider()

            with self.assertRaises(TimeoutError):
                provider.embed("alpha")

    def test_create_embedding_provider_supports_azure_openai_sdk(self) -> None:
        fake_openai = type("FakeOpenAI", (), {"AsyncAzureOpenAI": staticmethod(lambda **kwargs: object())})()
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "test-key",
                "SAM_AZURE_EMBEDDING_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
            },
            clear=True,
        ), patch.dict("sys.modules", {"openai": fake_openai}):
            provider = create_embedding_provider("azure_openai_sdk")

        self.assertIsInstance(provider, AzureOpenAISDKEmbeddingProvider)

    def test_sentence_transformer_embedding_provider_encodes_batches(self) -> None:
        captured: dict[str, object] = {}

        class FakeSentenceTransformer:
            def __init__(self, model_name: str, *, device: str | None = None, trust_remote_code: bool = False) -> None:
                captured["model_name"] = model_name
                captured["device"] = device
                captured["trust_remote_code"] = trust_remote_code

            def encode(self, texts, **kwargs):
                captured["encode"] = kwargs
                return [
                    [float(index), float(len(text))]
                    for index, text in enumerate(texts)
                ]

        fake_module = type("FakeSentenceTransformers", (), {"SentenceTransformer": FakeSentenceTransformer})()

        with patch.dict(
            "os.environ",
            {
                "SAM_SENTENCE_TRANSFORMER_MODEL": "/models/qwen3-embedding-0.6b",
                "SAM_SENTENCE_TRANSFORMER_DEVICE": "cpu",
                "SAM_SENTENCE_TRANSFORMER_BATCH_SIZE": "2",
                "SAM_SENTENCE_TRANSFORMER_NORMALIZE": "1",
            },
            clear=True,
        ), patch.dict("sys.modules", {"sentence_transformers": fake_module}):
            provider = SentenceTransformerEmbeddingProvider()
            embeddings = provider.embed_many(["alpha", "beta"])

        self.assertEqual(captured["model_name"], "/models/qwen3-embedding-0.6b")
        self.assertEqual(captured["device"], "cpu")
        self.assertTrue(captured["trust_remote_code"])
        self.assertEqual(captured["encode"]["batch_size"], 2)
        self.assertTrue(captured["encode"]["normalize_embeddings"])
        self.assertTrue(captured["encode"]["show_progress_bar"])
        self.assertEqual(embeddings, [[0.0, 5.0], [1.0, 4.0]])

    def test_create_embedding_provider_supports_sentence_transformers(self) -> None:
        class FakeSentenceTransformer:
            def __init__(self, model_name: str, **kwargs) -> None:
                self.model_name = model_name

            def encode(self, texts, **kwargs):
                return [[1.0, 0.0] for _ in texts]

        fake_module = type("FakeSentenceTransformers", (), {"SentenceTransformer": FakeSentenceTransformer})()

        with patch.dict("os.environ", {}, clear=True), patch.dict("sys.modules", {"sentence_transformers": fake_module}):
            provider = create_embedding_provider("sentence_transformers")

        self.assertIsInstance(provider, SentenceTransformerEmbeddingProvider)

    def test_embedding_config_diagnostic_supports_sentence_transformers(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_SENTENCE_TRANSFORMER_MODEL": "/models/qwen3-embedding-0.6b",
                "SAM_SENTENCE_TRANSFORMER_DEVICE": "cpu",
            },
            clear=True,
        ), patch("importlib.util.find_spec", return_value=object()):
            status = inspect_embedding_provider_config("sentence_transformers")

        self.assertTrue(status["ready"])
        self.assertIn("SAM_SENTENCE_TRANSFORMER_MODEL", status["configured_optional"])
        self.assertEqual(status["missing_packages"], [])

    def test_embedding_config_reports_missing_sentence_transformers_package(self) -> None:
        with patch("importlib.util.find_spec", return_value=None):
            status = inspect_embedding_provider_config("sentence_transformers")

        self.assertFalse(status["ready"])
        self.assertIn("sentence-transformers", status["missing_packages"])
        self.assertIn("sentence-transformers", status["install_hint"])

    def test_local_embedding_plan_checks_package_and_model_path(self) -> None:
        model_dir = Path(self.temp_dir.name) / "qwen3"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        with patch("importlib.util.find_spec", return_value=object()):
            plan = build_local_embedding_plan(str(model_dir))

        self.assertTrue(plan["ready"])
        model = plan["model"]
        assert isinstance(model, dict)
        self.assertTrue(model["ready"])
        self.assertIn("config.json", model["found_marker_files"])
        self.assertIn("--query-limit 30", plan["run_command"])

        output_dir = Path(self.temp_dir.name) / "local_embedding_plan"
        json_path, markdown_path = write_local_embedding_plan(plan, output_dir)
        self.assertTrue(json_path.exists())
        self.assertTrue(markdown_path.exists())

    def test_local_embedding_plan_reports_missing_model_path(self) -> None:
        missing_dir = Path(self.temp_dir.name) / "missing-qwen3"
        with patch("importlib.util.find_spec", return_value=object()):
            plan = build_local_embedding_plan(str(missing_dir))

        self.assertFalse(plan["ready"])
        model = plan["model"]
        assert isinstance(model, dict)
        self.assertFalse(model["exists"])
        self.assertIn("本地模型目录", plan["notes"][0])

    def test_embedding_config_diagnostic_supports_azure_openai_sdk(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "sdk-secret",
                "SAM_AZURE_EMBEDDING_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
                "SAM_AZURE_EMBEDDING_DIMENSIONS": "1024",
            },
            clear=True,
        ), patch("importlib.util.find_spec", return_value=object()):
            status = inspect_embedding_provider_config("azure_openai_sdk")

        rendered = json.dumps(status, ensure_ascii=False)
        self.assertTrue(status["ready"])
        self.assertNotIn("sdk-secret", rendered)
        self.assertNotIn("https://example.test", rendered)
        self.assertIn("SAM_AZURE_EMBEDDING_DIMENSIONS", status["configured_optional"])

    def test_embedding_config_can_reuse_official_baseline_aliases(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "embedding-secret",
                "RAPTOR_AZURE_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
                "RAPTOR_API_VERSION": "2023-07-01-preview",
            },
            clear=True,
        ), patch("importlib.util.find_spec", return_value=object()):
            status = inspect_embedding_provider_config("azure_openai_sdk")

        self.assertTrue(status["ready"])
        self.assertEqual(
            status["alias_sources"]["SAM_AZURE_EMBEDDING_API_KEY"],
            "OPENAI_API_KEY",
        )
        self.assertEqual(
            status["alias_sources"]["SAM_AZURE_EMBEDDING_ENDPOINT"],
            "RAPTOR_AZURE_ENDPOINT",
        )
        self.assertEqual(
            status["alias_sources"]["SAM_AZURE_EMBEDDING_API_VERSION"],
            "RAPTOR_API_VERSION",
        )

    def test_embedding_config_diagnostic_reports_missing_openai_package_for_sdk(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "sdk-secret",
                "SAM_AZURE_EMBEDDING_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
            },
            clear=True,
        ), patch("importlib.util.find_spec", return_value=None):
            status = inspect_embedding_provider_config("azure_openai_sdk")

        self.assertFalse(status["ready"])
        self.assertIn("openai", status["missing_packages"])
        self.assertIn("python -m pip install", status["install_hint"])

    def test_embedding_config_diagnostic_treats_placeholder_key_as_missing(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_API_KEY": "replace-with-embedding-api-key",
                "SAM_AZURE_EMBEDDING_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
            },
            clear=True,
        ), patch("importlib.util.find_spec", return_value=object()):
            status = inspect_embedding_provider_config("azure_openai_sdk")

        self.assertFalse(status["ready"])
        self.assertIn("SAM_AZURE_EMBEDDING_API_KEY", status["missing"])

    def test_chat_config_diagnostic_treats_placeholder_key_as_missing(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_CHAT_API_KEY": "replace-with-chat-api-key",
                "SAM_AZURE_CHAT_ENDPOINT": "https://example.test/api/modelhub/online/v2/crawl",
            },
            clear=True,
        ):
            status = inspect_chat_provider_config("azure_openai")

        self.assertFalse(status["ready"])
        self.assertIn("SAM_AZURE_CHAT_API_KEY", status["missing"])

    def test_provider_env_aliases_map_gpt54_config_to_sam_chat_config(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "GPT54_API_KEY": "chat-secret",
                "GPT54_BASE_URL": "https://example.test/api/modelhub/online/v2/crawl",
                "GPT54_API_VERSION": "2024-02-01",
                "GPT54_MODEL": "gpt-5.4-2026-03-05",
            },
            clear=True,
        ):
            status = inspect_chat_provider_config("azure_openai")

        self.assertTrue(status["ready"])
        self.assertEqual(status["alias_sources"]["SAM_AZURE_CHAT_API_KEY"], "GPT54_API_KEY")
        self.assertIn("SAM_AZURE_CHAT_MODEL", status["configured_optional"])
        self.assertNotIn("chat-secret", json.dumps(status))

    def test_provider_env_aliases_map_generic_embedding_config_to_sam_config(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "EMBEDDING_API_KEY": "embedding-secret",
                "EMBEDDING_BASE_URL": "https://example.test/gpt/openapi/online/v2/crawl",
                "EMBEDDING_MODEL": "text-embedding-3-large",
                "EMBEDDING_DIMENSIONS": "1024",
            },
            clear=True,
        ), patch("importlib.util.find_spec", return_value=object()):
            status = inspect_embedding_provider_config("azure_openai_sdk")

        self.assertTrue(status["ready"])
        self.assertEqual(status["alias_sources"]["SAM_AZURE_EMBEDDING_API_KEY"], "EMBEDDING_API_KEY")
        self.assertIn("SAM_AZURE_EMBEDDING_DIMENSIONS", status["configured_optional"])
        self.assertNotIn("embedding-secret", json.dumps(status))

    def test_provider_env_aliases_do_not_override_explicit_sam_values(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_CHAT_API_KEY": "explicit-chat-secret",
                "GPT54_API_KEY": "alias-chat-secret",
            },
            clear=True,
        ):
            applied = apply_provider_env_aliases()

            self.assertEqual(os.environ["SAM_AZURE_CHAT_API_KEY"], "explicit-chat-secret")
            self.assertNotIn("SAM_AZURE_CHAT_API_KEY", applied)

    def test_load_env_file_ignores_comments_and_preserves_existing_values(self) -> None:
        env_path = Path(self.temp_dir.name) / ".env.local"
        env_path.write_text(
            "\n".join(
                [
                    "# SAM 本地模型配置",
                    "SAM_AZURE_EMBEDDING_API_KEY=file-secret",
                    "export SAM_AZURE_EMBEDDING_ENDPOINT=\"https://example.test/embedding\"",
                    "EMPTY_VALUE=",
                ]
            ),
            encoding="utf-8",
        )

        with patch.dict("os.environ", {"SAM_AZURE_EMBEDDING_API_KEY": "existing-secret"}, clear=True):
            loaded = load_env_file(env_path)

            self.assertEqual(os.environ["SAM_AZURE_EMBEDDING_API_KEY"], "existing-secret")
            self.assertEqual(os.environ["SAM_AZURE_EMBEDDING_ENDPOINT"], "https://example.test/embedding")
            self.assertEqual(os.environ["EMPTY_VALUE"], "")

        self.assertEqual(
            loaded,
            {
                "SAM_AZURE_EMBEDDING_API_KEY": False,
                "SAM_AZURE_EMBEDDING_ENDPOINT": True,
                "EMPTY_VALUE": True,
            },
        )

    def test_load_env_file_expands_simple_env_references(self) -> None:
        env_path = Path(self.temp_dir.name) / ".env.local"
        env_path.write_text(
            "\n".join(
                [
                    "export GPT54_BASE_URL=\"https://example.test/gpt/openapi/online/v2/crawl\"",
                    "export GPT54_API_VERSION=\"2023-07-01-preview\"",
                    "export RAPTOR_AZURE_ENDPOINT=\"$GPT54_BASE_URL\"",
                    "export RAPTOR_API_VERSION=\"${GPT54_API_VERSION}\"",
                ]
            ),
            encoding="utf-8",
        )

        with patch.dict("os.environ", {}, clear=True):
            load_env_file(env_path)

            self.assertEqual(
                os.environ["RAPTOR_AZURE_ENDPOINT"],
                "https://example.test/gpt/openapi/online/v2/crawl",
            )
            self.assertEqual(os.environ["RAPTOR_API_VERSION"], "2023-07-01-preview")

    def test_load_default_env_file_uses_configured_env_path(self) -> None:
        env_path = Path(self.temp_dir.name) / ".env.local"
        env_path.write_text(
            "\n".join(
                [
                    "SAM_EMBEDDING_PROVIDER=azure_openai_sdk",
                    "SAM_AZURE_EMBEDDING_API_KEY=local-secret",
                    "SAM_AZURE_EMBEDDING_ENDPOINT=https://example.test/gpt/openapi/online/v2/crawl",
                    "SAM_AZURE_EMBEDDING_MODEL=text-embedding-3-large",
                ]
            ),
            encoding="utf-8",
        )

        with patch.dict(
            "os.environ",
            {
                "SAM_ENV_FILE": str(env_path),
            },
            clear=True,
        ), patch("importlib.util.find_spec", return_value=object()):
            loaded = load_default_env_file()
            status = inspect_embedding_provider_config()

        self.assertTrue(loaded["SAM_AZURE_EMBEDDING_API_KEY"])
        self.assertEqual(status["provider"], "azure_openai_sdk")
        self.assertTrue(status["ready"])
        self.assertNotIn("local-secret", json.dumps(status))

    def test_load_default_env_file_can_be_disabled(self) -> None:
        env_path = Path(self.temp_dir.name) / ".env.local"
        env_path.write_text("SAM_EMBEDDING_PROVIDER=azure_openai_sdk\n", encoding="utf-8")

        with patch.dict(
            "os.environ",
            {
                "SAM_ENV_FILE": str(env_path),
                "SAM_AUTO_LOAD_ENV": "0",
            },
            clear=True,
        ):
            loaded = load_default_env_file()
            status = inspect_embedding_provider_config()

        self.assertEqual(loaded, {})
        self.assertEqual(status["provider"], "local")
        self.assertTrue(status["ready"])

    def test_create_env_template_writes_placeholders_without_overwrite(self) -> None:
        env_path = Path(self.temp_dir.name) / ".env.local"

        write_env_template(env_path)
        content = env_path.read_text(encoding="utf-8")

        self.assertIn("replace-with-embedding-api-key", content)
        self.assertIn("replace-with-chat-api-key", content)
        self.assertNotIn("real-embedding-api-key", content)
        self.assertNotIn("real-chat-api-key", content)
        with self.assertRaises(FileExistsError):
            write_env_template(env_path)

    def test_opening_plan_audit_writes_progress_reports(self) -> None:
        root = Path(self.temp_dir.name) / "repo"
        (root / "src/sam").mkdir(parents=True)
        (root / "outputs/runs/fair_ablation_hotpotqa_300").mkdir(parents=True)
        (root / "src/sam/models.py").write_text("# model", encoding="utf-8")
        (root / "src/sam/store.py").write_text("# store", encoding="utf-8")
        (root / "src/sam/graph.py").write_text("# graph", encoding="utf-8")
        (root / "src/sam/retriever.py").write_text("# retriever", encoding="utf-8")
        (root / "src/sam/reranker.py").write_text("# reranker", encoding="utf-8")
        (root / "outputs/runs/fair_ablation_hotpotqa_300/ablation_metrics.json").write_text(
            json.dumps({"sam_full": {"evidence_recall": 0.6, "answer_hit_rate": 0.5}}),
            encoding="utf-8",
        )

        audit = build_opening_plan_audit(root)
        output_dir = Path(self.temp_dir.name) / "audit_docs"
        json_path, markdown_path = write_opening_plan_audit(audit, output_dir)

        self.assertEqual(audit["module_count"], 5)
        self.assertTrue(json_path.exists())
        self.assertTrue(markdown_path.exists())
        self.assertIn("SAM 开题计划进度审计", markdown_path.read_text(encoding="utf-8"))
        associative = [
            module for module in audit["modules"]
            if module["module_id"] == "associative_retrieval"
        ][0]
        self.assertTrue(any(item["exists"] for item in associative["experiment_evidence"]))

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

    def test_cached_embedding_provider_reports_embedding_progress(self) -> None:
        class BatchEmbeddingProvider(EmbeddingProvider):
            @property
            def cache_namespace(self) -> str:
                return "batch-progress"

            def __init__(self) -> None:
                self.batches: list[list[str]] = []

            def embed(self, text: str) -> list[float]:
                return self.embed_many([text])[0]

            def embed_many(self, texts: list[str]) -> list[list[float]]:
                self.batches.append(list(texts))
                return [[float(len(text))] for text in texts]

        events: list[tuple[str, int | None]] = []

        def fake_progress(iterable, total=None, desc="", enabled=True, progress_factory=None):
            events.append((desc, total))
            return iterable

        inner = BatchEmbeddingProvider()
        provider = CachedEmbeddingProvider(inner, Path(self.temp_dir.name) / "embedding_cache.sqlite")
        with patch("sam.embedding.progress_iter", side_effect=fake_progress):
            with patch.dict("os.environ", {"SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE": "2"}, clear=False):
                embeddings = provider.embed_many(["alpha", "beta", "alpha", "gamma"])
        provider.close()

        self.assertEqual(embeddings[0], embeddings[2])
        self.assertEqual(inner.batches, [["alpha", "beta"], ["gamma"]])
        self.assertIn(("检查embedding缓存", 4), events)
        self.assertIn(("生成缺失embedding", 2), events)

    def test_documents_to_nodes_reports_node_build_progress(self) -> None:
        class FixedEmbeddingProvider(EmbeddingProvider):
            def embed(self, text: str) -> list[float]:
                return [0.0]

            def embed_many(self, texts: list[str]) -> list[list[float]]:
                return [[float(index)] for index, _text in enumerate(texts)]

        events: list[tuple[str, int | None]] = []

        def fake_progress(iterable, total=None, desc="", enabled=True, progress_factory=None):
            events.append((desc, total))
            return iterable

        documents = [
            DatasetDocument(
                id="doc1",
                dataset="unit",
                title="第一段",
                text="alpha text",
                source="test",
                tags=[],
                keywords=[],
            ),
            DatasetDocument(
                id="doc2",
                dataset="unit",
                title="第二段",
                text="beta text",
                source="test",
                tags=[],
                keywords=[],
            ),
        ]
        with patch("sam.datasets.progress_iter", side_effect=fake_progress):
            nodes = documents_to_nodes(documents, FixedEmbeddingProvider())

        self.assertEqual(len(nodes), 2)
        self.assertIn(("构建MemoryNode", 2), events)

    def test_cached_embedding_provider_flushes_successes_before_later_failure(self) -> None:
        class FailingEmbeddingProvider(EmbeddingProvider):
            @property
            def cache_namespace(self) -> str:
                return "failing-online"

            def __init__(self) -> None:
                self.calls = 0

            def embed(self, text: str) -> list[float]:
                return self.embed_many([text])[0]

            def embed_many(self, texts: list[str]) -> list[list[float]]:
                embeddings: list[list[float]] = []
                for text in texts:
                    self.calls += 1
                    if text == "gamma":
                        raise RuntimeError("qpm limit")
                    embeddings.append([float(self.calls), float(len(text))])
                return embeddings

        cache_path = Path(self.temp_dir.name) / "embedding_cache.sqlite"
        provider = CachedEmbeddingProvider(FailingEmbeddingProvider(), cache_path)
        with patch.dict("os.environ", {"SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE": "1"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "qpm limit"):
                provider.embed_many(["alpha", "beta", "gamma"])
        provider.close()

        with sqlite3.connect(cache_path) as connection:
            cached_rows = connection.execute(
                "SELECT text_sha1 FROM embedding_cache ORDER BY text_sha1"
            ).fetchall()

        expected_cached = {
            hashlib.sha1("alpha".encode("utf-8")).hexdigest(),
            hashlib.sha1("beta".encode("utf-8")).hexdigest(),
        }
        self.assertEqual({str(row[0]) for row in cached_rows}, expected_cached)

    def test_embedding_run_plan_counts_dataset_texts_and_cache_hits(self) -> None:
        dataset_path = Path(self.temp_dir.name) / "sample.json"
        documents, queries = load_builtin_benchmark_sample()
        save_sam_dataset(
            dataset_path,
            documents=documents[:3],
            queries=queries[:1],
            dataset_info={"name": "unit"},
            processing={"source_script": "test"},
        )
        cache_path = Path(self.temp_dir.name) / "embedding_cache.sqlite"
        provider = CachedEmbeddingProvider(LocalHashEmbeddingProvider(), cache_path)
        provider.embed_many(
            [
                f"{documents[0].title}\n{documents[0].text}",
                f"{documents[1].title}\n{documents[1].text}",
            ]
        )
        provider.close()

        plan = build_embedding_run_plan(
            dataset_path=dataset_path,
            provider_name="local",
            cache_path=cache_path,
            batch_size=2,
        )

        self.assertEqual(plan["document_text_count"], 3)
        self.assertEqual(plan["summary_text_count"], 1)
        self.assertEqual(plan["unique_text_count"], 4)
        self.assertEqual(plan["cache_hit_count"], 2)
        self.assertEqual(plan["cache_miss_count"], 2)
        self.assertEqual(plan["estimated_batch_count"], 1)
        self.assertEqual(plan["will_call_provider"], True)

    def test_embedding_run_plan_can_include_runtime_query_and_raptor_texts(self) -> None:
        dataset_path = Path(self.temp_dir.name) / "runtime_sample.json"
        documents = [
            DatasetDocument(
                id="d1",
                dataset="unit",
                title="Alpha",
                text="Alpha evidence text",
                source="unit",
                tags=[],
                keywords=["alpha", "evidence"],
                metadata={"query_id": "q1", "entities": ["EntityA"]},
            ),
            DatasetDocument(
                id="d2",
                dataset="unit",
                title="Beta",
                text="Beta evidence text",
                source="unit",
                tags=[],
                keywords=["beta", "evidence"],
                metadata={"query_id": "q1", "entities": ["EntityB"]},
            ),
            DatasetDocument(
                id="d3",
                dataset="unit",
                title="Gamma",
                text="Gamma evidence text",
                source="unit",
                tags=[],
                keywords=["gamma", "evidence"],
                metadata={"query_id": "q2", "entities": ["EntityA"]},
            ),
        ]
        queries = [
            EvaluationQuery(
                id="q1",
                dataset="unit",
                question="Question one",
                answer="Alpha",
                supporting_doc_ids=["d1"],
                candidate_doc_ids=["d1", "d2"],
            ),
            EvaluationQuery(
                id="q2",
                dataset="unit",
                question="Question two",
                answer="Gamma",
                supporting_doc_ids=["d3"],
                candidate_doc_ids=["d3"],
            ),
        ]
        save_sam_dataset(
            dataset_path,
            documents=documents,
            queries=queries,
            dataset_info={"name": "unit"},
            processing={"source_script": "test"},
        )

        plan = build_embedding_run_plan(
            dataset_path=dataset_path,
            provider_name="local",
            include_query_summaries=False,
            include_query_texts=True,
            include_raptor_summaries=True,
        )

        self.assertEqual(plan["document_text_count"], 3)
        self.assertEqual(plan["query_text_count"], 2)
        self.assertEqual(plan["raptor_summary_text_count"], 3)
        self.assertEqual(plan["unique_text_count"], 8)

    def test_embedding_run_plan_supports_sentence_transformer_namespace(self) -> None:
        dataset_path = Path(self.temp_dir.name) / "sample.json"
        cache_path = Path(self.temp_dir.name) / "sentence_transformers_cache.sqlite"
        documents, queries = load_builtin_benchmark_sample()
        save_sam_dataset(
            dataset_path,
            documents=documents[:2],
            queries=queries[:1],
            dataset_info={"name": "unit"},
            processing={"source_script": "test"},
        )
        with sqlite3.connect(cache_path) as connection:
            connection.execute(
                """
                CREATE TABLE embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    text_sha1 TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

        with patch.dict(
            "os.environ",
            {
                "SAM_SENTENCE_TRANSFORMER_MODEL": "/models/qwen3",
                "SAM_SENTENCE_TRANSFORMER_DEVICE": "cpu",
                "SAM_SENTENCE_TRANSFORMER_BATCH_SIZE": "2",
                "SAM_SENTENCE_TRANSFORMER_NORMALIZE": "1",
            },
            clear=True,
        ):
            plan = build_embedding_run_plan(
                dataset_path=dataset_path,
                provider_name="sentence_transformers",
                cache_path=cache_path,
            )

        self.assertEqual(plan["batch_size"], 2)
        self.assertEqual(plan["cache_namespace_mode"], "exact")
        self.assertEqual(plan["estimated_batch_count"], 2)

    def test_embedding_run_plan_matches_azure_sdk_input_mode_namespace(self) -> None:
        class FakeAzureNamespaceEmbeddingProvider(EmbeddingProvider):
            @property
            def cache_namespace(self) -> str:
                return (
                    "azure_sdk:https://example.test/gpt/openapi/online/v2/crawl:"
                    "2023-07-01-preview:text-embedding-3-large:1024:single"
                )

            def embed(self, text: str) -> list[float]:
                return [1.0, float(len(text))]

        dataset_path = Path(self.temp_dir.name) / "sample.json"
        cache_path = Path(self.temp_dir.name) / "azure_sdk_cache.sqlite"
        documents, queries = load_builtin_benchmark_sample()
        save_sam_dataset(
            dataset_path,
            documents=documents[:1],
            queries=queries[:1],
            dataset_info={"name": "unit"},
            processing={"source_script": "test"},
        )
        cached = CachedEmbeddingProvider(FakeAzureNamespaceEmbeddingProvider(), cache_path)
        cached.embed_many([f"{documents[0].title}\n{documents[0].text}"])
        cached.close()

        with patch.dict(
            "os.environ",
            {
                "SAM_AZURE_EMBEDDING_ENDPOINT": "https://example.test/gpt/openapi/online/v2/crawl",
                "SAM_AZURE_EMBEDDING_API_VERSION": "2023-07-01-preview",
                "SAM_AZURE_EMBEDDING_MODEL": "text-embedding-3-large",
                "SAM_AZURE_EMBEDDING_DIMENSIONS": "1024",
                "SAM_AZURE_EMBEDDING_INPUT_MODE": "single",
            },
            clear=True,
        ):
            plan = build_embedding_run_plan(
                dataset_path=dataset_path,
                provider_name="azure_openai_sdk",
                cache_path=cache_path,
                include_query_summaries=False,
            )

        self.assertEqual(plan["cache_hit_count"], 1)
        self.assertEqual(plan["cache_miss_count"], 0)

    def test_warm_embedding_cache_writes_missing_dataset_vectors(self) -> None:
        dataset_path = Path(self.temp_dir.name) / "sample.json"
        documents, queries = load_builtin_benchmark_sample()
        save_sam_dataset(
            dataset_path,
            documents=documents[:2],
            queries=queries[:1],
            dataset_info={"name": "unit"},
            processing={"source_script": "test"},
        )
        cache_path = Path(self.temp_dir.name) / "embedding_cache.sqlite"

        first = warm_embedding_cache(
            dataset_path=dataset_path,
            provider_name="local",
            cache_path=cache_path,
            batch_size=2,
        )
        second = warm_embedding_cache(
            dataset_path=dataset_path,
            provider_name="local",
            cache_path=cache_path,
            batch_size=2,
        )

        self.assertEqual(first["before"]["cache_miss_count"], 3)
        self.assertEqual(first["warmed_text_count"], 3)
        self.assertEqual(first["after"]["cache_miss_count"], 0)
        self.assertEqual(second["before"]["cache_miss_count"], 0)
        self.assertEqual(second["warmed_text_count"], 0)

    def test_warm_embedding_cache_respects_max_texts_and_resumes(self) -> None:
        dataset_path = Path(self.temp_dir.name) / "sample.json"
        documents, queries = load_builtin_benchmark_sample()
        save_sam_dataset(
            dataset_path,
            documents=documents[:3],
            queries=queries[:1],
            dataset_info={"name": "unit"},
            processing={"source_script": "test"},
        )
        cache_path = Path(self.temp_dir.name) / "embedding_cache.sqlite"

        first = warm_embedding_cache(
            dataset_path=dataset_path,
            provider_name="local",
            cache_path=cache_path,
            include_query_summaries=False,
            max_texts=2,
        )
        second = warm_embedding_cache(
            dataset_path=dataset_path,
            provider_name="local",
            cache_path=cache_path,
            include_query_summaries=False,
            max_texts=2,
        )

        self.assertEqual(first["before"]["cache_miss_count"], 3)
        self.assertEqual(first["requested_text_count"], 2)
        self.assertEqual(first["warmed_text_count"], 2)
        self.assertEqual(first["after"]["cache_miss_count"], 1)
        self.assertEqual(second["before"]["cache_miss_count"], 1)
        self.assertEqual(second["requested_text_count"], 1)
        self.assertEqual(second["warmed_text_count"], 1)
        self.assertEqual(second["after"]["cache_miss_count"], 0)

    def test_warm_embedding_cache_ignores_global_cache_env_when_creating_inner_provider(self) -> None:
        dataset_path = Path(self.temp_dir.name) / "sample.json"
        documents, queries = load_builtin_benchmark_sample()
        save_sam_dataset(
            dataset_path,
            documents=documents[:2],
            queries=queries[:1],
            dataset_info={"name": "unit"},
            processing={"source_script": "test"},
        )
        target_cache_path = Path(self.temp_dir.name) / "target_embedding_cache.sqlite"
        unrelated_env_cache_path = Path(self.temp_dir.name) / "env_embedding_cache.sqlite"

        with patch.dict(
            "os.environ",
            {"SAM_EMBEDDING_CACHE_PATH": str(unrelated_env_cache_path)},
            clear=False,
        ):
            result = warm_embedding_cache(
                dataset_path=dataset_path,
                provider_name="local",
                cache_path=target_cache_path,
                include_query_summaries=False,
            )

        self.assertEqual(result["warmed_text_count"], 2)
        self.assertEqual(result["after"]["cache_hit_count"], 2)
        self.assertEqual(result["after"]["cache_miss_count"], 0)

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
        relation_pattern = engine.relation_pattern_for_case("case_path_match")
        matches = engine.retrieve_cases(
            "bridge evidence should use a shared entity and then a keyword bridge",
            top_k=2,
            relation_pattern=["shared_entity", "keyword_overlap"],
        )

        self.assertEqual(relation_pattern, ["shared_entity", "keyword_overlap"])
        self.assertEqual(matches[0].case_id, "case_path_match")
        self.assertGreater(matches[0].metadata["path_pattern_score"], 0.0)
        self.assertEqual(
            matches[0].metadata["matched_relation_path"],
            ["shared_entity", "keyword_overlap"],
        )
        self.assertGreater(matches[0].metadata["relation_path_count"], 0)
        self.assertEqual(
            matches[0].metadata["longest_relation_path"],
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

    def test_memory_consolidator_preserves_structural_evidence_without_support_hit(self) -> None:
        query = self.queries[0]
        support_node_ids = {
            node.id
            for node in self.store.get_nodes()
            if node.metadata.get("original_doc_id") in query.supporting_doc_ids
        }
        non_support_hits = [
            RetrievalHit(
                node=node,
                score=0.7,
                similarity_score=0.6,
                graph_score=0.4,
                usage_score=0.0,
                confidence_score=node.confidence,
                path=[node.id],
                reason="单元测试结构证据",
                metadata={"candidate_path_count": 3},
            )
            for node in self.store.get_nodes()
            if node.id not in support_node_ids
            and node.metadata.get("node_type") != "query_summary"
        ][:2]
        self.assertTrue(non_support_hits)

        record = MemoryConsolidator(self.store, self.embedding).consolidate_query(
            query=query,
            mode="sam_full",
            hits=non_support_hits,
            support_node_ids=support_node_ids,
            answer_status="insufficient_evidence",
        )

        self.assertIsNotNone(record)
        assert record is not None
        consolidated = self.store.get_node(record.node_id)
        self.assertIsNotNone(consolidated)
        assert consolidated is not None
        self.assertEqual(consolidated.metadata["consolidation_source"], "structural_activation")
        self.assertEqual(consolidated.metadata["support_node_ids"], [])
        self.assertEqual(
            consolidated.metadata["evidence_node_ids"],
            [hit.node.id for hit in non_support_hits],
        )

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
        self.assertEqual(result["source_case_hit_count"], 1)
        self.assertEqual(result["consolidated_case_hit_count"], 1)
        self.assertEqual(result["support_overlap_hit_count"], 1)
        self.assertEqual(result["source_case_hit_rate"], 1.0)
        self.assertIn("structure_match_hit_count", result)
        self.assertIn("bad_case_counts", result)
        self.assertTrue(result["cases"][0]["top_match"]["is_consolidated_case"])
        self.assertTrue(result["cases"][0]["source_case_hit"])
        self.assertIn("bad_case_type", result["cases"][0])
        self.assertIn("path_pattern_score", result["cases"][0]["top_match"])
        self.assertTrue(result["successful_cases"])
        self.assertEqual(result["successful_cases"][0]["bad_case_type"], "success")
        self.assertEqual(result["failed_cases"], [])
        self.assertIn("bad_case_explanation", result["cases"][0])

        _, markdown_path = write_analogy_reuse_reports(
            output_dir=Path(self.temp_dir.name) / "analogy_report",
            result=result,
        )
        markdown = markdown_path.read_text(encoding="utf-8")
        self.assertIn("## 类比命中案例", markdown)
        self.assertIn("## 类比失败案例", markdown)
        self.assertIn("本次未发现失败案例", markdown)

    def test_analogy_reuse_report_includes_failed_case_explanation(self) -> None:
        result = {
            "query_count": 1,
            "source_case_hit_count": 0,
            "source_case_hit_rate": 0.0,
            "consolidated_case_hit_count": 0,
            "consolidated_case_hit_rate": 0.0,
            "support_overlap_hit_count": 0,
            "support_overlap_hit_rate": 0.0,
            "structure_pattern_available_count": 1,
            "structure_match_hit_count": 0,
            "structure_match_hit_rate": 0.0,
            "average_top_match_score": 0.42,
            "bad_case_counts": {"wrong_case": 1},
            "successful_cases": [],
            "failed_cases": [
                {
                    "query_id": "q_probe",
                    "bad_case_type": "wrong_case",
                    "bad_case_explanation": "Top-1 历史案例不是当前来源案例。",
                    "top_match": {"case_id": "q_other"},
                }
            ],
            "cases": [],
        }

        _, markdown_path = write_analogy_reuse_reports(
            output_dir=Path(self.temp_dir.name) / "analogy_failed_report",
            result=result,
        )

        markdown = markdown_path.read_text(encoding="utf-8")
        self.assertIn("## 类比命中案例", markdown)
        self.assertIn("| 无 | 无 | 0.000 | 无 | 无 | 无 |", markdown)
        self.assertIn("## 类比失败案例", markdown)
        self.assertIn("q_probe", markdown)
        self.assertIn("wrong_case", markdown)
        self.assertIn("Top-1 历史案例不是当前来源案例。", markdown)

    def test_sam_with_analogy_reuses_consolidated_support_as_retrieval_signal(self) -> None:
        self.store.reset()
        now = utc_now_iso()

        def node(
            node_id: str,
            case_id: str,
            text: str,
            *,
            node_type: str = "document",
            support_ids: list[str] | None = None,
        ) -> MemoryNode:
            return MemoryNode(
                id=node_id,
                text=text,
                summary=text,
                keywords=text.lower().split()[:10],
                tags=[node_type],
                source="unit-test",
                created_at=now,
                last_accessed_at=None,
                usage_count=0,
                confidence=0.86,
                embedding=self.embedding.embed(text),
                metadata={
                    "query_id": case_id,
                    "title": node_id,
                    "node_type": node_type,
                    "support_node_ids": support_ids or [],
                    "answer": "answer-token" if support_ids else None,
                    "support_titles": ["old_support"] if support_ids else [],
                },
            )

        old_bridge = node(
            "old_bridge",
            "old_case",
            "river festival bridge evidence connects a person to an office role",
        )
        old_support = node(
            "old_support",
            "old_case",
            "archived support paragraph contains answer-token for the office role",
        )
        consolidated = node(
            "old_consolidated",
            "old_case",
            "successful old case: river festival bridge evidence used archived support",
            node_type="consolidated_memory",
            support_ids=["old_support"],
        )
        current_seed = node(
            "current_seed",
            "new_case",
            "river festival bridge question asks which office role was held",
        )
        distractor = node(
            "distractor",
            "new_case",
            "river festival bridge unrelated background without the archived answer",
        )
        self.store.upsert_nodes([old_bridge, old_support, consolidated, current_seed, distractor])
        self.store.upsert_edges(
            [
                MemoryEdge(
                    source_id="old_consolidated",
                    target_id="old_support",
                    relation_type="consolidates_support",
                    weight=0.9,
                    reason="历史成功案例的支持证据",
                    created_at=now,
                    updated_at=now,
                    activation_count=2,
                    last_activated_at=now,
                )
            ]
        )
        retriever = Retriever(self.store, self.embedding, self.graph)

        hits = retriever.retrieve(
            "river festival bridge asks for the office role",
            mode="sam_with_analogy",
            top_k=3,
            seed_k=1,
            hops=1,
            candidate_doc_ids=[
                "old_bridge",
                "old_support",
                "old_consolidated",
                "current_seed",
                "distractor",
            ],
        )

        support_hit = next(hit for hit in hits if hit.node.id == "old_support")
        self.assertEqual(support_hit.metadata["analogy_case_id"], "old_case")
        self.assertEqual(support_hit.metadata["analogy_support_node_id"], "old_support")
        self.assertIn("类比案例", support_hit.reason)
        self.assertIn("analogy_component", support_hit.metadata["score_breakdown"])

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

    def test_shared_memory_coordinator_resolves_conflicting_handoffs_with_versions(self) -> None:
        coordinator = SharedMemoryCoordinator(self.store, self.embedding)
        weak = coordinator.write_handoff(
            source_agent_id="planner",
            target_agent_id="writer",
            text="候选答案应写成 Alpha，但证据链不完整。",
            session_id="s-conflict",
            task_id="task-conflict",
            confidence=0.55,
        )
        strong = coordinator.write_handoff(
            source_agent_id="retriever",
            target_agent_id="writer",
            text="候选答案应写成 Beta，证据链覆盖两个 supporting facts。",
            session_id="s-conflict",
            task_id="task-conflict",
            confidence=0.92,
        )

        resolution = coordinator.resolve_conflict(
            resolver_agent_id="verifier",
            session_id="s-conflict",
            task_id="task-conflict",
            topic="final_answer",
            candidate_node_ids=[weak.node_id, strong.node_id],
        )

        resolution_node = self.store.get_nodes([resolution.node_id])[0]
        self.assertEqual(resolution_node.metadata["node_type"], "agent_conflict_resolution")
        self.assertEqual(resolution_node.metadata["selected_node_id"], strong.node_id)
        self.assertEqual(resolution_node.metadata["rejected_node_ids"], [weak.node_id])
        self.assertEqual(resolution_node.metadata["memory_version"], 3)

        updated = {
            node.id: node
            for node in self.store.get_nodes([weak.node_id, strong.node_id])
        }
        self.assertEqual(updated[strong.node_id].metadata["conflict_status"], "selected")
        self.assertEqual(updated[weak.node_id].metadata["conflict_status"], "rejected")
        self.assertEqual(
            updated[weak.node_id].metadata["resolved_by_node_id"],
            resolution.node_id,
        )

        metrics = coordinator.collaboration_metrics(
            session_id="s-conflict",
            task_id="task-conflict",
        )
        self.assertEqual(metrics["handoff_count"], 2)
        self.assertEqual(metrics["conflict_resolution_count"], 1)
        self.assertEqual(metrics["max_memory_version"], 3)
        self.assertEqual(metrics["participating_agent_count"], 3)

    def test_shared_memory_query_filters_rejected_and_latest_versions(self) -> None:
        coordinator = SharedMemoryCoordinator(self.store, self.embedding)
        old_writer = coordinator.write_handoff(
            source_agent_id="retriever",
            target_agent_id="writer",
            text="旧版本证据交接：候选答案是 Alpha。",
            session_id="s-version",
            task_id="task-version",
            confidence=0.7,
        )
        new_writer = coordinator.write_handoff(
            source_agent_id="retriever",
            target_agent_id="writer",
            text="新版本证据交接：候选答案是 Beta，证据更完整。",
            session_id="s-version",
            task_id="task-version",
            confidence=0.86,
        )
        verifier = coordinator.write_handoff(
            source_agent_id="writer",
            target_agent_id="verifier",
            text="writer 输出 Beta。",
            session_id="s-version",
            task_id="task-version",
            confidence=0.82,
        )
        coordinator.resolve_conflict(
            resolver_agent_id="verifier",
            session_id="s-version",
            task_id="task-version",
            topic="candidate_answer",
            candidate_node_ids=[old_writer.node_id, new_writer.node_id],
        )

        writer_hits = coordinator.query_memory(
            "候选答案是什么？",
            layers={"session"},
            session_id="s-version",
            task_id="task-version",
            include_other_sessions=False,
            agent_id="writer",
            source_agent_id="retriever",
            latest_version_only=True,
        )

        self.assertEqual([hit.id for hit in writer_hits], [new_writer.node_id])
        self.assertEqual(writer_hits[0].metadata["conflict_status"], "selected")

        rejected_hits = coordinator.query_memory(
            "Alpha",
            layers={"session"},
            session_id="s-version",
            task_id="task-version",
            include_other_sessions=False,
            agent_id="writer",
            conflict_statuses={"rejected"},
            include_rejected=True,
        )

        self.assertEqual([hit.id for hit in rejected_hits], [old_writer.node_id])

        verifier_hits = coordinator.query_memory(
            "writer 输出",
            layers={"session"},
            session_id="s-version",
            task_id="task-version",
            include_other_sessions=False,
            agent_id="verifier",
        )

        self.assertEqual([hit.id for hit in verifier_hits], [verifier.node_id])

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
        self.assertEqual(result["collaboration_metrics"]["handoff_count"], 2)
        self.assertEqual(result["collaboration_metrics"]["max_memory_version"], 4)
        self.assertEqual(result["collaboration_metrics"]["participating_agent_count"], 4)
        output_dir = Path(self.temp_dir.name) / "agent_workflow"
        json_path, markdown_path = write_agent_workflow_reports([result], output_dir)
        self.assertTrue(json_path.exists())
        self.assertTrue(markdown_path.exists())
        markdown = markdown_path.read_text(encoding="utf-8")
        self.assertIn("Handoff 数", markdown)
        self.assertIn("最大版本", markdown)

    def test_multi_agent_workflow_auto_resolves_failed_writer_conflict(self) -> None:
        class WrongAnswerChatClient(ChatClient):
            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                return "wrong-answer"

        case = {
            "query_id": "workflow_conflict_case",
            "question": "Which answer is identified by the retriever handoff?",
            "answer": "author",
            "support_hits_by_method": {"sam_full": 1},
            "final_answers": {"sam_full": {"status": "found_in_retrieved_context"}},
            "methods": {
                "sam_full": [
                    {
                        "title": "Workflow evidence",
                        "text": "The retriever handoff says the answer is author.",
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
            generator=ContextAnswerGenerator(WrongAnswerChatClient()),
            method="sam_full",
        )

        result = workflow.run_case(case)

        self.assertEqual(result["verifier"]["status"], "failed")
        self.assertTrue(result["conflict_resolution_node_ids"])
        self.assertEqual(result["collaboration_metrics"]["conflict_resolution_count"], 1)
        self.assertEqual(result["collaboration_metrics"]["max_memory_version"], 5)
        resolution = self.store.get_nodes(result["conflict_resolution_node_ids"])[0]
        self.assertEqual(resolution.metadata["node_type"], "agent_conflict_resolution")
        self.assertEqual(resolution.metadata["topic"], "answer_generation")

    def test_agent_workflow_audit_detects_rejected_memory_contamination(self) -> None:
        results = [
            {
                "query_id": "clean_case",
                "verifier": {"status": "passed"},
                "writer_memory": [
                    {"node_id": "selected_mem", "conflict_status": "selected"},
                ],
                "verifier_memory": [],
                "collaboration_metrics": {
                    "memory_count": 4,
                    "handoff_count": 2,
                    "conflict_resolution_count": 1,
                    "max_memory_version": 4,
                },
            },
            {
                "query_id": "contaminated_case",
                "verifier": {"status": "failed"},
                "writer_memory": [
                    {"node_id": "rejected_mem", "conflict_status": "rejected"},
                ],
                "verifier_memory": [
                    {"node_id": "verifier_rejected", "conflict_status": "rejected"},
                ],
                "collaboration_metrics": {
                    "memory_count": 5,
                    "handoff_count": 2,
                    "conflict_resolution_count": 1,
                    "max_memory_version": 5,
                },
            },
        ]

        audit = agent_workflow.audit_agent_workflow_results(results)

        self.assertEqual(audit["summary"]["case_count"], 2)
        self.assertEqual(audit["summary"]["passed_count"], 1)
        self.assertEqual(audit["summary"]["rejected_memory_used_count"], 2)
        self.assertEqual(audit["summary"]["contaminated_case_count"], 1)
        contaminated = audit["cases"][1]
        self.assertEqual(contaminated["query_id"], "contaminated_case")
        self.assertEqual(
            contaminated["rejected_memory_node_ids"],
            ["rejected_mem", "verifier_rejected"],
        )

        output_dir = Path(self.temp_dir.name) / "agent_workflow_audit"
        json_path, markdown_path = agent_workflow.write_agent_workflow_audit(audit, output_dir)
        self.assertTrue(json_path.exists())
        self.assertIn("共享记忆污染案例数", markdown_path.read_text(encoding="utf-8"))

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
                        "text": "The current bridge evidence identifies an author.",
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
        self.assertGreaterEqual(comparison["variants"]["shared_memory"]["context_answer_hit_count"], 1)
        self.assertGreaterEqual(
            comparison["variants"]["shared_memory"]["average_supplemental_context_count"],
            1.0,
        )
        self.assertEqual(comparison["delta"]["shared_memory_vs_baseline_answer_hits"], 1)
        self.assertEqual(comparison["case_deltas"][0]["shared_memory_status"], "improved")
        self.assertTrue(comparison["answers"]["shared_memory_with_analogy"][0]["metadata"]["analogy_hints"])

        output_dir = Path(self.temp_dir.name) / "agent_generation"
        json_path, markdown_path = write_agent_generation_comparison_reports(comparison, output_dir)
        self.assertTrue(json_path.exists())
        self.assertTrue(markdown_path.exists())
        self.assertTrue((output_dir / "generation_bad_cases" / "generation_bad_cases.json").exists())
        self.assertIn("上下文含答案数", markdown_path.read_text(encoding="utf-8"))

    def test_agent_generation_variants_capture_generation_errors(self) -> None:
        class RateLimitedChatClient(ChatClient):
            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                raise RuntimeError("HTTP 429 from https://example.test/chat/completions")

        case = {
            "query_id": "rate_limit_case",
            "question": "Who held the role?",
            "answer": "Chief of Protocol",
            "support_hits_by_method": {"sam_full": 1},
            "methods": {
                "sam_full": [
                    {
                        "title": "Evidence",
                        "text": "Shirley Temple served as Chief of Protocol.",
                        "reason": "向量种子节点",
                    }
                ]
            },
        }
        generator = ContextAnswerGenerator(RateLimitedChatClient())
        workflow = MultiAgentResearchWorkflow(
            coordinator=SharedMemoryCoordinator(self.store, self.embedding),
            generator=generator,
            method="sam_full",
        )

        comparison = compare_agent_generation_variants(
            [case],
            workflow=workflow,
            generator=generator,
            method="sam_full",
        )

        self.assertEqual(comparison["query_count"], 1)
        baseline_answer = comparison["answers"]["baseline"][0]
        self.assertEqual(baseline_answer["metadata"]["generation_error"]["type"], "RuntimeError")
        self.assertEqual(comparison["variants"]["baseline"]["answer_hit_count"], 0)
        rendered = json.dumps(comparison, ensure_ascii=False)
        self.assertNotIn("https://example.test", rendered)

        output_dir = Path(self.temp_dir.name) / "agent_generation_error"
        json_path, markdown_path = write_agent_generation_comparison_reports(comparison, output_dir)
        self.assertTrue(json_path.exists())
        self.assertTrue(markdown_path.exists())
        bad_case_path = output_dir / "generation_bad_cases" / "generation_bad_cases.json"
        self.assertTrue(bad_case_path.exists())
        bad_cases = json.loads(bad_case_path.read_text(encoding="utf-8"))
        self.assertEqual(bad_cases[0]["categories"], ["generation_error"])

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
                    "text": "Evidence text contains gold string.",
                    "path": ["n1"],
                }
            ],
        )

        self.assertTrue(answer.answer_hit)
        self.assertEqual(answer.metadata["answer_judgment"]["status"], "llm_equivalent")
        self.assertEqual(answer.metadata["answer_judgment"]["metadata"]["judge"], "fixed")

    def test_context_answer_generator_requires_retrieved_context_grounding(self) -> None:
        class GoldFromMemoryChatClient(ChatClient):
            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                return "最终答案：Chief of Protocol。依据：模型记忆。"

        generator = ContextAnswerGenerator(GoldFromMemoryChatClient())
        answer = generator.generate_from_hits(
            query_id="grounding-case",
            method="sam_full",
            question="What government position was held?",
            gold_answer="Chief of Protocol",
            hits=[
                {
                    "title": "Kiss and Tell",
                    "text": "Kiss and Tell stars Shirley Temple as Corliss Archer.",
                    "path": ["n1"],
                }
            ],
        )

        self.assertFalse(answer.answer_hit)
        self.assertTrue(answer.metadata["answer_judgment"]["answer_hit"])
        self.assertFalse(answer.metadata["context_answer_judgment"]["answer_hit"])
        self.assertTrue(answer.metadata["ungrounded_answer_hit"])

    def test_context_answer_generator_grounds_on_shared_memory_context(self) -> None:
        class SharedMemoryChatClient(ChatClient):
            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                return "最终答案：Chief of Protocol。依据：共享记忆上下文。"

        generator = ContextAnswerGenerator(SharedMemoryChatClient())
        answer = generator.generate_from_hits(
            query_id="shared-grounding-case",
            method="sam_full",
            question="What government position was held?",
            gold_answer="Chief of Protocol",
            hits=[
                {
                    "title": "Kiss and Tell",
                    "text": "Kiss and Tell stars Shirley Temple as Corliss Archer.",
                    "path": ["n1"],
                }
            ],
            supplemental_contexts=[
                {
                    "node_id": "agent-memory-1",
                    "title": "共享记忆:retriever:session",
                    "text": "Retriever handoff states that Shirley Temple later served as Chief of Protocol.",
                    "reason": "multi_agent_shared_memory",
                }
            ],
        )

        self.assertTrue(answer.answer_hit)
        self.assertTrue(answer.metadata["context_answer_judgment"]["answer_hit"])
        self.assertEqual(answer.metadata["supplemental_context_count"], 1)
        self.assertIn("共享记忆:retriever:session", answer.context_titles)

    def test_context_answer_generator_prompt_requires_direct_answer_extraction(self) -> None:
        captured: dict[str, object] = {}

        class CapturingChatClient(ChatClient):
            def complete(self, messages: list[dict[str, object]], max_tokens: int = 500) -> str:
                captured["messages"] = messages
                return "最终答案：Greenwich Village, New York City。依据：[1]"

        generator = ContextAnswerGenerator(CapturingChatClient())
        answer = generator.generate_from_hits(
            query_id="prompt-case",
            method="sam_full",
            question="Where is the director based?",
            gold_answer="Greenwich Village, New York City",
            hits=[
                {
                    "title": "Adriana Trigiani",
                    "text": "Adriana Trigiani is based in Greenwich Village, New York City.",
                    "path": ["n1"],
                }
            ],
        )

        messages = captured["messages"]
        assert isinstance(messages, list)
        rendered = json.dumps(messages, ensure_ascii=False)
        self.assertIn("最短、最直接的答案短语", rendered)
        self.assertIn("不要因为证据分散就过早回答", rendered)
        self.assertIn("最终答案：<最短答案短语>", rendered)
        self.assertTrue(answer.answer_hit)

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

    def test_heuristic_chat_client_extracts_answer_from_context(self) -> None:
        answer = HeuristicChatClient().complete(
            [
                {
                    "role": "user",
                    "content": (
                        "问题：Which city is identified?\n\n"
                        "上下文：\n"
                        "[1] Evidence\n"
                        "The answer is Shanghai.\n\n"
                        "请输出最终答案。"
                    ),
                }
            ]
        )

        self.assertEqual(answer, "Shanghai")

    def test_heuristic_chat_client_extracts_short_fact_from_context(self) -> None:
        answer = HeuristicChatClient().complete(
            [
                {
                    "role": "user",
                    "content": (
                        "问题：Which city is identified?\n\n"
                        "上下文：\n"
                        "[1] Evidence\n"
                        "The university is located in Shanghai. The city is Shanghai.\n\n"
                        "请输出最终答案。"
                    ),
                }
            ]
        )

        self.assertEqual(answer, "Shanghai")

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

    def test_generation_badcase_analyzer_distinguishes_missing_context_answer(self) -> None:
        answers = [
            {
                "query_id": "gen-q2",
                "method": "sam_full",
                "question": "What role did she hold?",
                "gold_answer": "Chief of Protocol",
                "generated_answer": "United States Ambassador to Ghana",
                "answer_hit": False,
                "context_titles": ["Kiss and Tell (1945 film)"],
                "metadata": {
                    "answer_judgment": {
                        "answer_hit": False,
                        "status": "not_matched",
                        "score": 0.0,
                    },
                    "context_answer_judgment": {
                        "answer_hit": False,
                        "status": "not_matched",
                        "score": 0.0,
                    },
                },
            }
        ]

        bad_cases = GenerationBadCaseAnalyzer().analyze(answers)

        self.assertEqual(len(bad_cases), 1)
        self.assertIn("retrieval_context_missing_answer", bad_cases[0].categories)
        self.assertNotIn("context_available_but_generation_failed", bad_cases[0].categories)

    def test_generation_badcase_analyzer_flags_ungrounded_generated_answer(self) -> None:
        answers = [
            {
                "query_id": "gen-q3",
                "method": "sam_full",
                "question": "What role did she hold?",
                "gold_answer": "Chief of Protocol",
                "generated_answer": "Chief of Protocol",
                "answer_hit": False,
                "context_titles": ["Kiss and Tell (1945 film)"],
                "metadata": {
                    "ungrounded_answer_hit": True,
                    "answer_judgment": {
                        "answer_hit": True,
                        "status": "exact_or_substring_match",
                        "score": 1.0,
                    },
                    "context_answer_judgment": {
                        "answer_hit": False,
                        "status": "not_matched",
                        "score": 0.0,
                    },
                },
            }
        ]

        bad_cases = GenerationBadCaseAnalyzer().analyze(answers)

        self.assertEqual(len(bad_cases), 1)
        self.assertIn("ungrounded_generated_answer", bad_cases[0].categories)
        self.assertNotIn("generated_answer_not_equivalent", bad_cases[0].categories)
        self.assertIn("外部知识", bad_cases[0].diagnosis)
        self.assertIn("context_answer_judgment", bad_cases[0].metadata)

    def test_retrieval_generation_pipeline_writes_end_to_end_outputs(self) -> None:
        documents, queries = load_builtin_benchmark_sample()
        output_dir = Path(self.temp_dir.name) / "pipeline"

        summary = run_retrieval_generation_pipeline(
            documents=documents,
            queries=queries[:2],
            output_dir=output_dir,
            embedding_provider=self.embedding,
            chat_client=HeuristicChatClient(),
            answer_judge=RuleBasedAnswerJudge(),
            retrieval_methods=["embedding_topk", "sam_full"],
            generation_method="sam_full",
            top_k=2,
            seed_k=1,
            hops=2,
        )

        self.assertEqual(summary["query_count"], 2)
        self.assertIn("retrieval", summary)
        self.assertIn("generation", summary)
        self.assertIn("relation_judge", summary)
        self.assertFalse(summary["relation_judge"]["enabled"])
        self.assertTrue((output_dir / "metrics.json").exists())
        self.assertTrue((output_dir / "cases.json").exists())
        self.assertTrue((output_dir / "generated_answers.json").exists())
        self.assertTrue((output_dir / "generation_bad_cases.json").exists())
        self.assertTrue((output_dir / "relation_judge_usage.json").exists())
        self.assertTrue((output_dir / "pipeline_summary.json").exists())
        generated = json.loads((output_dir / "generated_answers.json").read_text(encoding="utf-8"))
        self.assertIn("context_answer_judgment", generated[0]["metadata"])

    def test_retrieval_generation_pipeline_passes_query_planner_and_reranker_profile(self) -> None:
        class FixedQueryPlanner:
            def plan(self, query: EvaluationQuery) -> QueryPlan:
                return QueryPlan(
                    retrieval_query=f"{query.question} planned bridge evidence",
                    keywords=["planned", "bridge"],
                    entities=[],
                    reason="端到端测试规划器",
                    metadata={"planner": "fixed_pipeline"},
                )

        documents, queries = load_builtin_benchmark_sample()
        output_dir = Path(self.temp_dir.name) / "pipeline_advanced"

        summary = run_retrieval_generation_pipeline(
            documents=documents,
            queries=queries[:1],
            output_dir=output_dir,
            embedding_provider=self.embedding,
            chat_client=HeuristicChatClient(),
            answer_judge=RuleBasedAnswerJudge(),
            retrieval_methods=["sam_full"],
            generation_method="sam_full",
            top_k=2,
            seed_k=1,
            hops=2,
            query_planner=FixedQueryPlanner(),
            reranker_profile="graph_heavy",
        )

        cases = json.loads((output_dir / "cases.json").read_text(encoding="utf-8"))
        sam_hits = cases[0]["methods"]["sam_full"]
        self.assertEqual(summary["reranker_profile"], "graph_heavy")
        self.assertEqual(cases[0]["query_plan"]["metadata"]["planner"], "fixed_pipeline")
        self.assertTrue(all(hit["reranker_profile"] == "graph_heavy" for hit in sam_hits))

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

    def test_scifact_adapter_reads_official_jsonl_directory(self) -> None:
        source_root = Path(self.temp_dir.name) / "scifact"
        source_root.mkdir(parents=True)
        (source_root / "corpus.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "doc_id": 1001,
                            "title": "Cancer immunotherapy response",
                            "abstract": [
                                "T cell activation is associated with durable immunotherapy response.",
                                "The cohort included patients with melanoma.",
                            ],
                            "structured": False,
                        }
                    ),
                    json.dumps(
                        {
                            "doc_id": 1002,
                            "title": "Unrelated enzyme study",
                            "abstract": ["The enzyme regulates metabolic flux in yeast cells."],
                            "structured": False,
                        }
                    ),
                    json.dumps(
                        {
                            "doc_id": 1003,
                            "title": "Melanoma checkpoint blockade",
                            "abstract": ["Checkpoint blockade improves survival in selected melanoma patients."],
                            "structured": False,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (source_root / "claims_dev.jsonl").write_text(
            json.dumps(
                {
                    "id": 7,
                    "claim": "T cell activation is linked to durable response in melanoma immunotherapy.",
                    "evidence": {
                        "1001": [
                            {
                                "label": "SUPPORT",
                                "sentences": [0],
                            }
                        ]
                    },
                    "cited_doc_ids": [1001, 1003],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        documents, queries, manifest = load_scifact_sample(
            source_root,
            split="dev",
            sample_size=1,
            negative_docs_per_query=1,
            max_corpus_docs=3,
        )

        self.assertEqual(len(queries), 1)
        self.assertGreaterEqual(len(documents), 2)
        self.assertEqual(queries[0].id, "scifact_claim_7")
        self.assertEqual(queries[0].supporting_doc_ids, ["scifact_doc_1001"])
        self.assertIn("scifact_doc_1003", queries[0].candidate_doc_ids)
        self.assertEqual(queries[0].metadata["claim_id"], 7)
        self.assertEqual(queries[0].metadata["evidence_labels"]["scifact_doc_1001"], "SUPPORT")
        support_doc = [document for document in documents if document.id == "scifact_doc_1001"][0]
        self.assertEqual(support_doc.metadata["source_id"], "scifact_doc_1001")
        self.assertEqual(support_doc.metadata["rationale_sentence_indices"], [0])
        self.assertIn("T cell activation", support_doc.metadata["rationale_text"])
        self.assertEqual(manifest["split"], "dev")


if __name__ == "__main__":
    unittest.main()
