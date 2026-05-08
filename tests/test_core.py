from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.datasets import load_builtin_benchmark_sample, load_novelqa_sample
from sam.dataset_format import load_sam_dataset, save_sam_dataset, summarize_sam_dataset
from sam.embedding import LocalHashEmbeddingProvider
from sam.evaluator import Evaluator
from sam.graph import GraphBuilder
from sam.retriever import Retriever
from sam.store import MemoryStore


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

    def test_edges_are_created_on_demand(self) -> None:
        seed = self.store.get_nodes([self.nodes[0].id])
        edges = self.graph.build_edges_on_demand(seed)
        self.assertTrue(edges)
        self.assertTrue(any(edge.reason for edge in edges))

    def test_vector_and_associative_retrieval_return_hits(self) -> None:
        query = self.queries[0]
        retriever = Retriever(self.store, self.embedding, self.graph)
        candidate_ids = [
            node.id
            for node in self.store.get_nodes()
            if node.metadata["original_doc_id"] in query.candidate_doc_ids
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

    def test_evaluation_produces_gain(self) -> None:
        result = self.evaluator.evaluate(self.queries, top_k=2, seed_k=1, hops=2)
        self.assertGreaterEqual(result.associative_recall, result.vector_recall)
        self.assertGreaterEqual(result.associative_gain, 1)

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
        self.assertEqual(manifest["selected_books"][0]["book_id"], "B00")


if __name__ == "__main__":
    unittest.main()
