from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from sam.embedding import EmbeddingProvider
from sam.models import DatasetDocument, EvaluationQuery, MemoryNode, utc_now_iso
from sam.text import extract_keywords, stable_id


DATASET_REFERENCES = {
    "multihop_rag": {
        "name": "MultiHop-RAG",
        "homepage": "https://github.com/yixuantt/MultiHop-RAG",
        "note": "跨文档、多跳 RAG 评测数据集，证据分布在 2 到 4 篇文档中。",
    },
    "hotpotqa": {
        "name": "HotpotQA",
        "homepage": "https://hotpotqa.github.io/",
        "note": "经典多跳问答数据集，提供 supporting facts。",
    },
    "musique": {
        "name": "MuSiQue",
        "homepage": "https://github.com/stonybrooknlp/musique",
        "note": "通过单跳问题组合构造的多跳问答数据集。",
    },
}


def load_builtin_benchmark_sample() -> tuple[list[DatasetDocument], list[EvaluationQuery]]:
    """加载公开基准结构兼容的小样本。

    当前仓库不能假设网络和第三方 datasets 包可用，所以先内置一份
    “公开多跳问答基准风格”的极小样本，字段设计对齐 MultiHop-RAG/HotpotQA/MuSiQue：
    每个问题有候选文档、答案、支持文档集合。后续接入真实下载器时不用改评测流程。
    """

    raw_cases = [
        {
            "id": "mh_local_001",
            "dataset": "multihop_rag",
            "question": "Which city hosts the university where the researcher who introduced Graphiti-style temporal memory studied?",
            "answer": "Shanghai",
            "supporting_doc_ids": ["mh_local_001_doc_a", "mh_local_001_doc_b"],
            "documents": [
                {
                    "id": "mh_local_001_doc_a",
                    "title": "Temporal memory researcher profile",
                    "text": "Lin Chen introduced a Graphiti-style temporal memory prototype for agent systems. The profile notes that Lin Chen studied at Fudan University before working on dynamic memory graphs.",
                    "keywords": ["graphiti", "temporal", "memory", "lin", "chen", "fudan"],
                    "entities": ["Lin Chen", "Graphiti", "Fudan University"],
                },
                {
                    "id": "mh_local_001_doc_b",
                    "title": "Fudan University location",
                    "text": "Fudan University is a major research university located in Shanghai. The university is frequently referenced in computer science research collaborations.",
                    "keywords": ["fudan", "university", "shanghai", "research"],
                    "entities": ["Fudan University", "Shanghai"],
                },
                {
                    "id": "mh_local_001_doc_c",
                    "title": "Temporal databases overview",
                    "text": "Temporal databases record facts over time and often use validity intervals, transaction time, and historical snapshots.",
                    "keywords": ["temporal", "database", "time", "snapshot"],
                    "entities": ["Temporal Database"],
                },
                {
                    "id": "mh_local_001_doc_d",
                    "title": "Agent planning systems",
                    "text": "Agent planning systems decompose user goals into actions, tool calls, and intermediate checkpoints.",
                    "keywords": ["agent", "planning", "tool", "checkpoint"],
                    "entities": ["Agent Planning"],
                },
            ],
        },
        {
            "id": "mh_local_002",
            "dataset": "musique",
            "question": "What ability is evaluated by the benchmark associated with the dataset composed from single-hop questions?",
            "answer": "multi-hop reasoning",
            "supporting_doc_ids": ["mh_local_002_doc_a", "mh_local_002_doc_b"],
            "documents": [
                {
                    "id": "mh_local_002_doc_a",
                    "title": "MuSiQue construction",
                    "text": "MuSiQue constructs complex questions by composing connected single-hop questions. The design is intended to reduce shortcut solving in question answering.",
                    "keywords": ["musique", "single-hop", "composition", "question", "answering"],
                    "entities": ["MuSiQue", "single-hop question composition", "multi-hop reasoning"],
                },
                {
                    "id": "mh_local_002_doc_b",
                    "title": "Multi-hop reasoning benchmark",
                    "text": "This benchmark evaluates multi-hop reasoning: whether a system can connect multiple supporting facts instead of answering from one isolated passage.",
                    "keywords": ["multi-hop", "reasoning", "benchmark", "supporting", "facts"],
                    "entities": ["multi-hop reasoning", "supporting facts"],
                },
                {
                    "id": "mh_local_002_doc_c",
                    "title": "Summarization benchmark",
                    "text": "Summarization benchmarks evaluate whether a system can compress a long document while preserving key points.",
                    "keywords": ["summarization", "benchmark", "compress", "document"],
                    "entities": ["summarization"],
                },
                {
                    "id": "mh_local_002_doc_d",
                    "title": "Dialogue safety dataset",
                    "text": "Dialogue safety datasets focus on detecting harmful instructions, policy violations, and unsafe responses.",
                    "keywords": ["dialogue", "safety", "policy", "responses"],
                    "entities": ["dialogue safety"],
                },
            ],
        },
        {
            "id": "mh_local_003",
            "dataset": "hotpotqa",
            "question": "What evidence-chain problem is addressed by the architecture inspired by the brain structure used in long-term memory?",
            "answer": "multi-hop retrieval",
            "supporting_doc_ids": ["mh_local_003_doc_a", "mh_local_003_doc_b"],
            "documents": [
                {
                    "id": "mh_local_003_doc_a",
                    "title": "HippoRAG inspiration",
                    "text": "HippoRAG is inspired by hippocampal and neocortical memory organization. It combines graph traversal with retrieval-augmented generation to address multi-hop retrieval.",
                    "keywords": ["hipporag", "hippocampal", "neocortical", "graph"],
                    "entities": ["HippoRAG", "hippocampus", "neocortex", "multi-hop retrieval"],
                },
                {
                    "id": "mh_local_003_doc_b",
                    "title": "Multi-hop retrieval challenge",
                    "text": "Multi-hop retrieval requires finding several connected pieces of evidence. Plain top-k vector search often misses one part of the evidence chain.",
                    "keywords": ["multi-hop", "retrieval", "evidence", "chain", "vector"],
                    "entities": ["multi-hop retrieval", "evidence chain"],
                },
                {
                    "id": "mh_local_003_doc_c",
                    "title": "Long context window",
                    "text": "Long context models can place many passages in a prompt, but they may still suffer from lost-in-the-middle behavior.",
                    "keywords": ["long", "context", "prompt", "lost"],
                    "entities": ["long context", "lost in the middle"],
                },
                {
                    "id": "mh_local_003_doc_d",
                    "title": "Static keyword index",
                    "text": "A static keyword index maps words to documents, but it does not explicitly represent semantic paths between memories.",
                    "keywords": ["keyword", "index", "static", "documents"],
                    "entities": ["keyword index"],
                },
            ],
        },
    ]

    documents: list[DatasetDocument] = []
    queries: list[EvaluationQuery] = []
    for case in raw_cases:
        candidate_ids: list[str] = []
        for document in case["documents"]:
            candidate_ids.append(document["id"])
            documents.append(
                DatasetDocument(
                    id=document["id"],
                    dataset=case["dataset"],
                    title=document["title"],
                    text=document["text"],
                    source=DATASET_REFERENCES[case["dataset"]]["name"],
                    tags=[case["dataset"], "benchmark_sample"],
                    keywords=document["keywords"],
                    metadata={
                        "query_id": case["id"],
                        "title": document["title"],
                        "entities": document["entities"],
                        "dataset_reference": DATASET_REFERENCES[case["dataset"]],
                    },
                )
            )
        queries.append(
            EvaluationQuery(
                id=case["id"],
                dataset=case["dataset"],
                question=case["question"],
                answer=case["answer"],
                supporting_doc_ids=case["supporting_doc_ids"],
                candidate_doc_ids=candidate_ids,
            )
        )
    return documents, queries


def load_multihop_rag_from_huggingface(cache_path: str | Path) -> dict[str, Any]:
    """尝试下载 MultiHop-RAG 的 Hugging Face 数据集元信息。

    第一版不把下载作为强依赖；如果网络不可用，调用方应回退到内置样本。
    """

    url = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/raw/main/README.md"
    target = Path(cache_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=20) as response:
        content = response.read().decode("utf-8")
    target.write_text(content, encoding="utf-8")
    return {"url": url, "cache_path": str(target), "bytes": len(content.encode("utf-8"))}


def documents_to_nodes(
    documents: list[DatasetDocument],
    embedding_provider: EmbeddingProvider,
) -> list[MemoryNode]:
    nodes: list[MemoryNode] = []
    for document in documents:
        text = f"{document.title}\n{document.text}"
        keywords = document.keywords or extract_keywords(text)
        node_id = stable_id("mem", document.id)
        nodes.append(
            MemoryNode(
                id=node_id,
                text=document.text,
                summary=document.text[:180],
                keywords=keywords,
                tags=document.tags,
                source=document.source,
                created_at=utc_now_iso(),
                usage_count=0,
                confidence=0.86,
                embedding=embedding_provider.embed(text),
                metadata={
                    **document.metadata,
                    "original_doc_id": document.id,
                    "dataset": document.dataset,
                    "title": document.title,
                },
            )
        )
    return nodes


def write_dataset_manifest(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(DATASET_REFERENCES, ensure_ascii=False, indent=2), encoding="utf-8")
