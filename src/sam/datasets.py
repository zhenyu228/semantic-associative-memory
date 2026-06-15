from __future__ import annotations

import json
import os
import tarfile
import zipfile
import urllib.request
from pathlib import Path
from typing import Any

from sam.embedding import EmbeddingProvider
from sam.models import DatasetDocument, EvaluationQuery, MemoryNode, utc_now_iso
from sam.progress import progress_iter
from sam.text import extract_keywords, stable_id


DATASET_REFERENCES = {
    "hotpotqa_real": {
        "name": "HotpotQA dev distractor",
        "homepage": "https://hotpotqa.github.io/",
        "download_url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "paper": "https://arxiv.org/abs/1809.09600",
        "license": "CC BY-SA 4.0",
        "note": "公开多跳问答数据集，提供 paragraph context 和 sentence-level supporting facts。",
    },
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
    "novelqa": {
        "name": "NovelQA",
        "homepage": "https://novelqa.github.io/",
        "huggingface": "https://huggingface.co/datasets/NovelQA/NovelQA",
        "paper": "https://arxiv.org/abs/2403.12766",
        "license": "Apache-2.0",
        "note": "长篇小说问答数据集，面向超过 200K token 的长文本理解和检索评测；原始数据需要用户同意访问条件后本地提供。",
    },
    "scifact": {
        "name": "SciFact",
        "homepage": "https://github.com/allenai/scifact",
        "download_url": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
        "paper": "https://aclanthology.org/2020.emnlp-main.609/",
        "note": "科学事实核验数据集，包含 claim、科学论文摘要语料、gold evidence abstract 和 rationale sentence，适合评估跨论文证据检索效果。",
    },
}


def download_hotpotqa_dev(raw_path: str | Path) -> Path:
    """从 HotpotQA 官方地址下载 dev distractor 数据。

    数据文件较大，默认保存到 data/raw；该目录被 gitignore 排除。
    """

    target = Path(raw_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    url = DATASET_REFERENCES["hotpotqa_real"]["download_url"]
    with urllib.request.urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())
    return target


def download_scifact_data(raw_root: str | Path) -> Path:
    """下载并解压 SciFact 官方数据。

    官方数据包含 `corpus.jsonl`、`claims_train.jsonl`、`claims_dev.jsonl`
    和 `claims_test.jsonl`。原始数据保存在 `data/raw` 下，不进入仓库。
    """

    target_root = Path(raw_root)
    target_root.mkdir(parents=True, exist_ok=True)
    if _find_scifact_file(target_root, "corpus.jsonl") and _find_scifact_file(target_root, "claims_dev.jsonl"):
        return target_root

    archive_path = target_root / "scifact_data.tar.gz"
    if not archive_path.exists() or archive_path.stat().st_size < 10_000:
        with urllib.request.urlopen(DATASET_REFERENCES["scifact"]["download_url"], timeout=120) as response:
            archive_path.write_bytes(response.read())
    _extract_tar_safely(archive_path, target_root)
    return target_root


def load_scifact_sample(
    source_path: str | Path,
    split: str = "dev",
    sample_size: int = 50,
    negative_docs_per_query: int = 20,
    max_corpus_docs: int = 0,
) -> tuple[list[DatasetDocument], list[EvaluationQuery], dict[str, Any]]:
    """把 SciFact 官方 jsonl 数据转换为 SAM 统一格式。

    评测单位是 scientific claim。支持证据来自 `evidence` 字段中的
    gold evidence abstracts；候选文档由 gold evidence、cited documents
    和基于词项重叠选出的 hard negatives 构成。
    """

    if split not in {"train", "dev", "test"}:
        raise ValueError("SciFact split 只能是 train、dev 或 test")
    source_root = Path(source_path)
    corpus_path = _find_scifact_file(source_root, "corpus.jsonl")
    claims_path = _find_scifact_file(source_root, f"claims_{split}.jsonl")
    if not corpus_path or not claims_path:
        raise FileNotFoundError(
            f"SciFact 路径缺少 corpus.jsonl 或 claims_{split}.jsonl：{source_path}"
        )

    raw_corpus = _read_jsonl(corpus_path)
    if max_corpus_docs > 0:
        raw_corpus = raw_corpus[:max_corpus_docs]
    corpus_by_id = {
        str(item["doc_id"]): item
        for item in raw_corpus
        if isinstance(item, dict) and item.get("doc_id") is not None
    }
    claim_rows = [
        item
        for item in _read_jsonl(claims_path)
        if isinstance(item, dict) and item.get("claim") and isinstance(item.get("evidence"), dict) and item.get("evidence")
    ][:sample_size]
    if not claim_rows:
        raise ValueError(f"claims_{split}.jsonl 中没有可评估的带 evidence 样本")

    selected_doc_ids: set[str] = set()
    per_query_candidates: dict[str, list[str]] = {}
    rationale_by_doc: dict[str, set[int]] = {}
    labels_by_claim: dict[str, dict[str, str]] = {}
    rationale_text_by_claim: dict[str, dict[str, list[str]]] = {}

    for claim in claim_rows:
        claim_id = str(claim["id"])
        evidence_doc_ids = [str(doc_id) for doc_id in claim.get("evidence", {}).keys()]
        cited_doc_ids = [str(doc_id) for doc_id in claim.get("cited_doc_ids", [])]
        candidate_doc_ids = _unique_existing_doc_ids([*evidence_doc_ids, *cited_doc_ids], corpus_by_id)
        candidate_doc_ids.extend(
            _select_scifact_negative_docs(
                claim_text=str(claim["claim"]),
                corpus_by_id=corpus_by_id,
                excluded=set(candidate_doc_ids),
                limit=negative_docs_per_query,
            )
        )
        candidate_doc_ids = list(dict.fromkeys(candidate_doc_ids))
        selected_doc_ids.update(candidate_doc_ids)
        query_id = f"scifact_claim_{claim_id}"
        per_query_candidates[query_id] = [f"scifact_doc_{doc_id}" for doc_id in candidate_doc_ids]
        labels_by_claim[query_id] = {}
        rationale_text_by_claim[query_id] = {}
        for doc_id, rationales in claim.get("evidence", {}).items():
            raw_doc_id = str(doc_id)
            if raw_doc_id not in corpus_by_id:
                continue
            labels = []
            sentence_indices: list[int] = []
            for rationale in rationales if isinstance(rationales, list) else []:
                if not isinstance(rationale, dict):
                    continue
                labels.append(str(rationale.get("label", "")))
                sentence_indices.extend(int(index) for index in rationale.get("sentences", []) if isinstance(index, int))
            rationale_by_doc.setdefault(raw_doc_id, set()).update(sentence_indices)
            labels_by_claim[query_id][f"scifact_doc_{raw_doc_id}"] = labels[0] if labels else ""
            rationale_text_by_claim[query_id][f"scifact_doc_{raw_doc_id}"] = _scifact_rationale_texts(
                corpus_by_id[raw_doc_id],
                sentence_indices,
            )

    documents = [
        _scifact_document(raw_doc_id, corpus_by_id[raw_doc_id], sorted(rationale_by_doc.get(raw_doc_id, set())))
        for raw_doc_id in sorted(selected_doc_ids, key=_natural_doc_sort_key)
        if raw_doc_id in corpus_by_id
    ]
    queries: list[EvaluationQuery] = []
    for claim in claim_rows:
        claim_id = str(claim["id"])
        query_id = f"scifact_claim_{claim_id}"
        supporting_doc_ids = [
            f"scifact_doc_{doc_id}"
            for doc_id in claim.get("evidence", {}).keys()
            if str(doc_id) in corpus_by_id
        ]
        candidate_doc_ids = per_query_candidates.get(query_id, [])
        if not supporting_doc_ids or not candidate_doc_ids:
            continue
        evidence_labels = labels_by_claim.get(query_id, {})
        label_summary = sorted({label for label in evidence_labels.values() if label})
        queries.append(
            EvaluationQuery(
                id=query_id,
                dataset="scifact",
                question=str(claim["claim"]),
                answer=";".join(label_summary),
                supporting_doc_ids=supporting_doc_ids,
                candidate_doc_ids=candidate_doc_ids,
                metadata={
                    "claim_id": int(claim["id"]),
                    "split": split,
                    "cited_doc_ids": [f"scifact_doc_{doc_id}" for doc_id in claim.get("cited_doc_ids", [])],
                    "evidence_labels": evidence_labels,
                    "rationale_text_by_doc": rationale_text_by_claim.get(query_id, {}),
                    "retrieval_task": "scientific_claim_evidence_retrieval",
                },
            )
        )

    manifest = {
        "dataset": DATASET_REFERENCES["scifact"],
        "source_path": str(source_path),
        "split": split,
        "sample_size": len(queries),
        "document_count": len(documents),
        "negative_docs_per_query": negative_docs_per_query,
        "max_corpus_docs": max_corpus_docs,
        "selection_policy": "选择带 gold evidence 的 claim；候选集包含 evidence docs、cited docs 和词项重叠 hard negatives",
    }
    return documents, queries, manifest


def load_hotpotqa_real_sample(
    raw_path: str | Path,
    sample_size: int = 8,
    max_scan: int = 800,
) -> tuple[list[DatasetDocument], list[EvaluationQuery], dict[str, Any]]:
    """从真实 HotpotQA dev distractor 中抽取一小批桥接型样本。

    抽样原则写入 manifest：优先选择 supporting paragraph 之间存在标题提及的样本。
    这类样本最适合检验“先命中一跳证据，再沿语义边补回另一跳证据”。
    """

    raw_data = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    documents: list[DatasetDocument] = []
    queries: list[EvaluationQuery] = []
    selected: list[dict[str, Any]] = []

    for index, item in enumerate(raw_data[:max_scan]):
        support_titles = list(dict.fromkeys(title for title, _ in item["supporting_facts"]))
        if len(support_titles) < 2:
            continue
        title_to_sentences = {title: sentences for title, sentences in item["context"]}
        if any(title not in title_to_sentences for title in support_titles):
            continue

        support_text = {
            title: " ".join(title_to_sentences[title])
            for title in support_titles
        }
        has_bridge_mention = any(
            other_title.lower() in text.lower()
            for title, text in support_text.items()
            for other_title in support_titles
            if other_title != title
        )
        if not has_bridge_mention:
            continue

        query_id = f"hotpotqa_{item['_id']}"
        candidate_doc_ids: list[str] = []
        supporting_doc_ids: list[str] = []
        for paragraph_index, (title, sentences) in enumerate(item["context"]):
            doc_id = f"{query_id}_doc_{paragraph_index}"
            candidate_doc_ids.append(doc_id)
            if title in support_titles:
                supporting_doc_ids.append(doc_id)
            text = " ".join(sentences)
            entities = _extract_title_entities(title, text, [context_title for context_title, _ in item["context"]])
            documents.append(
                DatasetDocument(
                    id=doc_id,
                    dataset="hotpotqa_real",
                    title=title,
                    text=text,
                    source="HotpotQA dev distractor",
                    tags=["hotpotqa_real", item.get("type", "unknown"), item.get("level", "unknown")],
                    keywords=extract_keywords(f"{title} {text}", limit=10),
                    metadata={
                        "query_id": query_id,
                        "title": title,
                        "entities": entities,
                        "hotpotqa_id": item["_id"],
                        "paragraph_index": paragraph_index,
                        "dataset_reference": DATASET_REFERENCES["hotpotqa_real"],
                        "is_supporting": title in support_titles,
                    },
                )
            )
        queries.append(
            EvaluationQuery(
                id=query_id,
                dataset="hotpotqa_real",
                question=item["question"],
                answer=item["answer"],
                supporting_doc_ids=supporting_doc_ids,
                candidate_doc_ids=candidate_doc_ids,
            )
        )
        selected.append(
            {
                "index": index,
                "query_id": query_id,
                "hotpotqa_id": item["_id"],
                "question": item["question"],
                "answer": item["answer"],
                "type": item.get("type"),
                "level": item.get("level"),
                "support_titles": support_titles,
                "candidate_count": len(candidate_doc_ids),
                "selection_reason": "supporting paragraphs mention each other's titles, suitable for bridge-style graph expansion",
            }
        )
        if len(queries) >= sample_size:
            break

    if not queries:
        raise ValueError("未能从 HotpotQA 中抽取到符合条件的样本，请增大 max_scan")

    manifest = {
        "dataset": DATASET_REFERENCES["hotpotqa_real"],
        "raw_path": str(raw_path),
        "sample_size": len(queries),
        "max_scan": max_scan,
        "selected_examples": selected,
    }
    return documents, queries, manifest


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


def load_novelqa_sample(
    source_path: str | Path,
    sample_size: int = 8,
    max_books: int = 1,
    chunk_chars: int = 1800,
    chunk_overlap: int = 180,
    max_chunks_per_book: int = 80,
    split: str = "data",
) -> tuple[list[DatasetDocument], list[EvaluationQuery], dict[str, Any]]:
    """把本地 NovelQA zip 或目录转换成 SAM 统一数据结构。

    NovelQA 是 gated dataset，本函数只读取用户本地已经获得的数据，不负责登录或下载。
    """

    reader = _NovelQAReader(Path(source_path))
    if split not in {"data", "demonstration"}:
        raise ValueError("NovelQA split 只能是 data 或 demonstration")
    if split == "demonstration":
        qa_files = reader.list_files("Demonstration/", ".json")
    else:
        qa_files = reader.list_files("Data/", ".json")
        if not qa_files:
            qa_files = reader.list_files("Demonstration/", ".json")
    selected_books: list[dict[str, Any]] = []
    documents: list[DatasetDocument] = []
    queries: list[EvaluationQuery] = []

    for qa_file in qa_files:
        if len(selected_books) >= max_books or len(queries) >= sample_size:
            break
        book_id = Path(qa_file).stem
        book_file = reader.find_book_file(book_id, split=split)
        if not book_file:
            continue
        book_text = reader.read_text(book_file)
        qa_payload = reader.read_json(qa_file)
        qa_items = _normalize_novelqa_items(qa_payload)
        if not qa_items:
            continue

        chunks = _chunk_text(book_text, chunk_chars=chunk_chars, overlap=chunk_overlap)[:max_chunks_per_book]
        candidate_doc_ids: list[str] = []
        chunk_by_doc_id: dict[str, str] = {}
        for chunk_index, chunk in enumerate(chunks):
            doc_id = f"novelqa_{book_id}_chunk_{chunk_index:04d}"
            candidate_doc_ids.append(doc_id)
            chunk_by_doc_id[doc_id] = chunk
            title = f"{book_id} chunk {chunk_index:04d}"
            documents.append(
                DatasetDocument(
                    id=doc_id,
                    dataset="novelqa",
                    title=title,
                    text=chunk,
                    source="NovelQA",
                    tags=["novelqa", "long_context", "novel_chunk"],
                    keywords=extract_keywords(f"{title} {chunk}", limit=12),
                    metadata={
                        "book_id": book_id,
                        "chunk_index": chunk_index,
                        "dataset_reference": DATASET_REFERENCES["novelqa"],
                        "title": title,
                    },
                )
            )

        added_questions = 0
        for item in qa_items:
            if len(queries) >= sample_size:
                break
            if not isinstance(item, dict) or "Question" not in item:
                continue
            qid = str(item.get("QID") or item.get("qid") or f"Q{len(queries):04d}")
            options = dict(item.get("Options") or item.get("options") or {})
            answer = _extract_novelqa_answer(item)
            supporting_doc_ids = _match_evidence_chunks(item.get("Evidences"), chunk_by_doc_id)
            queries.append(
                EvaluationQuery(
                    id=f"novelqa_{book_id}_{qid}",
                    dataset="novelqa",
                    question=str(item["Question"]),
                    answer=answer,
                    supporting_doc_ids=supporting_doc_ids,
                    candidate_doc_ids=list(candidate_doc_ids),
                    metadata={
                        "book_id": book_id,
                        "qid": qid,
                        "aspect": item.get("Aspect") or item.get("aspect"),
                        "complexity": item.get("Complexity") or item.get("Complex") or item.get("complexity"),
                        "options": options,
                        "gold": item.get("Gold") or item.get("gold"),
                        "evidence_count": len(item.get("Evidences") or []),
                        "raw_answer": item.get("Answer") or item.get("answer"),
                        "retrieval_query": _novelqa_retrieval_query(item, options),
                        "source_file": qa_file,
                        "book_file": book_file,
                        "note": "NovelQA 公开格式通常不提供可直接映射到 chunk 的 gold evidence，本阶段主要评估答案/选项覆盖。",
                    },
                )
            )
            added_questions += 1

        selected_books.append(
            {
                "book_id": book_id,
                "book_file": book_file,
                "qa_file": qa_file,
                "chunk_count": len(chunks),
                "question_count": added_questions,
            }
        )

    if not queries:
        raise ValueError("没有从 NovelQA 本地路径中读取到可用问题，请确认 zip/目录结构包含 Books 和 Data/Demonstration")

    manifest = {
        "dataset": DATASET_REFERENCES["novelqa"],
        "source_path": str(source_path),
        "sample_size": len(queries),
        "max_books": max_books,
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "max_chunks_per_book": max_chunks_per_book,
        "split": split,
        "selected_books": selected_books,
    }
    return documents, queries, manifest


def documents_to_nodes(
    documents: list[DatasetDocument],
    embedding_provider: EmbeddingProvider,
) -> list[MemoryNode]:
    nodes: list[MemoryNode] = []
    texts = [f"{document.title}\n{document.text}" for document in documents]
    embeddings = embedding_provider.embed_many(texts)
    for document, text, embedding in progress_iter(
        zip(documents, texts, embeddings, strict=True),
        total=len(documents),
        desc="构建MemoryNode",
    ):
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
                last_accessed_at=None,
                usage_count=0,
                confidence=0.86,
                embedding=embedding,
                metadata={
                    **document.metadata,
                    "original_doc_id": document.id,
                    "dataset": document.dataset,
                    "title": document.title,
                },
            )
        )
    return nodes


def build_query_summary_nodes(
    document_nodes: list[MemoryNode],
    embedding_provider: EmbeddingProvider,
) -> list[MemoryNode]:
    """为每个查询上下文创建摘要记忆节点。

    摘要节点不是 gold evidence，只作为层级图中的中间记忆，帮助 SAM
    从一个种子文档跳转到同题上下文中的相关文档。
    """

    groups: dict[str, list[MemoryNode]] = {}
    for node in document_nodes:
        query_id = node.metadata.get("query_id")
        if query_id:
            groups.setdefault(str(query_id), []).append(node)

    summary_payloads: list[tuple[str, list[MemoryNode], str, list[str], list[str]]] = []
    for query_id, nodes in groups.items():
        ordered_nodes = sorted(nodes, key=lambda node: str(node.metadata.get("paragraph_index", node.id)))
        title_terms = [str(node.metadata.get("title", "")) for node in ordered_nodes]
        keyword_terms = sorted({keyword for node in ordered_nodes for keyword in node.keywords[:6]})
        summary_text = "\n".join(
            f"{node.metadata.get('title', node.id)}: {node.summary}"
            for node in ordered_nodes
        )
        text = (
            f"查询上下文摘要：{query_id}\n"
            f"候选标题：{'; '.join(title_terms)}\n"
            f"关键词：{', '.join(keyword_terms[:32])}\n"
            f"{summary_text}"
        )
        summary_payloads.append((query_id, ordered_nodes, text, title_terms, keyword_terms))

    summary_nodes: list[MemoryNode] = []
    embeddings = embedding_provider.embed_many([payload[2] for payload in summary_payloads])
    for (query_id, ordered_nodes, text, _title_terms, _keyword_terms), embedding in zip(
        summary_payloads,
        embeddings,
        strict=True,
    ):
        node_id = stable_id("summary", query_id)
        summary_nodes.append(
            MemoryNode(
                id=node_id,
                text=text,
                summary=text[:240],
                keywords=extract_keywords(text, limit=16),
                tags=["summary_memory", *sorted({tag for node in ordered_nodes for tag in node.tags})],
                source="SAM query summary memory",
                created_at=utc_now_iso(),
                last_accessed_at=None,
                usage_count=0,
                confidence=0.78,
                embedding=embedding,
                metadata={
                    "node_type": "query_summary",
                    "query_id": query_id,
                    "dataset": ordered_nodes[0].metadata.get("dataset"),
                    "title": f"Query summary: {query_id}",
                    "child_node_ids": [node.id for node in ordered_nodes],
                    "child_original_doc_ids": [
                        str(node.metadata.get("original_doc_id"))
                        for node in ordered_nodes
                        if node.metadata.get("original_doc_id")
                    ],
                    "child_titles": title_terms,
                    "summary_strategy": "title_keyword_context_summary",
                },
            )
        )
    return summary_nodes


def write_dataset_manifest(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(DATASET_REFERENCES, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_title_entities(title: str, text: str, all_titles: list[str]) -> list[str]:
    entities = {title}
    lowered = text.lower()
    for candidate in all_titles:
        if candidate != title and candidate.lower() in lowered:
            entities.add(candidate)
    return sorted(entities)


class _NovelQAReader:
    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path
        self.archive: zipfile.ZipFile | None = None
        if source_path.is_file() and source_path.suffix.lower() == ".zip":
            self.archive = zipfile.ZipFile(source_path)
        elif not source_path.exists():
            raise FileNotFoundError(f"NovelQA 路径不存在：{source_path}")

    def list_files(self, prefix: str, suffix: str) -> list[str]:
        if self.archive:
            return sorted(
                name
                for name in self.archive.namelist()
                if prefix in name and name.lower().endswith(suffix)
            )
        return sorted(
            str(path.relative_to(self.source_path))
            for path in self.source_path.rglob(f"*{suffix}")
            if prefix.rstrip("/") in str(path.relative_to(self.source_path))
        )

    def find_book_file(self, book_id: str, split: str = "data") -> str | None:
        candidates = self.list_files("Demonstration/", ".txt") if split == "demonstration" else self.list_files("Books/", ".txt")
        for candidate in candidates:
            if Path(candidate).stem == book_id:
                return candidate
        for candidate in candidates:
            if book_id in Path(candidate).stem or Path(candidate).stem in book_id:
                return candidate
        return None

    def read_text(self, name: str) -> str:
        if self.archive:
            return self.archive.read(name).decode("utf-8", errors="ignore")
        return (self.source_path / name).read_text(encoding="utf-8", errors="ignore")

    def read_json(self, name: str) -> Any:
        return json.loads(self.read_text(name))


def _chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    chunks: list[str] = []
    start = 0
    step = max(1, chunk_chars - overlap)
    while start < len(cleaned):
        chunk = cleaned[start : start + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def _extract_novelqa_answer(item: dict[str, Any]) -> str:
    for key in ["Answer", "answer", "Gold", "gold", "Label", "label"]:
        value = item.get(key)
        if value is not None:
            return str(value)
    return ""


def _novelqa_retrieval_query(item: dict[str, Any], options: dict[str, Any]) -> str:
    question = str(item.get("Question") or item.get("question") or "")
    aspect = item.get("Aspect") or item.get("aspect")
    complexity = item.get("Complexity") or item.get("Complex") or item.get("complexity")
    question_keywords = " ".join(extract_keywords(question, limit=10))
    parts = [
        question,
        f"Question keywords: {question_keywords}" if question_keywords else "",
        f"Aspect: {aspect}" if aspect else "",
        f"Complexity: {complexity}" if complexity else "",
    ]
    return " ".join(part for part in parts if part).strip()


def _normalize_novelqa_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        normalized: list[dict[str, Any]] = []
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            item = dict(value)
            item.setdefault("QID", key)
            normalized.append(item)
        return normalized
    return []


def _match_evidence_chunks(evidences: Any, chunk_by_doc_id: dict[str, str]) -> list[str]:
    if not isinstance(evidences, list):
        return []
    matched: list[str] = []
    for evidence in evidences:
        if not isinstance(evidence, dict):
            continue
        evidence_text = str(evidence.get("Evidence") or evidence.get("evidence") or "")
        if not evidence_text.strip():
            continue
        doc_id = _best_evidence_chunk(evidence_text, chunk_by_doc_id)
        if doc_id and doc_id not in matched:
            matched.append(doc_id)
    return matched


def _best_evidence_chunk(evidence_text: str, chunk_by_doc_id: dict[str, str]) -> str | None:
    normalized_evidence = " ".join(evidence_text.lower().split())
    if not normalized_evidence:
        return None
    for doc_id, chunk in chunk_by_doc_id.items():
        if normalized_evidence in " ".join(chunk.lower().split()):
            return doc_id

    evidence_keywords = set(extract_keywords(evidence_text, limit=24))
    best_doc_id: str | None = None
    best_score = 0.0
    for doc_id, chunk in chunk_by_doc_id.items():
        chunk_keywords = set(extract_keywords(chunk, limit=80))
        score = len(evidence_keywords & chunk_keywords) / max(1, len(evidence_keywords))
        if score > best_score:
            best_score = score
            best_doc_id = doc_id
    return best_doc_id if best_score >= 0.35 else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _find_scifact_file(source_root: Path, filename: str) -> Path | None:
    if source_root.is_file() and source_root.name == filename:
        return source_root
    if source_root.is_dir():
        direct = source_root / filename
        if direct.exists():
            return direct
        matches = sorted(source_root.rglob(filename))
        return matches[0] if matches else None
    return None


def _extract_tar_safely(archive_path: Path, target_root: Path) -> None:
    target_root = target_root.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_target = (target_root / member.name).resolve()
            if os.path.commonpath([str(target_root), str(member_target)]) != str(target_root):
                raise ValueError(f"压缩包包含非法路径：{member.name}")
        archive.extractall(target_root)


def _unique_existing_doc_ids(doc_ids: list[str], corpus_by_id: dict[str, dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for doc_id in doc_ids:
        normalized = str(doc_id)
        if normalized in corpus_by_id and normalized not in result:
            result.append(normalized)
    return result


def _select_scifact_negative_docs(
    *,
    claim_text: str,
    corpus_by_id: dict[str, dict[str, Any]],
    excluded: set[str],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []
    claim_terms = set(extract_keywords(claim_text, limit=32))
    scored: list[tuple[float, str]] = []
    fallback: list[str] = []
    for doc_id, item in corpus_by_id.items():
        if doc_id in excluded:
            continue
        fallback.append(doc_id)
        document_text = _scifact_document_text(item)
        document_terms = set(extract_keywords(f"{item.get('title', '')} {document_text}", limit=64))
        overlap = len(claim_terms & document_terms)
        if overlap:
            score = overlap / max(1, len(claim_terms))
            scored.append((score, doc_id))
    scored.sort(key=lambda item: (-item[0], _natural_doc_sort_key(item[1])))
    negatives = [doc_id for _score, doc_id in scored[:limit]]
    if len(negatives) >= limit:
        return negatives
    for doc_id in sorted(fallback, key=_natural_doc_sort_key):
        if doc_id not in negatives:
            negatives.append(doc_id)
        if len(negatives) >= limit:
            break
    return negatives


def _scifact_document(raw_doc_id: str, item: dict[str, Any], rationale_sentence_indices: list[int]) -> DatasetDocument:
    title = str(item.get("title") or f"SciFact document {raw_doc_id}")
    text = _scifact_document_text(item)
    rationale_text = " ".join(_scifact_rationale_texts(item, rationale_sentence_indices))
    return DatasetDocument(
        id=f"scifact_doc_{raw_doc_id}",
        dataset="scifact",
        title=title,
        text=text,
        source="SciFact",
        tags=["scifact", "scientific_claim_verification", "paper_abstract"],
        keywords=extract_keywords(f"{title} {text}", limit=16),
        metadata={
            "doc_id": raw_doc_id,
            "source_id": f"scifact_doc_{raw_doc_id}",
            "section": "abstract",
            "title": title,
            "structured": bool(item.get("structured", False)),
            "sentence_count": len(item.get("abstract") or []),
            "rationale_sentence_indices": rationale_sentence_indices,
            "rationale_text": rationale_text,
            "dataset_reference": DATASET_REFERENCES["scifact"],
        },
    )


def _scifact_document_text(item: dict[str, Any]) -> str:
    abstract = item.get("abstract") or []
    if isinstance(abstract, list):
        return " ".join(str(sentence) for sentence in abstract if str(sentence).strip())
    return str(abstract)


def _scifact_rationale_texts(item: dict[str, Any], sentence_indices: list[int]) -> list[str]:
    abstract = item.get("abstract") or []
    if not isinstance(abstract, list):
        return []
    texts: list[str] = []
    for index in sorted(set(sentence_indices)):
        if 0 <= index < len(abstract):
            texts.append(str(abstract[index]))
    return texts


def _natural_doc_sort_key(value: str) -> tuple[int, str]:
    try:
        return (int(value), value)
    except ValueError:
        return (0, value)
