# Semantic Associative Memory KG

Local repository for the master's thesis project:

**基于语义联想机制的动态知识图谱记忆系统方法与实现**

## Project Understanding

This project aims to build a dynamic knowledge-graph memory system for LLM agents. The central idea is to move beyond plain RAG-style chunk retrieval by representing past information as memory nodes and semantic relations, then using associative retrieval to reactivate relevant prior knowledge when a new task arrives.

The opening report defines four core research questions:

1. How to construct and update a dynamic knowledge graph that stores semantic relations between memories.
2. How to design semantic activation and recall so relevant memories can be retrieved from a growing memory space.
3. How to support shared semantic memory and coordination among multiple agents.
4. How to trigger analogy-based reasoning by retrieving structurally or contextually similar past cases.

For the first runnable milestone, the implementation should stay intentionally small:

1. Store each new piece of knowledge as a memory node with text, summary, keywords, tags, source, timestamp, usage count, and embedding.
2. Build semantic edges on demand instead of constructing the full graph upfront.
3. Retrieve in two stages: vector similarity first, then graph-based associative expansion.
4. Rank memories by relevance, graph proximity, confidence, freshness, and usage frequency.
5. Expose simple read/write APIs that can later be used by Reading Agent, Summary Agent, and other research agents.

## Practical Notes From Review Feedback

The opening defense raised two important implementation risks:

1. Graph construction may be expensive. A practical MVP should use lazy or on-demand graph construction: only frequently accessed, repeatedly retrieved, or high-value memories should receive richer semantic links.
2. Stacked memory modules must prove useful. The system should track usage frequency and retrieval contribution, then use these signals to adjust retrieval priority and evaluate whether the memory hierarchy improves performance.

## Initial Milestone

Before the midterm assessment, the goal is to produce an early but demonstrable result rather than a complete thesis system:

1. A minimal local memory store.
2. Basic memory ingestion and node representation.
3. Embedding-based candidate retrieval.
4. Lightweight graph expansion over semantic edges.
5. A small experiment showing that associative retrieval can recover related cross-document information better than plain top-k vector retrieval in at least one toy or small literature-reading scenario.

No implementation code has been added yet.

