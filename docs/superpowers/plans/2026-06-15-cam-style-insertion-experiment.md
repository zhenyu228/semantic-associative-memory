# CAM Style Insertion Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CAM Figure 3(a)-style online insertion time experiment for SAM, with separate curves for low-level non-LLM insertion and high-level LLM refinement.

**Architecture:** Keep low-level graph construction in `GraphBuilder` non-LLM. Add a small `SemanticCompressor` abstraction used only for high-level refinement timing. Extend the insertion-time benchmark with CAM-style batch sizes, 512-token chunk metadata, CAM reference baselines, and a publication-style figure.

**Tech Stack:** Python, unittest, matplotlib/seaborn, existing SAM dataset format, optional Azure/OpenAI-compatible chat provider through existing `sam.llm` interfaces.

---

### Task 1: Semantic Compressor Interface

**Files:**
- Create: `src/sam/semantic_compressor.py`
- Modify: `tests/test_core.py`

- [ ] Add a failing test that verifies an extractive compressor compresses several memory nodes into one high-level summary with source node ids and token metadata.
- [ ] Implement `SemanticCompressor`, `ExtractiveSemanticCompressor`, and `CompressionResult`.
- [ ] Run: `conda run --no-capture-output -n sam python -m unittest tests.test_core.SamCoreTest.test_extractive_semantic_compressor_returns_summary_memory`

### Task 2: LLM Compressor Wrapper

**Files:**
- Modify: `src/sam/semantic_compressor.py`
- Modify: `tests/test_core.py`

- [ ] Add a failing test using a fake chat function to verify LLM compression prompt construction, result metadata, and no API dependency in tests.
- [ ] Implement `LLMSemanticCompressor` with an injectable callable so production can use GPT-5.4 while tests remain local.
- [ ] Run the targeted compressor tests.

### Task 3: CAM-Style Time Benchmark

**Files:**
- Modify: `src/sam/insertion_time_experiment.py`
- Modify: `tests/test_core.py`

- [ ] Add a failing test for `run_cam_style_insertion_benchmark` that checks batch sizes, 512-token chunk scope, CAM reference rows, `sam_online_no_llm`, and `sam_online_with_refinement`.
- [ ] Implement the benchmark by timing low-level insertion/local graph update and optional high-level compression.
- [ ] Keep reported timing fields explicit: embedding included/excluded, LLM refinement included/excluded, chunk token size.
- [ ] Run the targeted insertion benchmark tests.

### Task 4: CAM-Style Script And Figure

**Files:**
- Create: `scripts/run_cam_style_insertion_experiment.py`
- Modify: `src/sam/insertion_time_experiment.py`

- [ ] Add CLI arguments for dataset file, batch sizes `1,100,200,300,400,500`, chunk token size `512`, compressor mode `extractive|llm`, and output figure paths.
- [ ] Plot y-axis in hours and include CAM reference horizontal lines for GraphRAG/RAPTOR/CAM offline plus SAM measured curves.
- [ ] Run a small smoke command with batch sizes `1,10`.

### Task 5: Documentation, Verification, Commit

**Files:**
- Modify: `docs/evidence_rescue_experiment.md`
- Possibly add: `docs/figures/sam_cam_style_insertion_time.png`

- [ ] Run formal experiment on available long-text/abstract data.
- [ ] Update docs with the command, scope, and interpretation.
- [ ] Run `git diff --check` and `conda run --no-capture-output -n sam python -m unittest tests.test_core`.
- [ ] Commit with `feat: add cam style insertion time experiment` and push.
