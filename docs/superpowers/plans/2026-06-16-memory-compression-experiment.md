# Memory Compression Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run Experiment 2: memory reconstruction and high-level compression comparison, directly following Experiment 1's finding that dense low-level graph expansion has diminishing returns and high noise.

**Architecture:** Extend the existing `sam.insight_experiment` module instead of creating a parallel experiment stack. The experiment fixes warmup retrieval and consolidated memories, then compares no reconstruction, flat memory, keyword clustering, embedding clustering, and SAM hybrid reconstruction using compression, traceability, trace noise, and retrieval-budget metrics.

**Tech Stack:** Python 3, existing SAM package, SQLite-backed `MemoryStore`, existing embedding providers, `unittest`, Markdown/JSON reports under `outputs/runs/` and durable docs under `docs/`.

---

### Task 1: Add Compression Noise And Budget Metrics

**Files:**
- Modify: `src/sam/insight_experiment.py`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Add a failing unit test for trace noise and budget metrics**

Add assertions to `test_insight_reconstruction_comparison_reports_control_methods` that each strategy includes:

```python
self.assertIn("trace_edge_count", strategies["sam_hybrid_reconstruction"])
self.assertIn("trace_edge_reduction_rate", strategies["sam_hybrid_reconstruction"])
self.assertIn("retrieval_unit_reduction_rate", strategies["sam_hybrid_reconstruction"])
self.assertIn("trace_noise_rate", strategies["sam_hybrid_reconstruction"])
self.assertIn("effective_trace_precision", strategies["sam_hybrid_reconstruction"])
self.assertGreaterEqual(strategies["sam_hybrid_reconstruction"]["retrieval_unit_reduction_rate"], 0.0)
self.assertLessEqual(strategies["sam_hybrid_reconstruction"]["trace_noise_rate"], 1.0)
```

- [ ] **Step 2: Run the targeted test and verify failure**

Run:

```bash
/Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python -m unittest tests.test_core.SamCoreTest.test_insight_reconstruction_comparison_reports_control_methods
```

Expected: FAIL because the new metric keys do not exist yet.

- [ ] **Step 3: Implement metric calculations in `_reconstruction_strategy_metrics`**

Add these metrics:

```python
raw_trace_edge_count = sum(len(node.metadata.get("evidence_node_ids", [])) for node in consolidated_nodes)
trace_edge_count = sum(len(group.evidence_node_ids) for group in groups)
retrieval_unit_reduction_rate = 1.0 - _safe_divide(reconstructed_units, consolidated_count)
trace_edge_reduction_rate = 1.0 - _safe_divide(trace_edge_count, raw_trace_edge_count)
trace_noise_count = len(grouped_evidence_ids - support_node_ids)
trace_noise_rate = _safe_divide(trace_noise_count, len(grouped_evidence_ids))
effective_trace_precision = _safe_divide(len(grouped_evidence_ids & support_node_ids), len(grouped_evidence_ids))
```

For `no_reconstruction`, keep `trace_edge_count=0`, `trace_noise_rate=0`, and `retrieval_unit_reduction_rate=0` so the method remains a null baseline.

- [ ] **Step 4: Update the quality score to include trace noise and budget reduction**

Adjust `quality` so it rewards support traceability and compression while penalizing trace noise. Keep the current score shape but include:

```python
noise_penalty = 1.0 + trace_noise_rate
quality_cost_score = quality / (time_penalty * redundancy_penalty * noise_penalty)
```

- [ ] **Step 5: Run the targeted test and verify pass**

Run the same targeted test. Expected: PASS.

### Task 2: Update Experiment 2 Markdown Report

**Files:**
- Modify: `src/sam/insight_experiment.py`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Extend report table columns**

In `_comparison_markdown`, add columns for:

```text
检索单元减少率
Trace边数
Trace边减少率
Trace噪声率
有效Trace精度
```

- [ ] **Step 2: Add a unit test assertion for new report labels**

Extend the existing markdown assertions:

```python
self.assertIn("Trace噪声率", markdown)
self.assertIn("检索单元减少率", markdown)
self.assertIn("有效Trace精度", markdown)
```

- [ ] **Step 3: Run targeted test**

Run:

```bash
/Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python -m unittest tests.test_core.SamCoreTest.test_insight_reconstruction_comparison_reports_control_methods
```

Expected: PASS.

### Task 3: Add Experiment 2 Report Document

**Files:**
- Create: `docs/experiment2_memory_compression_report.md`
- Modify: `docs/insight_reconstruction_comparison_experiment.md`

- [ ] **Step 1: Create the report skeleton**

Include:

```markdown
# 实验二：记忆重构与高层压缩有效性实验报告

## 实验目的

## 与实验一的关系

## SAM 压缩框架

## 对照方法

## 评测指标

## 实验结果

## 结论与后续工作
```

- [ ] **Step 2: Add reproducible commands**

Include smoke, HotpotQA300, QASPER30, and LitSearch30 commands using `scripts/run_insight_reconstruction_comparison.py`.

- [ ] **Step 3: Leave result sections ready for measured numbers**

Use measured result sections only after experiments run; before final commit, replace draft text with actual values from output JSON/Markdown.

### Task 4: Run Experiments

**Files:**
- Output only under `outputs/runs/` unless docs are updated with summarized results.

- [ ] **Step 1: Run local smoke**

```bash
/Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python scripts/run_insight_reconstruction_comparison.py \
  --dataset-file data/processed/hotpotqa_midterm30_sam_sample.json \
  --limit 5 \
  --embedding-provider local_hash \
  --top-k 4 \
  --seed-k 1 \
  --hops 1 \
  --run-name experiment2_smoke_local
```

Expected: script completes and writes `outputs/runs/experiment2_smoke_local/insight_reconstruction_comparison.md`.

- [ ] **Step 2: Run HotpotQA300 with cached Azure embedding**

```bash
SAM_EMBEDDING_CACHE_WRITE_BATCH_SIZE=200 /Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python scripts/run_insight_reconstruction_comparison.py \
  --dataset-file data/processed/hotpotqa_midterm300_sam_sample.json \
  --limit 300 \
  --embedding-provider azure_openai_sdk \
  --embedding-cache \
  --embedding-cache-path outputs/runs/hotpotqa300_real_embedding_cache_warmup/embedding_cache.sqlite \
  --embedding-concurrency 20 \
  --top-k 5 \
  --seed-k 2 \
  --hops 1 \
  --embedding-threshold 0.82 \
  --hybrid-threshold 0.34 \
  --run-name experiment2_hotpotqa300_real_embedding
```

Expected: script completes and reports all five strategies.

- [ ] **Step 3: Run QASPER30 and LitSearch30 if runtime is acceptable**

Use the same script with `data/processed/qasper_validation30_sam_sample.json` and `data/processed/litsearch_query30_sam_sample.json`.

### Task 5: Verify, Commit, Push

**Files:**
- Modify: `src/sam/insight_experiment.py`
- Modify: `tests/test_core.py`
- Modify/Create docs from Task 3

- [ ] **Step 1: Run targeted test**

```bash
/Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python -m unittest tests.test_core.SamCoreTest.test_insight_reconstruction_comparison_reports_control_methods
```

- [ ] **Step 2: Run full test suite**

```bash
/Users/bytedance/miniconda3/bin/conda run --no-capture-output -n sam python -m unittest tests.test_core
```

- [ ] **Step 3: Stage only source, test, docs**

Do not stage `outputs/` or `reports/graph_artifact.json` / `reports/graph_view.html`.

- [ ] **Step 4: Commit and push**

```bash
git commit -m "feat: add memory compression experiment metrics"
git push
```
