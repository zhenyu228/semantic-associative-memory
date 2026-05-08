from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam.datasets import (  # noqa: E402
    load_builtin_benchmark_sample,
    load_multihop_rag_from_huggingface,
    write_dataset_manifest,
)
from sam.dataset_format import load_sam_dataset, save_sam_dataset, summarize_sam_dataset  # noqa: E402
from sam.datasets import DATASET_REFERENCES, download_hotpotqa_dev, load_hotpotqa_real_sample  # noqa: E402
from sam.embedding import create_embedding_provider  # noqa: E402
from sam.evaluator import Evaluator  # noqa: E402
from sam.graph import GraphBuilder  # noqa: E402
from sam.store import MemoryStore  # noqa: E402
from sam.visualization import export_graph_artifacts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 SAM 两阶段检索 demo")
    parser.add_argument("--db", default="data/sam_demo.sqlite", help="SQLite 数据库路径")
    parser.add_argument("--report-dir", default="reports", help="实验报告输出目录")
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "builtin"], help="实验数据来源")
    parser.add_argument("--dataset-file", default="data/processed/hotpotqa_sam_sample.json", help="SAM 统一数据格式文件")
    parser.add_argument("--sample-size", type=int, default=8, help="真实数据集抽样数量")
    parser.add_argument("--max-scan", type=int, default=800, help="真实数据集最大扫描样本数")
    parser.add_argument("--case-index", type=int, default=None, help="HTML 页面默认聚焦的 HotpotQA 原始 index")
    parser.add_argument("--rebuild-dataset", action="store_true", help="重新生成 SAM 统一数据格式文件")
    parser.add_argument("--embedding-provider", default=None, help="local 或 openai")
    parser.add_argument("--top-k", type=int, default=4, help="最终返回文档数")
    parser.add_argument("--seed-k", type=int, default=1, help="联想检索种子节点数")
    parser.add_argument("--hops", type=int, default=2, help="图扩展跳数")
    parser.add_argument("--reset", action="store_true", help="运行前清空本地记忆库")
    parser.add_argument(
        "--try-download",
        action="store_true",
        help="尝试下载公开数据集元信息；失败后自动使用内置小样本",
    )
    return parser.parse_args()


def _query_id_for_case_index(
    dataset_payload,
    queries,
    case_index: int,
) -> str:
    selected_examples = dataset_payload.get("processing", {}).get("manifest", {}).get("selected_examples", [])
    for position, example in enumerate(selected_examples):
        if int(example.get("index")) == case_index:
            query_id = example.get("query_id")
            if query_id:
                return str(query_id)
            if position < len(queries):
                return str(queries[position].id)
            break
    available = ", ".join(str(example.get("index")) for example in selected_examples)
    raise ValueError(f"没有找到 case-index={case_index}。可用 index: {available}")


def main() -> None:
    args = parse_args()
    store = MemoryStore(ROOT / args.db)
    if args.reset:
        store.reset()

    report_dir = ROOT / args.report_dir
    write_dataset_manifest(report_dir / "dataset_references.json")
    if args.try_download:
        try:
            info = load_multihop_rag_from_huggingface(report_dir / "multihop_rag_readme.md")
            print(f"已下载公开数据集元信息：{info}")
        except Exception as exc:
            print(f"公开数据集元信息下载失败，使用内置小样本兜底：{exc}")

    embedding_provider = create_embedding_provider(args.embedding_provider)
    graph_builder = GraphBuilder(store)
    evaluator = Evaluator(store, embedding_provider, graph_builder)

    if args.dataset == "hotpotqa":
        dataset_path = ROOT / args.dataset_file
        if args.rebuild_dataset or not dataset_path.exists():
            raw_path = download_hotpotqa_dev(ROOT / "data/raw/hotpot_dev_distractor_v1.json")
            documents, queries, manifest = load_hotpotqa_real_sample(
                raw_path=raw_path,
                sample_size=args.sample_size,
                max_scan=args.max_scan,
            )
            save_sam_dataset(
                path=dataset_path,
                documents=documents,
                queries=queries,
                dataset_info=DATASET_REFERENCES["hotpotqa_real"],
                processing={
                    "source_script": "scripts/run_demo.py",
                    "raw_path": str(raw_path),
                    "sample_size": args.sample_size,
                    "max_scan": args.max_scan,
                    "selection_policy": "选择 supporting paragraph 之间存在标题提及的 bridge-style 样本",
                    "manifest": manifest,
                },
            )
        documents, queries, dataset_payload = load_sam_dataset(dataset_path)
        focus_query_id = None
        if args.case_index is not None:
            focus_query_id = _query_id_for_case_index(dataset_payload, queries, args.case_index)
        manifest = dataset_payload["processing"].get("manifest", {})
        (report_dir / "hotpotqa_sample_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"使用 SAM 数据格式文件：{dataset_path}")
        print(json.dumps(summarize_sam_dataset(dataset_path), ensure_ascii=False, indent=2))
        if args.case_index is not None:
            print(f"HTML 页面将默认聚焦 HotpotQA 原始 index={args.case_index} 对应的样本")
    else:
        documents, queries = load_builtin_benchmark_sample()
        focus_query_id = None
        print(f"使用内置兜底样本：{len(queries)} 条")

    nodes = evaluator.ingest(documents)
    result = evaluator.evaluate(
        queries=queries,
        top_k=args.top_k,
        seed_k=args.seed_k,
        hops=args.hops,
    )
    json_path, markdown_path = evaluator.write_reports(result, report_dir)
    graph_paths = export_graph_artifacts(
        nodes=nodes,
        edges=store.get_edges(),
        queries=queries,
        output_dir=report_dir,
        retrieval_cases=result.cases,
        focus_query_id=focus_query_id,
    )

    print("SAM demo 已完成")
    print(f"查询数量：{result.query_count}")
    print(f"纯向量证据召回率：{result.vector_recall:.3f}")
    print(f"联想检索证据召回率：{result.associative_recall:.3f}")
    print(f"联想检索新增有效证据数：{result.associative_gain}")
    print(f"JSON 结果：{json_path}")
    print(f"Markdown 报告：{markdown_path}")
    print(f"图谱 HTML：{graph_paths['html']}")
    print(f"图谱 JSON：{graph_paths['json']}")


if __name__ == "__main__":
    main()
