from __future__ import annotations

import html
import json
from pathlib import Path

from sam.models import EvaluationQuery, MemoryEdge, MemoryNode, RetrievalHit


def export_graph_artifacts(
    nodes: list[MemoryNode],
    edges: list[MemoryEdge],
    queries: list[EvaluationQuery],
    output_dir: str | Path,
    retrieval_cases: list[dict[str, object]] | None = None,
) -> dict[str, Path]:
    """导出可检查的图谱产物：JSON、Mermaid、HTML/SVG。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    graph_json = target / "graph_artifact.json"
    mermaid_md = target / "graph_mermaid.md"
    html_path = target / "graph_view.html"

    node_payload = [_node_payload(node) for node in nodes]
    edge_payload = [_edge_payload(edge) for edge in edges]
    query_payload = [
        {
            "id": query.id,
            "dataset": query.dataset,
            "question": query.question,
            "answer": query.answer,
            "supporting_doc_ids": query.supporting_doc_ids,
            "candidate_doc_ids": query.candidate_doc_ids,
        }
        for query in queries
    ]
    graph_json.write_text(
        json.dumps(
            {
                "nodes": node_payload,
                "edges": edge_payload,
                "queries": query_payload,
                "retrieval_cases": retrieval_cases or [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    mermaid_md.write_text(_to_mermaid(node_payload, edge_payload), encoding="utf-8")
    html_path.write_text(_to_html(node_payload, edge_payload, retrieval_cases or []), encoding="utf-8")
    return {"json": graph_json, "mermaid": mermaid_md, "html": html_path}


def _node_payload(node: MemoryNode) -> dict[str, object]:
    return {
        "id": node.id,
        "title": node.metadata.get("title", node.id),
        "dataset": node.metadata.get("dataset"),
        "query_id": node.metadata.get("query_id"),
        "is_supporting": bool(node.metadata.get("is_supporting", False)),
        "usage_count": node.usage_count,
        "keywords": node.keywords,
        "entities": node.metadata.get("entities", []),
        "snippet": node.text[:240],
        "source": node.source,
    }


def _edge_payload(edge: MemoryEdge) -> dict[str, object]:
    return {
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "relation_type": edge.relation_type,
        "weight": edge.weight,
        "reason": edge.reason,
        "metadata": edge.metadata,
    }


def _to_mermaid(nodes: list[dict[str, object]], edges: list[dict[str, object]]) -> str:
    lines = ["# SAM 图谱 Mermaid 视图", "", "```mermaid", "flowchart LR"]
    for node in nodes:
        label = str(node["title"]).replace('"', "'")
        shape_left, shape_right = ("[[", "]]") if node["is_supporting"] else ("[", "]")
        lines.append(f'  {node["id"]}{shape_left}"{label}"{shape_right}')
    for edge in edges:
        relation = str(edge["relation_type"])
        weight = float(edge["weight"])
        if weight < 0.2:
            continue
        lines.append(
            f'  {edge["source_id"]} -- "{relation}:{weight:.2f}" --> {edge["target_id"]}'
        )
    lines.append("```")
    return "\n".join(lines)


def _to_html(
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
    retrieval_cases: list[dict[str, object]],
) -> str:
    by_query: dict[str, list[dict[str, object]]] = {}
    for node in nodes:
        by_query.setdefault(str(node["query_id"]), []).append(node)
    cases_by_query = {str(case["query_id"]): case for case in retrieval_cases}
    graph_blocks = "\n".join(
        _query_graph_html(
            query_id=query_id,
            query_nodes=query_nodes,
            query_edges=[
                edge
                for edge in edges
                if edge["source_id"] in {node["id"] for node in query_nodes}
                and edge["target_id"] in {node["id"] for node in query_nodes}
            ],
            case=cases_by_query.get(query_id),
        )
        for query_id, query_nodes in by_query.items()
    )

    node_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(node['title']))}</td>"
        f"<td>{html.escape(str(node['query_id']))}</td>"
        f"<td>{'是' if node['is_supporting'] else '否'}</td>"
        f"<td>{html.escape(', '.join(map(str, node['entities'])))}</td>"
        f"<td>{html.escape(str(node['snippet']))}</td>"
        "</tr>"
        for node in nodes
    )
    edge_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(edge['source_id']))}</td>"
        f"<td>{html.escape(str(edge['target_id']))}</td>"
        f"<td>{html.escape(str(edge['relation_type']))}</td>"
        f"<td>{float(edge['weight']):.3f}</td>"
        f"<td>{html.escape(str(edge['reason']))}</td>"
        "</tr>"
        for edge in edges
        if float(edge["weight"]) >= 0.2
    )
    case_blocks = "\n".join(_case_to_html(case) for case in retrieval_cases)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>SAM 图谱运行产物</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; background: #f7f9fc; }}
    h1, h2, h3 {{ margin: 18px 0 10px; }}
    .hint {{ color: #52606d; line-height: 1.6; }}
    .legend span {{ display: inline-block; margin-right: 18px; }}
    .query-card {{ background: white; border: 1px solid #d7dde5; border-radius: 8px; margin: 18px 0 26px; padding: 16px; }}
    .question {{ color: #334e68; line-height: 1.5; margin-bottom: 12px; }}
    .graph {{ overflow-x: auto; border: 1px solid #d7dde5; background: #fbfcfe; border-radius: 6px; }}
    .edge-note {{ color: #52606d; font-size: 13px; margin: 8px 0 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px; vertical-align: top; }}
    th {{ background: #f3f6f9; text-align: left; }}
    code {{ background: #eef2f7; padding: 2px 4px; }}
    .support-pill {{ color: #8a5a00; font-weight: 700; }}
    .candidate-pill {{ color: #1f5f8b; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>SAM 图谱运行产物</h1>
  <p class="hint">黄色节点是真实 HotpotQA supporting paragraph，蓝色节点是候选干扰段落。箭头表示系统实际创建的关键语义边。这个文件由代码自动生成，不是手绘示意图。</p>
  <p class="legend"><span>黄色：支持证据</span><span>蓝色：候选文档</span><span>橙色边：共享实体</span><span>蓝色边：关键词重叠</span><span>紫色边：embedding 相似</span></p>
  <h2>按问题拆分的图谱</h2>
  {graph_blocks}
  <h2>检索案例</h2>
  {case_blocks}
  <h2>节点明细</h2>
  <table><thead><tr><th>标题</th><th>问题 ID</th><th>支持证据</th><th>实体</th><th>文本片段</th></tr></thead><tbody>{node_rows}</tbody></table>
  <h2>语义边明细</h2>
  <table><thead><tr><th>起点</th><th>终点</th><th>关系</th><th>权重</th><th>原因</th></tr></thead><tbody>{edge_rows}</tbody></table>
</body>
</html>
"""


def _query_graph_html(
    query_id: str,
    query_nodes: list[dict[str, object]],
    query_edges: list[dict[str, object]],
    case: dict[str, object] | None,
) -> str:
    width = 1240
    height = 360
    node_width = 160
    node_height = 58
    support_nodes = [node for node in query_nodes if node["is_supporting"]]
    candidate_nodes = [node for node in query_nodes if not node["is_supporting"]]
    ordered_nodes = [*support_nodes, *candidate_nodes]
    positions: dict[str, tuple[int, int]] = {}
    for index, node in enumerate(ordered_nodes):
        row = 0 if index < 5 else 1
        col = index if index < 5 else index - 5
        x = 70 + col * 230
        y = 90 + row * 150
        positions[str(node["id"])] = (x, y)

    path_edges = _path_edge_keys(case)
    important_edges = _select_important_edges(query_edges, path_edges)
    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        "<defs><marker id='arrow' markerWidth='10' markerHeight='10' refX='9' refY='3' orient='auto' markerUnits='strokeWidth'><path d='M0,0 L0,6 L9,3 z' fill='#555'/></marker></defs>",
    ]
    for edge in important_edges:
        source = positions.get(str(edge["source_id"]))
        target = positions.get(str(edge["target_id"]))
        if not source or not target:
            continue
        x1, y1 = source[0] + node_width / 2, source[1] + node_height / 2
        x2, y2 = target[0] + node_width / 2, target[1] + node_height / 2
        color = _edge_color(str(edge["relation_type"]))
        stroke_width = 2.2 if _edge_key(edge) in path_edges else 1.4
        svg_parts.append(
            f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='{stroke_width}' marker-end='url(#arrow)' opacity='0.68'/>"
        )
    for node in ordered_nodes:
        x, y = positions[str(node["id"])]
        fill = "#fff2bd" if node["is_supporting"] else "#e6f2ff"
        stroke = "#bd7b00" if node["is_supporting"] else "#2f75a8"
        title = html.escape(_short_title(str(node["title"]), 36))
        role = "supporting" if node["is_supporting"] else "candidate"
        svg_parts.append(
            f"<rect x='{x}' y='{y}' width='{node_width}' height='{node_height}' rx='8' fill='{fill}' stroke='{stroke}' stroke-width='2'/>"
        )
        svg_parts.append(
            f"<foreignObject x='{x + 8}' y='{y + 8}' width='{node_width - 16}' height='28'><div xmlns='http://www.w3.org/1999/xhtml' style='font-size:12px;font-weight:700;line-height:14px;text-align:center;color:#102a43;'>{title}</div></foreignObject>"
        )
        svg_parts.append(
            f"<text x='{x + node_width / 2}' y='{y + 47}' text-anchor='middle' font-size='11'>{role}</text>"
        )
    svg_parts.append("</svg>")

    edge_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(_title_for(query_nodes, str(edge['source_id'])))}</td>"
        f"<td>{html.escape(_title_for(query_nodes, str(edge['target_id'])))}</td>"
        f"<td>{html.escape(str(edge['relation_type']))}</td>"
        f"<td>{float(edge['weight']):.3f}</td>"
        f"<td>{html.escape(str(edge['reason']))}</td>"
        "</tr>"
        for edge in important_edges
    )
    question = html.escape(str(case["question"])) if case else query_id
    answer = html.escape(str(case["answer"])) if case else ""
    return f"""
  <section class="query-card">
    <h3>{html.escape(query_id)}</h3>
    <div class="question"><b>问题：</b>{question}<br><b>答案：</b>{answer}</div>
    <div class="graph">{''.join(svg_parts)}</div>
    <p class="edge-note">为避免拥挤，此处只展示检索路径边、supporting 相关边和高权重边；完整图数据见 <code>graph_artifact.json</code>。</p>
    <table><thead><tr><th>起点</th><th>终点</th><th>关系</th><th>权重</th><th>原因</th></tr></thead><tbody>{edge_rows}</tbody></table>
  </section>
"""


def _select_important_edges(
    query_edges: list[dict[str, object]],
    path_edges: set[tuple[str, str]],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for edge in query_edges:
        if _edge_key(edge) in path_edges:
            selected.append(edge)
            continue
        if edge["relation_type"] == "shared_entity" and float(edge["weight"]) >= 0.5:
            selected.append(edge)
    if len(selected) < 8:
        remaining = [
            edge
            for edge in query_edges
            if edge not in selected and float(edge["weight"]) >= 0.45
        ]
        remaining.sort(key=lambda edge: float(edge["weight"]), reverse=True)
        selected.extend(remaining[: 8 - len(selected)])
    return selected[:12]


def _path_edge_keys(case: dict[str, object] | None) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not case:
        return keys
    for mode in ["vector", "associative"]:
        for hit in case.get(mode, []):
            path = [str(node_id) for node_id in hit.get("path", [])]
            for left, right in zip(path, path[1:], strict=False):
                keys.add((left, right))
                keys.add((right, left))
    return keys


def _edge_key(edge: dict[str, object]) -> tuple[str, str]:
    return (str(edge["source_id"]), str(edge["target_id"]))


def _title_for(nodes: list[dict[str, object]], node_id: str) -> str:
    for node in nodes:
        if str(node["id"]) == node_id:
            return str(node["title"])
    return node_id


def _short_title(title: str, limit: int) -> str:
    return title if len(title) <= limit else f"{title[: limit - 1]}…"


def _case_to_html(case: dict[str, object]) -> str:
    rows = []
    for mode in ["vector", "associative"]:
        for hit in case.get(mode, []):
            rows.append(
                "<tr>"
                f"<td>{'纯向量' if mode == 'vector' else '联想检索'}</td>"
                f"<td>{html.escape(str(hit['title']))}</td>"
                f"<td>{'是' if hit['is_supporting'] else '否'}</td>"
                f"<td>{html.escape(' -> '.join(map(str, hit['path'])))}</td>"
                f"<td>{html.escape(str(hit['reason']))}</td>"
                "</tr>"
            )
    return (
        f"<h3>{html.escape(str(case['query_id']))}</h3>"
        f"<p><b>问题：</b>{html.escape(str(case['question']))}<br><b>答案：</b>{html.escape(str(case['answer']))}</p>"
        "<table><thead><tr><th>模式</th><th>文档</th><th>支持证据</th><th>路径</th><th>原因</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _edge_color(relation_type: str) -> str:
    if relation_type == "shared_entity":
        return "#b45309"
    if relation_type == "keyword_overlap":
        return "#2563eb"
    if relation_type == "embedding_similarity":
        return "#6d28d9"
    return "#64748b"
