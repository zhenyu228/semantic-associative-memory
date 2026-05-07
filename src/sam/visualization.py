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
    width = 1300
    height = max(720, len(nodes) * 46)
    by_query: dict[str, list[dict[str, object]]] = {}
    for node in nodes:
        by_query.setdefault(str(node["query_id"]), []).append(node)

    positions: dict[str, tuple[int, int]] = {}
    y = 70
    for _, query_nodes in by_query.items():
        for index, node in enumerate(query_nodes):
            x = 120 + (index % 5) * 230
            positions[str(node["id"])] = (x, y)
        y += 150

    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        "<defs><marker id='arrow' markerWidth='10' markerHeight='10' refX='9' refY='3' orient='auto' markerUnits='strokeWidth'><path d='M0,0 L0,6 L9,3 z' fill='#555'/></marker></defs>",
    ]
    for edge in edges:
        if float(edge["weight"]) < 0.2:
            continue
        source = positions.get(str(edge["source_id"]))
        target = positions.get(str(edge["target_id"]))
        if not source or not target:
            continue
        x1, y1 = source
        x2, y2 = target
        color = _edge_color(str(edge["relation_type"]))
        svg_parts.append(
            f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='{1.2 + float(edge['weight']) * 2:.2f}' marker-end='url(#arrow)' opacity='0.72'/>"
        )
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 - 6
        svg_parts.append(
            f"<text x='{mid_x}' y='{mid_y}' font-size='11' fill='{color}'>{html.escape(str(edge['relation_type']))}</text>"
        )
    for node in nodes:
        x, y = positions[str(node["id"])]
        fill = "#ffe8a3" if node["is_supporting"] else "#d8ecff"
        stroke = "#bd7b00" if node["is_supporting"] else "#3a78a0"
        title = html.escape(str(node["title"])[:26])
        svg_parts.append(f"<circle cx='{x}' cy='{y}' r='34' fill='{fill}' stroke='{stroke}' stroke-width='2'/>")
        svg_parts.append(
            f"<text x='{x}' y='{y - 4}' text-anchor='middle' font-size='11' font-weight='700'>{title}</text>"
        )
        svg_parts.append(
            f"<text x='{x}' y='{y + 12}' text-anchor='middle' font-size='10'>{'support' if node['is_supporting'] else 'candidate'}</text>"
        )
    svg_parts.append("</svg>")

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
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }}
    h1, h2 {{ margin: 18px 0 10px; }}
    .hint {{ color: #52606d; line-height: 1.6; }}
    .legend span {{ display: inline-block; margin-right: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px; vertical-align: top; }}
    th {{ background: #f3f6f9; text-align: left; }}
    .graph {{ overflow-x: auto; border: 1px solid #d7dde5; background: #fbfcfe; }}
    code {{ background: #eef2f7; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>SAM 图谱运行产物</h1>
  <p class="hint">黄色节点是真实 HotpotQA supporting paragraph，蓝色节点是候选干扰段落。边上的文字表示系统实际创建的语义关系。这个文件由代码自动生成，不是手绘示意图。</p>
  <p class="legend"><span>黄色：支持证据</span><span>蓝色：候选文档</span><span>箭头：语义边</span></p>
  <div class="graph">{''.join(svg_parts)}</div>
  <h2>检索案例</h2>
  {case_blocks}
  <h2>节点明细</h2>
  <table><thead><tr><th>标题</th><th>问题 ID</th><th>支持证据</th><th>实体</th><th>文本片段</th></tr></thead><tbody>{node_rows}</tbody></table>
  <h2>语义边明细</h2>
  <table><thead><tr><th>起点</th><th>终点</th><th>关系</th><th>权重</th><th>原因</th></tr></thead><tbody>{edge_rows}</tbody></table>
</body>
</html>
"""


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
