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
    focus_query_id: str | None = None,
) -> dict[str, Path]:
    """导出可检查的图谱产物：JSON、Mermaid、HTML/SVG。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    graph_json = target / "graph_artifact.json"
    mermaid_md = target / "graph_mermaid.md"
    html_path = target / "graph_view.html"

    node_payload = [_node_payload(node) for node in nodes]
    edge_payload = [_edge_payload(edge) for edge in edges]
    for index, edge in enumerate(edge_payload):
        edge["index"] = index
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
    html_path.write_text(
        _to_html(node_payload, edge_payload, retrieval_cases or [], focus_query_id=focus_query_id),
        encoding="utf-8",
    )
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
        "text": node.text,
        "summary": node.summary,
        "tags": node.tags,
        "created_at": node.created_at,
        "confidence": node.confidence,
        "snippet": node.text[:240],
        "source": node.source,
        "metadata": node.metadata,
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
    focus_query_id: str | None = None,
) -> str:
    by_query: dict[str, list[dict[str, object]]] = {}
    for node in nodes:
        by_query.setdefault(str(node["query_id"]), []).append(node)
    cases_by_query = {str(case["query_id"]): case for case in retrieval_cases}
    default_query_id = focus_query_id or (str(retrieval_cases[0]["query_id"]) if retrieval_cases else "")
    case_options = "\n".join(
        f'<option value="{html.escape(str(case["query_id"]))}" {"selected" if str(case["query_id"]) == default_query_id else ""}>index={html.escape(str(_case_index(case)))} | {html.escape(str(case["question"])[:90])}</option>'
        for case in retrieval_cases
    )
    graph_data_json = json.dumps(
        {
            "nodes": {str(node["id"]): node for node in nodes},
            "edges": {str(edge["index"]): edge for edge in edges},
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
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

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>SAM 图谱运行产物</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; background: #f7f9fc; }}
    .layout {{ display: grid; grid-template-columns: minmax(0, 1fr) 420px; gap: 18px; align-items: start; }}
    .side-panel {{ position: sticky; top: 16px; background: white; border: 1px solid #d7dde5; border-radius: 8px; padding: 16px; max-height: 88vh; overflow: auto; }}
    .side-panel pre {{ white-space: pre-wrap; word-break: break-word; background: #f5f7fa; padding: 10px; border-radius: 6px; font-size: 12px; }}
    h1, h2, h3 {{ margin: 18px 0 10px; }}
    .hint {{ color: #52606d; line-height: 1.6; }}
    .legend span {{ display: inline-block; margin-right: 18px; }}
    .query-card {{ background: white; border: 1px solid #d7dde5; border-radius: 8px; margin: 18px 0 26px; padding: 16px; }}
    .question {{ color: #334e68; line-height: 1.5; margin-bottom: 12px; }}
    .method-grid {{ display: flex; flex-direction: column; gap: 16px; }}
    .method-card {{ border: 1px solid #d7dde5; border-radius: 8px; padding: 12px; background: #fff; }}
    .method-card h4 {{ margin: 0 0 8px; }}
    .answer-card {{ background: #f8fafc; border: 1px solid #e1e8f0; border-radius: 6px; padding: 10px; margin-bottom: 10px; line-height: 1.5; }}
    .toolbar {{ background: white; border: 1px solid #d7dde5; border-radius: 8px; padding: 12px; margin: 16px 0; }}
    .toolbar select {{ width: 100%; max-width: 920px; padding: 8px; font-size: 14px; }}
    .graph {{ overflow-x: auto; border: 1px solid #d7dde5; background: #fbfcfe; border-radius: 6px; }}
    .edge-note {{ color: #52606d; font-size: 13px; margin: 8px 0 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px; vertical-align: top; }}
    th {{ background: #f3f6f9; text-align: left; }}
    code {{ background: #eef2f7; padding: 2px 4px; }}
    .support-pill {{ color: #8a5a00; font-weight: 700; }}
    .candidate-pill {{ color: #1f5f8b; font-weight: 700; }}
    .clickable, .edge-click {{ cursor: pointer; }}
    .clickable:hover {{ filter: brightness(0.95); }}
    .edge-click:hover {{ opacity: 1; stroke-width: 3.2; }}
    @media (max-width: 1100px) {{ .layout {{ display: block; }} .method-card {{ margin-bottom: 16px; }} .side-panel {{ position: static; margin-bottom: 16px; }} }}
  </style>
</head>
<body>
  <h1>SAM 图谱运行产物</h1>
  <p class="hint">黄色节点是真实 HotpotQA supporting paragraph，蓝色节点是候选干扰段落。箭头表示系统实际创建的关键语义边。这个文件由代码自动生成，不是手绘示意图。</p>
  <p class="legend"><span>黄色：支持证据</span><span>蓝色：候选文档</span><span>橙色边：共享实体</span><span>蓝色边：关键词重叠</span><span>紫色边：embedding 相似</span></p>
  <div class="toolbar">
    <label for="case-select"><b>选择样本：</b></label>
    <select id="case-select" onchange="selectCase(this.value)">
      {case_options}
    </select>
  </div>
  <div class="layout">
    <main>
      <h2>方法对比图</h2>
      {graph_blocks}
    </main>
    <aside class="side-panel" id="detail-panel">
      <h2>详情面板</h2>
      <p>点击图中的节点查看完整 MemoryNode；点击边查看建边原因。</p>
    </aside>
  </div>
  <script>
    const graphData = {graph_data_json};
    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}
    function showNode(nodeId) {{
      const node = graphData.nodes[nodeId];
      const panel = document.getElementById("detail-panel");
      panel.innerHTML = `
        <h2>节点详情</h2>
        <p><b>标题：</b>${{escapeHtml(node.title)}}</p>
        <p><b>节点 ID：</b><code>${{escapeHtml(node.id)}}</code></p>
        <p><b>是否支持证据：</b>${{node.is_supporting ? "是" : "否"}}</p>
        <p><b>来源：</b>${{escapeHtml(node.source)}}</p>
        <p><b>Query ID：</b><code>${{escapeHtml(node.query_id)}}</code></p>
        <p><b>关键词：</b>${{escapeHtml((node.keywords || []).join(", "))}}</p>
        <p><b>实体：</b>${{escapeHtml((node.entities || []).join(", "))}}</p>
        <p><b>MemoryNode 内容：</b></p>
        <pre>${{escapeHtml(JSON.stringify({{
          id: node.id,
          title: node.title,
          summary: node.summary,
          text: node.text,
          tags: node.tags,
          usage_count: node.usage_count,
          confidence: node.confidence,
          created_at: node.created_at,
          metadata: node.metadata
        }}, null, 2))}}</pre>
      `;
    }}
    function showEdge(edgeIndex) {{
      const edge = graphData.edges[String(edgeIndex)];
      const source = graphData.nodes[edge.source_id] || {{}};
      const target = graphData.nodes[edge.target_id] || {{}};
      const panel = document.getElementById("detail-panel");
      panel.innerHTML = `
        <h2>边详情</h2>
        <p><b>起点：</b>${{escapeHtml(source.title || edge.source_id)}}</p>
        <p><b>终点：</b>${{escapeHtml(target.title || edge.target_id)}}</p>
        <p><b>关系类型：</b><code>${{escapeHtml(edge.relation_type)}}</code></p>
        <p><b>边权：</b>${{Number(edge.weight).toFixed(3)}}</p>
        <p><b>为什么可以连起来：</b>${{escapeHtml(edge.reason)}}</p>
        <p><b>完整 MemoryEdge 内容：</b></p>
        <pre>${{escapeHtml(JSON.stringify(edge, null, 2))}}</pre>
      `;
    }}
    function selectCase(queryId) {{
      for (const card of document.querySelectorAll(".query-card")) {{
        card.style.display = card.dataset.queryId === queryId ? "block" : "none";
      }}
      document.getElementById("detail-panel").innerHTML = `
        <h2>详情面板</h2>
        <p>当前样本：<code>${{escapeHtml(queryId)}}</code></p>
        <p>点击图中的节点查看完整 MemoryNode；点击边查看建边原因。</p>
      `;
    }}
    selectCase("{html.escape(default_query_id)}");
  </script>
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

    vector_hit_ids = {str(hit["node_id"]) for hit in case.get("vector", [])} if case else set()
    associative_hit_ids = {str(hit["node_id"]) for hit in case.get("associative", [])} if case else set()
    vector_svg = _method_svg(
        ordered_nodes=ordered_nodes,
        query_edges=query_edges,
        positions=positions,
        width=width,
        height=height,
        node_width=node_width,
        node_height=node_height,
        case=case,
        method="vector",
        hit_ids=vector_hit_ids,
    )
    associative_svg = _method_svg(
        ordered_nodes=ordered_nodes,
        query_edges=query_edges,
        positions=positions,
        width=width,
        height=height,
        node_width=node_width,
        node_height=node_height,
        case=case,
        method="associative",
        hit_ids=associative_hit_ids,
    )

    question = html.escape(str(case["question"])) if case else query_id
    answer = html.escape(str(case["answer"])) if case else ""
    vector_answer = _method_answer_html(case, "vector") if case else ""
    associative_answer = _method_answer_html(case, "associative") if case else ""
    return f"""
  <section class="query-card" data-query-id="{html.escape(query_id)}">
    <h3>{html.escape(query_id)}</h3>
    <div class="question"><b>问题：</b>{question}<br><b>标准答案：</b>{answer}</div>
    <div class="method-grid">
      <div class="method-card">
        <h4>纯向量检索</h4>
        {vector_answer}
        <div class="graph">{vector_svg}</div>
      </div>
      <div class="method-card">
        <h4>SAM 联想图检索</h4>
        {associative_answer}
        <div class="graph">{associative_svg}</div>
      </div>
    </div>
    <p class="edge-note">粗边表示该方法实际检索路径；带红色描边的节点表示该方法最终 top-k 选择的节点。点击节点/边可在右侧查看完整信息。</p>
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


def _method_svg(
    ordered_nodes: list[dict[str, object]],
    query_edges: list[dict[str, object]],
    positions: dict[str, tuple[int, int]],
    width: int,
    height: int,
    node_width: int,
    node_height: int,
    case: dict[str, object] | None,
    method: str,
    hit_ids: set[str],
) -> str:
    method_path_edges = _method_path_edge_keys(case, method)
    important_edges = _select_important_edges(query_edges, method_path_edges)
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
        is_path = _edge_key(edge) in method_path_edges
        stroke_width = 3.0 if is_path else 1.2
        opacity = 0.88 if is_path else 0.24
        svg_parts.append(
            f"<line class='edge-click' onclick='showEdge({edge['index']})' x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='{stroke_width}' marker-end='url(#arrow)' opacity='{opacity}'/>"
        )
    for node in ordered_nodes:
        x, y = positions[str(node["id"])]
        fill = "#fff2bd" if node["is_supporting"] else "#e6f2ff"
        stroke = "#d92d20" if str(node["id"]) in hit_ids else ("#bd7b00" if node["is_supporting"] else "#2f75a8")
        stroke_width = 3 if str(node["id"]) in hit_ids else 2
        title = html.escape(_short_title(str(node["title"]), 36))
        role = "supporting" if node["is_supporting"] else "candidate"
        selected = "selected" if str(node["id"]) in hit_ids else ""
        svg_parts.append(
            f"<rect class='clickable' onclick='showNode(\"{html.escape(str(node['id']))}\")' x='{x}' y='{y}' width='{node_width}' height='{node_height}' rx='8' fill='{fill}' stroke='{stroke}' stroke-width='{stroke_width}'/>"
        )
        svg_parts.append(
            f"<foreignObject x='{x + 8}' y='{y + 8}' width='{node_width - 16}' height='28'><div xmlns='http://www.w3.org/1999/xhtml' style='font-size:12px;font-weight:700;line-height:14px;text-align:center;color:#102a43;'>{title}</div></foreignObject>"
        )
        svg_parts.append(
            f"<text x='{x + node_width / 2}' y='{y + 47}' text-anchor='middle' font-size='11'>{role} {selected}</text>"
        )
    svg_parts.append("</svg>")
    return "".join(svg_parts)


def _method_path_edge_keys(case: dict[str, object] | None, method: str) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not case:
        return keys
    for hit in case.get(method, []):
        path = [str(node_id) for node_id in hit.get("path", [])]
        for left, right in zip(path, path[1:], strict=False):
            keys.add((left, right))
            keys.add((right, left))
    return keys


def _method_answer_html(case: dict[str, object], method: str) -> str:
    answer_key = f"{method}_final_answer"
    support_key = f"{method}_support_hits"
    answer = case.get(answer_key, {})
    status = answer.get("status") if isinstance(answer, dict) else None
    value = answer.get("answer") if isinstance(answer, dict) else ""
    evidence = answer.get("evidence_title") if isinstance(answer, dict) else None
    status_text = "找到标准答案字符串" if status == "found_in_retrieved_context" else "未找到标准答案字符串"
    return (
        '<div class="answer-card">'
        f"<b>方法最终答案：</b>{html.escape(str(value))}<br>"
        f"<b>答案状态：</b>{html.escape(status_text)}<br>"
        f"<b>命中 gold 支持证据：</b>{html.escape(str(case.get(support_key, 0)))}<br>"
        f"<b>答案来源节点：</b>{html.escape(str(evidence or '无'))}"
        "</div>"
    )


def _case_index(case: dict[str, object]) -> object:
    query_id = str(case.get("query_id", ""))
    for part in query_id.split("_"):
        if part.isdigit():
            return part
    # HotpotQA 的原始 index 不总在 query_id 中；旧数据则回退显示 query_id 尾部。
    return query_id.split("_")[-1]


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
