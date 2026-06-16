from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
PNG_PATH = ASSET_DIR / "sam_memory_development_process.png"
PDF_PATH = ASSET_DIR / "sam_memory_development_process.pdf"


COLORS = {
    "navy": "#15284b",
    "blue": "#9fb7d9",
    "blue_light": "#d8e7fb",
    "green": "#79ad47",
    "green_light": "#dfeecd",
    "orange": "#e8873a",
    "orange_light": "#f8dbc2",
    "purple": "#7a4aa0",
    "purple_light": "#e5d8ef",
    "red": "#c85c5c",
    "gray": "#6f6f6f",
    "dark": "#111111",
}


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.unicode_minus": False,
        }
    )
    fig, ax = plt.subplots(figsize=(16, 5.8), dpi=180)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 43.5)
    ax.axis("off")

    outer = FancyBboxPatch(
        (2.5, 4.2),
        95,
        35.0,
        boxstyle="round,pad=0.35,rounding_size=3.6",
        linewidth=2.3,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(outer)
    ax.plot([76.5, 76.5], [4.4, 39.0], color=COLORS["navy"], lw=2.2, ls="--")

    ax.text(
        39.0,
        36.1,
        "SAM Memory Development: Dynamic Associative Compression",
        ha="center",
        va="center",
        fontsize=14,
        fontstyle="italic",
        fontweight="bold",
    )
    ax.text(
        87.0,
        36.1,
        "Memory Retrieval",
        ha="center",
        va="center",
        fontsize=14,
        fontstyle="italic",
        fontweight="bold",
    )

    draw_panel_a(ax, 6.5, 10.0)
    draw_transition(ax, (27.0, 23.2), (33.0, 23.2), "Activation")
    draw_panel_b(ax, 32.2, 10.0)
    draw_transition(ax, (52.6, 23.2), (58.6, 23.2), "Reconstruction")
    draw_panel_c(ax, 58.0, 10.0)
    draw_panel_d(ax, 79.2, 10.0)

    draw_caption(
        ax,
        17.0,
        "(a) Low-level Evidence Graph",
    )
    draw_caption(
        ax,
        42.5,
        "(b) Query-activated Consolidation",
    )
    draw_caption(
        ax,
        67.0,
        "(c) Associative Compression",
    )
    draw_caption(
        ax,
        88.5,
        "(d) Compressed Retrieval",
    )
    fig.savefig(PNG_PATH, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(PNG_PATH)
    print(PDF_PATH)


def plane_point(x: float, y: float, w: float, h: float, u: float, v: float) -> tuple[float, float]:
    skew = 3.0
    return x + u * w + v * skew, y + v * h


def draw_plane(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    alpha: float = 1.0,
    edge: str = COLORS["gray"],
) -> None:
    skew = 3.0
    patch = Polygon(
        [(x, y), (x + w, y), (x + w + skew, y + h), (x + skew, y + h)],
        closed=True,
        fill=False,
        edgecolor=edge,
        linewidth=1.25,
        linestyle=(0, (4, 3)),
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(
        x + w + skew - 0.35,
        y + h * 0.7,
        label,
        ha="right",
        va="center",
        fontsize=10.4,
        fontweight="bold",
        zorder=11,
        bbox={
            "boxstyle": "round,pad=0.08",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        },
    )


def label_box(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    fontsize: float = 9.0,
    ha: str = "center",
    va: str = "center",
) -> None:
    """给流程注释加白底，避免文字和节点、边混在一起。"""

    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        color=color,
        fontsize=fontsize,
        fontstyle="italic",
        zorder=12,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )


def node(
    ax,
    xy: tuple[float, float],
    *,
    color: str = COLORS["blue"],
    edge: str = COLORS["navy"],
    radius: float = 0.65,
    lw: float = 1.4,
    z: int = 5,
) -> None:
    ax.add_patch(Circle(xy, radius, facecolor=color, edgecolor=edge, linewidth=lw, zorder=z))


def edge(
    ax,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    color: str = COLORS["navy"],
    lw: float = 1.25,
    ls: str = "-",
    alpha: float = 1.0,
    z: int = 3,
) -> None:
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, ls=ls, alpha=alpha, zorder=z)


def arrow(
    ax,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    color: str = COLORS["navy"],
    lw: float = 1.4,
    rad: float = 0.0,
    style: str = "-|>",
    ls: str = "-",
    alpha: float = 1.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle=style,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=12,
            lw=lw,
            linestyle=ls,
            color=color,
            alpha=alpha,
            zorder=4,
        )
    )


def draw_base_graph(ax, x: float, y: float, *, active: bool = False, noise: bool = False):
    w, h = 15.0, 7.0
    draw_plane(ax, x, y, w, h, r"$G_0$")
    coords = {
        "a": (0.12, 0.20),
        "b": (0.33, 0.34),
        "c": (0.50, 0.24),
        "d": (0.66, 0.45),
        "e": (0.38, 0.65),
        "f": (0.19, 0.58),
        "g": (0.76, 0.18),
    }
    pts = {name: plane_point(x, y, w, h, *uv) for name, uv in coords.items()}
    for left, right in [("a", "b"), ("b", "c"), ("c", "d"), ("b", "e"), ("e", "d"), ("f", "b"), ("f", "e"), ("c", "g")]:
        edge(ax, pts[left], pts[right], color=COLORS["navy"], lw=1.3)
    pts["new1"] = plane_point(x, y, w, h, 0.87, 0.32)
    pts["new2"] = plane_point(x, y, w, h, 0.78, 0.03)
    edge(ax, pts["d"], pts["new1"], color="#7ba6d8", lw=1.2)
    edge(ax, pts["g"], pts["new2"], color="#7ba6d8", lw=1.2)
    active_nodes = {"b", "c", "d"} if active else set()
    for name in ["a", "b", "c", "d", "e", "f", "g"]:
        pt = pts[name]
        if noise and name == "g":
            node(ax, pt, color="#f0d6d6", edge=COLORS["red"])
        elif name in active_nodes:
            node(ax, pt, color=COLORS["green_light"], edge=COLORS["green"], lw=1.7)
        else:
            node(ax, pt)
    node(ax, pts["new1"], color=COLORS["blue_light"], edge="#6d99cc")
    node(ax, pts["new2"], color=COLORS["blue_light"], edge="#6d99cc")
    return pts


def draw_panel_a(ax, x: float, y: float) -> None:
    pts = draw_base_graph(ax, x, y)
    draw_plane(ax, x + 4.9, y + 9.0, 8.2, 4.0, r"$G_1$", alpha=0.9)
    draw_plane(ax, x + 7.2, y + 16.1, 5.8, 3.0, r"$G_2$", alpha=0.75)
    label_box(ax, x + 13.7, y + 5.6, "New\nMemoryItems", color="#1e78c8", fontsize=8.4)
    arrow(ax, (x + 13.7, y + 4.9), (pts["new1"][0] + 0.1, pts["new1"][1] + 0.25), color="#1e78c8", rad=-0.22, lw=1.0)


def draw_panel_b(ax, x: float, y: float) -> None:
    pts = draw_base_graph(ax, x, y, active=True, noise=True)
    g1x, g1y, g1w, g1h = x + 4.8, y + 9.0, 9.0, 4.2
    g2x, g2y, g2w, g2h = x + 7.3, y + 16.2, 5.8, 3.0
    draw_plane(ax, g1x, g1y, g1w, g1h, r"$G_1$")
    draw_plane(ax, g2x, g2y, g2w, g2h, r"$G_2$", alpha=0.75)
    c1 = plane_point(g1x, g1y, g1w, g1h, 0.26, 0.42)
    c2 = plane_point(g1x, g1y, g1w, g1h, 0.58, 0.50)
    edge(ax, c1, c2, color=COLORS["orange"], lw=1.5)
    node(ax, c1, color=COLORS["orange_light"], edge=COLORS["orange"], radius=0.70, lw=1.7)
    node(ax, c2, color=COLORS["orange_light"], edge=COLORS["orange"], radius=0.70, lw=1.7)
    for src, dst in [(pts["b"], c1), (pts["c"], c1), (pts["d"], c2)]:
        edge(ax, src, dst, color=COLORS["navy"], lw=1.0, ls=":")
    label_box(ax, x + 13.1, y + 5.6, "Consolidation", color=COLORS["orange"], fontsize=8.8)
    label_box(ax, x + 2.6, y + 17.8, "Query", color=COLORS["green"], fontsize=9.2)
    arrow(ax, (x + 2.7, y + 16.9), (pts["b"][0], pts["b"][1] + 0.9), color=COLORS["green"], lw=1.2, ls="--")
    ax.add_patch(Circle((pts["g"][0], pts["g"][1]), 1.05, fill=False, edgecolor=COLORS["red"], lw=1.2, ls="--"))


def draw_panel_c(ax, x: float, y: float) -> None:
    pts = draw_base_graph(ax, x, y, active=True, noise=True)
    g1x, g1y, g1w, g1h = x + 4.7, y + 9.0, 9.2, 4.3
    g2x, g2y, g2w, g2h = x + 7.3, y + 16.1, 6.0, 3.2
    draw_plane(ax, g1x, g1y, g1w, g1h, r"$G_1$")
    draw_plane(ax, g2x, g2y, g2w, g2h, r"$G_2$")
    c1 = plane_point(g1x, g1y, g1w, g1h, 0.20, 0.36)
    c2 = plane_point(g1x, g1y, g1w, g1h, 0.55, 0.48)
    c3 = plane_point(g1x, g1y, g1w, g1h, 0.78, 0.25)
    for c in [c1, c2, c3]:
        node(ax, c, color=COLORS["orange_light"], edge=COLORS["orange"], radius=0.62, lw=1.5)
    for p1, p2 in [(c1, c2), (c2, c3)]:
        edge(ax, p1, p2, color=COLORS["orange"], lw=1.3, ls="--")
    insight = plane_point(g2x, g2y, g2w, g2h, 0.42, 0.46)
    insight2 = plane_point(g2x, g2y, g2w, g2h, 0.68, 0.38)
    node(ax, insight, color=COLORS["purple_light"], edge=COLORS["purple"], radius=0.72, lw=1.8)
    node(ax, insight2, color=COLORS["purple_light"], edge=COLORS["purple"], radius=0.58, lw=1.6)
    edge(ax, insight, insight2, color=COLORS["purple"], lw=1.3)
    for c in [c1, c2, c3]:
        edge(ax, c, insight, color=COLORS["purple"], lw=1.1, ls=":")
    for src in [pts["b"], pts["c"], pts["d"]]:
        edge(ax, src, c1, color=COLORS["navy"], lw=0.9, ls=":")
    label_box(ax, x + 8.6, y + 21.0, "Compressed\nMemory", color=COLORS["purple"], fontsize=8.8)
    label_box(ax, x + 13.4, y + 6.6, "Pruned path", color=COLORS["red"], fontsize=8.2)
    arrow(ax, (pts["g"][0] + 0.8, pts["g"][1] + 0.2), (x + 12.9, y + 5.9), color=COLORS["red"], lw=1.0, ls="--")


def draw_panel_d(ax, x: float, y: float) -> None:
    pts = draw_base_graph(ax, x, y, active=True)
    g1x, g1y, g1w, g1h = x + 4.8, y + 9.0, 8.8, 4.2
    g2x, g2y, g2w, g2h = x + 7.2, y + 16.2, 5.8, 3.0
    draw_plane(ax, g1x, g1y, g1w, g1h, r"$G_1$")
    draw_plane(ax, g2x, g2y, g2w, g2h, r"$G_2$")
    c1 = plane_point(g1x, g1y, g1w, g1h, 0.30, 0.42)
    c2 = plane_point(g1x, g1y, g1w, g1h, 0.63, 0.47)
    insight = plane_point(g2x, g2y, g2w, g2h, 0.45, 0.48)
    for c in [c1, c2]:
        node(ax, c, color=COLORS["green_light"], edge=COLORS["green"], radius=0.62, lw=1.7)
    node(ax, insight, color=COLORS["green_light"], edge=COLORS["green"], radius=0.72, lw=1.8)
    edge(ax, c1, c2, color=COLORS["green"], lw=1.5)
    for c in [c1, c2]:
        edge(ax, c, insight, color=COLORS["green"], lw=1.1, ls=":")
    for src in [pts["b"], pts["c"], pts["d"]]:
        edge(ax, src, c1, color=COLORS["green"], lw=1.1, ls=":")
    label_box(ax, x + 0.7, y + 16.6, "Query", color=COLORS["green"], fontsize=9.2)
    arrow(ax, (x + 1.6, y + 16.0), (insight[0] - 0.5, insight[1] - 0.1), color=COLORS["green"], lw=1.2, ls="--")
    for p in [insight, c1, pts["b"], pts["c"], pts["d"]]:
        check(ax, p[0] + 0.85, p[1] - 0.45, scale=0.48)


def draw_transition(ax, p1: tuple[float, float], p2: tuple[float, float], label: str) -> None:
    arrow(ax, p1, p2, color=COLORS["navy"], lw=1.4, rad=-0.28)
    label_box(ax, (p1[0] + p2[0]) / 2, p1[1] + 2.4, label, color=COLORS["navy"], fontsize=9.4)


def draw_caption(ax, x: float, title: str) -> None:
    ax.text(x, 7.2, title, ha="center", va="top", fontsize=10.5, fontweight="bold")


def check(ax, x: float, y: float, *, scale: float = 1.0) -> None:
    ax.plot(
        [x, x + 0.35 * scale, x + 1.10 * scale],
        [y, y - 0.45 * scale, y + 0.65 * scale],
        color=COLORS["green"],
        lw=1.8,
        solid_capstyle="round",
        zorder=8,
    )


if __name__ == "__main__":
    main()
