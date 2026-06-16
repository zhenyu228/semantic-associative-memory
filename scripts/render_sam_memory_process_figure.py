from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
PNG_PATH = ASSET_DIR / "sam_memory_development_process.png"
PDF_PATH = ASSET_DIR / "sam_memory_development_process.pdf"


COLORS = {
    "navy": "#15284b",
    "blue": "#a9bddb",
    "blue_light": "#dceaf9",
    "green": "#74a943",
    "green_light": "#e1efcf",
    "orange": "#e8873a",
    "orange_light": "#f8dbc2",
    "purple": "#7a4aa0",
    "purple_light": "#eadcf1",
    "gray": "#707070",
    "gray_light": "#f4f4f4",
    "red": "#c85c5c",
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

    fig, ax = plt.subplots(figsize=(17, 6.7), dpi=180)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 62)
    ax.axis("off")

    outer = FancyBboxPatch(
        (2.0, 3.5),
        116,
        55.5,
        boxstyle="round,pad=0.35,rounding_size=3.2",
        linewidth=2.2,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(outer)
    for x in [41.5, 80.0]:
        ax.plot([x, x], [4.2, 58.3], color=COLORS["navy"], lw=1.8, ls=(0, (5, 4)), alpha=0.95)

    draw_panel_initial(ax, 6.0)
    draw_panel_update(ax, 45.0)
    draw_panel_retrieval(ax, 84.0)

    draw_caption(ax, 22.0, "(a) Initial Three-layer Memory Construction")
    draw_caption(ax, 61.0, "(b) Incremental Local Memory Update")
    draw_caption(ax, 100.0, "(c) Associative Hierarchical Retrieval")

    fig.savefig(PNG_PATH, bbox_inches="tight", pad_inches=0.14)
    fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.14)
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
    subtitle: str,
    *,
    alpha: float = 1.0,
) -> None:
    skew = 3.0
    patch = Polygon(
        [(x, y), (x + w, y), (x + w + skew, y + h), (x + skew, y + h)],
        closed=True,
        fill=False,
        edgecolor=COLORS["gray"],
        linewidth=1.25,
        linestyle=(0, (4, 3)),
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(
        x + w + skew - 0.45,
        y + h * 0.72,
        label,
        ha="right",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        zorder=12,
        bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.text(
        x + w + skew - 0.45,
        y + h * 0.36,
        subtitle,
        ha="right",
        va="center",
        fontsize=6.5,
        color=COLORS["gray"],
        zorder=12,
        bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )


def node(
    ax,
    xy: tuple[float, float],
    *,
    color: str,
    edge_color: str,
    radius: float = 0.62,
    lw: float = 1.45,
    alpha: float = 1.0,
    z: int = 6,
) -> None:
    ax.add_patch(Circle(xy, radius, facecolor=color, edgecolor=edge_color, linewidth=lw, alpha=alpha, zorder=z))


def edge(
    ax,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    color: str = COLORS["navy"],
    lw: float = 1.2,
    ls: str = "-",
    alpha: float = 1.0,
    z: int = 4,
) -> None:
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, ls=ls, alpha=alpha, zorder=z)


def arrow(
    ax,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    color: str = COLORS["navy"],
    lw: float = 1.35,
    ls: str = "-",
    rad: float = 0.0,
    alpha: float = 1.0,
    z: int = 5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=12,
            lw=lw,
            linestyle=ls,
            color=color,
            alpha=alpha,
            zorder=z,
        )
    )


def label_box(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    fontsize: float = 8.4,
    ha: str = "center",
    weight: str = "normal",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        color=color,
        fontsize=fontsize,
        fontstyle="italic",
        fontweight=weight,
        zorder=16,
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )


def draw_doc_icon(ax, x: float, y: float) -> None:
    for i in range(3):
        ax.add_patch(
            Rectangle(
                (x + i * 0.55, y + i * 0.35),
                3.4,
                4.2,
                facecolor="white",
                edgecolor=COLORS["gray"],
                linewidth=1.0,
                zorder=3,
            )
        )
        ax.plot([x + 0.55 + i * 0.55, x + 2.65 + i * 0.55], [y + 3.2 + i * 0.35, y + 3.2 + i * 0.35], color=COLORS["gray"], lw=0.7, zorder=4)
        ax.plot([x + 0.55 + i * 0.55, x + 3.05 + i * 0.55], [y + 2.35 + i * 0.35, y + 2.35 + i * 0.35], color=COLORS["gray"], lw=0.7, zorder=4)


def graph_layout(ox: float) -> dict[str, object]:
    g0 = (ox + 1.8, 10.0, 25.0, 8.0)
    g1 = (ox + 7.2, 29.0, 18.0, 6.2)
    g2 = (ox + 10.4, 45.0, 13.8, 5.0)

    g0_coords = {
        "a": (0.12, 0.20),
        "b": (0.31, 0.36),
        "c": (0.46, 0.23),
        "d": (0.63, 0.45),
        "e": (0.35, 0.65),
        "f": (0.18, 0.58),
        "g": (0.70, 0.15),
        "h": (0.78, 0.37),
        "new1": (0.88, 0.31),
        "new2": (0.80, 0.02),
    }
    g1_coords = {
        "l1": (0.24, 0.40),
        "l2": (0.52, 0.48),
        "l3": (0.75, 0.30),
    }
    g2_coords = {
        "t1": (0.34, 0.48),
        "t2": (0.66, 0.46),
    }

    return {
        "g0": g0,
        "g1": g1,
        "g2": g2,
        "g0_pts": {k: plane_point(*g0, u, v) for k, (u, v) in g0_coords.items()},
        "g1_pts": {k: plane_point(*g1, u, v) for k, (u, v) in g1_coords.items()},
        "g2_pts": {k: plane_point(*g2, u, v) for k, (u, v) in g2_coords.items()},
    }


def draw_layered_memory(
    ax,
    ox: float,
    *,
    include_new: bool,
    active_g0: set[str] | None = None,
    active_g1: set[str] | None = None,
    active_g2: set[str] | None = None,
    update_g1: set[str] | None = None,
    update_g2: set[str] | None = None,
):
    active_g0 = active_g0 or set()
    active_g1 = active_g1 or set()
    active_g2 = active_g2 or set()
    update_g1 = update_g1 or set()
    update_g2 = update_g2 or set()
    layout = graph_layout(ox)
    g0 = layout["g0"]
    g1 = layout["g1"]
    g2 = layout["g2"]
    g0_pts = layout["g0_pts"]
    g1_pts = layout["g1_pts"]
    g2_pts = layout["g2_pts"]

    draw_plane(ax, *g0, r"$G_0$", "Evidence")
    draw_plane(ax, *g1, r"$G_1$", "Local memory")
    draw_plane(ax, *g2, r"$G_2$", "High-level")

    base_edges = [("a", "b"), ("b", "c"), ("c", "d"), ("b", "e"), ("e", "d"), ("f", "b"), ("f", "e"), ("c", "g"), ("d", "h")]
    for left, right in base_edges:
        edge(ax, g0_pts[left], g0_pts[right], color=COLORS["navy"], lw=1.15)
    if include_new:
        edge(ax, g0_pts["h"], g0_pts["new1"], color="#6c9bcc", lw=1.2)
        edge(ax, g0_pts["g"], g0_pts["new2"], color="#6c9bcc", lw=1.2)

    for name in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        if name in active_g0:
            node(ax, g0_pts[name], color=COLORS["green_light"], edge_color=COLORS["green"], radius=0.66, lw=1.8)
        else:
            node(ax, g0_pts[name], color=COLORS["blue"], edge_color=COLORS["navy"])

    if include_new:
        for name in ["new1", "new2"]:
            if name in active_g0:
                node(ax, g0_pts[name], color=COLORS["green_light"], edge_color=COLORS["green"], radius=0.66, lw=1.8)
            else:
                node(ax, g0_pts[name], color=COLORS["blue_light"], edge_color="#6c9bcc", radius=0.64, lw=1.6)

    for name in ["l1", "l2", "l3"]:
        if name in active_g1:
            node(ax, g1_pts[name], color=COLORS["green_light"], edge_color=COLORS["green"], radius=0.70, lw=1.9)
        elif name in update_g1:
            node(ax, g1_pts[name], color="#fff2d9", edge_color=COLORS["orange"], radius=0.74, lw=2.0)
        else:
            node(ax, g1_pts[name], color=COLORS["orange_light"], edge_color=COLORS["orange"], radius=0.68, lw=1.65)

    for left, right in [("l1", "l2"), ("l2", "l3")]:
        edge(ax, g1_pts[left], g1_pts[right], color=COLORS["orange"], lw=1.25)

    for name in ["t1", "t2"]:
        if name in active_g2:
            node(ax, g2_pts[name], color=COLORS["green_light"], edge_color=COLORS["green"], radius=0.73, lw=2.0)
        elif name in update_g2:
            node(ax, g2_pts[name], color="#f3e4f7", edge_color=COLORS["purple"], radius=0.75, lw=2.1)
        else:
            node(ax, g2_pts[name], color=COLORS["purple_light"], edge_color=COLORS["purple"], radius=0.70, lw=1.75)
    edge(ax, g2_pts["t1"], g2_pts["t2"], color=COLORS["purple"], lw=1.25)
    return layout


def draw_panel_initial(ax, ox: float) -> None:
    layout = draw_layered_memory(ax, ox, include_new=False)
    g0_pts = layout["g0_pts"]
    g1_pts = layout["g1_pts"]
    g2_pts = layout["g2_pts"]

    draw_doc_icon(ax, ox + 0.2, 47.2)
    label_box(ax, ox + 4.0, 55.0, "Corpus", color=COLORS["navy"], fontsize=8.2, weight="bold")
    arrow(ax, (ox + 5.0, 48.0), (g0_pts["f"][0] - 0.2, g0_pts["f"][1] + 1.0), color=COLORS["navy"], lw=1.15, rad=-0.25)

    fan_in_groups = {
        "l1": ["b", "e"],
        "l2": ["c", "d"],
        "l3": ["g", "h"],
    }
    for dst, sources in fan_in_groups.items():
        for src in sources:
            arrow(ax, (g0_pts[src][0], g0_pts[src][1] + 0.65), (g1_pts[dst][0], g1_pts[dst][1] - 0.82), color=COLORS["orange"], lw=0.95, ls=":", alpha=0.9)
    for src, dst in [("l1", "t1"), ("l2", "t1"), ("l3", "t2")]:
        arrow(ax, (g1_pts[src][0], g1_pts[src][1] + 0.7), (g2_pts[dst][0], g2_pts[dst][1] - 0.8), color=COLORS["purple"], lw=1.0, ls=":", alpha=0.95)
    label_box(ax, ox + 20.6, 26.0, "bottom-up\naggregation", color=COLORS["gray"], fontsize=7.6)


def draw_panel_update(ax, ox: float) -> None:
    layout = draw_layered_memory(ax, ox, include_new=True, update_g1={"l3"}, update_g2={"t2"})
    g0_pts = layout["g0_pts"]
    g1_pts = layout["g1_pts"]
    g2_pts = layout["g2_pts"]

    label_box(ax, ox + 26.2, 18.3, "New\nMemoryItems", color="#1e78c8", fontsize=8.0)
    arrow(ax, (ox + 25.1, 17.6), (g0_pts["new1"][0] + 0.2, g0_pts["new1"][1] + 0.4), color="#1e78c8", lw=1.0, rad=-0.20)
    for src in ["h", "new1", "new2"]:
        arrow(ax, (g0_pts[src][0], g0_pts[src][1] + 0.72), (g1_pts["l3"][0], g1_pts["l3"][1] - 0.88), color=COLORS["orange"], lw=1.08, ls=":")
    arrow(ax, (g1_pts["l3"][0], g1_pts["l3"][1] + 0.75), (g2_pts["t2"][0], g2_pts["t2"][1] - 0.8), color=COLORS["purple"], lw=1.2, ls=":")
    label_box(ax, ox + 12.8, 25.1, "local update\nwithout global rebuild", color=COLORS["orange"], fontsize=7.5)


def draw_panel_retrieval(ax, ox: float) -> None:
    layout = draw_layered_memory(
        ax,
        ox,
        include_new=True,
        active_g0={"d", "h", "new1"},
        active_g1={"l3"},
        active_g2={"t2"},
    )
    g0_pts = layout["g0_pts"]
    g1_pts = layout["g1_pts"]
    g2_pts = layout["g2_pts"]

    label_box(ax, ox + 4.0, 52.5, "Query", color=COLORS["green"], fontsize=9.0, weight="bold")
    arrow(ax, (ox + 6.0, 51.4), (g2_pts["t2"][0] - 0.6, g2_pts["t2"][1] + 0.25), color=COLORS["green"], lw=1.5, ls="--")
    arrow(ax, (g2_pts["t2"][0], g2_pts["t2"][1] - 0.75), (g1_pts["l3"][0], g1_pts["l3"][1] + 0.85), color=COLORS["green"], lw=1.35, ls=":")
    arrow(ax, (g1_pts["l3"][0], g1_pts["l3"][1] - 0.75), (g0_pts["h"][0], g0_pts["h"][1] + 0.8), color=COLORS["green"], lw=1.35, ls=":")
    arrow(ax, (g1_pts["l3"][0] - 0.4, g1_pts["l3"][1] - 0.8), (g0_pts["new1"][0], g0_pts["new1"][1] + 0.8), color=COLORS["green"], lw=1.35, ls=":")
    for name in ["d", "h", "new1"]:
        check(ax, g0_pts[name][0] + 0.7, g0_pts[name][1] - 0.4, scale=0.47)
    label_box(ax, ox + 16.6, 25.3, "associate down\nthrough memory paths", color=COLORS["green"], fontsize=7.6)
    label_box(ax, ox + 22.5, 8.0, "bottom evidence", color=COLORS["green"], fontsize=7.4)


def draw_caption(ax, x: float, title: str) -> None:
    ax.text(x, 6.7, title, ha="center", va="top", fontsize=10.2, fontweight="bold")


def check(ax, x: float, y: float, *, scale: float = 1.0) -> None:
    ax.plot(
        [x, x + 0.36 * scale, x + 1.08 * scale],
        [y, y - 0.44 * scale, y + 0.66 * scale],
        color=COLORS["green"],
        lw=1.8,
        solid_capstyle="round",
        zorder=18,
    )


if __name__ == "__main__":
    main()
