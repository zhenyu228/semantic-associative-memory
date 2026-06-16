from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "assets"
PNG_PATH = OUT_DIR / "progress_and_plan.png"
PDF_PATH = OUT_DIR / "progress_and_plan.pdf"

W, H = 1920, 1080

NAVY = "#2F4A7A"
NAVY_DARK = "#17345F"
BLUE_LIGHT = "#EEF4FB"
BLUE_BORDER = "#B9C9DF"
TEAL = "#0C9A9A"
ORANGE = "#E9792F"
GREEN = "#3D8B57"
GRAY = "#5C6675"
LIGHT_GRAY = "#F7F9FC"
TEXT = "#111827"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size, index=0)
    return ImageFont.load_default()


FONT_TITLE = font(62, bold=True)
FONT_SUBTITLE = font(34)
FONT_CARD_TITLE = font(34, bold=True)
FONT_BULLET = font(28)
FONT_FOOTER = font(32, bold=True)
FONT_SMALL = font(22)


def draw_text_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_width: int,
    fill: str,
    fnt: ImageFont.FreeTypeFont,
    line_gap: int = 10,
) -> int:
    x, y = xy
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for ch in paragraph:
            candidate = current + ch
            if draw.textlength(candidate, font=fnt) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = ch
        if current:
            lines.append(current)

    for line in lines:
        draw.text((x, y), line, fill=fill, font=fnt)
        y += fnt.size + line_gap
    return y


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    bullets: list[str],
    accent: str,
    icon: str,
) -> None:
    x1, y1, x2, y2 = box
    radius = 24
    draw.rounded_rectangle(box, radius=radius, fill="white", outline=BLUE_BORDER, width=3)
    draw.rounded_rectangle((x1, y1, x2, y1 + 86), radius=radius, fill=accent)
    draw.rectangle((x1, y1 + 48, x2, y1 + 86), fill=accent)

    draw.ellipse((x1 + 28, y1 + 22, x1 + 78, y1 + 72), fill="white")
    draw.text((x1 + 43, y1 + 27), icon, fill=accent, font=FONT_SMALL, anchor="ma")
    draw.text((x1 + 96, y1 + 25), title, fill="white", font=FONT_CARD_TITLE)

    y = y1 + 126
    for bullet in bullets:
        draw.ellipse((x1 + 36, y + 9, x1 + 48, y + 21), fill=accent)
        y = draw_text_wrapped(draw, (x1 + 66, y), bullet, x2 - x1 - 104, TEXT, FONT_BULLET, line_gap=8)
        y += 22


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    # Header.
    draw.rounded_rectangle((70, 64, 130, 124), radius=12, fill=NAVY)
    draw.text((160, 50), "阶段性进展与后续安排", fill=NAVY, font=FONT_TITLE)
    draw.line((70, 155, W - 70, 155), fill=BLUE_BORDER, width=3)
    draw.text(
        (78, 192),
        "围绕动态知识图谱记忆系统，当前工作已从局部建图验证推进到分层记忆重构与跨数据集分析。",
        fill=TEXT,
        font=FONT_SUBTITLE,
    )

    card_w = 540
    card_h = 560
    gap = 55
    x0 = 105
    y0 = 295
    boxes = [
        (x0, y0, x0 + card_w, y0 + card_h),
        (x0 + card_w + gap, y0, x0 + 2 * card_w + gap, y0 + card_h),
        (x0 + 2 * (card_w + gap), y0, x0 + 3 * card_w + 2 * gap, y0 + card_h),
    ]

    draw_card(
        draw,
        boxes[0],
        "已完成工作",
        [
            "完成三个公开数据集接入",
            "实现 SAM 原型：局部建图、图扩展检索、分层记忆重构",
            "完成三组实验：证据补回、图密度分析、方法对比",
            "跑通在线插入效率实验，验证动态更新成本优势",
        ],
        NAVY,
        "1",
    )
    draw_card(
        draw,
        boxes[1],
        "阶段性发现",
        [
            "图结构能补回部分向量检索遗漏证据",
            "图不是越密越好，过多边会带来噪声和边际收益下降",
            "SAM 在多跳问答与论文问答上表现较强",
            "科研检索暴露出跨论文全局关系建模不足",
        ],
        TEAL,
        "2",
    )
    draw_card(
        draw,
        boxes[2],
        "后续安排",
        [
            "强化路径有效性评分，减少无效图扩展",
            "完善 G0→G1→G2 高层记忆重构机制",
            "增强跨论文实体归一化与关系组织能力",
            "扩大论文问答与科研检索实验规模，补充案例分析",
        ],
        ORANGE,
        "3",
    )

    # Bottom conclusion bar.
    footer = (110, 905, W - 110, 1015)
    draw.rounded_rectangle(footer, radius=18, fill=BLUE_LIGHT, outline=BLUE_BORDER, width=3)
    draw.text((145, 936), "下一阶段重点：", fill=NAVY_DARK, font=FONT_FOOTER)
    draw.text(
        (395, 936),
        "从“局部证据补全”推进到“跨文档高层语义记忆与可解释推理链重建”",
        fill=TEAL,
        font=FONT_FOOTER,
    )

    img.save(PNG_PATH)
    img.save(PDF_PATH, "PDF", resolution=200.0)
    print(f"saved {PNG_PATH}")
    print(f"saved {PDF_PATH}")


if __name__ == "__main__":
    main()
