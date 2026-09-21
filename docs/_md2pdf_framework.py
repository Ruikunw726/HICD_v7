# -*- coding: utf-8 -*-
"""Convert PhD_thesis_framework.md to a readable CJK PDF."""
from __future__ import annotations

import os
import platform
import re
import sys
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

MD_PATH = sys.argv[1]
PDF_PATH = sys.argv[2]

CJK_TTF_CANDIDATES = {
    "Windows": [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ],
}


def resolve_cjk_font():
    for p in CJK_TTF_CANDIDATES.get(platform.system(), []):
        if os.path.exists(p):
            try:
                pdfmetrics.registerFont(TTFont("CJK", p, subfontIndex=0))
                return "CJK"
            except Exception:
                continue
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


CJK = resolve_cjk_font()

# Try bold variant
CJK_BOLD = CJK
bold_path = r"C:\Windows\Fonts\msyhbd.ttc"
if platform.system() == "Windows" and os.path.exists(bold_path):
    try:
        pdfmetrics.registerFont(TTFont("CJK-Bold", bold_path, subfontIndex=0))
        CJK_BOLD = "CJK-Bold"
    except Exception:
        pass


def md_inline(text: str) -> str:
    """Minimal markdown inline -> reportlab XML."""
    t = escape(text)
    t = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
    return t


def split_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def is_table_sep(line: str) -> bool:
    s = line.strip()
    return bool(s) and set(s) <= set("|-: ") and "-" in s


styles = getSampleStyleSheet()
body = ParagraphStyle(
    "BodyCJK",
    parent=styles["BodyText"],
    fontName=CJK,
    fontSize=9.5,
    leading=14,
    spaceAfter=4,
    alignment=TA_LEFT,
)
h1 = ParagraphStyle(
    "H1CJK",
    parent=styles["Heading1"],
    fontName=CJK_BOLD,
    fontSize=16,
    leading=22,
    spaceBefore=12,
    spaceAfter=8,
    textColor=colors.HexColor("#1a2744"),
)
h2 = ParagraphStyle(
    "H2CJK",
    parent=styles["Heading2"],
    fontName=CJK_BOLD,
    fontSize=13,
    leading=18,
    spaceBefore=10,
    spaceAfter=6,
    textColor=colors.HexColor("#243b6b"),
)
h3 = ParagraphStyle(
    "H3CJK",
    parent=styles["Heading3"],
    fontName=CJK_BOLD,
    fontSize=11,
    leading=16,
    spaceBefore=8,
    spaceAfter=4,
    textColor=colors.HexColor("#2f4a80"),
)
h4 = ParagraphStyle(
    "H4CJK",
    parent=styles["Heading4"],
    fontName=CJK_BOLD,
    fontSize=10,
    leading=14,
    spaceBefore=6,
    spaceAfter=3,
)
quote_style = ParagraphStyle(
    "QuoteCJK",
    parent=body,
    leftIndent=8,
    rightIndent=8,
    textColor=colors.HexColor("#444444"),
    borderColor=colors.HexColor("#9db0d0"),
    borderWidth=0,
    borderPadding=4,
    backColor=colors.HexColor("#f3f6fb"),
    spaceBefore=4,
    spaceAfter=6,
)
code_style = ParagraphStyle(
    "CodeCJK",
    parent=body,
    fontName="Courier",
    fontSize=8,
    leading=11,
    leftIndent=6,
    rightIndent=6,
    backColor=colors.HexColor("#f5f5f5"),
    spaceBefore=4,
    spaceAfter=6,
)
title_style = ParagraphStyle(
    "TitleCJK",
    parent=styles["Title"],
    fontName=CJK_BOLD,
    fontSize=18,
    leading=24,
    alignment=TA_CENTER,
    spaceAfter=4,
)
subtitle_style = ParagraphStyle(
    "SubCJK",
    parent=body,
    alignment=TA_CENTER,
    fontSize=10,
    textColor=colors.HexColor("#555555"),
    spaceAfter=12,
)
bullet_style = ParagraphStyle(
    "BulletCJK",
    parent=body,
    leftIndent=12,
    spaceAfter=2,
)
table_cell = ParagraphStyle(
    "CellCJK",
    parent=body,
    fontSize=8,
    leading=11,
    spaceAfter=0,
    spaceBefore=0,
)
table_head = ParagraphStyle(
    "CellHeadCJK",
    parent=table_cell,
    fontName=CJK_BOLD,
    textColor=colors.white,
)


def parse_md(md: str):
    lines = md.splitlines()
    i = 0
    flow = []
    n = len(lines)
    first_h1 = True

    while i < n:
        line = lines[i]
        raw = line.rstrip("\n")

        # skip mermaid / fenced
        if raw.strip().startswith("```"):
            lang = raw.strip()[3:].strip()
            i += 1
            block = []
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1  # closing fence
            if lang == "mermaid":
                flow.append(
                    Paragraph(
                        "【结构示意图见源 Markdown 中 mermaid 代码块】",
                        ParagraphStyle(
                            "MermaidNote",
                            parent=body,
                            fontSize=8.5,
                            textColor=colors.HexColor("#666666"),
                            alignment=TA_CENTER,
                            backColor=colors.HexColor("#eef2f8"),
                            spaceBefore=4,
                            spaceAfter=6,
                        ),
                    )
                )
            elif block:
                text = "<br/>".join(escape(b) for b in block)
                flow.append(Paragraph(f"<font face='Courier'>{text}</font>", code_style))
            continue

        if not raw.strip():
            i += 1
            continue

        if raw.strip() == "---":
            flow.append(Spacer(1, 4))
            flow.append(
                HRFlowable(
                    width="100%",
                    thickness=0.6,
                    color=colors.HexColor("#c5d0e0"),
                    spaceBefore=4,
                    spaceAfter=8,
                )
            )
            i += 1
            continue

        # headings
        m = re.match(r"^(#{1,4})\s+(.*)$", raw.strip())
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            style = {1: h1, 2: h2, 3: h3, 4: h4}[level]
            if level == 1 and first_h1:
                flow.append(Spacer(1, 6))
                flow.append(Paragraph(md_inline(text), title_style))
                first_h1 = False
            else:
                flow.append(Paragraph(md_inline(text), style))
            i += 1
            continue

        # table
        if raw.strip().startswith("|") and i + 1 < n and is_table_sep(lines[i + 1]):
            headers = split_table_row(raw)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_table_row(lines[i]))
                i += 1
            ncols = len(headers)
            data = [
                [Paragraph(md_inline(headers[c]) if c < len(headers) else "", table_head) for c in range(ncols)]
            ]
            for r in rows:
                data.append(
                    [
                        Paragraph(md_inline(r[c]) if c < len(r) else "", table_cell)
                        for c in range(ncols)
                    ]
                )
            # column widths
            usable = A4[0] - 30 * mm
            widths = [usable / ncols] * ncols
            # shrink first col if many cols
            if ncols >= 5:
                widths = [usable * 0.18] + [usable * 0.82 / (ncols - 1)] * (ncols - 1)
            elif ncols == 4:
                widths = [usable * 0.22, usable * 0.26, usable * 0.26, usable * 0.26]
            elif ncols == 3:
                widths = [usable * 0.28, usable * 0.36, usable * 0.36]
            elif ncols == 2:
                widths = [usable * 0.32, usable * 0.68]

            tbl = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c4a7c")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, -1), CJK),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7f9fc")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7f9fc"), colors.white]),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c5d0e0")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            flow.append(tbl)
            flow.append(Spacer(1, 6))
            continue

        # blockquote
        if raw.strip().startswith(">"):
            parts = []
            while i < n and lines[i].strip().startswith(">"):
                parts.append(lines[i].strip().lstrip(">").strip())
                i += 1
            flow.append(Paragraph(md_inline(" ".join(parts)), quote_style))
            continue

        # checkbox / list items (collect consecutive)
        if re.match(r"^(\s*)([-*+]|\d+\.)\s+", raw):
            items = []
            while i < n and re.match(r"^(\s*)([-*+]|\d+\.)\s+", lines[i]):
                lm = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", lines[i])
                indent = len(lm.group(1))
                marker = lm.group(2)
                content = lm.group(3).strip()
                # checkbox
                content = re.sub(r"^\[([ xX])\]\s*", "", content)
                bullet = "•" if not marker[0].isdigit() else marker
                items.append((indent, bullet, content))
                i += 1
            for indent, bullet, content in items:
                pad = 10 + min(indent, 20)
                st = ParagraphStyle(
                    f"li{indent}",
                    parent=bullet_style,
                    leftIndent=pad,
                )
                flow.append(Paragraph(f"{bullet} {md_inline(content)}", st))
            flow.append(Spacer(1, 3))
            continue

        # default paragraph
        # join consecutive non-empty plain lines
        buf = [raw.strip()]
        i += 1
        while i < n:
            nxt = lines[i]
            if (
                not nxt.strip()
                or nxt.strip().startswith("#")
                or nxt.strip().startswith("|")
                or nxt.strip().startswith(">")
                or nxt.strip().startswith("```")
                or nxt.strip() == "---"
                or re.match(r"^(\s*)([-*+]|\d+\.)\s+", nxt)
            ):
                break
            buf.append(nxt.strip())
            i += 1
        flow.append(Paragraph(md_inline(" ".join(buf)), body))

    return flow


def chrome(canv, doc):
    canv.saveState()
    canv.setFont(CJK, 8)
    canv.setFillColor(colors.HexColor("#666666"))
    canv.drawString(15 * mm, 10 * mm, "博士论文框架（内部对齐稿）")
    canv.drawRightString(A4[0] - 15 * mm, 10 * mm, f"{doc.page}")
    canv.restoreState()


def main():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md = f.read()

    # strip YAML-like header already in file; keep content
    story = parse_md(md)

    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title="博士论文框架与小论文安排",
        author="HICD",
    )
    doc.build(story, onFirstPage=chrome, onLaterPages=chrome)
    print(f"OK: {PDF_PATH}")


if __name__ == "__main__":
    main()
