# -*- coding: utf-8 -*-
"""Convert Ch1_2 problem formulation md to PDF: red accents, black text, no blue."""
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
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

MD_PATH = sys.argv[1]
PDF_PATH = sys.argv[2]

RED = colors.HexColor("#B00020")
DARK_RED = colors.HexColor("#8B0000")
LIGHT_RED_BG = colors.HexColor("#FFF5F5")
HEADER_BG = colors.HexColor("#B00020")
GRID = colors.HexColor("#D0D0D0")
GRAY = colors.HexColor("#555555")
BLACK = colors.black

CJK_TTF_CANDIDATES = {
    "Windows": [
        r"C:\Windows\Fonts\msyh.ttc",
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
CJK_BOLD = CJK
bold_path = r"C:\Windows\Fonts\msyhbd.ttc"
if os.path.exists(bold_path):
    try:
        pdfmetrics.registerFont(TTFont("CJK-Bold", bold_path, subfontIndex=0))
        CJK_BOLD = "CJK-Bold"
    except Exception:
        pass


def md_inline(text: str) -> str:
    t = escape(text)
    t = re.sub(r"`([^`]+)`", r"<font face='Courier' color='#333333'>\1</font>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<font color='#B00020'><b>\1</b></font>", t)
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
    leading=14.5,
    spaceAfter=5,
    textColor=BLACK,
    alignment=TA_LEFT,
)
h1 = ParagraphStyle(
    "H1CJK",
    parent=styles["Heading1"],
    fontName=CJK_BOLD,
    fontSize=15,
    leading=22,
    spaceBefore=14,
    spaceAfter=8,
    textColor=RED,
)
h2 = ParagraphStyle(
    "H2CJK",
    parent=styles["Heading2"],
    fontName=CJK_BOLD,
    fontSize=12,
    leading=17,
    spaceBefore=10,
    spaceAfter=5,
    textColor=DARK_RED,
)
h3 = ParagraphStyle(
    "H3CJK",
    parent=styles["Heading3"],
    fontName=CJK_BOLD,
    fontSize=10.5,
    leading=15,
    spaceBefore=8,
    spaceAfter=4,
    textColor=DARK_RED,
)
quote_style = ParagraphStyle(
    "QuoteCJK",
    parent=body,
    leftIndent=8,
    rightIndent=8,
    textColor=BLACK,
    backColor=LIGHT_RED_BG,
    borderPadding=5,
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
    textColor=RED,
    spaceAfter=6,
)
subtitle_style = ParagraphStyle(
    "SubCJK",
    parent=body,
    alignment=TA_CENTER,
    fontSize=9.5,
    textColor=GRAY,
    spaceAfter=10,
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
    textColor=BLACK,
)
table_head = ParagraphStyle(
    "CellHeadCJK",
    parent=table_cell,
    fontName=CJK_BOLD,
    textColor=colors.white,
)
mermaid_note = ParagraphStyle(
    "MermaidNote",
    parent=body,
    fontSize=8.5,
    textColor=GRAY,
    alignment=TA_CENTER,
    backColor=colors.HexColor("#F5F5F5"),
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
    backColor=colors.HexColor("#F5F5F5"),
    spaceBefore=4,
    spaceAfter=6,
    textColor=BLACK,
)


def parse_md(md: str):
    lines = md.splitlines()
    i = 0
    flow = []
    n = len(lines)
    first_h1 = True

    while i < n:
        raw = lines[i].rstrip("\n")

        if raw.strip().startswith("```"):
            lang = raw.strip()[3:].strip()
            i += 1
            block = []
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            if lang == "mermaid":
                # Render mermaid as a clean CJK step list (Courier has no CJK glyphs)
                step_style = ParagraphStyle(
                    "MermaidStep",
                    parent=body,
                    fontName=CJK,
                    fontSize=9,
                    leading=13,
                    leftIndent=10,
                    spaceAfter=2,
                    textColor=BLACK,
                    backColor=LIGHT_RED_BG,
                    borderPadding=3,
                )
                arrow_style = ParagraphStyle(
                    "MermaidArrow",
                    parent=body,
                    fontName=CJK_BOLD,
                    fontSize=10,
                    leading=12,
                    alignment=TA_CENTER,
                    textColor=RED,
                    spaceBefore=1,
                    spaceAfter=1,
                )
                for b in block:
                    s = b.strip()
                    if not s or s in ("flowchart TB", "flowchart LR", "graph TB", "graph LR"):
                        continue
                    if s in ("```", "```mermaid"):
                        continue
                    # strip mermaid syntax decorations
                    s2 = re.sub(r"^[A-Za-z0-9_]+(\[[\"\']?)", "", s)
                    s2 = s2.rstrip("]").rstrip("\"'")
                    s2 = s2.replace("[", "").replace("]", "")
                    s2 = s2.replace("{", "").replace("}", "")
                    s2 = s2.replace("(", "").replace(")", "")
                    s2 = s2.replace("-->", "→").replace("-.->", "→")
                    s2 = re.sub(r"^[A-Za-z0-9_]+\s*", "", s2) if re.match(r"^[A-Za-z0-9_]+\[", s.strip()) else s2
                    # keep readable content after arrows or as labels
                    label = s
                    mlab = re.search(r"\[(.*)\]", s)
                    if mlab:
                        label = mlab.group(1).strip("\"'")
                    elif "-->" in s or "-.->" in s:
                        parts = re.split(r"-\.->|-->", s)
                        label = " → ".join(p.strip().strip("[]\"'") for p in parts if p.strip())
                    else:
                        label = s.strip().strip("[]\"'")
                    label = label.replace("&", " & ")
                    if not label or set(label) <= set("-=| "):
                        continue
                    if "→" in label or label.startswith("→"):
                        flow.append(Paragraph(escape(label), arrow_style))
                    else:
                        flow.append(Paragraph(f"<font color='#B00020'>■</font> {escape(label)}", step_style))
            elif block:
                has_cjk = any(re.search(r"[一-鿿]", b) for b in block)
                if has_cjk or lang in ("text", ""):
                    step_style = ParagraphStyle(
                        "FlowStep",
                        parent=body,
                        fontName=CJK,
                        fontSize=9,
                        leading=13,
                        leftIndent=8,
                        spaceAfter=2,
                        textColor=BLACK,
                        backColor=LIGHT_RED_BG,
                        borderPadding=3,
                    )
                    arrow_style = ParagraphStyle(
                        "FlowArrow",
                        parent=body,
                        fontName=CJK_BOLD,
                        fontSize=10,
                        leading=12,
                        alignment=TA_CENTER,
                        textColor=RED,
                        spaceBefore=1,
                        spaceAfter=1,
                    )
                    for b in block:
                        s = b.strip()
                        if not s or set(s) <= set("-=| "):
                            continue
                        s = s.replace("-->", "→").replace("↓", "↓")
                        s = re.sub(r"\s+", " ", s)
                        if "→" in s or s == "↓" or s.startswith("↓"):
                            flow.append(Paragraph(escape(s), arrow_style))
                        else:
                            flow.append(
                                Paragraph(
                                    f"<font color='#B00020'>■</font> {escape(s)}",
                                    step_style,
                                )
                            )
                else:
                    text = "<br/>".join(escape(b) for b in block)
                    flow.append(
                        Paragraph(
                            f"<font face='Courier' color='#000000'>{text}</font>",
                            code_style,
                        )
                    )
            continue

        if not raw.strip():
            i += 1
            continue

        if raw.strip() == "---":
            flow.append(Spacer(1, 3))
            flow.append(
                HRFlowable(
                    width="100%",
                    thickness=0.7,
                    color=RED,
                    spaceBefore=4,
                    spaceAfter=8,
                )
            )
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", raw.strip())
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            style = {1: h1, 2: h2, 3: h3}.get(level, h3)
            if level == 1 and first_h1:
                flow.append(Spacer(1, 4))
                flow.append(Paragraph(md_inline(text), title_style))
                first_h1 = False
            else:
                flow.append(Paragraph(md_inline(text), style))
            i += 1
            continue

        if raw.strip().startswith("|") and i + 1 < n and is_table_sep(lines[i + 1]):
            headers = split_table_row(raw)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_table_row(lines[i]))
                i += 1
            ncols = len(headers)
            data = [
                [
                    Paragraph(md_inline(headers[c]) if c < len(headers) else "", table_head)
                    for c in range(ncols)
                ]
            ]
            for r in rows:
                data.append(
                    [
                        Paragraph(md_inline(r[c]) if c < len(r) else "", table_cell)
                        for c in range(ncols)
                    ]
                )
            usable = A4[0] - 28 * mm
            if ncols >= 5:
                widths = [usable * 0.16] + [usable * 0.84 / (ncols - 1)] * (ncols - 1)
            elif ncols == 4:
                widths = [usable * 0.22, usable * 0.26, usable * 0.26, usable * 0.26]
            elif ncols == 3:
                widths = [usable * 0.28, usable * 0.36, usable * 0.36]
            elif ncols == 2:
                widths = [usable * 0.32, usable * 0.68]
            else:
                widths = [usable / ncols] * ncols

            tbl = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, -1), CJK),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_RED_BG]),
                        ("GRID", (0, 0), (-1, -1), 0.3, GRID),
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

        if raw.strip().startswith(">"):
            parts = []
            while i < n and lines[i].strip().startswith(">"):
                parts.append(lines[i].strip().lstrip(">").strip())
                i += 1
            flow.append(Paragraph(md_inline(" ".join(parts)), quote_style))
            continue

        if re.match(r"^(\s*)([-*+]|\d+\.)\s+", raw):
            while i < n and re.match(r"^(\s*)([-*+]|\d+\.)\s+", lines[i]):
                lm = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", lines[i])
                indent = len(lm.group(1))
                marker = lm.group(2)
                content = lm.group(3).strip()
                content = re.sub(r"^\[([ xX])\]\s*", "", content)
                bullet = "•" if not marker[0].isdigit() else marker
                pad = 10 + min(indent, 24)
                st = ParagraphStyle(f"li{indent}", parent=bullet_style, leftIndent=pad)
                flow.append(Paragraph(f"{bullet} {md_inline(content)}", st))
                i += 1
            flow.append(Spacer(1, 3))
            continue

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
    canv.setFillColor(GRAY)
    canv.drawString(14 * mm, 10 * mm, "论文框架梳理")
    canv.drawRightString(A4[0] - 14 * mm, 10 * mm, f"{doc.page}")
    canv.setStrokeColor(RED)
    canv.setLineWidth(0.4)
    canv.line(14 * mm, 13 * mm, A4[0] - 14 * mm, 13 * mm)
    canv.restoreState()


def main():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md = f.read()
    story = parse_md(md)
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title="论文框架梳理",
        author="HICD",
    )
    doc.build(story, onFirstPage=chrome, onLaterPages=chrome)
    print(f"OK: {PDF_PATH}")


if __name__ == "__main__":
    main()
