# -*- coding: utf-8 -*-
"""组会精读 PPT：圆角卡片 + 模块示意图 + 结果柱状图，文字精简，面向马磊。"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

OUT = Path(r"F:\mambacd\home\HICD_v7\docs\ISPRS2024_精读_王瑞坤.pptx")
ASSETS = Path(r"F:\mambacd\home\HICD_v7\docs\assets")
FIG_ARCH = ASSETS / "hginet_flowchart.png"

PURPLE = RGBColor(0x6E, 0x0F, 0x6D)
RED = RGBColor(0xC0, 0x00, 0x00)
BRIGHT_RED = RGBColor(0xFF, 0x00, 0x00)
INK = RGBColor(0x1A, 0x1A, 0x1A)
BLUE = RGBColor(0x1F, 0x4E, 0x79)
TEAL = RGBColor(0x0D, 0x73, 0x7A)
ORANGE = RGBColor(0xC4, 0x5C, 0x26)
GRAY = RGBColor(0x66, 0x66, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_PURPLE = RGBColor(0xF3, 0xE8, 0xF3)
LIGHT_BLUE = RGBColor(0xE8, 0xF0, 0xF8)
LIGHT_RED = RGBColor(0xFB, 0xEB, 0xEB)
LIGHT_GREEN = RGBColor(0xE8, 0xF5, 0xE9)
LIGHT_YELLOW = RGBColor(0xFF, 0xF8, 0xE1)
LIGHT_GRAY = RGBColor(0xF5, 0xF5, 0xF5)
CARD_BG = RGBColor(0xFA, 0xFA, 0xFC)
CARD_BORDER = RGBColor(0xD0, 0xC4, 0xD8)

FONT = "微软雅黑"
L = Emu(300000)
R = Emu(8800000)
FULL_W = Emu(R - L)
HALF = Emu(4100000)
RIGHT = Emu(4650000)


def set_cjk(run, name=FONT):
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    succ = {
        "a:ea": ("a:cs", "a:sym", "a:hlinkClick", "a:hlinkMouseOver", "a:rtl", "a:extLst"),
        "a:cs": ("a:sym", "a:hlinkClick", "a:hlinkMouseOver", "a:rtl", "a:extLst"),
    }
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.insert_element_before(el, *succ[tag])
        el.set("typeface", name)


def style(run, size=14, bold=False, color=INK, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    set_cjk(run, name)


def rounded_card(slide, x, y, w, h, fill=CARD_BG, line=CARD_BORDER, radius=True):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(shape_type, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    sh.line.width = Pt(1)
    try:
        sh.adjustments[0] = 0.1
    except Exception:
        pass
    return sh


def add_text(slide, x, y, w, h, segments, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    if isinstance(segments, str):
        segments = [(segments, {})]
    p = tf.paragraphs[0]
    p.alignment = align
    for text, kw in segments:
        if text == "\n":
            p = tf.add_paragraph()
            p.alignment = kw.get("align", align)
            continue
        run = p.add_run()
        run.text = text
        style(run, size=kw.get("size", 14), bold=kw.get("bold", False),
              color=kw.get("color", INK), name=kw.get("font", FONT))
    return box


def card_text(slide, x, y, w, h, title, lines, *, fill=CARD_BG, title_color=PURPLE,
              line_color=CARD_BORDER, title_size=14, body_size=12):
    """圆角卡片 + 标题 + 少量要点（每行一条，短句）"""
    rounded_card(slide, x, y, w, h, fill=fill, line=line_color)
    pad = Emu(120000)
    add_text(slide, Emu(x + pad), Emu(y + 70000), Emu(w - 2 * pad), Emu(280000),
             [(title, {"size": title_size, "bold": True, "color": title_color})])
    body_y = Emu(y + 380000)
    body_h = Emu(h - 450000)
    box = slide.shapes.add_textbox(Emu(x + pad), body_y, Emu(w - 2 * pad), body_h)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for line in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = ("· " if not line.startswith(" ") else "") + line.strip()
        style(run, size=body_size, color=INK)


def header(slide, page=None, tag=None):
    add_text(slide, L, Emu(90000), Emu(1800000), Emu(340000),
             [("论文阅读", {"size": 18, "bold": True, "color": PURPLE})])
    if tag:
        add_text(slide, Emu(2000000), Emu(100000), Emu(6500000), Emu(300000),
                 [(tag, {"size": 11, "color": GRAY})], align=PP_ALIGN.RIGHT)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, L, Emu(450000), FULL_W, Emu(16000))
    bar.fill.solid()
    bar.fill.fore_color.rgb = PURPLE
    bar.line.fill.background()
    if page:
        add_text(slide, Emu(8300000), Emu(7000000), Emu(550000), Emu(240000),
                 [(str(page), {"size": 10, "color": GRAY})], align=PP_ALIGN.RIGHT)


def slide_title(slide, text, y=Emu(550000)):
    add_text(slide, L, y, FULL_W, Emu(380000),
             [(text, {"size": 20, "bold": True, "color": PURPLE})])


def new(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def arrow(slide, x1, y1, x2, y2, color=PURPLE, width=1.5):
    conn = slide.shapes.add_connector(1, x1, y1, x2, y2)  # straight
    conn.line.color.rgb = color
    conn.line.width = Pt(width)
    try:
        conn.line.end_arrowhead = True
    except Exception:
        pass
    return conn


def mini_box(slide, x, y, w, h, label, *, fill=LIGHT_BLUE, line=BLUE, font_size=11,
             bold=True, font_color=INK):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    sh.line.width = Pt(1.2)
    try:
        sh.adjustments[0] = 0.15
    except Exception:
        pass
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = label
    style(run, size=font_size, bold=bold, color=font_color)
    return sh


def pic(slide, path, x, y, w, h=None):
    p = Path(path)
    if not p.exists():
        return None
    if h is None:
        return slide.shapes.add_picture(str(p), x, y, width=w)
    return slide.shapes.add_picture(str(p), x, y, width=w, height=h)


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def bar_chart(slide, x, y, w, h, categories, series, *, title=None,
              colors=None, max_val=None, y_label=None):
    """纯形状柱状图：series = list of (name, [values])"""
    if title:
        add_text(slide, x, Emu(y - 280000), w, Emu(260000),
                 [(title, {"size": 13, "bold": True, "color": PURPLE})])
    # plot area
    pad_left = Emu(500000)
    pad_bottom = Emu(500000)
    pad_top = Emu(200000)
    plot_x = Emu(x + pad_left)
    plot_y = Emu(y + pad_top)
    plot_w = Emu(w - pad_left - 100000)
    plot_h = Emu(h - pad_top - pad_bottom)
    # background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, plot_x, plot_y, plot_w, plot_h)
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(0xF7, 0xF7, 0xFA)
    bg.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)

    all_vals = [v for _, vals in series for v in vals]
    vmax = max_val or (max(all_vals) * 1.15 if all_vals else 1)
    n_cat = len(categories)
    n_ser = len(series)
    group_w = plot_w / max(n_cat, 1)
    bar_w = int(group_w * 0.7 / max(n_ser, 1))
    if colors is None:
        colors = [PURPLE, TEAL, ORANGE, BLUE]

    for ci, cat in enumerate(categories):
        gx = Emu(plot_x + ci * group_w + int(group_w * 0.15))
        for si, (sname, vals) in enumerate(series):
            v = vals[ci]
            bh = int(plot_h * (v / vmax)) if vmax else 0
            bx = Emu(gx + si * bar_w)
            by = Emu(plot_y + plot_h - bh)
            bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, bx, by, Emu(max(bar_w - 40000, 80000)), Emu(bh))
            bar.fill.solid()
            bar.fill.fore_color.rgb = colors[si % len(colors)]
            bar.line.fill.background()
            # value label
            add_text(slide, Emu(bx - 40000), Emu(by - 220000), Emu(bar_w + 80000), Emu(200000),
                     [(f"{v:.1f}" if isinstance(v, float) else str(v), {"size": 9, "color": colors[si % len(colors)], "bold": True})],
                     align=PP_ALIGN.CENTER)
        # category label
        add_text(slide, Emu(plot_x + ci * group_w), Emu(plot_y + plot_h + 30000), Emu(group_w), Emu(220000),
                 [(cat, {"size": 10, "color": INK})], align=PP_ALIGN.CENTER)
    # legend
    lx = plot_x
    ly = Emu(plot_y + plot_h + 250000)
    for si, (sname, _) in enumerate(series):
        sw = Emu(140000)
        sq = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(lx + si * 1400000), ly, sw, Emu(140000))
        sq.fill.solid()
        sq.fill.fore_color.rgb = colors[si % len(colors)]
        sq.line.fill.background()
        add_text(slide, Emu(lx + si * 1400000 + 180000), Emu(ly - 30000), Emu(1100000), Emu(200000),
                 [(sname, {"size": 10, "color": GRAY})])
    if y_label:
        add_text(slide, Emu(x), Emu(y + 20000), Emu(450000), Emu(plot_h),
                 [(y_label, {"size": 10, "color": GRAY})], align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)


def main():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # ========== 1 封面 ==========
    s = new(prs)
    rounded_card(s, Emu(500000), Emu(1200000), Emu(8000000), Emu(4800000),
                 fill=LIGHT_PURPLE, line=PURPLE)
    add_text(s, Emu(700000), Emu(1450000), Emu(7600000), Emu(400000),
             [("文献精读 · 组会汇报", {"size": 16, "bold": True, "color": PURPLE})],
             align=PP_ALIGN.CENTER)
    add_text(s, Emu(700000), Emu(2000000), Emu(7600000), Emu(900000),
             [("HGINet：用「图」建模双时相关系的\n语义变化检测方法",
               {"size": 24, "bold": True, "color": INK})],
             align=PP_ALIGN.CENTER)
    add_text(s, Emu(700000), Emu(3100000), Emu(7600000), Emu(800000),
             [("Long et al., ISPRS J. Photogramm. Remote Sens., 2024\n福州大学 / 特文特大学 ITC",
               {"size": 14, "color": GRAY})],
             align=PP_ALIGN.CENTER)
    add_text(s, Emu(700000), Emu(4200000), Emu(7600000), Emu(400000),
             [("汇报人：王瑞坤", {"size": 16, "bold": True})],
             align=PP_ALIGN.CENTER)
    add_text(s, Emu(700000), Emu(4800000), Emu(7600000), Emu(350000),
             [("核心：问题 → 怎么做 → 效果如何 → 启发", {"size": 13, "color": PURPLE, "bold": True})],
             align=PP_ALIGN.CENTER)
    notes(s, "开场：一篇2024顶刊，讲语义变化检测如何用图关系提升效果。")

    # ========== 2 一句话问题 ==========
    s = new(prs)
    header(s, 2)
    slide_title(s, "这篇论文要解决什么问题？")
    # three big cards
    cards = [
        (Emu(350000), "要回答", ["双时相影像上：", "哪里变了？", "变成什么类别？"], LIGHT_BLUE, BLUE),
        (Emu(3200000), "难点", ["变化区域占比很小", "复杂城区易误判", "类别之间易混淆"], LIGHT_RED, RED),
        (Emu(6050000), "论文抓手", ["显式建模「关系」", "时序相关 + 图交互", "而不是只做像素差"], LIGHT_YELLOW, ORANGE),
    ]
    for x, title, lines, fill, tc in cards:
        card_text(s, x, Emu(1200000), Emu(2600000), Emu(2800000),
                  title, lines, fill=fill, title_color=tc, title_size=16, body_size=14)
        # arrows between cards
    add_text(s, Emu(2900000), Emu(2400000), Emu(300000), Emu(300000),
             [("→", {"size": 22, "bold": True, "color": PURPLE})], align=PP_ALIGN.CENTER)
    add_text(s, Emu(5750000), Emu(2400000), Emu(300000), Emu(300000),
             [("→", {"size": 22, "bold": True, "color": PURPLE})], align=PP_ALIGN.CENTER)
    rounded_card(s, Emu(350000), Emu(4300000), Emu(8300000), Emu(1800000),
                 fill=WHITE, line=PURPLE)
    add_text(s, Emu(550000), Emu(4500000), Emu(7900000), Emu(300000),
             [("一句话", {"size": 14, "bold": True, "color": BRIGHT_RED})])
    add_text(s, Emu(550000), Emu(4900000), Emu(7900000), Emu(900000),
             [("语义变化检测不仅要「看出变化」，还要「说清变成什么」；作者认为关键在于建模双时相特征之间的关系。",
               {"size": 16, "bold": True})])
    notes(s, "三张卡：任务—难点—抓手。核心句放底部大卡。")

    # ========== 3 为什么难：示意 ==========
    s = new(prs)
    header(s, 3)
    slide_title(s, "为什么普通方法不够？—— 从 BCD 到 SCD")
    # BCD schematic
    rounded_card(s, Emu(350000), Emu(1100000), Emu(4000000), Emu(2600000),
                 fill=LIGHT_GRAY, line=GRAY)
    add_text(s, Emu(500000), Emu(1200000), Emu(3700000), Emu(280000),
             [("二值变化检测 BCD", {"size": 14, "bold": True, "color": GRAY})])
    mini_box(s, Emu(550000), Emu(1650000), Emu(1000000), Emu(450000), "T1", fill=WHITE, line=GRAY)
    mini_box(s, Emu(550000), Emu(2300000), Emu(1000000), Emu(450000), "T2", fill=WHITE, line=GRAY)
    mini_box(s, Emu(1750000), Emu(1950000), Emu(900000), Emu(800000), "对比", fill=WHITE, line=GRAY)
    mini_box(s, Emu(2850000), Emu(1950000), Emu(1100000), Emu(800000), "变/不变", fill=LIGHT_GRAY, line=GRAY, font_size=10)
    add_text(s, Emu(500000), Emu(3000000), Emu(3600000), Emu(400000),
             [("只回答：变没变", {"size": 13, "color": GRAY})])

    # SCD schematic
    rounded_card(s, Emu(4550000), Emu(1100000), Emu(4100000), Emu(2600000),
                 fill=LIGHT_PURPLE, line=PURPLE)
    add_text(s, Emu(4700000), Emu(1200000), Emu(3800000), Emu(280000),
             [("语义变化检测 SCD（本文）", {"size": 14, "bold": True, "color": PURPLE})])
    mini_box(s, Emu(4750000), Emu(1650000), Emu(900000), Emu(450000), "T1", fill=WHITE, line=PURPLE)
    mini_box(s, Emu(4750000), Emu(2300000), Emu(900000), Emu(450000), "T2", fill=WHITE, line=PURPLE)
    mini_box(s, Emu(5800000), Emu(1950000), Emu(1000000), Emu(800000), "联合\n推理", fill=WHITE, line=PURPLE, font_size=10)
    mini_box(s, Emu(7000000), Emu(1700000), Emu(1300000), Emu(550000), "哪里变", fill=LIGHT_RED, line=RED, font_size=10)
    mini_box(s, Emu(7000000), Emu(2400000), Emu(1300000), Emu(550000), "变成什么", fill=LIGHT_YELLOW, line=ORANGE, font_size=10)
    add_text(s, Emu(4700000), Emu(3000000), Emu(3800000), Emu(400000),
             [("同时回答：位置 + 类别（from→to）", {"size": 13, "color": PURPLE, "bold": True})])

    # bottom insight cards
    card_text(s, Emu(350000), Emu(4000000), Emu(4000000), Emu(2200000),
              "已有方法的问题",
              ["只做像素差 → 关系信息丢失", "先分类再比 → 误差累积", "变化类少、背景多 → 指标虚高"],
              fill=LIGHT_RED, title_color=RED, body_size=13)
    card_text(s, Emu(4550000), Emu(4000000), Emu(4100000), Emu(2200000),
              "HGINet 的思路",
              ["先提好特征（MPANet）", "再稳住未变化（TC）", "最后图交互理清变化（SDI）"],
              fill=LIGHT_GREEN, title_color=TEAL, body_size=13)
    notes(s, "左BCD右SCD对照，底部两点：问题在哪、思路是什么。")

    # ========== 4 总框架 真图 ==========
    s = new(prs)
    header(s, 4)
    slide_title(s, "方法总览：三步走（论文架构图）")
    if FIG_ARCH.exists():
        pic(s, FIG_ARCH, Emu(350000), Emu(1000000), Emu(8300000), Emu(2600000))
    add_text(s, Emu(350000), Emu(3650000), Emu(8300000), Emu(250000),
             [("图：HGINet 整体架构（官方代码/论文 Fig.1）", {"size": 10, "color": GRAY})])
    # three step cards
    steps = [
        (Emu(350000), "① MPANet", "提取双时相\n多层级特征", LIGHT_BLUE, BLUE),
        (Emu(3050000), "② TC", "建模时序相关\n稳住「没变」", LIGHT_GREEN, TEAL),
        (Emu(5750000), "③ SDI", "图卷积交互\n理清「怎么变」", LIGHT_YELLOW, ORANGE),
    ]
    for x, t, body, fill, tc in steps:
        rounded_card(s, x, Emu(4100000), Emu(2500000), Emu(1800000), fill=fill, line=tc)
        add_text(s, Emu(x + 120000), Emu(4250000), Emu(2260000), Emu(350000),
                 [(t, {"size": 16, "bold": True, "color": tc})], align=PP_ALIGN.CENTER)
        add_text(s, Emu(x + 120000), Emu(4700000), Emu(2260000), Emu(900000),
                 [(body, {"size": 14, "color": INK})], align=PP_ALIGN.CENTER)
    add_text(s, Emu(350000), Emu(6100000), Emu(8300000), Emu(400000),
             [("输出：T1/T2 分割图 + 语义变化图（from→to）", {"size": 14, "bold": True, "color": PURPLE})],
             align=PP_ALIGN.CENTER)
    notes(s, "先看图再点三步：特征—不变—变化。")

    # ========== 5 模块1 MPANet 示意 ==========
    s = new(prs)
    header(s, 5)
    slide_title(s, "模块① MPANet：把特征「提清楚」")
    # schematic pipeline
    y0 = Emu(1300000)
    mini_box(s, Emu(400000), y0, Emu(1400000), Emu(700000), "T1 / T2\n影像", fill=LIGHT_GRAY, line=GRAY, font_size=12)
    mini_box(s, Emu(2200000), y0, Emu(1600000), Emu(700000), "PVT-V2\n骨干", fill=LIGHT_BLUE, line=BLUE, font_size=12)
    mini_box(s, Emu(4200000), y0, Emu(1400000), Emu(700000), "F1–F4\n多层级", fill=WHITE, line=BLUE, font_size=12)
    mini_box(s, Emu(6000000), y0, Emu(2400000), Emu(700000), "MSCF + SFSR\n上下文 + 边界 refine", fill=LIGHT_GREEN, line=TEAL, font_size=11)
    for x1, x2 in [(1800000, 2200000), (3800000, 4200000), (5600000, 6000000)]:
        add_text(s, Emu(x1 - 20000), Emu(y0 + 220000), Emu(400000), Emu(300000),
                 [("→", {"size": 18, "bold": True, "color": PURPLE})], align=PP_ALIGN.CENTER)

    card_text(s, Emu(350000), Emu(2400000), Emu(4000000), Emu(2200000),
              "核心想法",
              ["Transformer 骨干提取多尺度特征", "远程上下文（条带注意力）+ 近程细节", "高低层融合，边界更干净"],
              fill=LIGHT_BLUE, title_color=BLUE, body_size=13)
    card_text(s, Emu(4550000), Emu(2400000), Emu(4100000), Emu(2200000),
              "为什么需要",
              ["地物大小、形状差异大", "只靠卷积长程信息不够", "特征不好，后面关系也学不准"],
              fill=CARD_BG, title_color=PURPLE, body_size=13)
    card_text(s, Emu(350000), Emu(4800000), Emu(8300000), Emu(1400000),
              "一句话",
              ["先把「看得清」的特征做出来，后面的关系建模才有基础。"],
              fill=LIGHT_YELLOW, title_color=ORANGE, body_size=15)
    notes(s, "MPANet不展开公式，只讲：多尺度+上下文+边界。")

    # ========== 6 模块2 TC 示意 ==========
    s = new(prs)
    header(s, 6)
    slide_title(s, "模块② TC：把「没变」的部分稳住")
    # schematic
    rounded_card(s, Emu(350000), Emu(1100000), Emu(8300000), Emu(2800000),
                 fill=CARD_BG, line=CARD_BORDER)
    mini_box(s, Emu(550000), Emu(1500000), Emu(1300000), Emu(600000), "T1特征", fill=LIGHT_BLUE, line=BLUE)
    mini_box(s, Emu(550000), Emu(2400000), Emu(1300000), Emu(600000), "T2特征", fill=LIGHT_GREEN, line=TEAL)
    mini_box(s, Emu(2400000), Emu(1900000), Emu(1800000), Emu(1000000), "Q / K / V\n交叉参照", fill=WHITE, line=PURPLE, font_size=12)
    mini_box(s, Emu(4800000), Emu(1900000), Emu(1800000), Emu(1000000), "全局相关\n（non-local）", fill=LIGHT_PURPLE, line=PURPLE, font_size=12)
    mini_box(s, Emu(7000000), Emu(1900000), Emu(1300000), Emu(1000000), "未变化\n更相似", fill=LIGHT_YELLOW, line=ORANGE, font_size=12)
    for x1, x2 in [(1850000, 2400000), (4200000, 4800000), (6600000, 7000000)]:
        add_text(s, Emu(x1), Emu(2200000), Emu(400000), Emu(300000),
                 [("→", {"size": 18, "bold": True, "color": PURPLE})])
    add_text(s, Emu(550000), Emu(3300000), Emu(7800000), Emu(350000),
             [("T1、T2 互相「对答案」：该一样的地方尽量一样，减少假变化", {"size": 14, "bold": True, "color": INK})],
             align=PP_ALIGN.CENTER)

    card_text(s, Emu(350000), Emu(4200000), Emu(4000000), Emu(2000000),
              "核心想法",
              ["变化检测里「没变」占大多数", "显式利用双时相相关", "假变化减少，变化类更可信"],
              fill=LIGHT_GREEN, title_color=TEAL, body_size=13)
    card_text(s, Emu(4550000), Emu(4200000), Emu(4100000), Emu(2000000),
              "类比",
              ["像两个人对答案：", "同一道题答案一致 → 大概率没错", "不一致的地方 → 才可能是变化"],
              fill=CARD_BG, title_color=PURPLE, body_size=13)
    notes(s, "TC讲类比：对答案。强调未变化主导。")

    # ========== 7 模块3 SDI 示意 ==========
    s = new(prs)
    header(s, 7)
    slide_title(s, "模块③ SDI：用「图」理清怎么变（核心）")
    # graph schematic
    rounded_card(s, Emu(350000), Emu(1050000), Emu(5000000), Emu(3200000),
                 fill=CARD_BG, line=PURPLE)
    add_text(s, Emu(500000), Emu(1150000), Emu(4700000), Emu(280000),
             [("示意图：把多层特征看成图上的节点", {"size": 12, "bold": True, "color": PURPLE})])
    # nodes
    mini_box(s, Emu(700000), Emu(1700000), Emu(1400000), Emu(550000), "语义特征层", fill=LIGHT_BLUE, line=BLUE, font_size=10)
    mini_box(s, Emu(2400000), Emu(1700000), Emu(1400000), Emu(550000), "差分特征层", fill=LIGHT_RED, line=RED, font_size=10)
    mini_box(s, Emu(1500000), Emu(2600000), Emu(1600000), Emu(550000), "二者组合", fill=LIGHT_YELLOW, line=ORANGE, font_size=10)
    mini_box(s, Emu(3400000), Emu(2600000), Emu(1400000), Emu(550000), "GCN\n交互推理", fill=LIGHT_PURPLE, line=PURPLE, font_size=11)
    mini_box(s, Emu(3400000), Emu(1700000), Emu(1400000), Emu(550000), "变化图", fill=LIGHT_GREEN, line=TEAL, font_size=11)
    # simple edges as text arrows
    add_text(s, Emu(2050000), Emu(1800000), Emu(350000), Emu(300000),
             [("↔", {"size": 16, "bold": True, "color": PURPLE})])
    add_text(s, Emu(1200000), Emu(2250000), Emu(350000), Emu(300000),
             [("↘", {"size": 16, "bold": True, "color": PURPLE})])
    add_text(s, Emu(3000000), Emu(2300000), Emu(350000), Emu(300000),
             [("→", {"size": 16, "bold": True, "color": PURPLE})])
    add_text(s, Emu(4800000), Emu(2000000), Emu(350000), Emu(300000),
             [("→", {"size": 16, "bold": True, "color": PURPLE})])

    card_text(s, Emu(5600000), Emu(1050000), Emu(3050000), Emu(3200000),
              "核心想法",
              ["不只做像素相减", "多层信息互相「对话」", "图擅长表达不规则关系", "复杂场景变化更说得清"],
              fill=LIGHT_YELLOW, title_color=ORANGE, body_size=13)

    card_text(s, Emu(350000), Emu(4500000), Emu(4000000), Emu(1800000),
              "消融告诉你",
              ["SDI 贡献最大", "SECOND：OS +3.9%", "F1scd +4.6%"],
              fill=LIGHT_RED, title_color=RED, body_size=14)
    card_text(s, Emu(4550000), Emu(4500000), Emu(4100000), Emu(1800000),
              "一句话",
              ["关系建模比「再加深网络」更关键——至少在这篇的任务里是这样。"],
              fill=LIGHT_PURPLE, title_color=PURPLE, body_size=14)
    notes(s, "SDI是全文最亮点：图交互+消融支撑。示意图讲节点是多层特征。")

    # ========== 8 结果图 ==========
    s = new(prs)
    header(s, 8)
    slide_title(s, "实验效果：四个数据集（论文最优结果）")
    bar_chart(
        s, Emu(350000), Emu(1150000), Emu(5200000), Emu(3200000),
        ["SECOND", "HRSCD", "Fuzhou", "Xiamen"],
        [("F1scd (%)", [59.5, 64.1, 64.5, 84.9])],
        title="变化类 F1（越高越好）",
        colors=[PURPLE],
        max_val=100,
    )
    bar_chart(
        s, Emu(5700000), Emu(1150000), Emu(3000000), Emu(3200000),
        ["SE", "HR", "FZ", "XM"],
        [("SeK (%)", [19.3, 14.6, 18.3, 51.1])],
        title="SeK（关注变化类）",
        colors=[TEAL],
        max_val=60,
    )
    card_text(s, Emu(350000), Emu(4600000), Emu(4000000), Emu(1800000),
              "怎么读这两张图",
              ["厦门高：变化类型少、目标较规整", "福州/公开集更低：更复杂", "SeK 防止背景把分数抬高"],
              fill=LIGHT_BLUE, title_color=BLUE, body_size=12)
    card_text(s, Emu(4550000), Emu(4600000), Emu(4100000), Emu(1800000),
              "对比结论",
              ["9 个基线中总体最优", "复杂数据集上优势更明显", "福州→厦门可迁移"],
              fill=LIGHT_GREEN, title_color=TEAL, body_size=12)
    notes(s, "结果用柱状图，先读F1再读SeK；强调厦门数字高不代表任务简单。")

    # ========== 9 消融图 ==========
    s = new(prs)
    header(s, 9)
    slide_title(s, "模块有没有用？—— 看消融")
    bar_chart(
        s, Emu(350000), Emu(1200000), Emu(5000000), Emu(3000000),
        ["Baseline", "+MSCF", "+SFSR", "+TC", "+SDI", "完整"],
        [("相对贡献示意（越大越好）", [55.0, 56.5, 57.2, 57.8, 59.5, 59.5])],
        title="SECOND 上 F1scd 变化趋势（示意）",
        colors=[ORANGE],
        max_val=65,
    )
    add_text(s, Emu(350000), Emu(4300000), Emu(5000000), Emu(300000),
             [("说明：SDI 单独增益最显著（论文报告 OS+3.9 / F1scd+4.6）", {"size": 11, "color": GRAY})])
    card_text(s, Emu(5600000), Emu(1200000), Emu(3050000), Emu(2800000),
              "作者局限",
              ["复杂边界仍易错", "公开数据标签噪声大", "未做实例级清单"],
              fill=LIGHT_RED, title_color=RED, body_size=13)
    card_text(s, Emu(350000), Emu(4800000), Emu(8300000), Emu(1500000),
              "对读论文的提醒",
              ["先问「去掉谁掉最多」，再问「为什么」——关系结构（图）在这里比单纯加层更关键。"],
              fill=LIGHT_YELLOW, title_color=ORANGE, body_size=14)
    notes(s, "消融柱状为示意趋势，精确数字以论文为准，口头补充SDI增益。")

    # ========== 10 启发 ==========
    s = new(prs)
    header(s, 10)
    slide_title(s, "启发与可讨论的问题")
    card_text(s, Emu(350000), Emu(1100000), Emu(2650000), Emu(2400000),
              "启发 1", ["变化检测的瓶颈常在「关系」", "时序相关值得单独设计"], fill=LIGHT_BLUE, title_color=BLUE, body_size=13)
    card_text(s, Emu(3150000), Emu(1100000), Emu(2650000), Emu(2400000),
              "启发 2", ["图结构适合不规则关系", "关键是：图建在哪一层？"], fill=LIGHT_PURPLE, title_color=PURPLE, body_size=13)
    card_text(s, Emu(5950000), Emu(1100000), Emu(2700000), Emu(2400000),
              "启发 3", ["「未变化」不是配角", "稳住背景，变化才可信"], fill=LIGHT_GREEN, title_color=TEAL, body_size=13)

    rounded_card(s, Emu(350000), Emu(3800000), Emu(8300000), Emu(2400000),
                 fill=WHITE, line=PURPLE)
    add_text(s, Emu(550000), Emu(4000000), Emu(7900000), Emu(350000),
             [("组会可讨论（2 个即可）", {"size": 15, "bold": True, "color": BRIGHT_RED})])
    add_text(s, Emu(550000), Emu(4450000), Emu(7900000), Emu(1400000),
             [("1. 关系应该建在「特征」上，还是建在「语义/目标结构」上？\n2. 当变化目标很稀疏、类别组合极不均衡时，是结构更重要，还是样本与策略更重要？",
               {"size": 15, "color": INK})])
    notes(s, "启发三卡+底部两个讨论题，给马磊开口讨论的空间。")

    # ========== 11 文献 ==========
    s = new(prs)
    header(s, 11)
    slide_title(s, "文献信息")
    card_text(s, Emu(350000), Emu(1100000), Emu(8300000), Emu(2800000),
              "引用",
              ["Long J., Li M., Wang X., Stein A., 2024.",
               "ISPRS J. Photogramm. Remote Sens. 211, 318–335.",
               "DOI: 10.1016/j.isprsjprs.2024.04.012",
               "代码: github.com/long123524/HGINet-torch"],
              fill=LIGHT_GRAY, title_color=PURPLE, body_size=14)
    card_text(s, Emu(350000), Emu(4200000), Emu(8300000), Emu(1600000),
              "全文依据",
              ["Zotero 本地 PDF（QBLZQPBN，18 页）已通读；架构图为论文/官方代码 Fig.1。"],
              fill=CARD_BG, title_color=GRAY, body_size=13)
    add_text(s, Emu(350000), Emu(6100000), Emu(8300000), Emu(400000),
             [("谢谢，请老师批评指正", {"size": 18, "bold": True, "color": PURPLE})],
             align=PP_ALIGN.CENTER)
    notes(s, "收尾。")

    prs.save(str(OUT))
    print("SAVED", OUT)
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    n = sum(1 for sl in prs.slides for sh in sl.shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE)
    print("slides", len(prs.slides), "pictures", n)


if __name__ == "__main__":
    main()
