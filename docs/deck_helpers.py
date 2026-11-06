import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
D = json.loads((DOCS / "deck_data.json").read_text())
INFO, PERF, RUL, ATT, TIM, IMP = D["info"], D["perf"], D["rules"], D["attributes"], D["timing"], D["importance"]

NAVY, INK, ICE, TINT, ORANGE, SLATE, WHITE = "1E2761", "1B1F3B", "D7E1F5", "EEF2FB", "E8734A", "5B6B8C", "FFFFFF"
BLUE, C_ORANGE, AQUA, YELLOW, VIOLET = "2A78D6", "EB6834", "1BAF7A", "EDA100", "4A3AA7"
HEAD, BODY, MONO = "Cambria", "Calibri", "Consolas"
DS = ["pima", "early", "cdc"]
DS_NAME = {"pima": "Pima", "early": "Early Stage", "cdc": "CDC"}
DS_FULL = {"pima": "Pima Indians Diabetes", "early": "Early Stage Diabetes Risk", "cdc": "CDC Health Indicators"}
EXPL = [("rule_consequent", "Rules: THEN-part", BLUE), ("rule_antecedent", "Rules: IF-part", C_ORANGE),
        ("rule_full", "Rules: IF + THEN", AQUA), ("shap_lgbm", "SHAP on LightGBM", YELLOW)]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
LAYOUT = prs.slide_layouts[5]
count = [0]


def rgb(h):
    return RGBColor.from_string(h)


def text(slide, x, y, w, h, paras, size=14, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space=4, bullets=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        runs = [(para, {})] if isinstance(para, str) else para
        if bullets:
            runs = [("•  ", {"color": ORANGE, "bold": True})] + list(runs)
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.color.rgb = rgb(o.get("color", color))
    return box


def card(slide, x, y, w, h, fill=TINT, radius=0.06):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = radius
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def dot(slide, x, y, d, fill, label=None, size=14, color=WHITE):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    if label:
        tf = s.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = label
        r.font.size, r.font.bold, r.font.name = Pt(size), True, BODY
        r.font.color.rgb = rgb(color)
    return s


def arrow(slide, x, y, w=0.35, h=0.3, fill=ORANGE):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False


def new_slide(kicker, title, sub=None, notes="", dark=False):
    count[0] += 1
    s = prs.slides.add_slide(LAYOUT)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(INK if dark else WHITE)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.5), Inches(12.1), Inches(0.75)
    tf = t.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.text = title
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    f = p.runs[0].font
    f.name, f.size, f.bold = HEAD, Pt(28), True
    f.color.rgb = rgb(WHITE if dark else NAVY)
    text(s, 12.0, 7.02, 0.75, 0.3, str(count[0]), size=10, color=SLATE, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def code_box(slide, x, y, w, h, path, start, end=None, max_lines=40, size=11.5, skip_doc=False):
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    i = next(k for k, l in enumerate(lines) if start in l)
    j = len(lines) if end is None else next(k for k in range(i + 1, len(lines)) if end in lines[k])
    j = min(j, i + max_lines)
    while not lines[j - 1].strip():
        j -= 1
    chunk, in_doc = [], False
    for k in range(i, j):
        q = lines[k].count('"""')
        if skip_doc and (in_doc or q):
            if q == 1:
                in_doc = not in_doc
            continue
        chunk.append((k + 1, lines[k]))
    card(slide, x, y, w, h, fill="1E2430", radius=0.03)
    for n, c in enumerate(["FF5F57", "FEBC2E", "28C840"]):
        dot(slide, x + 0.2 + n * 0.2, y + 0.16, 0.12, c)
    text(slide, x + 0.95, y + 0.1, w - 1.2, 0.3,
         [[(path.replace("\\", "/"), {"color": "CFD6E4", "bold": True}),
           (f"    lines {i + 1} to {j}", {"color": "8F9BB3"})]], size=11, font=MONO)
    paras = []
    for no, l in chunk:
        body, comment = (l.split("  #", 1) + [None])[:2] if "  #" in l else (l, None)
        runs = [(f"{no:>3}  ", {"color": "5D6B84"}), (body, {"color": "E6E9EF"})]
        if comment is not None:
            runs.append(("  #" + comment, {"color": "A5D6A7"}))
        paras.append(runs)
    text(slide, x + 0.25, y + 0.5, w - 0.5, h - 0.6, paras, size=size, font=MONO, space=0)


def legend(slide, x, y, items, gap=2.55):
    for n, (label, color) in enumerate(items):
        dot(slide, x + n * gap, y + 0.05, 0.16, color)
        text(slide, x + n * gap + 0.24, y - 0.02, gap - 0.3, 0.3, label, size=12, color=SLATE)


def _axis_style(chart, vmin, vmax, fmt, unit=None):
    va, ca = chart.value_axis, chart.category_axis
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E3E6EC")
    va.format.line.fill.background()
    va.tick_labels.font.size = ca.tick_labels.font.size = Pt(11)
    va.tick_labels.font.color.rgb = ca.tick_labels.font.color.rgb = rgb(SLATE)
    va.tick_labels.font.name = ca.tick_labels.font.name = BODY
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = fmt, False
    ca.format.line.color.rgb = rgb("C5CBD8")
    ca.has_major_gridlines = False


def _axis_title(axis, title):
    axis.has_title = True
    tf = axis.axis_title.text_frame
    tf.text = title
    f = tf.paragraphs[0].runs[0].font
    f.size, f.bold, f.name = Pt(11), False, BODY
    f.color.rgb = rgb(SLATE)


def bar_chart(slide, x, y, w, h, cats, series, vmin=0, vmax=1, fmt="0.00", labels=True, ytitle=None,
              horizontal=False, label_size=10):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals, _ in series:
        cd.add_series(name, vals)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    chart = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    chart.has_title, chart.has_legend = False, False
    chart.font.name = BODY
    plot = chart.plots[0]
    plot.gap_width, plot.overlap = 70, -8
    for ser, (_, _, color) in zip(plot.series, series):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = rgb(color)
        ser.invert_if_negative = False
    if labels:
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format, dl.number_format_is_linked = fmt, False
        dl.font.size, dl.font.name = Pt(label_size), BODY
        dl.font.color.rgb = rgb(INK)
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
    _axis_style(chart, vmin, vmax, fmt)
    if horizontal:
        chart.category_axis.reverse_order = True
        chart.value_axis.tick_label_position
    if ytitle:
        _axis_title(chart.value_axis, ytitle)
    return chart


def line_chart(slide, x, y, w, h, cats, series, vmin=0, vmax=1, fmt="0.0", xtitle=None, ytitle=None):
    cd = CategoryChartData()
    cd.categories = [str(c) for c in cats]
    for name, vals, _ in series:
        cd.add_series(name, vals)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    chart.has_title, chart.has_legend = False, False
    chart.font.name = BODY
    for ser, (_, _, color) in zip(chart.plots[0].series, series):
        ser.smooth = False
        ser.format.line.color.rgb = rgb(color)
        ser.format.line.width = Pt(2.5)
        ser.marker.style, ser.marker.size = XL_MARKER_STYLE.CIRCLE, 7
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(color)
        ser.marker.format.line.color.rgb = rgb(WHITE)
    _axis_style(chart, vmin, vmax, fmt)
    if xtitle:
        _axis_title(chart.category_axis, xtitle)
    if ytitle:
        _axis_title(chart.value_axis, ytitle)
    return chart


def picture(slide, path, x, y, w, crop=None, name="crop"):
    im = Image.open(ROOT / path)
    if crop:
        im = im.crop(crop)
        out = DOCS / "screens" / f"_{name}.png"
        im.save(out)
        path = out.relative_to(ROOT)
    h = w * im.height / im.width
    slide.shapes.add_picture(str(ROOT / path), Inches(x), Inches(y), Inches(w), Inches(h))
    return h


def stat(slide, x, y, w, h, big, small, fill=TINT, big_color=NAVY, small_color=SLATE, big_size=40):
    card(slide, x, y, w, h, fill)
    text(slide, x + 0.25, y + 0.2, w - 0.5, 0.8, big, size=big_size, bold=True, color=big_color, font=HEAD)
    text(slide, x + 0.25, y + 0.2 + big_size / 72 * 1.35, w - 0.5, h - 1.0, small, size=14, color=small_color)


def r2(v):
    return f"{v:.2f}"


def cons(ds, ex, k, metric="spearman_local", study=RUL):
    return study[ds][ex][str(k)][metric]


def auc(ds, model):
    return PERF[ds][model]["roc_auc"][0]


rules = INFO["rules"]
lo, hi = rules[0], rules[-1]
subs = INFO["subset_sizes"]
n_seeds = len(INFO["seeds"])
recs = {d["dataset"]: d for d in D["datasets"]}
sign_vals = [cons(ds, "rule_consequent", k, "sign_agreement") for ds in DS for k in rules]
