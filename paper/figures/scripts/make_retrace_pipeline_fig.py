"""Draw the publication-ready pipeline overview for ReTrace.

Outputs
-------
figures/fig_retrace_pipeline.svg
figures/fig_retrace_pipeline.pdf

The SVG is the editable source. The PDF is the vector artifact for LaTeX inclusion.
Conventions follow the C4 figure scripts: Liberation Serif registered under a single
family name, a three-step ink ramp for text and rules, rounded cards, and a post-pass that
rewrites the SVG font-family to Times New Roman so the figure matches the body text.
"""

from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Path as RlPath
from reportlab.graphics.shapes import Polygon, Rect, String
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parents[1]
SVG_OUT = ROOT / "fig_retrace_pipeline.svg"
PDF_OUT = ROOT / "fig_retrace_pipeline.pdf"

# Designed for a 475 pt two-column width; the smallest 11 pt label renders at ~6.9 pt.
W, H = 760, 310

FONT_DIR = Path("/usr/share/fonts/truetype/liberation2")
pdfmetrics.registerFont(TTFont("SaberTimes", str(FONT_DIR / "LiberationSerif-Regular.ttf")))
pdfmetrics.registerFont(TTFont("SaberTimes-Bold", str(FONT_DIR / "LiberationSerif-Bold.ttf")))
pdfmetrics.registerFont(TTFont("SaberTimes-Italic", str(FONT_DIR / "LiberationSerif-Italic.ttf")))
pdfmetrics.registerFont(
    TTFont("SaberTimes-BoldItalic", str(FONT_DIR / "LiberationSerif-BoldItalic.ttf"))
)

INK = HexColor("#17202B")
MUTED = HexColor("#596579")
HAIRLINE = HexColor("#D8DEE7")

# ReTrace's identity is the paper's `oursemph` blue; it is reserved for the behavioural
# path, so a reader can see at a glance which arrows carry the watermark evidence.
BASIN = HexColor("#005AB5")
BASIN_DARK = HexColor("#003C7A")
BASIN_FILL = HexColor("#E8F1FB")
BASIN_SOFT = HexColor("#F5F9FE")
BASIN_LINE = HexColor("#9FC3E8")

# Prior work is deliberately neutral so blue stays reserved for the response channel.
PRIOR = HexColor("#7B6651")
PRIOR_FILL = HexColor("#F7F2EA")
PRIOR_LINE = HexColor("#CDBDAA")

NEUTRAL_FILL = HexColor("#F5F6F8")
NEUTRAL_LINE = HexColor("#C8CED7")

DEFER = HexColor("#B2182B")
DEFER_FILL = HexColor("#FBEAEB")
DEFER_LINE = HexColor("#E3A6AC")

MASK_FILL = HexColor("#E9ECF1")


def label(drawing, x, y, text, *, size=12, color=INK, font="SaberTimes", anchor="start"):
    drawing.add(
        String(x, y, text, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor)
    )


def rounded_card(drawing, x, y, w, h, *, fill, stroke, radius=7, width=1.1):
    drawing.add(
        Rect(x, y, w, h, rx=radius, ry=radius, fillColor=fill, strokeColor=stroke,
             strokeWidth=width)
    )


def arrow(drawing, x1, y1, x2, y2, *, color=MUTED, width=1.4, head=6):
    drawing.add(Line(x1, y1, x2 - head, y2, strokeColor=color, strokeWidth=width))
    drawing.add(
        Polygon([x2, y2, x2 - head, y2 + head * 0.62, x2 - head, y2 - head * 0.62],
                fillColor=color, strokeColor=color)
    )


def check(drawing, x, y, *, color=BASIN, width=1.6):
    path = RlPath()
    path.moveTo(x - 4, y)
    path.lineTo(x - 1, y - 3)
    path.lineTo(x + 5, y + 4)
    path.strokeColor = color
    path.strokeWidth = width
    path.fillColor = None
    drawing.add(path)


def cross(drawing, x, y, *, color=DEFER, width=1.6):
    drawing.add(Line(x - 3.5, y - 3.5, x + 3.5, y + 3.5, strokeColor=color, strokeWidth=width))
    drawing.add(Line(x - 3.5, y + 3.5, x + 3.5, y - 3.5, strokeColor=color, strokeWidth=width))


def badge(drawing, x, y, *, color, fill, mark="check", radius=8):
    drawing.add(Circle(x, y, radius, fillColor=fill, strokeColor=color, strokeWidth=1.1))
    (check if mark == "check" else cross)(drawing, x, y, color=color, width=1.5)


def formula(drawing, x, y, pieces, *, anchor="start"):
    """Times-based notation with true script positioning: (text, font, size, dy, color)."""
    widths = [pdfmetrics.stringWidth(t, f, s) for t, f, s, _dy, _c in pieces]
    total = sum(widths)
    cursor = x - total / 2 if anchor == "middle" else (x - total if anchor == "end" else x)
    for (t, f, s, dy, c), wd in zip(pieces, widths):
        label(drawing, cursor, y + dy, t, size=s, color=c, font=f)
        cursor += wd


def token_row(drawing, x, y, cells, *, cw=26, ch=17, gap=3):
    """A strip of token cells; a cell whose text is None is drawn as [MASK]."""
    for k, (text, fill, stroke, color) in enumerate(cells):
        cx = x + k * (cw + gap)
        rounded_card(drawing, cx, y, cw, ch, fill=fill, stroke=stroke, radius=3, width=0.9)
        label(drawing, cx + cw / 2, y + 5, text, size=10.5, color=color, anchor="middle")
    return x + len(cells) * (cw + gap) - gap


def panel_title(drawing, x, y, n, text):
    label(drawing, x, y, n, size=12, color=BASIN, font="SaberTimes-Bold")
    label(drawing, x + 22, y, text, size=12, color=INK, font="SaberTimes-Bold")


def main():
    d = Drawing(W, H)
    d.add(Rect(0, 0, W, H, fillColor=white, strokeColor=None))

    # A strict column grid, shared by both rows of panel (a), so the two pipelines start
    # and end in the same place and differ only in the middle. Every stacked element is
    # placed at a symmetric offset from its row's centre line.
    M = 24
    C1, W1 = M, 96
    C2, W2 = 140, 136
    C3, W3 = 300, 92
    C4, W4 = 416, 200
    C5, W5 = 624, 112
    YA, YB = 256, 186          # row centres, panel (a)
    YC = 74                    # row centre, panel (b)
    DY = 18                    # symmetric stack offset
    OUT = 30                   # symmetric branch offset

    # ------------------------------------------------------------- panel (a)
    panel_title(d, M, 296, "(a)", "What the verifier reads")

    # prior work: three stages, and a long arrow that carries the point -- no model call
    label(d, M, YA + 22, "Prior work", size=12, color=PRIOR, font="SaberTimes-Bold")
    rounded_card(d, C1, YA - 15, W1, 30, fill=NEUTRAL_FILL, stroke=NEUTRAL_LINE)
    label(d, C1 + W1 / 2, YA - 4, "final text", size=12, color=INK, anchor="middle")
    arrow(d, C1 + W1 + 4, YA, C2 - 2, YA, color=PRIOR)
    rounded_card(d, C2, YA - 15, W2, 30, fill=PRIOR_FILL, stroke=PRIOR_LINE)
    label(d, C2 + W2 / 2, YA + 2, "hash of token ids", size=12, color=PRIOR, anchor="middle")
    label(d, C2 + W2 / 2, YA - 10, "green list / parity", size=10.5, color=MUTED, anchor="middle")
    arrow(d, C2 + W2 + 4, YA, C5 - 2, YA, color=PRIOR)
    label(d, (C2 + W2 + C5) / 2, YA + 6, "no model call", size=10.5, color=MUTED,
          font="SaberTimes-Italic", anchor="middle")
    rounded_card(d, C5, YA - 15, W5, 30, fill=NEUTRAL_FILL, stroke=NEUTRAL_LINE)
    label(d, C5 + W5 / 2, YA - 4, "verdict", size=12, color=INK, anchor="middle")

    # ReTrace: the same endpoints, with the model in the middle
    label(d, M, YB + 28, "ReTrace", size=12, color=BASIN, font="SaberTimes-Bold")
    rounded_card(d, C1, YB - 22, W1, 44, fill=NEUTRAL_FILL, stroke=NEUTRAL_LINE)
    label(d, C1 + W1 / 2, YB + 4, "final text", size=12, color=INK, anchor="middle")
    label(d, C1 + W1 / 2, YB - 12, "y", size=12, color=MUTED, font="SaberTimes-Italic",
          anchor="middle")

    for off, sub in ((+DY, "u"), (-DY, "v")):
        cy = YB + off
        rounded_card(d, C2, cy - 13, W2, 26, fill=BASIN_SOFT, stroke=BASIN_LINE)
        formula(d, C2 + W2 / 2, cy - 4,
                [("keyed corruption ", "SaberTimes", 10.5, 0, MUTED),
                 ("C", "SaberTimes-Italic", 10, 0, BASIN),
                 (sub, "SaberTimes-Italic", 7, -3, BASIN)], anchor="middle")
        arrow(d, C1 + W1 + 4, YB, C2 - 2, cy, color=BASIN, width=1.2)
        arrow(d, C2 + W2 + 4, cy, C3 - 2, YB, color=BASIN, width=1.2)

    rounded_card(d, C3, YB - 22, W3, 44, fill=BASIN_FILL, stroke=BASIN)
    label(d, C3 + W3 / 2, YB + 5, "diffusion LM", size=12, color=BASIN_DARK, anchor="middle")
    label(d, C3 + W3 / 2, YB - 9, "re-denoise", size=10.5, color=MUTED, anchor="middle")
    arrow(d, C3 + W3 + 4, YB, C4 - 2, YB, color=BASIN)

    formula(d, C4, YB + 4,
            [("g", "SaberTimes-Italic", 12, 0, BASIN_DARK),
             ("i", "SaberTimes-Italic", 8, -3, BASIN_DARK),
             (" = log p (y", "SaberTimes", 11, 0, INK),
             ("i", "SaberTimes-Italic", 7, -3, INK),
             (" | C", "SaberTimes", 11, 0, INK),
             ("v", "SaberTimes-Italic", 7, -3, INK),
             (") \u2212 log p (y", "SaberTimes", 11, 0, INK),
             ("i", "SaberTimes-Italic", 7, -3, INK),
             (" | C", "SaberTimes", 11, 0, INK),
             ("u", "SaberTimes-Italic", 7, -3, INK),
             (")", "SaberTimes", 11, 0, INK)])
    label(d, C4, YB - 12, "the model's response, not the symbols", size=10.5, color=MUTED)
    arrow(d, C4 + W4 - 8, YB, C5 - 2, YB, color=BASIN)
    rounded_card(d, C5, YB - 15, W5, 30, fill=BASIN_FILL, stroke=BASIN)
    label(d, C5 + W5 / 2, YB - 4, "verdict", size=12, color=BASIN_DARK, anchor="middle")

    d.add(Line(M, 148, W - M, 148, strokeColor=HAIRLINE, strokeWidth=1))

    # ------------------------------------------------------------- panel (b)
    panel_title(d, M, 128, "(b)", "How the watermark is placed")
    label(d, 250, 128, "every emitted token is drawn from the model's own conditional",
          size=12, color=MUTED)

    rounded_card(d, M, YC - 16, 118, 32, fill=NEUTRAL_FILL, stroke=NEUTRAL_LINE)
    label(d, M + 59, YC + 5, "model conditional", size=12, color=INK, anchor="middle")
    formula(d, M + 59, YC - 9,
            [("p", "SaberTimes-Italic", 11, 0, MUTED),
             ("\u03b8", "SaberTimes-Italic", 7, -3, MUTED),
             ("( \u00b7 | x )  unchanged", "SaberTimes", 9, 0, MUTED)], anchor="middle")

    arrow(d, M + 122, YC, M + 148, YC, color=MUTED)
    rounded_card(d, M + 152, YC - 16, 92, 32, fill=NEUTRAL_FILL, stroke=NEUTRAL_LINE)
    label(d, M + 198, YC + 5, "draw token", size=12, color=INK, anchor="middle")
    formula(d, M + 198, YC - 9,
            [("v", "SaberTimes-Italic", 11, 0, MUTED),
             (" ~ p", "SaberTimes-Italic", 10, 0, MUTED),
             ("\u03b8", "SaberTimes-Italic", 7, -3, MUTED)], anchor="middle")

    arrow(d, M + 248, YC, M + 274, YC, color=MUTED)
    rounded_card(d, M + 278, YC - 18, 152, 36, fill=BASIN_SOFT, stroke=BASIN_LINE)
    formula(d, M + 354, YC + 3,
            [("w", "SaberTimes-Italic", 11, 0, BASIN_DARK),
             ("i", "SaberTimes-Italic", 7, -3, BASIN_DARK),
             (" \u03b5", "SaberTimes-Italic", 11, 0, BASIN_DARK),
             ("i", "SaberTimes-Italic", 7, -3, BASIN_DARK),
             (" g", "SaberTimes-Italic", 11, 0, BASIN_DARK),
             ("i", "SaberTimes-Italic", 7, -3, BASIN_DARK),
             ("(v) > 0 ?", "SaberTimes", 11, 0, INK)], anchor="middle")
    label(d, M + 354, YC - 12, "keyed reconstruction response", size=12, color=MUTED,
          anchor="middle")

    # accept path
    arrow(d, M + 434, YC + 6, M + 458, YC + OUT, color=BASIN, width=1.3)
    rounded_card(d, M + 462, YC + OUT - 14, 96, 28, fill=BASIN_FILL, stroke=BASIN)
    badge(d, M + 478, YC + OUT, color=BASIN, fill=white, mark="check")
    label(d, M + 492, YC + OUT - 4, "emit", size=12, color=BASIN_DARK)
    # reject path loops back to the draw box: a retry is free, the row is unchanged
    arrow(d, M + 434, YC - 6, M + 458, YC - OUT, color=DEFER, width=1.3)
    badge(d, M + 470, YC - OUT, color=DEFER, fill=white, mark="cross")
    label(d, M + 482, YC - OUT - 4, "redraw, at most R times", size=12, color=DEFER)
    d.add(Line(M + 470, YC - OUT - 10, M + 198, YC - OUT - 10, strokeColor=DEFER,
               strokeWidth=1.1))
    arrow(d, M + 198, YC - OUT - 10, M + 198, YC - 20, color=DEFER, width=1.1, head=5)
    label(d, M + 334, YC - OUT - 20,
          "a retry costs no forward pass: the conditional row is unchanged",
          size=12, color=MUTED, anchor="middle")
    label(d, M + 566, YC + OUT - 4,
          "exhausted: one fresh draw, emitted as-is", size=10.5, color=MUTED)

    renderSVG.drawToFile(d, str(SVG_OUT))
    svg = SVG_OUT.read_text(encoding="utf-8")
    for a, b in (
        ("font-family: SaberTimes-BoldItalic;",
         "font-family: 'Times New Roman'; font-weight: bold; font-style: italic;"),
        ("font-family: SaberTimes-Bold;", "font-family: 'Times New Roman'; font-weight: bold;"),
        ("font-family: SaberTimes-Italic;", "font-family: 'Times New Roman'; font-style: italic;"),
        ("font-family: SaberTimes;", "font-family: 'Times New Roman';"),
    ):
        svg = svg.replace(a, b)
    SVG_OUT.write_text(svg, encoding="utf-8")
    renderPDF.drawToFile(d, str(PDF_OUT))
    print(SVG_OUT)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
