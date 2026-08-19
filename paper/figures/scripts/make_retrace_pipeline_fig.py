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
W, H = 760, 330

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
        label(drawing, cx + cw / 2, y + 5, text, size=9, color=color, anchor="middle")
    return x + len(cells) * (cw + gap) - gap


def panel_title(drawing, x, y, n, text):
    label(drawing, x, y, n, size=12, color=BASIN, font="SaberTimes-Bold")
    label(drawing, x + 22, y, text, size=12, color=INK, font="SaberTimes-Bold")


def main():
    d = Drawing(W, H)
    d.add(Rect(0, 0, W, H, fillColor=white, strokeColor=None))

    # ------------------------------------------------------------- panel (a)
    # What the verifier reads. Prior work inspects symbols; ReTrace interrogates the
    # model. Blue is reserved for the response path so the contrast is visible at a glance.
    panel_title(d, 24, 306, "(a)", "What the verifier reads")

    label(d, 24, 282, "Prior work", size=11, color=PRIOR, font="SaberTimes-Bold")
    rounded_card(d, 24, 250, 88, 26, fill=NEUTRAL_FILL, stroke=NEUTRAL_LINE)
    label(d, 68, 259, "final text", size=11, color=INK, anchor="middle")
    arrow(d, 116, 263, 150, 263, color=PRIOR)
    rounded_card(d, 152, 250, 126, 26, fill=PRIOR_FILL, stroke=PRIOR_LINE)
    label(d, 215, 265, "hash of token ids", size=10, color=PRIOR, anchor="middle")
    label(d, 215, 254, "green list / parity", size=9, color=MUTED, anchor="middle")

    label(d, 24, 228, "ReTrace", size=11, color=BASIN, font="SaberTimes-Bold")
    rounded_card(d, 24, 166, 88, 48, fill=NEUTRAL_FILL, stroke=NEUTRAL_LINE)
    label(d, 68, 194, "final text", size=11, color=INK, anchor="middle")
    label(d, 68, 180, "y", size=11, color=MUTED, font="SaberTimes-Italic", anchor="middle")

    for cy, sub in ((194, "u"), (162, "v")):
        rounded_card(d, 130, cy - 12, 116, 24, fill=BASIN_SOFT, stroke=BASIN_LINE)
        formula(d, 188, cy - 4,
                [("keyed corruption ", "SaberTimes", 9, 0, MUTED),
                 ("C", "SaberTimes-Italic", 10, 0, BASIN),
                 (sub, "SaberTimes-Italic", 7, -3, BASIN)], anchor="middle")
        arrow(d, 114, 190, 128, cy, color=BASIN, width=1.2)
        arrow(d, 248, cy, 268, 190, color=BASIN, width=1.2)

    rounded_card(d, 270, 166, 78, 48, fill=BASIN_FILL, stroke=BASIN)
    label(d, 309, 195, "diffusion LM", size=10, color=BASIN_DARK, anchor="middle")
    label(d, 309, 181, "re-denoise", size=9, color=MUTED, anchor="middle")

    arrow(d, 350, 190, 378, 190, color=BASIN)
    formula(d, 384, 196,
            [("g", "SaberTimes-Italic", 13, 0, BASIN_DARK),
             ("i", "SaberTimes-Italic", 8, -3, BASIN_DARK),
             ("(y", "SaberTimes", 12, 0, INK),
             ("i", "SaberTimes-Italic", 8, -3, INK),
             (") = log p (y", "SaberTimes", 12, 0, INK),
             ("i", "SaberTimes-Italic", 8, -3, INK),
             (" | C", "SaberTimes", 12, 0, INK),
             ("v", "SaberTimes-Italic", 8, -3, INK),
             (") \u2212 log p (y", "SaberTimes", 12, 0, INK),
             ("i", "SaberTimes-Italic", 8, -3, INK),
             (" | C", "SaberTimes", 12, 0, INK),
             ("u", "SaberTimes-Italic", 8, -3, INK),
             (")", "SaberTimes", 12, 0, INK)])
    label(d, 384, 178, "the evidence is the model's response, not the symbols",
          size=9, color=MUTED)

    d.add(Line(24, 148, W - 24, 148, strokeColor=HAIRLINE, strokeWidth=1))

    # ------------------------------------------------------------- panel (b)
    # How it is placed. The proposal is never replaced; the key only orders commits.
    panel_title(d, 24, 128, "(b)", "How the watermark is placed")
    label(d, 236, 128, "the model proposes; the key only chooses what commits now",
          size=10, color=MUTED)

    label(d, 24, 104, "block being decoded", size=9, color=MUTED)
    cells = [("The", NEUTRAL_FILL, NEUTRAL_LINE, INK),
             ("model", NEUTRAL_FILL, NEUTRAL_LINE, INK),
             ("[M]", MASK_FILL, NEUTRAL_LINE, MUTED),
             ("[M]", MASK_FILL, NEUTRAL_LINE, MUTED),
             ("[M]", MASK_FILL, NEUTRAL_LINE, MUTED)]
    token_row(d, 24, 74, cells)

    arrow(d, 172, 82, 196, 82, color=MUTED)
    rounded_card(d, 200, 66, 106, 32, fill=NEUTRAL_FILL, stroke=NEUTRAL_LINE)
    label(d, 253, 86, "model proposal", size=10, color=INK, anchor="middle")
    formula(d, 253, 72,
            [("x", "SaberTimes-Italic", 11, 0, MUTED),
             ("i", "SaberTimes-Italic", 7, -3, MUTED),
             ("  unchanged", "SaberTimes", 9, 0, MUTED)], anchor="middle")

    arrow(d, 310, 82, 332, 82, color=MUTED)
    rounded_card(d, 336, 64, 156, 36, fill=BASIN_SOFT, stroke=BASIN_LINE)
    formula(d, 414, 84,
            [("w", "SaberTimes-Italic", 11, 0, BASIN_DARK),
             ("i", "SaberTimes-Italic", 7, -3, BASIN_DARK),
             (" \u03b5", "SaberTimes-Italic", 11, 0, BASIN_DARK),
             ("i", "SaberTimes-Italic", 7, -3, BASIN_DARK),
             (" g", "SaberTimes-Italic", 11, 0, BASIN_DARK),
             ("i", "SaberTimes-Italic", 7, -3, BASIN_DARK),
             ("(x", "SaberTimes", 11, 0, INK),
             ("i", "SaberTimes-Italic", 7, -3, INK),
             (") > 0 ?", "SaberTimes", 11, 0, INK)], anchor="middle")
    label(d, 414, 70, "answered by the model's own token", size=8, color=MUTED,
          anchor="middle")

    arrow(d, 496, 88, 516, 106, color=BASIN, width=1.3)
    arrow(d, 496, 76, 516, 46, color=DEFER, width=1.3)

    rounded_card(d, 520, 94, 118, 28, fill=BASIN_FILL, stroke=BASIN)
    badge(d, 536, 108, color=BASIN, fill=white, mark="check")
    label(d, 550, 104, "commit now", size=11, color=BASIN_DARK)

    rounded_card(d, 520, 32, 118, 28, fill=DEFER_FILL, stroke=DEFER_LINE)
    badge(d, 536, 46, color=DEFER, fill=white, mark="cross")
    label(d, 550, 42, "defer", size=11, color=DEFER)

    # the two notes sit under the objects they qualify, not in the margin, so nothing
    # collides with the outcome cards
    label(d, 253, 54, "no token is ever substituted", size=9, color=BASIN_DARK,
          font="SaberTimes-Italic", anchor="middle")
    label(d, 579, 22, "re-proposed next step", size=9, color=MUTED, anchor="middle")

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
