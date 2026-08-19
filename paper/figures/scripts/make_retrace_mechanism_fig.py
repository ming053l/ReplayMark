"""Draw the step-by-step mechanism figure for ReTrace.

Outputs
-------
figures/fig_retrace_mechanism.svg
figures/fig_retrace_mechanism.pdf

Idiom follows the C4 mechanism figures: horizontal lanes, one per decoding rule, each with a
header pill and a sequence of step columns; commits are marked with a check badge and a
deferral with a cross, so the two lanes can be read against each other position by position.
The two lanes see the *same* model proposals -- that is the point of the figure. Only the
order of commitment differs, and the bottom strip shows how that order becomes evidence.
"""

from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Path as RlPath
from reportlab.graphics.shapes import Polygon, Rect, String
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parents[1]
SVG_OUT = ROOT / "fig_retrace_mechanism.svg"
PDF_OUT = ROOT / "fig_retrace_mechanism.pdf"

W, H = 760, 316

FONT_DIR = Path("/usr/share/fonts/truetype/liberation2")
pdfmetrics.registerFont(TTFont("SaberTimes", str(FONT_DIR / "LiberationSerif-Regular.ttf")))
pdfmetrics.registerFont(TTFont("SaberTimes-Bold", str(FONT_DIR / "LiberationSerif-Bold.ttf")))
pdfmetrics.registerFont(TTFont("SaberTimes-Italic", str(FONT_DIR / "LiberationSerif-Italic.ttf")))

INK = HexColor("#17202B")
MUTED = HexColor("#596579")
HAIRLINE = HexColor("#D8DEE7")

BASIN = HexColor("#005AB5")
BASIN_DARK = HexColor("#003C7A")
BASIN_FILL = HexColor("#E8F1FB")
BASIN_SOFT = HexColor("#F5F9FE")
BASIN_LINE = HexColor("#9FC3E8")

PRIOR = HexColor("#7B6651")
PRIOR_FILL = HexColor("#F7F2EA")
PRIOR_LINE = HexColor("#CDBDAA")

NEUTRAL_FILL = HexColor("#F5F6F8")
NEUTRAL_LINE = HexColor("#C8CED7")
MASK_FILL = HexColor("#E9ECF1")

DEFER = HexColor("#B2182B")
DEFER_FILL = HexColor("#FBEAEB")
DEFER_LINE = HexColor("#E3A6AC")

CW, CH, GAP = 24, 18, 3          # token cell geometry, shared by every row


def label(d, x, y, text, *, size=11, color=INK, font="SaberTimes", anchor="start"):
    d.add(String(x, y, text, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor))


def rounded_card(d, x, y, w, h, *, fill, stroke, radius=6, width=1.0):
    d.add(Rect(x, y, w, h, rx=radius, ry=radius, fillColor=fill, strokeColor=stroke,
               strokeWidth=width))


def arrow(d, x1, y1, x2, y2, *, color=MUTED, width=1.3, head=5):
    d.add(Line(x1, y1, x2 - head, y2, strokeColor=color, strokeWidth=width))
    d.add(Polygon([x2, y2, x2 - head, y2 + head * 0.6, x2 - head, y2 - head * 0.6],
                  fillColor=color, strokeColor=color))


def check(d, x, y, *, color=BASIN, width=1.5):
    p = RlPath()
    p.moveTo(x - 3.5, y)
    p.lineTo(x - 1, y - 2.5)
    p.lineTo(x + 4, y + 3.5)
    p.strokeColor, p.strokeWidth, p.fillColor = color, width, None
    d.add(p)


def cross(d, x, y, *, color=DEFER, width=1.5):
    d.add(Line(x - 3, y - 3, x + 3, y + 3, strokeColor=color, strokeWidth=width))
    d.add(Line(x - 3, y + 3, x + 3, y - 3, strokeColor=color, strokeWidth=width))


def badge(d, x, y, *, color, fill, mark="check", radius=7):
    d.add(Circle(x, y, radius, fillColor=fill, strokeColor=color, strokeWidth=1.0))
    (check if mark == "check" else cross)(d, x, y, color=color)


def pill(d, x, y, w, text, *, fill, stroke, color, h=17):
    rounded_card(d, x, y - h / 2, w, h, fill=fill, stroke=stroke, radius=h / 2, width=1.0)
    label(d, x + w / 2, y - 3.5, text, size=9, color=color, anchor="middle")


def cells(d, x, y, states):
    """A block of token cells. Each state is (text, kind), kind in
    {mask, proposal, committed, wm, defer}."""
    style = {
        "mask": (MASK_FILL, NEUTRAL_LINE, MUTED),
        "proposal": (white, BASIN_LINE, MUTED),
        "committed": (NEUTRAL_FILL, NEUTRAL_LINE, INK),
        "wm": (BASIN_FILL, BASIN, BASIN_DARK),
        "defer": (DEFER_FILL, DEFER_LINE, DEFER),
    }
    for k, (text, kind) in enumerate(states):
        fill, stroke, color = style[kind]
        cx = x + k * (CW + GAP)
        rounded_card(d, cx, y, CW, CH, fill=fill, stroke=stroke, radius=3, width=0.9)
        label(d, cx + CW / 2, y + 5.5, text, size=8, color=color, anchor="middle")
    return x + len(states) * (CW + GAP) - GAP


def lane_header(d, x, y, *, tag, title, subtitle, color, fill, stroke):
    pill(d, x, y + 24, 96, tag, fill=fill, stroke=stroke, color=color)
    label(d, x, y - 2, title, size=11, color=INK, font="SaberTimes-Bold")
    label(d, x, y - 15, subtitle, size=9, color=MUTED)


def main():
    d = Drawing(W, H)
    d.add(Rect(0, 0, W, H, fillColor=white, strokeColor=None))

    LX = 24                     # lane header column
    S1, S2, S3 = 172, 366, 560  # step columns
    YA, YB, YV = 250, 176, 72   # lane centres and the verification strip

    label(d, LX, 302, "One block of the reference decoding schedule; both lanes see the "
                      "same model proposals.", size=10, color=MUTED)

    for x, t in ((S1, "step t"), (S2, "step t+1"), (S3, "step t+2")):
        label(d, x + 2.5 * (CW + GAP), 286, t, size=9, color=MUTED,
              font="SaberTimes-Italic", anchor="middle")

    # ------------------------------------------------------------------ lane A
    lane_header(d, LX, YA, tag="reference", title="Reference decoding",
                subtitle="commits by confidence", color=PRIOR, fill=PRIOR_FILL,
                stroke=PRIOR_LINE)

    a1 = [("the", "committed"), ("model", "committed"), ("fills", "proposal"),
          ("in", "proposal"), ("any", "proposal"), ("order", "proposal")]
    a2 = [("the", "committed"), ("model", "committed"), ("fills", "committed"),
          ("in", "proposal"), ("any", "proposal"), ("order", "proposal")]
    a3 = [("the", "committed"), ("model", "committed"), ("fills", "committed"),
          ("in", "committed"), ("any", "proposal"), ("order", "proposal")]
    for x, st, hit in ((S1, a1, 2), (S2, a2, 3), (S3, a3, 4)):
        cells(d, x, YA - 9, st)
        badge(d, x + hit * (CW + GAP) + CW / 2, YA + 20, color=PRIOR, fill=white)

    # ------------------------------------------------------------------ lane B
    lane_header(d, LX, YB, tag="ReTrace", title="Order-steered", color=BASIN,
                subtitle="commits what answers the key", fill=BASIN_SOFT, stroke=BASIN_LINE)

    b1 = [("the", "committed"), ("model", "committed"), ("fills", "defer"),
          ("in", "proposal"), ("any", "wm"), ("order", "proposal")]
    b2 = [("the", "committed"), ("model", "committed"), ("fills", "wm"),
          ("in", "proposal"), ("any", "committed"), ("order", "proposal")]
    b3 = [("the", "committed"), ("model", "committed"), ("fills", "committed"),
          ("in", "proposal"), ("any", "committed"), ("order", "wm")]
    for x, st, hit, mark in ((S1, b1, 4, "check"), (S2, b2, 2, "check"),
                             (S3, b3, 5, "check")):
        cells(d, x, YB - 9, st)
        badge(d, x + hit * (CW + GAP) + CW / 2, YB + 20, color=BASIN, fill=white,
              mark=mark)
    badge(d, S1 + 2 * (CW + GAP) + CW / 2, YB + 20, color=DEFER, fill=white, mark="cross")

    label(d, S1, YB - 26, "incompatible: deferred, token not replaced",
          size=8.5, color=DEFER)
    label(d, S2, YB - 26, "re-proposed with more of the block resolved",
          size=8.5, color=MUTED)
    label(d, S3, YB - 26, "compatible: committed now", size=8.5, color=BASIN_DARK)

    d.add(Line(LX, 132, W - LX, 132, strokeColor=HAIRLINE, strokeWidth=1))

    # ------------------------------------------------------- verification strip
    label(d, LX, 114, "Verification", size=11, color=BASIN, font="SaberTimes-Bold")
    label(d, 116, 114, "from the finished text alone, with no access to the trajectory",
          size=9, color=MUTED)

    fin = [("the", "committed"), ("model", "committed"), ("fills", "committed"),
           ("in", "committed"), ("any", "committed"), ("order", "committed")]
    end = cells(d, LX, YV - 9, fin)
    label(d, LX, YV - 26, "finished text", size=9, color=MUTED)

    arrow(d, end + 8, YV, end + 34, YV, color=BASIN)
    msk = [("[M]", "mask"), ("model", "committed"), ("[M]", "mask"),
           ("in", "committed"), ("[M]", "mask"), ("order", "committed")]
    end2 = cells(d, end + 40, YV - 9, msk)
    label(d, end + 40, YV - 26, "re-masked under the key", size=9, color=BASIN_DARK)

    arrow(d, end2 + 8, YV, end2 + 34, YV, color=BASIN)
    ix = end2 + 40
    for k, m in enumerate((1, None, 0, None, 1, None)):
        cx = ix + k * (CW + GAP)
        if m is None:
            rounded_card(d, cx, YV - 9, CW, CH, fill=NEUTRAL_FILL, stroke=NEUTRAL_LINE,
                         radius=3, width=0.9)
            label(d, cx + CW / 2, YV - 3.5, "—", size=8, color=MUTED, anchor="middle")
        else:
            rounded_card(d, cx, YV - 9, CW, CH, fill=BASIN_FILL if m else DEFER_FILL,
                         stroke=BASIN if m else DEFER_LINE, radius=3, width=0.9)
            label(d, cx + CW / 2, YV - 3.5, str(m), size=9,
                  color=BASIN_DARK if m else DEFER, anchor="middle")
    label(d, ix, YV - 26, "indicators m", size=9, color=MUTED)
    label(d, ix + 58, YV - 26, "i", size=7, color=MUTED, font="SaberTimes-Italic")

    arrow(d, ix + 6 * (CW + GAP) + 4, YV, ix + 6 * (CW + GAP) + 30, YV, color=BASIN)
    rounded_card(d, ix + 6 * (CW + GAP) + 36, YV - 15, 104, 30, fill=BASIN_FILL,
                 stroke=BASIN)
    label(d, ix + 6 * (CW + GAP) + 88, YV - 1, "count", size=10, color=BASIN_DARK,
          anchor="middle")
    label(d, ix + 6 * (CW + GAP) + 88, YV - 12, "Binomial(n, 1/2)", size=8, color=MUTED,
          anchor="middle")

    renderSVG.drawToFile(d, str(SVG_OUT))
    svg = SVG_OUT.read_text(encoding="utf-8")
    for a, b in (("font-family: SaberTimes-Bold;",
                  "font-family: 'Times New Roman'; font-weight: bold;"),
                 ("font-family: SaberTimes-Italic;",
                  "font-family: 'Times New Roman'; font-style: italic;"),
                 ("font-family: SaberTimes;", "font-family: 'Times New Roman';")):
        svg = svg.replace(a, b)
    SVG_OUT.write_text(svg, encoding="utf-8")
    renderPDF.drawToFile(d, str(PDF_OUT))
    print(SVG_OUT)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
