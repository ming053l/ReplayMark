"""Generate the Balanced Carrier Gate mechanism figure as editable SVG and vector PDF.

The figure follows the actual ResampleMark implementation: the RRB partitions the
block-masked base conditional by the sign of g_i, BCG applies the orientation-invariant
score 2 min(q+, q-), and only admitted positions are consumed by RAR/RRV.
"""

from html import escape
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SVG_OUT = ROOT / "fig_retrace_mechanism.svg"
PDF_OUT = ROOT / "fig_retrace_mechanism.pdf"
W, H = 960, 318

INK = "#17202B"
MUTED = "#596579"
LINE = "#CCD3DD"
LIGHT = "#F6F8FA"
BLUE = "#005AB5"
BLUE_DARK = "#003C7A"
BLUE_FILL = "#EAF3FC"
RED = "#B2182B"
RED_FILL = "#FBEAEC"
GREEN = "#287A4B"
GREEN_FILL = "#EAF5EE"
GOLD = "#9B6A16"
GOLD_FILL = "#FBF3E3"
GRAY = "#697384"


class SVG:
    def __init__(self):
        self.items = []

    def add(self, value):
        self.items.append(value)

    def text(self, x, y, value, size=13, fill=INK, weight="normal", anchor="start",
             style="normal"):
        self.add(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" font-style="{style}" text-anchor="{anchor}">'
            f'{escape(value)}</text>'
        )

    def rect(self, x, y, w, h, fill="white", stroke=LINE, radius=7, sw=1):
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=1, marker=False):
        extra = ' marker-end="url(#arrow)"' if marker else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def circle(self, x, y, r, fill, stroke=None, sw=1):
        stroke = stroke or fill
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"/>')

    def render(self):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"
 viewBox="0 0 {W} {H}">
<defs>
  <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4"
          orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L8,4 L0,8 z" fill="{MUTED}"/>
  </marker>
  <style>text {{ font-family: 'Times New Roman', 'Liberation Serif', serif; }}</style>
</defs>
<rect width="100%" height="100%" fill="white"/>
{"\n".join(self.items)}
</svg>'''


def panel_header(s, x, width, letter, title, subtitle, color):
    s.circle(x + 22, 59, 12, color)
    s.text(x + 22, 63.5, letter, 12, "white", "bold", "middle")
    s.text(x + 43, 57, title, 15, INK, "bold")
    s.text(x + 43, 74, subtitle, 10.5, MUTED)


def mass_bar(s, x, y, width, qminus, qplus):
    s.text(x, y - 9, "RRB sign-mass partition of pᵢ(base)", 11, MUTED, "bold")
    minus_w = width * qminus
    plus_w = width * qplus
    s.rect(x, y, width, 34, LIGHT, LINE, 7)
    s.add(f'<path d="M{x+1},{y+1} h{minus_w-1} v32 h{-minus_w+1} z" fill="{RED}"/>')
    s.add(f'<path d="M{x+minus_w},{y+1} h{plus_w-1} v32 h{-plus_w+1} z" fill="{BLUE}"/>')
    if qminus >= .12:
        s.text(x + minus_w / 2, y + 22, f"q₋ = {qminus:.2f}", 12, "white", "bold", "middle")
    s.text(x + minus_w + plus_w / 2, y + 22, f"q₊ = {qplus:.2f}", 12, "white", "bold", "middle")
    s.text(x + 2, y + 49, "gᵢ(z) < 0", 10.5, RED, "bold")
    s.text(x + width - 2, y + 49, "gᵢ(z) > 0", 10.5, BLUE, "bold", "end")


def orientation_card(s, x, y, width, sign, mass, retries=None):
    color = BLUE if sign == "+" else RED
    fill = BLUE_FILL if sign == "+" else RED_FILL
    s.rect(x, y, width, 38, fill, color, 7, 1)
    s.text(x + 12, y + 15, f"key asks {sign}", 10.5, color, "bold")
    s.text(x + 12, y + 30, f"one-draw mass = {mass:.2f}", 10.5, INK)
    if retries is not None:
        prob = 1 - (1 - mass) ** retries
        s.text(x + width - 10, y + 24, f"P₈ = {prob:.2f}", 11, color, "bold", "end")


def main():
    s = SVG()
    s.text(18, 25, "Balanced Carrier Gate (BCG): admit only orientation-robust response channels",
           15.5, INK, "bold")
    s.text(942, 25, "computed before εᵢ and wᵢ are consulted", 11, MUTED,
           anchor="end", style="italic")

    left_x, right_x, panel_y, panel_w, panel_h = 18, 490, 39, 452, 225
    s.rect(left_x, panel_y, panel_w, panel_h, "white", LINE, 10, 1.2)
    s.rect(right_x, panel_y, panel_w, panel_h, "white", LINE, 10, 1.2)
    panel_header(s, left_x, panel_w, "A", "Two-sided response", "usable for either keyed orientation", BLUE)
    panel_header(s, right_x, panel_w, "B", "One-sided response", "usable only for one orientation", RED)

    # A: balanced response mass.
    mass_bar(s, left_x + 24, 91, 404, .47, .53)
    s.rect(left_x + 24, 154, 404, 34, GREEN_FILL, GREEN, 7, 1.1)
    s.text(left_x + 38, 176, "Sᵢ = 2 min(.53, .47) = .94  >  sₘᵢₙ=.50", 12.5, GREEN, "bold")
    s.rect(left_x + 328, 160, 89, 22, GREEN, GREEN, 11)
    s.text(left_x + 372.5, 175, "BCG admits", 10.5, "white", "bold", "middle")
    orientation_card(s, left_x + 24, 199, 194, "+", .53)
    orientation_card(s, left_x + 234, 199, 194, "−", .47)

    # B: response mass concentrated on one sign.
    mass_bar(s, right_x + 24, 91, 404, .03, .97)
    s.rect(right_x + 24, 154, 404, 34, LIGHT, GRAY, 7, 1.1)
    s.text(right_x + 38, 176, "Sᵢ = 2 min(.97, .03) = .06  <  sₘᵢₙ=.50", 12.5, GRAY, "bold")
    s.rect(right_x + 326, 160, 91, 22, GRAY, GRAY, 11)
    s.text(right_x + 371.5, 175, "BCG rejects", 10.5, "white", "bold", "middle")
    orientation_card(s, right_x + 24, 199, 194, "+", .97, retries=8)
    orientation_card(s, right_x + 234, 199, 194, "−", .03, retries=8)

    # Downstream routing makes the module boundary concrete.
    s.line(left_x + panel_w/2, 264, left_x + panel_w/2, 277, MUTED, 1.2, marker=True)
    s.line(right_x + panel_w/2, 264, right_x + panel_w/2, 277, MUTED, 1.2, marker=True)
    s.rect(left_x, 280, panel_w, 25, GREEN_FILL, GREEN, 12, 1)
    s.text(left_x + panel_w/2, 297, "RAR may steer  •  RRV counts one Bernoulli trial",
           11.5, GREEN, "bold", "middle")
    s.rect(right_x, 280, panel_w, 25, LIGHT, GRAY, 12, 1)
    s.text(right_x + panel_w/2, 297, "reference sampling  •  no detector trial",
           11.5, GRAY, "bold", "middle")

    s.text(480, 316,
           "Orientation-free gate: Sᵢ is invariant to swapping the two RRB arms; εᵢ and wᵢ enter only after admission.",
           10.5, GOLD, "bold", "middle")

    SVG_OUT.write_text(s.render(), encoding="utf-8")
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(ROOT), str(SVG_OUT)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if not PDF_OUT.exists():
        raise RuntimeError("SVG-to-PDF conversion did not produce the expected file")
    print(SVG_OUT)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
