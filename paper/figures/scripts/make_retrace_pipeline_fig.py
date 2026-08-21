"""Generate the editable vector overview used as Figure 1.

The composition follows dgMARK's strongest visual idea: show the native DLM operation,
the precise watermark intervention, and the resulting decision with token-level examples.
Only the Python standard library is required. LibreOffice performs SVG-to-PDF conversion
when available; the SVG remains the authoritative editable source.
"""

from html import escape
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SVG_OUT = ROOT / "fig_retrace_pipeline.svg"
PDF_OUT = ROOT / "fig_retrace_pipeline.pdf"
W, H = 960, 360

INK = "#17202B"
MUTED = "#596579"
LINE = "#CCD3DD"
LIGHT = "#F6F8FA"
MASK = "#E7EBF0"
BLUE = "#285F8F"
BLUE_DARK = "#1F4668"
BLUE_FILL = "#EDF3F7"
BLUE_LINE = "#AABFCC"
RED = "#8A6D3B"
RED_FILL = "#F4F0E8"
GREEN = BLUE
GREEN_FILL = BLUE_FILL
GOLD = RED
GOLD_FILL = RED_FILL


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

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=1, dash=None, marker=False):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        if marker:
            extra += ' marker-end="url(#arrow)"'
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def circle(self, x, y, r, fill="white", stroke=LINE, sw=1):
        self.add(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def render(self):
        body = "\n".join(self.items)
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"
 viewBox="0 0 {W} {H}">
<defs>
  <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4"
          orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L8,4 L0,8 z" fill="{MUTED}"/>
  </marker>
  <style>
    text {{ font-family: 'Times New Roman', 'Liberation Serif', serif; }}
  </style>
</defs>
<rect width="100%" height="100%" fill="white"/>
{body}
</svg>'''


def panel(svg, x, width, number, title, subtitle, accent=BLUE):
    svg.rect(x, 40, width, 266, fill="white", stroke=LINE, radius=10, sw=1.2)
    svg.circle(x + 18, 61, 11, fill=accent, stroke=accent)
    svg.text(x + 18, 65, str(number), 12, "white", "bold", "middle")
    svg.text(x + 37, 59, title, 14, INK, "bold")
    svg.text(x + 37, 74, subtitle, 10.5, MUTED)


def token(svg, x, y, w, label, fill=LIGHT, stroke=LINE, color=INK, bold=False):
    svg.rect(x, y, w, 25, fill=fill, stroke=stroke, radius=4, sw=0.9)
    svg.text(x + w / 2, y + 17, label, 11.5, color, "bold" if bold else "normal", "middle")


def token_strip(svg, x, y, labels, widths=None, highlights=()):
    widths = widths or [31] * len(labels)
    cursor = x
    for i, (label, width) in enumerate(zip(labels, widths)):
        if label == "MASK":
            token(svg, cursor, y, width, label, MASK, LINE, MUTED)
        elif i in highlights:
            token(svg, cursor, y, width, label, BLUE_FILL, BLUE, BLUE_DARK, True)
        else:
            token(svg, cursor, y, width, label)
        cursor += width + 3


def tiny_table(svg, x, y, rows, widths, header, row_h=23):
    total = sum(widths)
    svg.rect(x, y, total, row_h * (len(rows) + 1), fill="white", stroke=LINE, radius=5)
    svg.add(f'<path d="M{x},{y + row_h} H{x + total}" stroke="{LINE}"/>')
    cursor = x
    for width in widths[:-1]:
        cursor += width
        svg.line(cursor, y, cursor, y + row_h * (len(rows) + 1), LINE, .8)
    cursor = x
    for label, width in zip(header, widths):
        svg.text(cursor + width / 2, y + 16, label, 10.5, MUTED, "bold", "middle")
        cursor += width
    for ridx, row in enumerate(rows):
        yy = y + row_h * (ridx + 1)
        if ridx % 2 == 1:
            svg.add(f'<rect x="{x + 1}" y="{yy}" width="{total - 2}" height="{row_h}" fill="{LIGHT}"/>')
        cursor = x
        for value, width in zip(row, widths):
            color = BLUE_DARK if value in {"+", "accept"} else RED if value in {"−", "reject"} else INK
            weight = "bold" if value in {"+", "−", "accept", "reject"} else "normal"
            svg.text(cursor + width / 2, yy + 16, value, 11, color, weight, "middle")
            cursor += width


def main():
    s = SVG()
    s.text(18, 24, "ReTrace: the DLM is both generator and keyed reconstruction probe", 15, INK, "bold")
    s.text(942, 24, "one block at a time", 11, MUTED, "normal", "end", "italic")

    xs = [18, 241, 474, 707]
    ws = [213, 223, 223, 235]
    panel(s, xs[0], ws[0], 1, "Reference DLM", "keep the native block schedule", MUTED)
    panel(s, xs[1], ws[1], 2, "Response Bank + Gate", "RRB records; BCG selects", GOLD)
    panel(s, xs[2], ws[2], 3, "RAR embedding", "response-aligned resampling", BLUE)
    panel(s, xs[3], ws[3], 4, "RRV verification", "replay, count, test", GREEN)

    for index, (left, right) in enumerate(zip(xs[:-1], xs[1:])):
        s.line(left + ws[index] + 2, 173, right - 4, 173, MUTED, 1.4, marker=True)

    # 1. Native blockwise decoding.
    x = xs[0]
    s.text(x + 14, 99, "Current state", 11, MUTED, "bold")
    token_strip(s, x + 14, 108, ["prompt", "fixed", "MASK", "MASK"], [48, 39, 43, 43])
    s.line(x + 106, 137, x + 106, 153, MUTED, 1.1, marker=True)
    s.rect(x + 55, 156, 102, 32, BLUE_FILL, BLUE_LINE, 7)
    s.text(x + 106, 177, "diffusion LM", 12, BLUE_DARK, "bold", "middle")
    s.line(x + 106, 190, x + 106, 203, MUTED, 1.1, marker=True)
    tiny_table(s, x + 40, 202,
               [["the", ".42"], ["a", ".25"], ["this", ".13"]],
               [76, 56], ["candidate", "pθ"], row_h=20)
    s.text(x + 106, 297, "sample once and commit", 10.5, MUTED, "normal", "middle")

    # 2. Reproducible Response Bank and Balanced Carrier Gate.
    x = xs[1]
    s.text(x + 14, 96, "Mask block + suffix; ablate the fixed prefix", 10.5, MUTED)
    s.text(x + 14, 118, "Cᵘ", 11, GOLD, "bold")
    token_strip(s, x + 39, 103, ["·", "fixed", "MASK"], [27, 47, 49])
    s.text(x + 14, 149, "Cᵛ", 11, GOLD, "bold")
    token_strip(s, x + 39, 134, ["prompt", "·", "MASK"], [52, 27, 49])
    s.line(x + 174, 129, x + 194, 129, MUTED, 1.1, marker=True)
    s.rect(x + 151, 155, 58, 31, GOLD_FILL, "#D9BB7B", 6)
    s.text(x + 180, 175, "DLM", 11.5, GOLD, "bold", "middle")
    s.text(x + 14, 196, "RRB entry  gᵢ(z)=log pᵛ−log pᵘ", 10.5, MUTED, "bold")
    tiny_table(s, x + 14, 202,
               [["the", "+.8", "+"], ["a", "−.6", "−"], ["this", "+.1", "+"]],
               [74, 60, 38], ["token", "gᵢ", "sign"], row_h=18)
    s.rect(x + 14, 292, 194, 8, LIGHT, LINE, 4)
    s.add(f'<rect x="{x + 15}" y="293" width="102" height="6" rx="3" fill="{BLUE}"/>')
    s.add(f'<rect x="{x + 117}" y="293" width="90" height="6" rx="3" fill="{RED}"/>')
    s.text(x + 14, 286, "q₊=.53", 10.5, BLUE_DARK, "bold")
    s.text(x + 208, 286, "q₋=.47  ⇒  carrier", 10.5, RED, "bold", "end")

    # 3. Response-guided resampling.
    x = xs[2]
    s.rect(x + 14, 91, 195, 34, BLUE_FILL, BLUE_LINE, 7)
    s.text(x + 26, 105, "keyed orientation", 10, MUTED)
    s.text(x + 197, 113, "wᵢ εᵢ = +1", 13, BLUE_DARK, "bold", "end")
    s.text(x + 14, 146, "Draw from the unchanged live row", 10.5, MUTED, "bold")
    tiny_table(s, x + 14, 153,
               [["1", "a", ".25", "−", "reject"], ["2", "the", ".42", "+", "accept"]],
               [26, 55, 44, 31, 53], ["try", "token", "pᵢ", "g", "decision"], row_h=25)
    s.text(x + 14, 244, "accept if sign matches and pᵢ ≥ κ max pᵢ", 10.5, INK)
    s.text(x + 14, 261, "after R misses: one unchecked fallback draw", 10.5, RED)
    s.text(x + 14, 276, "Committed block", 10.5, MUTED, "bold")
    token_strip(s, x + 14, 280, ["the", "next", "…"], [55, 55, 42], highlights=(0,))

    # 4. Verification.
    x = xs[3]
    s.text(x + 14, 96, "Finished text", 10.5, MUTED, "bold")
    token_strip(s, x + 14, 103, ["the", "next", "claim", "…"], [46, 48, 49, 32], highlights=(0, 2))
    s.line(x + 105, 132, x + 105, 150, MUTED, 1.1, marker=True)
    s.rect(x + 28, 153, 154, 34, GREEN_FILL, "#A9D1B8", 7)
    s.text(x + 105, 168, "re-mask and rebuild", 10.5, GREEN, "bold", "middle")
    s.text(x + 105, 181, "the same Response Bank", 10, MUTED, "normal", "middle")
    s.text(x + 14, 208, "Carrier matches mᵢ", 10.5, MUTED, "bold")
    for j, hit in enumerate([1, 1, 0, 1, 0, 1, 1]):
        cx = x + 28 + j * 26
        fill = GREEN_FILL if hit else RED_FILL
        stroke = GREEN if hit else RED
        s.circle(cx, 225, 9, fill, stroke, 1.1)
        s.text(cx, 229, "✓" if hit else "×", 11, stroke, "bold", "middle")
    s.rect(x + 14, 246, 207, 43, LIGHT, LINE, 7)
    s.text(x + 25, 263, "T = Σᵢ mᵢ", 12, INK, "bold")
    s.text(x + 25, 280, "pdet = Pr[Bin(n, ½) ≥ T]", 11, INK)
    s.rect(x + 137, 252, 75, 28, GREEN_FILL, GREEN, 14, 1.2)
    s.text(x + 174.5, 270, "detected", 11, GREEN, "bold", "middle")

    s.rect(18, 317, 924, 29, LIGHT, LINE, 6)
    s.text(35, 336, "REPRODUCIBLE", 10, GOLD, "bold")
    s.text(124, 336, "table uses only the fixed prefix", 10.5, MUTED)
    s.text(350, 336, "MODEL-PROPOSED", 10, BLUE, "bold")
    s.text(449, 336, "retries reuse one probability row", 10.5, MUTED)
    s.text(687, 336, "EXACT NULL", 10, GREEN, "bold")
    s.text(756, 336, "conditional Binomial(n, ½)", 10.5, MUTED)

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
