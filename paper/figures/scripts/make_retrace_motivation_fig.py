"""Generate the conceptual motivation figure for ReTrace's response channel.

Numbers are explicitly illustrative. The figure explains the mechanism, not an empirical
measurement: local model uncertainty supplies multiple proposals, context ablations assign
opposite response polarities, and a keyed orientation selects one polarity while BCG skips
positions that do not support both signs.
"""

from html import escape
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SVG_OUT = ROOT / "fig_retrace_motivation.svg"
PDF_OUT = ROOT / "fig_retrace_motivation.pdf"
W, H = 960, 286

INK = "#17202B"
MUTED = "#596579"
LINE = "#CCD3DD"
LIGHT = "#F6F8FA"
MASK = "#E7EBF0"
BLUE = "#285F8F"
BLUE_DARK = "#1F4668"
BLUE_FILL = "#EDF3F7"
RED = "#8A6D3B"
RED_FILL = "#F4F0E8"
GREEN = BLUE
GREEN_FILL = BLUE_FILL
GOLD = RED
GOLD_FILL = RED_FILL
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
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=1, marker=False):
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{marker_attr}/>')

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


def panel(s, x, width, number, title, subtitle, color):
    s.rect(x, 35, width, 211, "white", LINE, 10, 1.2)
    s.circle(x + 20, 56, 11, color)
    s.text(x + 20, 60, str(number), 11.5, "white", "bold", "middle")
    s.text(x + 39, 54, title, 14.5, INK, "bold")
    s.text(x + 39, 70, subtitle, 10.5, MUTED)


def token(s, x, y, width, value, fill=LIGHT, stroke=LINE, color=INK, bold=False):
    s.rect(x, y, width, 25, fill, stroke, 4, .9)
    s.text(x + width/2, y + 17, value, 11, color, "bold" if bold else "normal", "middle")


def candidate_row(s, x, y, token_name, probability, color=BLUE):
    s.text(x, y + 10, token_name, 10.5, INK)
    s.rect(x + 58, y, 139, 12, LIGHT, LINE, 6, .7)
    s.add(f'<rect x="{x+59}" y="{y+1}" width="{137*probability/.40}" height="10" '
          f'rx="5" fill="{color}"/>')
    s.text(x + 207, y + 10, f"{probability:.2f}", 10.5, MUTED, anchor="end")


def main():
    s = SVG()
    s.text(18, 22, "Where does ReTrace find room for a watermark?", 15.5, INK, "bold")
    s.text(942, 22, "illustrative probability row", 10.5, MUTED, anchor="end", style="italic")

    xs, ws = [18, 302, 632], [274, 320, 310]
    panel(s, xs[0], ws[0], 1, "The model has local choice", "several tokens remain plausible", GRAY)
    panel(s, xs[1], ws[1], 2, "Context assigns polarity", "RRB measures a replayable response", GOLD)
    panel(s, xs[2], ws[2], 3, "The key selects one side", "RAR embeds; RRV replays", BLUE)

    # Stage 1: a masked position with nontrivial conditional entropy.
    x = xs[0]
    token(s, x + 18, 84, 54, "The")
    token(s, x + 76, 84, 65, "result")
    token(s, x + 145, 84, 58, "MASK", MASK, LINE, MUTED)
    token(s, x + 207, 84, 43, ".")
    s.text(x + 18, 126, "live conditional  pᵢ", 10.5, MUTED, "bold")
    for j, (name, prob) in enumerate([("clear", .34), ("robust", .29),
                                      ("stable", .22), ("concise", .11)]):
        candidate_row(s, x + 25, 137 + 22*j, name, prob)
    s.text(x + 137, 237, "all are model proposals", 10.5, MUTED, anchor="middle")

    # Stage 2: two probe contexts assign candidate-specific response signs.
    x = xs[1]
    s.text(x + 18, 93, "two replayable probe contexts", 10.5, MUTED, "bold")
    s.rect(x + 18, 101, 131, 30, GOLD_FILL, "#D9BB7B", 6)
    s.text(x + 29, 119, "Cᵘ: near context hidden", 10.5, GOLD, "bold")
    s.rect(x + 159, 101, 143, 30, GOLD_FILL, "#D9BB7B", 6)
    s.text(x + 170, 119, "Cᵛ: far context hidden", 10.5, GOLD, "bold")
    s.text(x + 18, 151, "candidate", 10, MUTED, "bold")
    s.text(x + 167, 151, "gᵢ(z)=log pᵛ−log pᵘ", 10, MUTED, "bold", "middle")
    s.text(x + 285, 151, "set", 10, MUTED, "bold", "middle")
    rows = [("clear", "−0.71", "−"), ("robust", "+0.54", "+"),
            ("stable", "+0.26", "+"), ("concise", "−0.19", "−")]
    for j, (name, contrast, sign) in enumerate(rows):
        yy = 158 + 18*j
        if j % 2:
            s.add(f'<rect x="{x+17}" y="{yy-12}" width="286" height="17" fill="{LIGHT}"/>')
        color = BLUE if sign == "+" else RED
        s.text(x + 24, yy + 1, name, 10.5, INK)
        s.text(x + 167, yy + 1, contrast, 10.5, color, "bold", "middle")
        s.circle(x + 285, yy - 3, 8, BLUE_FILL if sign == "+" else RED_FILL, color, 1)
        s.text(x + 285, yy + 1, sign, 10.5, color, "bold", "middle")
    s.rect(x + 18, 232, 284, 8, LIGHT, LINE, 4)
    s.add(f'<rect x="{x+19}" y="233" width="127" height="6" rx="3" fill="{RED}"/>')
    s.add(f'<rect x="{x+146}" y="233" width="11" height="6" fill="{GRAY}"/>')
    s.add(f'<rect x="{x+157}" y="233" width="144" height="6" rx="3" fill="{BLUE}"/>')
    s.text(x + 18, 224, "q₋=.45", 10.5, RED, "bold")
    s.text(x + 160, 224, "q₊=.51", 10.5, BLUE, "bold")
    s.text(x + 302, 224, "Sᵢ=.90", 10.5, GREEN, "bold", "end")

    # Stage 3: orientation turns the response split into a keyed bit and replayable hit.
    x = xs[2]
    s.rect(x + 18, 85, 274, 34, BLUE_FILL, "#9BC3EA", 7)
    s.text(x + 31, 106, "secret orientation asks for  +  response", 11.5, BLUE_DARK, "bold")
    s.text(x + 18, 140, "RAR accepts from the requested set", 10.5, MUTED, "bold")
    token(s, x + 18, 149, 74, "clear", RED_FILL, RED, RED)
    s.text(x + 100, 166, "×", 14, RED, "bold")
    token(s, x + 120, 149, 78, "robust", BLUE_FILL, BLUE, BLUE_DARK, True)
    s.text(x + 207, 166, "✓", 14, GREEN, "bold")
    token(s, x + 227, 149, 65, "stable", BLUE_FILL, BLUE, BLUE_DARK)
    s.line(x + 155, 179, x + 155, 195, MUTED, 1.2, marker=True)
    s.rect(x + 45, 198, 220, 37, GREEN_FILL, GREEN, 8, 1.1)
    s.text(x + 155, 214, "emit “robust”  •  still a model proposal", 10.5, GREEN, "bold", "middle")
    s.text(x + 155, 229, "RRV replay: gᵢ(robust)>0  ⇒  mᵢ=1", 10.5, INK, "bold", "middle")

    s.rect(18, 255, 924, 25, LIGHT, LINE, 6)
    s.text(34, 272, "BALANCED", 10.5, GOLD, "bold")
    s.rect(110, 263, 96, 8, "white", LINE, 4)
    s.add(f'<rect x="111" y="264" width="45" height="6" rx="3" fill="{RED}"/>')
    s.add(f'<rect x="160" y="264" width="45" height="6" rx="3" fill="{BLUE}"/>')
    s.text(217, 272, "two-sided  →  carrier", 10.5, INK, "bold")
    s.text(458, 272, "ONE-SIDED", 10.5, MUTED, "bold")
    s.rect(548, 263, 96, 8, "white", LINE, 4)
    s.add(f'<rect x="549" y="264" width="8" height="6" rx="3" fill="{RED}"/>')
    s.add(f'<rect x="561" y="264" width="82" height="6" rx="3" fill="{BLUE}"/>')
    s.text(655, 272, "sharp  →  BCG skips", 10.5, INK, "bold")
    s.text(930, 272, "capacity = min(q₊,q₋)", 10.5, GOLD, "bold", "end")

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
