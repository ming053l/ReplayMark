"""Generate the three-panel empirical summary figure from reported paper values.

This is intentionally dependency-free and keeps the SVG as an editable source. Values
are the local rows reported in Table 1 and Sections 5.2/5.5; the plot does not pool runs
or interpolate missing measurements.
"""

from html import escape
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SVG_OUT = ROOT / "fig_retrace_evidence.svg"
PDF_OUT = ROOT / "fig_retrace_evidence.pdf"
W, H = 960, 326

INK = "#17202B"
MUTED = "#596579"
GRID = "#D9DFE7"
LIGHT = "#F6F8FA"
BLUE = "#005AB5"
BLUE_SOFT = "#83B9EA"
RED = "#B2182B"
GREEN = "#287A4B"
BROWN = "#8B6843"
GRAY = "#697384"


class SVG:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)

    def text(self, x, y, value, size=12, fill=INK, weight="normal", anchor="start",
             style="normal", rotate=None):
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.add(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" font-style="{style}" text-anchor="{anchor}"{transform}>'
            f'{escape(value)}</text>'
        )

    def line(self, x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                 f'stroke="{stroke}" stroke-width="{width}"{extra}/>' )

    def rect(self, x, y, w, h, fill="white", stroke=GRID, radius=6, width=1):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def circle(self, x, y, r, fill, stroke="white", width=1.2):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{width}"/>')

    def poly(self, points, fill, stroke="white", width=1.2):
        ps = " ".join(f"{x},{y}" for x, y in points)
        self.add(f'<polygon points="{ps}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def render(self):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"
viewBox="0 0 {W} {H}">
<style>text {{ font-family: 'Times New Roman', 'Liberation Serif', serif; }}</style>
<rect width="100%" height="100%" fill="white"/>
{"\n".join(self.items)}
</svg>'''


def title(s, x, letter, main, sub):
    s.text(x, 24, f"({letter})", 13, BLUE, "bold")
    s.text(x + 27, 24, main, 13.5, INK, "bold")
    s.text(x + 27, 40, sub, 10.5, MUTED)


def marker(s, x, y, kind, color, size=6):
    if kind == "circle":
        s.circle(x, y, size, color)
    elif kind == "diamond":
        s.poly([(x, y-size-1), (x+size+1, y), (x, y+size+1), (x-size-1, y)], color)
    elif kind == "square":
        s.rect(x-size, y-size, 2*size, 2*size, color, "white", 1, 1.2)
    elif kind == "triangle":
        s.poly([(x, y-size-1), (x+size+1, y+size), (x-size-1, y+size)], color)


def axes(s, x, y, w, h, xticks, yticks, xmap, ymap):
    for value, label in yticks:
        yy = ymap(value)
        s.line(x, yy, x+w, yy, GRID, .8)
        s.text(x-7, yy+4, label, 10.5, MUTED, anchor="end")
    for value, label in xticks:
        xx = xmap(value)
        s.line(xx, y, xx, y+h, GRID, .65)
        s.text(xx, y+h+16, label, 10.5, MUTED, anchor="middle")
    s.line(x, y+h, x+w, y+h, INK, 1)
    s.line(x, y, x, y+h, INK, 1)


def main():
    s = SVG()
    # subtle panel backgrounds and separators
    for x, w in [(12, 310), (329, 304), (640, 308)]:
        s.rect(x, 8, w, 305, "white", GRID, 8, 1)

    # (a) Joint frontier: every point corresponds to a row in the main comparison table.
    title(s, 25, "a", "Detection–quality frontier", "local comparison rows; TPR at 1% FPR")
    ax, ay, aw, ah = 67, 62, 231, 190
    xmin, xmax, ymin, ymax = .70, 1.70, .30, 1.02
    xm = lambda v: ax + (v-xmin)/(xmax-xmin)*aw
    ym = lambda v: ay + ah - (v-ymin)/(ymax-ymin)*ah
    axes(s, ax, ay, aw, ah,
         [(.75, ".75"), (1.0, "1.0"), (1.25, "1.25"), (1.5, "1.5")],
         [(.4, ".4"), (.6, ".6"), (.8, ".8"), (1.0, "1.0")], xm, ym)
    s.text(ax+aw/2, 286, "paired PPL ratio  ← better", 11, INK, anchor="middle")
    s.text(27, ay+ah/2, "TPR ↑", 11, INK, "bold", "middle", rotate=-90)

    retrace = [(.97, .43, "R1"), (1.14, .73, "R2"), (1.40, .83, "R4"),
               (1.61, .87, "R8")]
    s.add('<polyline points="' + " ".join(f"{xm(x)},{ym(y)}" for x,y,_ in retrace) +
          f'" fill="none" stroke="{BLUE_SOFT}" stroke-width="2"/>')
    for xv, yv, label in retrace:
        marker(s, xm(xv), ym(yv), "circle", BLUE, 5.5)
        dx, dy = {"R1":(7,14), "R2":(7,13), "R4":(6,-8), "R8":(-7,-9)}[label]
        s.text(xm(xv)+dx, ym(yv)+dy, label, 9.8, BLUE, "bold", "end" if dx < 0 else "start")

    for xv, yv, label, dx, dy in [(.87,.70,"floor .10",7,-8), (.74,.80,"floor .05",7,-8)]:
        marker(s, xm(xv), ym(yv), "square", GREEN, 5.5)
        s.text(xm(xv)+dx, ym(yv)+dy, label, 9.5, GREEN, "bold")

    baselines = [(1.03,.93,"KGW δ1","triangle",GRAY,-7,-9,"end"),
                 (1.21,1.00,"KGW δ3","triangle",GRAY,7,13,"start"),
                 (1.55,.74,"dgMARK k1","diamond",BROWN,-7,15,"end"),
                 (1.23,.86,"dgMARK b3","diamond",BROWN,-7,-8,"end")]
    for xv,yv,label,kind,color,dx,dy,anchor in baselines:
        marker(s, xm(xv), ym(yv), kind, color, 5.5)
        s.text(xm(xv)+dx, ym(yv)+dy, label, 9.5, color, "bold", anchor)
    s.text(178, 305, "● ReTrace   ■ +floor   ◆ dgMARK   ▲ KGW", 10, MUTED, anchor="middle")

    # (b) Two aligned microplots make the retry trade-off legible without a dual y-axis.
    title(s, 342, "b", "What retries change", "unfloored 512-token operating points")
    bx, bw = 379, 226
    xvals = [1,2,4,8]
    bxp = lambda v: bx + {1:0,2:1,4:2,8:3}[v] * (bw/3)

    # upper: detection
    y0, hh = 65, 79
    by = lambda v: y0 + hh - (v-.3)/(.6)*hh
    for v in [.4,.6,.8]:
        yy=by(v); s.line(bx,yy,bx+bw,yy,GRID,.75); s.text(bx-7,yy+4,f"{v:.1f}",10,MUTED,anchor="end")
    s.line(bx,y0+hh,bx+bw,y0+hh,INK,1); s.line(bx,y0,bx,y0+hh,INK,1)
    det=[.43,.73,.83,.87]
    s.add('<polyline points="'+" ".join(f"{bxp(r)},{by(v)}" for r,v in zip(xvals,det))+
          f'" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    for r,v in zip(xvals,det): marker(s,bxp(r),by(v),"circle",BLUE,5)
    s.text(bx+5,y0+13,"TPR@1%",10.5,BLUE,"bold")

    # lower: quality cost
    y1 = 180
    qy = lambda v: y1 + hh - (v-.7)/(1.0)*hh
    for v in [.8,1.2,1.6]:
        yy=qy(v); s.line(bx,yy,bx+bw,yy,GRID,.75); s.text(bx-7,yy+4,f"{v:.1f}",10,MUTED,anchor="end")
    s.line(bx,y1+hh,bx+bw,y1+hh,INK,1); s.line(bx,y1,bx,y1+hh,INK,1)
    ppl=[.97,1.14,1.40,1.61]
    s.add('<polyline points="'+" ".join(f"{bxp(r)},{qy(v)}" for r,v in zip(xvals,ppl))+
          f'" fill="none" stroke="{RED}" stroke-width="2"/>')
    for r,v in zip(xvals,ppl): marker(s,bxp(r),qy(v),"circle",RED,5)
    s.text(bx+5,y1+13,"PPL ratio",10.5,RED,"bold")
    for r in xvals:
        s.text(bxp(r), y1+hh+17, str(r), 10.5, MUTED, anchor="middle")
    s.text(bx+bw/2, 294, "retry budget R", 11, INK, anchor="middle")
    s.rect(355, 300, 250, 0, "none", "none")
    s.text(481, 308, "Floor (TPR / PPL):  R8 κ=.1 → .70/.87;  R16 κ=.05 → .80/.74",
           9.5, GREEN, "bold", "middle")

    # (c) Honest failure mode: paired bars at the same analytic decision threshold.
    title(s, 653, "c", "Post-edit signal collapses", "R=1, n=20; analytic 1% threshold")
    cx, cy, cw, ch = 687, 70, 237, 184
    cmax=.40
    cym=lambda v: cy+ch-v/cmax*ch
    for v in [0,.1,.2,.3,.4]:
        yy=cym(v); s.line(cx,yy,cx+cw,yy,GRID,.75); s.text(cx-7,yy+4,f"{v:.1f}",10,MUTED,anchor="end")
    s.line(cx,cy+ch,cx+cw,cy+ch,INK,1); s.line(cx,cy,cx,cy+ch,INK,1)
    labels=["clean","re-denoise\n10%","re-denoise\n20%","delete\n10%"]
    wm=[.35,.05,.05,.05]; ctrl=[0,.05,0,.05]
    for i,(lab,wv,cv) in enumerate(zip(labels,wm,ctrl)):
        center=cx+31+i*58
        top=cym(wv); s.rect(center-12,top,17,cy+ch-top,BLUE,"none",1,0)
        top2=cym(cv); s.rect(center+7,top2,17,max(1,cy+ch-top2),GRAY,"none",1,0)
        s.text(center-3.5,top-5,f"{wv:.2f}",9.5,BLUE,"bold","middle")
        parts=lab.split("\n")
        s.text(center+3,270,parts[0],9.5,MUTED,anchor="middle")
        if len(parts)>1: s.text(center+3,281,parts[1],9.5,MUTED,anchor="middle")
    s.text(653, cy+ch/2, "rate", 11, INK, "bold", "middle", rotate=-90)
    s.rect(708, 295, 10, 10, BLUE, "none", 1, 0); s.text(723,304,"watermarked TPR",10,MUTED)
    s.rect(822, 295, 10, 10, GRAY, "none", 1, 0); s.text(837,304,"control FPR",10,MUTED)

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
