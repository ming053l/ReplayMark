"""Render the carrier-suitability figures from results/43_viz.json.

One paper figure, three panels:
  (a) heatmap over one 512-token document (16 blocks x 32 positions) of the admission
      score S = 2 min(q+, q-): where the block-masked conditional keeps mass on both
      response signs, i.e. where a watermark bit can be written. Admitted cells
      (S > 0.5) are outlined.
  (b) every position of both documents in the (q+, q-) plane; the gate admits the
      corner q+ > 0.25 and q- > 0.25. Gray = rejected, blue = admitted.
  (c) the block-masked conditional at a strong carrier vs a one-sided position:
      top-12 tokens, bar color = response sign. A carrier needs mass on both colors.

Colors: single-hue sequential (white -> #18548C) for magnitude; blue vs orange
(#18548C / #C4622D) for the response-sign polarity; neutral gray for rejected.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

BLUE, ORANGE, GRAY = "#18548C", "#C4622D", "#9A9A9A"
D = json.load(open("/ssd2/ming/basinmark/results/43_viz.json"))
BLK = D["blk"]
NB = D["gen"] // BLK

cmap = LinearSegmentedColormap.from_list("seqblue", ["#FFFFFF", BLUE])
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})

fig = plt.figure(figsize=(9.2, 4.6))
gs = GridSpec(2, 3, height_ratios=[1.15, 1.0], hspace=0.52, wspace=0.34,
              left=0.06, right=0.985, top=0.92, bottom=0.13)

# ---- (a) heatmap, doc 0 ----
axa = fig.add_subplot(gs[0, :])
Sgrid = np.zeros((NB, BLK))
adm = np.zeros((NB, BLK), dtype=bool)
for r in D["docs"][0]:
    Sgrid[r["block"], r["pos"] % BLK] = r["S"]
    adm[r["block"], r["pos"] % BLK] = r["S"] > 0.5
im = axa.imshow(Sgrid, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                interpolation="nearest")
for b in range(NB):
    for j in range(BLK):
        if adm[b, j]:
            axa.add_patch(Rectangle((j - 0.5, b - 0.5), 1, 1, fill=False,
                                    edgecolor=ORANGE, linewidth=0.9))
axa.set_xlabel("position within block")
axa.set_ylabel("block")
axa.set_title(r"(a) admission score $S_i = 2\min(q_{i,+},\,q_{i,-})$ over one "
              r"512-token document; outlined cells are admitted ($S_i > 0.5$)",
              loc="left")
cb = fig.colorbar(im, ax=axa, fraction=0.025, pad=0.01)
cb.set_label(r"$S_i$")

# ---- (b) (q+, q-) plane ----
axb = fig.add_subplot(gs[1, 0])
qp = np.array([r["qp"] for d in D["docs"] for r in d])
qm = np.array([r["qm"] for d in D["docs"] for r in d])
sel = np.minimum(qp, qm) > 0.25
axb.scatter(qp[~sel], qm[~sel], s=4, c=GRAY, alpha=0.45, linewidths=0,
            label="rejected")
axb.scatter(qp[sel], qm[sel], s=5, c=BLUE, alpha=0.8, linewidths=0,
            label="admitted")
axb.axvline(0.25, ls="--", lw=0.8, c="#555555")
axb.axhline(0.25, ls="--", lw=0.8, c="#555555")
axb.plot([0, 1], [1, 0], lw=0.6, c="#BBBBBB")
axb.set_xlim(0, 1); axb.set_ylim(0, 1)
axb.set_xlabel(r"$q_{i,+}$ (mass responding $+$)")
axb.set_ylabel(r"$q_{i,-}$ (mass responding $-$)")
axb.set_title("(b) the balanced gate", loc="left")
axb.legend(frameon=False, loc="upper right", handletextpad=0.2)

# ---- (c) showcase conditionals ----
for col, (key, ttl) in enumerate(
        [("best", "(c) a strong carrier"), ("onesided", "(d) a one-sided position")]):
    ax = fig.add_subplot(gs[1, 1 + col])
    s = D["showcase"][key]
    probs = np.array(s["probs"])
    colors = [BLUE if g > 0 else (ORANGE if g < 0 else GRAY) for g in s["gsign"]]
    ax.bar(range(len(probs)), probs, color=colors, width=0.72)
    ax.set_xticks(range(len(probs)))
    labels = [t.replace("\n", "\\n").strip() or "␣" for t in s["toks"]]
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("probability" if col == 0 else "")
    ax.set_title(f"{ttl}  ($S_i={s['S']:.2f}$)", loc="left")
if True:
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="s", ls="", color=BLUE,
                               label="response $+$"),
                        Line2D([], [], marker="s", ls="", color=ORANGE,
                               label="response $-$")],
               loc="lower right", frameon=False, ncol=2, bbox_to_anchor=(0.99, 0.0))

fig.savefig("/ssd2/ming/basinmark/paper/figures/fig_carrier_map.pdf",
            bbox_inches="tight")
print("wrote paper/figures/fig_carrier_map.pdf")
