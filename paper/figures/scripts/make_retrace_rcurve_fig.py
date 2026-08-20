"""Detection against the retry budget -- the paper's operating-point figure.

Reads results/21_R_matched.json when present (final numbers land there as the sweep ends)
and falls back to the printed interim rows, so re-running after the sweep re-renders the
final figure. Matplotlib with Times to match the body text; baseline reference lines are
the LOCAL dgMARK and KGW reproductions, not paper-quoted numbers.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.family": "serif",
                            "font.serif": ["Liberation Serif", "Times New Roman"],
                            "font.size": 10, "axes.linewidth": 0.8})
OURS = "#005AB5"; PRIOR = "#7B6651"; FAIL = "#B2182B"; MUT = "#596579"

R = [1, 2, 4, 8]
tpr = {5: [None, 0.83, 0.90, None], 1: [0.38, 0.73, 0.83, None],
       0.1: [0.23, 0.63, 0.77, None]}
src = "/ssd1/ming/basinmark/results/21_R_matched.json"
if os.path.exists(src):
    d = json.load(open(src))
    for k, arm in ((2, "R2"), (4, "R4"), (8, "R8")):
        if arm in d.get("stats", {}):
            ps = np.array([r["p"] for r in d["stats"][arm]])
            i = R.index(k)
            tpr[5][i] = float(np.mean(ps < .05))
            tpr[1][i] = float(np.mean(ps < .01))
            tpr[0.1][i] = float(np.mean(ps < .001))

fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=200)
for a, mk, lab in ((5, "o", "TPR@5%"), (1, "s", "TPR@1%"), (0.1, "^", "TPR@0.1%")):
    xs = [r for r, v in zip(R, tpr[a]) if v is not None]
    ys = [v for v in tpr[a] if v is not None]
    ax.plot(xs, ys, marker=mk, ms=4.5, lw=1.4, color=OURS,
            alpha={5: .45, 1: 1.0, 0.1: .7}[a], label=lab)
ax.axhline(0.86, color=PRIOR, lw=1.1, ls="--")
ax.text(8.2, 0.865, "dgMARK 3-beam @1% (×1.23 ppl)", color=PRIOR, fontsize=7.5,
        ha="right", va="bottom")
ax.axhline(0.93, color=PRIOR, lw=1.1, ls=":")
ax.text(8.2, 0.935, "KGW δ=1 @1% (×1.03 ppl)", color=PRIOR, fontsize=7.5,
        ha="right", va="bottom")
ax.set_xscale("log", base=2); ax.set_xticks(R); ax.set_xticklabels(R)
ax.set_xlabel("retry budget R  (per-token draws; no extra model forwards)")
ax.set_ylabel("TPR at fixed FPR")
ax.set_ylim(0, 1.02); ax.set_xlim(0.9, 9)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Detection vs retry budget (held-out, n=30, 512 tokens)", fontsize=9)
fig.tight_layout()
fig.savefig("/ssd1/ming/basinmark/paper/figures/fig_retrace_rcurve.pdf")
print("figures/fig_retrace_rcurve.pdf")
