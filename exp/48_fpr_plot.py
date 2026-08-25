"""Redraw the human-text FPR calibration figure (Figure 5) from results/48_fpr.json in the
paper's style: Times serif, sized for 0.62\textwidth inclusion."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, BLACK = "#18548C", "#C4622D", "#3E7C4F", "#111111"
ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "results" / "48_fpr.json", encoding="utf-8") as handle:
    D = json.load(handle)
pv = {k: np.sort(np.array(v)) for k, v in D["p_values"].items()}
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10,
                     "font.family": "serif",
                     "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
fig, ax = plt.subplots(figsize=(3.9, 3.1))
alphas = np.logspace(-3.31, 0, 300)
ax.plot(alphas, alphas, "--", color=BLACK, lw=1.0, label="nominal ($y=x$)")
for (k, v), col in zip(sorted(pv.items()), [BLUE, ORANGE, GREEN]):
    emp = np.searchsorted(v, alphas, side="right") / len(v)
    ax.plot(alphas, np.maximum(emp, 1e-4), color=col, lw=1.0, alpha=0.85,
            label=f"key {k} ($n=600$)")
allp = np.sort(np.concatenate(list(pv.values())))
emp = np.searchsorted(allp, alphas, side="right") / len(allp)
ax.plot(alphas, np.maximum(emp, 1e-4), color=BLACK, lw=1.8,
        label=f"pooled ($n=1800$)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(4e-4, 1); ax.set_ylim(4e-4, 1)
ax.set_xlabel(r"nominal FPR $\alpha$")
ax.set_ylabel("empirical FPR")
ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.6)
fig.savefig(ROOT / "paper" / "figures" / "fig_fpr_calibration.pdf", bbox_inches="tight")
print("wrote fig_fpr_calibration.pdf")
