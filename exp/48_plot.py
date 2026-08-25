"""Render the carrier-count / CI figure from results/48_carrier_stats.json.

(a) per-document carrier counts |P| for each arm (every dot one document; bar = mean,
    whisker = 95% t-interval). Carrier selection ignores R/kappa, so control and
    watermarked arms of one model should overlap.
(b) per-document 95% Wilson intervals on the keyed match rate for the two watermarked
    arms, documents sorted; the 0.5 line is the null. Documents whose interval clears
    0.5 carry individually decisive evidence.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GRAY = "#18548C", "#C4622D", "#9A9A9A"
D = json.load(open("/ssd2/ming/basinmark/results/48_carrier_stats.json"))["rows"]

def wilson(h, n, z=1.96):
    if n == 0:
        return 0.5, 0.5
    p = h / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - w, c + w

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                     "font.family": "serif",
                     "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 2.9),
                               gridspec_kw=dict(width_ratios=[1, 1.35],
                                                wspace=0.28))

groups = [("llada_control", "LLaDA\ncontrol", GRAY),
          ("llada_R8k10", "LLaDA\n$R{=}8,\\kappa{=}.1$", BLUE),
          ("dream_control", "Dream\ncontrol", GRAY),
          ("dream_R16k05", "Dream\n$R{=}16,\\kappa{=}.05$", BLUE)]
rng = np.random.default_rng(0)
for x, (key, lab, col) in enumerate(groups):
    ns = np.array([r["n"] for r in D[key]])
    jit = rng.uniform(-0.16, 0.16, len(ns))
    axa.scatter(x + jit, ns, s=9, c=col, alpha=0.65, linewidths=0)
    m = ns.mean()
    ci = 1.96 * ns.std(ddof=1) / np.sqrt(len(ns))
    axa.errorbar([x + 0.30], [m], yerr=[ci], fmt="_", color="#222222",
                 capsize=3, markersize=11, elinewidth=1.1)
axa.set_xticks(range(len(groups)))
axa.set_xticklabels([g[1] for g in groups], fontsize=7)
axa.set_ylabel("carriers per document $|P|$")
axa.set_title("(a) carrier counts (dot = document; mean $\\pm$ 95% CI)", loc="left")

for off, (key, lab, col) in enumerate([("llada_R8k10", "LLaDA $R{=}8,\\kappa{=}0.1$", BLUE),
                                       ("dream_R16k05", "Dream $R{=}16,\\kappa{=}0.05$",
                                        ORANGE)]):
    rows = sorted(D[key], key=lambda r: r["rate"])
    xs = np.arange(len(rows)) + off * (len(D["llada_R8k10"]) + 4)
    for x, r in zip(xs, rows):
        lo, hi = wilson(r["hits"], r["n"])
        axb.plot([x, x], [lo, hi], color=col, lw=1.1, alpha=0.8)
        axb.plot([x], [r["rate"]], marker="o", ms=2.4, color=col)
    axb.plot([], [], color=col, lw=1.4, label=lab)
axb.axhline(0.5, ls="--", lw=0.9, c="#555555")
axb.text(len(D["llada_R8k10"]) + 1.2, 0.512, "null", fontsize=6.5,
         color="#555555", ha="center")
axb.set_ylim(0.3, 1.02)
axb.set_xticks([])
axb.set_xlabel("documents, sorted by match rate")
axb.set_ylabel("keyed match rate")
axb.set_title("(b) per-document 95% Wilson intervals", loc="left")
axb.legend(frameon=False, loc="lower right", fontsize=7, handlelength=1.2)

fig.savefig("/ssd2/ming/basinmark/paper/figures/fig_carrier_ci.pdf",
            bbox_inches="tight")
print("wrote paper/figures/fig_carrier_ci.pdf")
