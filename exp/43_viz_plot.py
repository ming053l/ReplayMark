"""Render the carrier-availability heat map from saved measurements."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle


BLUE = "#18548C"
ORANGE = "#C4622D"
DATA_PATH = "/ssd2/ming/basinmark/results/43_viz.json"
OUT_PATH = "/ssd2/ming/basinmark/paper/figures/fig_carrier_map.pdf"

with open(DATA_PATH, encoding="utf-8") as handle:
    data = json.load(handle)

block_size = data["blk"]
num_blocks = data["gen"] // block_size

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "font.size": 9.4,
    "axes.titlesize": 10.2,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
})

fig, axis = plt.subplots(figsize=(8.0, 2.05))
fig.subplots_adjust(left=0.09, right=0.94, top=0.91, bottom=0.22)

score_grid = np.zeros((num_blocks, block_size))
retained_grid = np.zeros((num_blocks, block_size), dtype=bool)
for row in data["docs"][0]:
    block = row["block"]
    offset = row["pos"] % block_size
    score_grid[block, offset] = row["S"]
    retained_grid[block, offset] = row["S"] > 0.5

colour_map = LinearSegmentedColormap.from_list("score_blue", ["#FFFFFF", BLUE])
image = axis.imshow(
    score_grid,
    aspect="auto",
    cmap=colour_map,
    vmin=0,
    vmax=1,
    interpolation="nearest",
)

for block in range(num_blocks):
    for offset in range(block_size):
        if not retained_grid[block, offset]:
            axis.add_patch(Rectangle(
                (offset - 0.5, block - 0.5),
                1,
                1,
                fill=False,
                edgecolor=ORANGE,
                linewidth=1.1,
            ))

axis.set_xticks([0, 5, 11, 17, 23, 31], [1, 6, 12, 18, 24, 32])
axis.set_yticks([0, 3, 7, 11, 15], [1, 4, 8, 12, 16])
axis.set_xlabel("position within block")
axis.set_ylabel("block")
axis.legend(
    handles=[Patch(
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=1.1,
        label="not used: below selection cutoff",
    )],
    loc="upper right",
    frameon=True,
    framealpha=0.94,
    borderpad=0.35,
    handlelength=1.15,
    fontsize=8.0,
)

colour_bar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.012)
colour_bar.set_ticks([0, 1])
colour_bar.set_ticklabels(["one dominates", "evenly divided"])
colour_bar.set_label("response balance")

fig.savefig(OUT_PATH, bbox_inches="tight")
print("wrote paper/figures/fig_carrier_map.pdf")
