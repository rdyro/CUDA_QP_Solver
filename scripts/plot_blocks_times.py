from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

fnames = ["32_threads.json", "64_threads.json"]

for i, fname in enumerate(fnames):
    fname = Path(fname)
    x, mu, std = map(np.array, json.loads(fname.read_text()))
    plt.plot(x, mu, marker=".", label=fname.with_suffix("").name.replace("_", " "), color=f"C{i}")
    plt.fill_between(x, mu - std, mu + std, alpha=0.3, color=f"C{i}")
plt.legend()
plt.ylim(0, plt.ylim()[1])
plt.xlabel("Number of blocks/QPs")
plt.ylabel("Runtime (ms)")
plt.grid(which="both", axis="y")
plt.title("Runtime depending on the number of QPs")
for i in range(1, 7):
    plt.plot([82 * i, 82 * i], plt.ylim(), color="black", alpha=0.3, linestyle="--")
plt.tight_layout()
opts = dict(bbox_inches="tight", pad_inches=0.1)
plt.savefig("figs/blocks_scan.png", dpi=200, **opts)
plt.savefig("figs/blocks_scan.pdf", **opts)
plt.savefig("figs/blocks_scan.svg", **opts)
plt.show()