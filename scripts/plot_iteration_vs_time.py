from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

fname = Path("time_vs_iters.json")
iters, mu, std = map(np.array, json.loads(fname.read_text()))

plt.plot(iters, mu, marker=".")
plt.fill_between(iters, mu - std, mu + std, alpha=0.3)
p = np.polyfit(iters, mu, 1)
plt.plot(iters, np.polyval(p, iters), label=f"Linear fit: {p[0]:.4e} $\\frac{{ms}}{{it}}$ x + {p[1]:.4e} ms", linestyle="--")
plt.ylim(0, plt.ylim()[1])
plt.xlim(0, plt.xlim()[1])
plt.minorticks_on()
plt.grid(which="major", visible=True)
plt.grid(which="minor", visible=True, linestyle=":")
plt.ylabel("Runtime (ms)")
plt.xlabel("ADMM Iterations")
plt.title("Algorithm Setup and Iteration Runtime")
plt.legend()
plt.tight_layout()
opts = dict(bbox_inches="tight", pad_inches=0.1)
plt.savefig("figs/iteration_scan.pdf", **opts)
plt.savefig("figs/iteration_scan.svg", **opts)
plt.savefig("figs/iteration_scan.png", dpi=200, **opts)
plt.show()