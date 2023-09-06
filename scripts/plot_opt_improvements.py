import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

df = pd.read_csv("speed_data.csv")
print(df.columns)

labels = ["without\noptimization", 
          "+ no bounds\nchecking", 
          "+ CUDA shared\nmemory", 
          "+ matrix\nreordering",
          "+ CUDA\nthreads"]
plt.bar(labels, df["mean"], yerr=df["std"], capsize=10.0, color="C0")
plt.ylabel("Runtime (ms)")
plt.xticks(rotation=-20)
plt.title("Solver Optimization Improvements")
plt.tight_layout()
opts = dict(bbox_inches="tight", pad_inches=0.1)
plt.savefig("figs/opt_improvements.png", dpi=200, **opts)
plt.savefig("figs/opt_improvements.svg", **opts)
plt.savefig("figs/opt_improvements.pdf", **opts)
plt.show()