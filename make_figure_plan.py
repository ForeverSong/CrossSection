"""Plan-view sketch of a pipe (manuscript/figures/fig1b_plan.pdf/.png): schematic, not to scale.  usage: python make_figure_plan.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "manuscript", "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(figsize=(4.6, 1.75))
ax.add_patch(plt.Rectangle((0, -1), 10, 2, fc="0.93", ec="k", lw=0.7))
s = np.linspace(0, 1, 200); x = 2.0 + 4.2 * s                               # exit at x = 2, tip at x = 6.2
w = 0.07 + 0.27 * (1 - s) ** 0.6 * (s < 1)                                  # half-width: W0/2 near the tip, wider downstream
mid = 0.10 * np.sin(5.5 * s) * s * (1 - s) * 4
ax.fill_between(x, mid - w, mid + w, color="w", ec="k", lw=0.7)
ax.add_patch(plt.Circle((2.0, 0), 0.13, fc="w", ec="k", lw=0.7))
for yy in (-0.75, -0.45, 0.45, 0.75):
    ax.annotate("", xy=(4.2 + 0.25 * abs(yy), 0.42 * np.sign(yy)), xytext=(9.6, yy), arrowprops=dict(arrowstyle="->", lw=0.5, color="0.35", connectionstyle=f"arc3,rad={0.12 * np.sign(yy)}"))
ax.annotate("", xy=(6.35, 0.0), xytext=(9.6, 0.0), arrowprops=dict(arrowstyle="->", lw=0.5, color="0.35"))
ax.annotate("exit", (2.0, -0.2), ha="center", va="top", fontsize=8); ax.annotate("tip", (6.25, 0.16), ha="left", fontsize=8)
ax.annotate("", xy=(2.0, -1.25), xytext=(6.2, -1.25), arrowprops=dict(arrowstyle="<->", lw=0.6)); ax.text(4.1, -1.2, "$l$", ha="center", va="bottom")
ax.annotate("", xy=(2.0, 1.25), xytext=(10.0, 1.25), arrowprops=dict(arrowstyle="<->", lw=0.6)); ax.text(6.0, 1.28, "$L$", ha="center", va="bottom")
ax.annotate("$W_0$", xy=(5.9, 0.07), xytext=(5.3, 0.62), fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.4))
ax.text(10.15, 0, "head $H$", rotation=90, va="center", fontsize=8); ax.text(0.3, 0.7, "roof (plan view)", fontsize=8)
ax.set_xlim(-0.2, 10.6); ax.set_ylim(-1.5, 1.6); ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig1b_plan.{ext}"), dpi=300)
