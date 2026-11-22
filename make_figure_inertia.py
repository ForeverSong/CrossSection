"""Manuscript figure for the section on inertia (manuscript/figures/fig11_inertia).

(a) Uniform gap with injection through the bed: head gradient, bed stress and roof stress over their creeping values
    against the injection Reynolds number; similarity solution of the Navier-Stokes equations (lines) and the first-order
    result (dashed).
(b) Head at the fold over its creeping value against J0 = c k H0/nu: curves for three pipe lengths in the small-scale
    box, and the 29 states at rest of Pol (2022).
Reads tmp/paper1_inertia_similarity.csv, paper1_inertia_curve.csv and paper1_inertia.csv (run_inertia.py).
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipetheory.inertia import wide_pipe_coefficients

HERE = os.path.dirname(os.path.abspath(__file__))
TMP, FIG = os.path.join(HERE, "..", "tmp"), os.path.join(HERE, "..", "manuscript", "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "mathtext.fontset": "cm",
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "legend.fontsize": 6.5})

fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.5))
c = wide_pipe_coefficients()

ax = axs[0]
sim = np.loadtxt(os.path.join(TMP, "paper1_inertia_similarity.csv"), delimiter=",", skiprows=1)
R = sim[:, 0]
for j, (lab, slope, y_lab) in enumerate((("head gradient", float(c["S_V"]) / 12, None), ("bed stress", float(c["bed_V"]) / 6, None),
                                         ("roof stress", float(c["roof_V"]) / 6, None)), start=1):
    ax.plot(R, sim[:, j], "k-", lw=1.0)
    ax.plot(R, 1 + slope * R, "k--", lw=0.6)
    ax.text(R[-1] + 0.08, sim[-1, j], lab, va="center", ha="left", fontsize=6.5)
ax.set_xlim(0, 5); ax.set_ylim(0.4, 2.2)
ax.set_xlabel(r"injection Reynolds number $Va/\nu$"); ax.set_ylabel("ratio to creeping flow")
ax.text(0.04, 0.93, "(a)", transform=ax.transAxes)

ax = axs[1]
cv = np.loadtxt(os.path.join(TMP, "paper1_inertia_curve.csv"), delimiter=",", skiprows=1)
for l, ls in ((0.10, ":"), (0.15, "-"), (0.20, "--")):
    s = cv[np.isclose(cv[:, 0], l)]
    ax.plot(s[:, 2], s[:, 3], "k" + ls, lw=0.9, label=rf"$l={100 * l:.0f}$ cm")
rows = list(csv.DictReader(open(os.path.join(TMP, "paper1_inertia.csv"))))
MARK = {"FPH": ("s", r"$d_{50}=0.185$ mm"), "B25": ("o", r"$d_{50}=0.228$ mm"), "FS35": ("^", r"$d_{50}=0.422$ mm")}
for sand, (m, lab) in MARK.items():
    rr = [r for r in rows if r["sand"] == sand and r["Hf_i"] not in ("", "nan")]
    ax.plot([float(r["J0"]) for r in rr], [float(r["Hf_i"]) / float(r["Hf_0"]) for r in rr], m, ms=3.6, mfc="0.75", mec="0.2", mew=0.5, label=lab)
ax.set_xscale("log"); ax.set_xlim(0.03, 3.0); ax.set_ylim(0.98, 1.5)
ax.set_xlabel(r"$J_0=c\,kH_0/\nu$"); ax.set_ylabel(r"$H_{\min}/H_0$")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.9), handlelength=2.2)
ax.text(0.04, 0.93, "(b)", transform=ax.transAxes)
for a_ in axs:
    a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
fig.tight_layout(w_pad=2.0)
fig.subplots_adjust(right=0.97)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig11_inertia.{ext}"), dpi=300)
print("written fig11_inertia")
