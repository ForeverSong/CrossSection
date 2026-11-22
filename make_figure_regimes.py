"""Figure: the three regimes of the pipe law with every measured state (manuscript/figures/fig7_regimes.pdf/.png).
Discharge and gradient are scaled with their values where the self-formed width equals the width of the tip,
S_x = w A / W0 (w = 2 beta, the width of the section in units of A/S) and Q_x = Lambda / S_x^3 (W0 = 30 d50), so that all sands share the same theoretical curve:
gradient set by the width of the tip (S > S_x), self-formed (Q S^3 = Lambda), and the plateaus S* = A/(c d50).
States: tmp/paper1_pipe_width_data.csv (small and medium scale; written by pipe_width_data.py) and the large-scale
test (gradient from large_scale_data.py, discharge 18-24 mL/s).     usage:  python make_figure_regimes.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section

HERE = os.path.dirname(os.path.abspath(__file__))
TMP, FIG = os.path.join(HERE, "..", "tmp"), os.path.join(HERE, "..", "manuscript", "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6, "mathtext.fontset": "cm"})
sec = threshold_section(0.77, 16, 161, 41)
pts = []
for r in csv.DictReader(open(os.path.join(TMP, "paper1_pipe_width_data.csv"))):
    law = PipeLaw(sec, float(r["tau_c"]), W0=30 * float(r["d50"]))
    pts.append((r["scale"], float(r["Q"]) / law.Q_x, float(r["S"]) / law.S_x, r["sand"]))
fph = PipeLaw(sec, 0.339 / (sec["a_max"] * sec["aavg_over_amax"]), W0=30 * 0.185e-3, nu=1.52e-6)
large = [(q / fph.Q_x, 0.111 / fph.S_x) for q in (18e-6, 24e-6)]
ref = PipeLaw(sec, 0.45, W0=30 * 0.2e-3)
q = np.logspace(-2.5, 4.3, 300)
s = ref.slope(q * ref.Q_x)[0] / ref.S_x

fig, ax = plt.subplots(figsize=(3.6, 2.9))
ax.loglog(q, s, "k-", lw=1.0)
ax.loglog(q[q < 1], q[q < 1] ** (-1 / 3), "k:", lw=0.8)
for c in (4, 7):
    y = 15.0 / (ref.beta * c)
    ax.loglog([y ** -3, q[-1]], [y, y], "k--", lw=0.7); ax.text(q[-1] * 0.9, y * 1.07, rf"$a={c}\,d_{{50}}$", ha="right", fontsize=7)
for scale, mk, fc, lab in (("small", "o", "w", "small scale"), ("medium", "s", "0.35", "medium scale")):
    p = np.array([(a, b) for sc, a, b, _ in pts if sc == scale])
    ax.loglog(p[:, 0], p[:, 1], mk, ms=3.5, mfc=fc, mec="k", mew=0.6, ls="none", label=lab)
ax.loglog([x for x, _ in large], [y for _, y in large], "k-", lw=2.2, solid_capstyle="butt", label="large scale")
ax.set_xlabel(r"$Q/Q_\times$"); ax.set_ylabel(r"$S/S_\times$"); ax.set_xlim(q[0], q[-1]); ax.set_ylim(0.08, 6)
ax.text(0.0042, 1.9, "limited by the\nwidth of the tip", fontsize=7); ax.text(60, 0.16, r"self-formed, $QS^3=\Lambda$", fontsize=7, rotation=-24)
ax.legend(frameon=False, fontsize=7, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig7_regimes.{ext}"), dpi=300)
for scale in ("small", "medium"):
    p = np.array([(a, b) for sc, a, b, _ in pts if sc == scale])
    print(f"{scale}: {len(p)} states, Q/Q_x {p[:, 0].min():.2g}-{p[:, 0].max():.2g}, S/S_x {p[:, 1].min():.2f}-{p[:, 1].max():.2f}, S (Q/Q_x)^(1/3)/S_x {np.min(p[:, 1] * p[:, 0] ** (1 / 3)):.2f}-{np.max(p[:, 1] * p[:, 0] ** (1 / 3)):.2f}")
print("large:", [(round(a, 1), round(b, 2)) for a, b in large], "| beta", round(float(ref.beta), 2), "| plateaus c=4, 7:", round(15 / (ref.beta * 4), 2), round(15 / (ref.beta * 7), 2))
