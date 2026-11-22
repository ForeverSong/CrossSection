"""Measured cross-section of a pipe against the threshold section (manuscript/figures/fig2c_section_data.pdf/.png).
Measured: Pol (2022), thesis Fig. 3.10, test B25-245 at x = 0.22 m (digitised: data/pol2022_fig3_10_B25_245_x022.csv).
Theory: self-formed threshold section for the friction coefficient of this sand, scaled to the largest measured depth and
placed on the axis of the main channel (one parameter).  Also printed: the size the theory gives, unscaled, for the smallest and
largest mean pipe gradient measured in this test (tmp/paper1_pipe_width_data.csv).          usage:  python make_figure_section_data.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipetheory.section import threshold_section

HERE = os.path.dirname(os.path.abspath(__file__))
TMP, FIG = os.path.join(HERE, "..", "tmp"), os.path.join(HERE, "..", "manuscript", "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6, "mathtext.fontset": "cm"})
d = np.loadtxt(os.path.join(HERE, "data", "pol2022_fig3_10_B25_245_x022.csv"), delimiter=",", skiprows=1)
o = np.argsort(d[:, 0]); y, dep = d[o, 0], np.maximum(d[o, 1], 0.0)
area = np.trapezoid(dep, y); yc = np.trapezoid(dep * y, y) / area
inside = y[dep > 0.1 * dep.max()]
rows = [r for r in csv.DictReader(open(os.path.join(TMP, "paper1_pipe_width_data.csv"))) if r["test"] == "B25_245"]
tau_c, d50 = float(rows[0]["tau_c"]), float(rows[0]["d50"]) * 1e3
S = sorted(float(r["S"]) for r in rows)
sec = threshold_section(np.tan(np.radians(37.6)), 16, 161, 41)
ys, a = np.asarray(sec["y"], float), np.asarray(sec["a"], float)
if ys.min() >= 0:
    ys, a = np.concatenate([-ys[::-1], ys]), np.concatenate([a[::-1], a])
A = 2 * tau_c / 9810 * 1e3                                                  # mm
im = int(np.argmax(dep)); dm = dep[im]


def edge(level, step):                                                    # where the main channel becomes shallower than `level`
    j = im
    while 0 < j < len(y) - 1 and dep[j] > level:
        j += step
    return y[j]


y0 = 0.5 * (edge(0.5 * dm, -1) + edge(0.5 * dm, 1))                        # axis of the main channel
fig, ax = plt.subplots(figsize=(3.6, 1.9))
ax.fill_between(y - y0, -dep, 0, color="0.8", lw=0, label="measured")
sc = dm / a.max()
ax.plot(ys * sc, -a * sc, "k-", lw=0.9, label="threshold section of the same depth")
for f in (0.75, 0.5, 0.25):
    print(f"width where deeper than {f:.2f} of the largest depth: measured {edge(f * dm, 1) - edge(f * dm, -1):.1f} mm, threshold section {np.ptp(ys[a >= f * a.max()]) * sc:.1f} mm")
for s_, in ((S[0],), (S[-1],)):
    print(f"theory, S = {s_:.3f}: depth {a.max() * A / s_:.2f} mm, width {np.ptp(ys) * A / s_:.1f} mm")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("across the pipe (mm)"); ax.set_ylabel("depth (mm)"); ax.set_xlim(-17, 8); ax.set_ylim(-2.4, 1.0)
ax.set_yticks([-2, -1, 0], ["2", "1", "0"]); ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=1, handlelength=1.8)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig2c_section_data.{ext}"), dpi=300)
print(f"measured: {len(y)} points, y {y.min():.1f}-{y.max():.1f} mm, largest depth {dep.max():.2f} mm = {dep.max() / d50:.1f} d50, width where deeper than 10% of that {np.ptp(inside):.1f} mm = {np.ptp(inside) / d50:.0f} d50,"
      f" area {area:.1f} mm2, mean depth over that width {area / np.ptp(inside):.2f} mm;  tau_c {tau_c:.3f} Pa, A {A:.3f} mm, 30 d50 = {30 * d50:.1f} mm")
