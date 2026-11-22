"""Figure for the large-scale test (manuscript/figures/fig6_large_scale.pdf/.png).
(a) head in the pipe against distance from the exit: transducers that the tip has passed, and the gradient S* predicted
    from the depth read at small scale;  (b) head that loads the tip to the small-scale value against the width over
    which the pipe drains the bed, for three pipe lengths, with the head at which the real pipe stopped.
Reads tmp/paper1_large_scale_data.txt, tmp/paper1_large_scale_width.txt and, if present, tmp/paper1_large_scale_taper.txt
(written by large_scale_data.py and run_large_scale_width.py).   usage:  python make_figure_large_scale.py
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TMP, FIG = os.path.join(HERE, "..", "tmp"), os.path.join(HERE, "..", "manuscript", "figures")
S_STAR, H_MEAS = 0.126, (1.26, 1.30)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6, "mathtext.fontset": "cm"})

txt = open(os.path.join(TMP, "paper1_large_scale_data.txt")).read()
line = next(l for l in txt.splitlines() if "head above P1 at 46.5" in l)
pts = np.array([[float(a), float(b)] for a, b in re.findall(r"\(([\d.]+) m from the exit\) (-?[\d.]+)", line)])
rows = []
for l in open(os.path.join(TMP, "paper1_large_scale_width.txt")):
    v = l.split()
    if len(v) == 11 and re.match(r"^\d", v[0]):
        rows.append([float(v[0]), float(v[1]), float(v[8]), float(v[9])])       # length, width, K, head for K_c
rows = np.array(rows)
taper = []
f = os.path.join(TMP, "paper1_large_scale_taper.txt")
if os.path.exists(f):
    for l in open(f):
        m = re.match(r"\s*([\d.]+)\s+((?:[\d.]+ )+)\s*(tip|bed)\s+((?:[\d.]+ ?)+?)\s{2,}", l)
        if m:
            w = [float(x) for x in m.group(4).split()]
            taper.append([float(m.group(1)), float(m.group(2).split()[-1]), w[-2]])   # length, head, mid-pipe width

fig, ax = plt.subplots(1, 2, figsize=(5.1, 2.35))
s = np.array([0.0, 3.8])
ax[0].plot(s, S_STAR * s, "k-", lw=0.9)
ax[0].plot(pts[:, 0], pts[:, 1], "ko", ms=4, mfc="w")
ax[0].annotate(r"$S^*=0.126$", (2.2, 0.30), fontsize=8); ax[0].annotate("measured", (2.75, 0.19), fontsize=8)
ax[0].set_xlabel("distance from the exit (m)"); ax[0].set_ylabel("head in the pipe (m)"); ax[0].set_xlim(0, 3.8); ax[0].set_ylim(-0.02, 0.5)
ax[1].axhspan(*H_MEAS, color="0.82", lw=0)
for (l, ls, mk) in ((0.8, ":", "v"), (1.65, "-", "o"), (3.0, "--", "s")):
    r = rows[np.isclose(rows[:, 0], l)]
    ax[1].semilogx(r[:, 1], r[:, 3], "k", ls=ls, lw=0.9, marker=mk, ms=3, mfc="w", label=f"$l={l:g}$ m")
for l, h, w in taper:
    if any(np.isclose(l, x) for x in (0.8, 1.65, 3.0)):
        ax[1].plot(w, h, "k*", ms=7)
ax[1].set_xlabel("drained width (m)"); ax[1].set_ylabel(r"head for $K=K_c$ (m)"); ax[1].set_ylim(0.3, 1.4); ax[1].set_xlim(0.04, 1.5)
ax[1].legend(frameon=False, fontsize=7, loc="lower right"); ax[1].annotate("pipe stopped", (0.045, 1.315), fontsize=7)
for b, t in zip(ax, "ab"):
    b.spines[["top", "right"]].set_visible(False); b.text(-0.34, 1.02, f"({t})", transform=b.transAxes)
fig.tight_layout(w_pad=1.2)
os.makedirs(FIG, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig6_large_scale.{ext}"), dpi=300)
print("points (a):", pts.tolist(), "| rows (b):", len(rows), "| self-consistent:", taper)
