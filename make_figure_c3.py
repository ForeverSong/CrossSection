"""Coefficient c3 of the scale law for two exit geometries (manuscript/figures/fig4b_c3.pdf/.png, tmp/paper1_c3_geometries.txt):
a hole in the roof (box of Pol 2022) and an exit along the downstream toe (model levee of Okamura & Kusube 2025).
c3 = (16 chi N_min)^(1/4) / L^(1/2), with N_min the fold of the coupled problem for a pipe of length l.
usage:  python make_figure_c3.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipetheory.aquifer import PipeResponse, pol_box, okamura_box
from pipetheory.coupled import fold, c3_from_N
from pipetheory.section import threshold_section

HERE = os.path.dirname(os.path.abspath(__file__))
TMP, FIG = os.path.join(HERE, "..", "tmp"), os.path.join(HERE, "..", "manuscript", "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6, "mathtext.fontset": "cm"})
chi = threshold_section(np.tan(np.radians(37.6)), 16, 161, 41)["chi"]
out = open(os.path.join(TMP, "paper1_c3_geometries.txt"), "w")
fig, ax = plt.subplots(figsize=(3.3, 2.4))
for name, box, ls in (("hole in the roof", pol_box(), "-"), ("exit at the toe", okamura_box(), "--")):
    L = box.seepage_length
    res = []
    for frac in np.linspace(0.08, 0.92, 15):
        try:
            R = PipeResponse(box, frac * L, rtol=1e-9)
            res.append((R.length / L, c3_from_N(fold(R)[0], chi, L)))
        except Exception as e:                                             # pipe too short or reaching the upstream boundary
            continue
    res = np.array(res)
    ax.plot(res[:, 0], res[:, 1], "k", ls=ls, lw=0.9, label=name)
    i = int(np.argmax(res[:, 1]))
    line = f"{name}: L = {L:.3f} m, c3 max {res[i, 1]:.2f} at l/L = {res[i, 0]:.2f}; " + " ".join(f"{a:.2f}:{b:.2f}" for a, b in res)
    print(line); out.write(line + "\n")
out.close()
ax.set_xlabel("$l/L$"); ax.set_ylabel("$c_3$"); ax.set_xlim(0, 1); ax.set_ylim(0, None); ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig4b_c3.{ext}"), dpi=300)
