"""Depth of the pipe against distance from the tip, all states at rest (manuscript/figures/fig9_depth_profile.pdf/.png).
Theory: largest depth of the self-formed threshold pipe computed for each of the 29 small-scale states at rest, nothing fitted.
Measurement: regression of all laser scans by Pol (2022, eq. 3.1), a_max = 10 (kappa s)^(1/3).   usage: python make_figure_depth_profile.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipetheory.aquifer import PipeResponse, pol_box
from pipetheory.coupled import solve_pipe_law, fold_law
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section
from pipe_width_data import read_csv

HERE = os.path.dirname(os.path.abspath(__file__))
TMP, FIG = os.path.join(HERE, "..", "tmp"), os.path.join(HERE, "..", "manuscript", "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6, "mathtext.fontset": "cm"})
sands = {r["sand"]: r for r in read_csv("pol2022_sands.csv")}
kk = {r["test"]: float(r["k_m_per_s"]) for r in read_csv("pol2022_table3_2.csv")}
rows = [r for r in csv.DictReader(open(os.path.join(TMP, "paper1_pol_states.csv"))) if r["variant"] == "base" and r["test"] in kk]
style = {"FPH": ("-", "0.0"), "B25": ("--", "0.35"), "FS35": (":", "0.0")}
fig, ax = plt.subplots(figsize=(3.6, 2.8))
x = np.logspace(-13.2, -10.6, 50)
ax.loglog(x, 10 * x ** (1 / 3) * 1e3, color="0.75", lw=5, solid_capstyle="butt", label="laser scans (regression)")
ratio, seen = {}, set()
for r in rows:
    sd = sands[r["sand"]]; d50 = float(sd["d50_mm"]) * 1e-3; k = kk[r["test"]]; kappa = k * 1e-6 / 9.81
    sec = threshold_section(np.tan(np.radians(float(sd["repose_deg"]))), 16, 161, 41)
    law = PipeLaw(sec, float(sd["tau_eq_Pa"]) / (sec["a_max"] * sec["aavg_over_amax"]))
    R = PipeResponse(pol_box(), float(r["l_cm"]) / 100, rtol=1e-9)
    Hf, sf = fold_law(R, law, k)
    H = max(float(r["H_net_cm"]) / 100, 1.001 * Hf)
    sol = solve_pipe_law(R, law, H, k, sf["h"] * H / Hf) or sf
    back = R.x_tip - R.x_st; amax = law.geometry(sol["S"])[1]
    m = (back > 0.02) & (back < R.length - 0.01)
    ls, c = style[r["sand"]]
    ax.loglog(kappa * back[m], amax[m] * 1e3, ls=ls, color=c, lw=0.6, label=f"theory, $d_{{50}}={float(sd['d50_mm']):.3f}$ mm" if r["sand"] not in seen else None)
    seen.add(r["sand"]); ratio.setdefault(r["sand"], []).extend(amax[m] / (10 * (kappa * back[m]) ** (1 / 3)))
ax.set_xlabel(r"$\kappa s$ (m$^3$)"); ax.set_ylabel(r"largest depth $a_{\max}$ (mm)"); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig9_depth_profile.{ext}"), dpi=300)
allr = np.concatenate([np.array(v) for v in ratio.values()])
line = f"{len(rows)} states; theory / regression over 2 cm from the tip to 1 cm from the exit: all {allr.mean():.2f} +- {allr.std():.2f} (range {allr.min():.2f}-{allr.max():.2f}); " + "; ".join(f"{s} {np.mean(v):.2f} +- {np.std(v):.2f}" for s, v in ratio.items())
print(line); open(os.path.join(TMP, "paper1_depth_profile_all.txt"), "w").write(line + "\n")
