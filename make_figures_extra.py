"""Three further figures for the manuscript (manuscript/figures/):
  fig3b_lumped        the lumped model of the fold: y^3 (H - gamma y) against the level that the sand sets;
  fig8_scale_effect   H_c/L against L for the two sands tested at two scales, measured and computed, with the
                      slopes -1/2 (bed criterion, self-formed) and -1/3;
  graphical_abstract  the threshold section and the regime diagram side by side (PNG, 1600 x 800 px).
Reads data/pol2022_table3_2.csv, tmp/paper1_pol_box_predictions.csv and tmp/paper1_pipe_width_data.csv.  The heads of
the medium- and large-scale tests and the computed heads are those quoted in the text (sources in the comments).
usage:  python make_figures_extra.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section
from pipe_width_data import read_csv

HERE = os.path.dirname(os.path.abspath(__file__))
TMP, FIG = os.path.join(HERE, "..", "tmp"), os.path.join(HERE, "..", "manuscript", "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6, "mathtext.fontset": "cm"})


def save(fig, name, png_only=False, dpi=300):
    for ext in (("png",) if png_only else ("pdf", "png")):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=dpi)
    plt.close(fig)


# --- lumped model -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.9, 2.3))
eta = np.linspace(0, 1.25, 400)
for h, ls in ((0.9, ":"), (1.0, "-"), (1.15, "--")):
    f = 256 / 27 * eta ** 3 * (h - eta)
    ax.plot(eta[f >= 0], f[f >= 0], "k", ls=ls, lw=0.9, label=f"$H/H_{{\\min}}={h:g}$")
ax.axhline(1.0, color="0.55", lw=0.7)
r = np.roots([1, -1.15, 0, 0, 27 / 256]); r = np.sort(r[np.isreal(r) & (r.real > 0)].real)
ax.plot(r, [1, 1], "ko", ms=3.5, mfc="w"); ax.plot([0.75], [1], "ko", ms=3.5)
ax.set_xlabel(r"$\gamma y/H_{\min}$"); ax.set_ylabel(r"$y^3(H-\gamma y)\,/\,(\Lambda l^2/\mathcal{G}_0k)$")
ax.set_xlim(0, 1.2); ax.set_ylim(0, 2.0); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=2.2)
fig.tight_layout(); save(fig, "fig3b_lumped")

# --- scale effect -------------------------------------------------------------------------------------------------
L_S, L_M, L_L = 0.352, 1.385, 7.2
hc = {}
for r in read_csv("pol2022_table3_2.csv"):
    hc.setdefault(r["sand"], []).append(float(r["Hc_corr_cm"]) / 100)
pred = [float(r["H_bed_cm"]) / 100 for r in csv.DictReader(open(os.path.join(TMP, "paper1_pol_box_predictions.csv")))
        if r["sink"] == "base" and float(r["W0_in_d50"]) == 30 and float(r["tortuosity"]) == 1.0 and r["sand"] == "FPH"]
fig, ax = plt.subplots(figsize=(3.7, 2.9))
f0 = np.mean(hc["FPH"]) / L_S
x = np.array([L_S, 10.0])
ax.loglog(x, f0 * (x / L_S) ** -0.5, "k:", lw=0.8); ax.text(4.2, f0 * (4.2 / L_S) ** -0.5 * 1.12, r"$-1/2$", fontsize=7)
ax.loglog(x, f0 * (x / L_S) ** (-1 / 3), "k--", lw=0.7); ax.text(4.2, f0 * (4.2 / L_S) ** (-1 / 3) * 1.1, r"$-1/3$", fontsize=7)
for s, mk in (("B25", "^"), ("FS35", "v")):                              # sands tested at one scale only
    ax.errorbar(L_S * (0.93 if s == "B25" else 1.07), np.mean(hc[s]) / L_S, yerr=np.std(hc[s]) / L_S, fmt=mk, ms=3.5, mfc="0.75", mec="0.4", ecolor="0.6", elinewidth=0.6, lw=0)
# FPH sand: small scale (Pol 2022, table 3.2) and large scale (1.42 m reported; 1.26-1.30 m equivalent head for the computed geometry)
ax.errorbar(L_S, f0, yerr=np.std(hc["FPH"]) / L_S, fmt="ko", ms=4.5, elinewidth=0.7, label="0.185 mm sand, measured")
ax.plot([L_L, L_L], [1.26 / L_L, 1.42 / L_L], "k-", lw=0.8); ax.plot(L_L, 1.42 / L_L, "ko", ms=4.5)
ax.plot(L_S * 1.0, np.mean(pred) / L_S, "ko", ms=4.5, mfc="w", label="computed")
for hcomp in (0.98, 1.01, 1.22):                                          # strip 0.05 m; width from the pipe law; whole bed
    ax.plot(L_L * 0.9, hcomp / L_L, "ko", ms=4.5, mfc="w")
# Baskarp sand: small scale 0.080 m (L = 0.344 m, three dense tests), medium scale 0.21 m; computed 0.79-0.83 of it
ax.plot([0.344, L_M], [0.080 / 0.344, 0.21 / L_M], "ks", ms=4, ls="none", label="0.132 mm sand, measured")
ax.plot([L_M * 0.92] * 2, [0.79 * 0.21 / L_M, 0.83 * 0.21 / L_M], "ks", ms=4, mfc="w")
ax.set_xlabel(r"$L$ (m)"); ax.set_ylabel(r"$H_c/L$"); ax.set_xlim(0.25, 10); ax.set_ylim(0.03, 0.4)
ax.set_xticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"]); ax.set_yticks([0.05, 0.1, 0.2, 0.4], ["0.05", "0.1", "0.2", "0.4"]); ax.minorticks_off()
ax.legend(frameon=False, fontsize=7, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); save(fig, "fig8_scale_effect")
print(f"H_c/L small scale: FPH {f0:.3f} +- {np.std(hc['FPH']) / L_S:.3f} (n={len(hc['FPH'])}), computed {np.mean(pred) / L_S:.3f} (n={len(pred)});"
      f" B25 {np.mean(hc['B25']) / L_S:.3f}, FS35 {np.mean(hc['FS35']) / L_S:.3f}; Baskarp {0.080 / 0.344:.3f} -> {0.21 / L_M:.3f};"
      f" FPH large {1.42 / L_L:.3f} (equivalent {1.26 / L_L:.3f}-{1.30 / L_L:.3f}); -1/2 at 7.2 m {f0 * (L_L / L_S) ** -0.5:.3f}, -1/3 {f0 * (L_L / L_S) ** (-1 / 3):.3f}")

# --- graphical abstract -------------------------------------------------------------------------------------------
sec = threshold_section(0.77, 16, 161, 41)
y, a = np.asarray(sec["y"], float), np.asarray(sec["a"], float)
if y.min() >= 0:
    y, a = np.concatenate([-y[::-1], y]), np.concatenate([a[::-1], a])
fig, ax = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [1.45, 1]})
ax[0].fill_between(y, -a, -1.6, color="0.85", lw=0); ax[0].fill_between([y.min() * 1.18, y.max() * 1.18], 0, 0.12, color="0.4", lw=0)
ax[0].plot(y, -a, "k-", lw=1.2); ax[0].plot([y.min() * 1.18, y.min()], [0, 0], "k-", lw=1.2); ax[0].plot([y.max(), y.max() * 1.18], [0, 0], "k-", lw=1.2)
ax[0].fill_between([y.min() * 1.18, y.min()], 0, -1.6, color="0.85", lw=0); ax[0].fill_between([y.max(), y.max() * 1.18], 0, -1.6, color="0.85", lw=0)
ax[0].text(0, 0.22, "roof", ha="center", fontsize=10); ax[0].text(0, -1.35, "sand", ha="center", fontsize=10)
ax[0].text(0, -0.45, r"bed at the threshold", ha="center", fontsize=10); ax[0].text(0, 0.75, r"$QS^3=\Lambda$", ha="center", fontsize=16)
ax[0].set_ylim(-1.6, 1.1); ax[0].set_xlim(y.min() * 1.18, y.max() * 1.18); ax[0].set_aspect("equal"); ax[0].axis("off")
ref = PipeLaw(sec, 0.45, W0=30 * 0.2e-3)
q = np.logspace(-2.5, 4.3, 300)
ax[1].loglog(q, ref.slope(q * ref.Q_x)[0] / ref.S_x, "k-", lw=1.3); ax[1].loglog(q[q < 1], q[q < 1] ** (-1 / 3), "k:", lw=0.9)
for r in csv.DictReader(open(os.path.join(TMP, "paper1_pipe_width_data.csv"))):
    law = PipeLaw(sec, float(r["tau_c"]), W0=30 * float(r["d50"]))
    ax[1].loglog(float(r["Q"]) / law.Q_x, float(r["S"]) / law.S_x, "o" if r["scale"] == "small" else "s", ms=5, mfc="w" if r["scale"] == "small" else "0.35", mec="k", mew=0.7)
fph = PipeLaw(sec, 0.339 / (sec["a_max"] * sec["aavg_over_amax"]), W0=30 * 0.185e-3, nu=1.52e-6)
ax[1].loglog([18e-6 / fph.Q_x, 24e-6 / fph.Q_x], [0.111 / fph.S_x] * 2, "k-", lw=3, solid_capstyle="butt")
ax[1].text(3e2, 0.12, "self-formed pipe", fontsize=9, rotation=-27); ax[1].text(4.0, 2.3, "pipes at 0.35, 1.4 and 7 m", fontsize=9)
ax[1].set_xlabel("discharge", fontsize=11); ax[1].set_ylabel("gradient in the pipe", fontsize=11); ax[1].set_xlim(q[0], q[-1]); ax[1].set_ylim(0.08, 6)
ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].minorticks_off(); ax[1].spines[["top", "right"]].set_visible(False)
fig.tight_layout(); save(fig, "graphical_abstract", png_only=True, dpi=200)
