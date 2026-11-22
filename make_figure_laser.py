"""Manuscript figure: sections of pipes at rest from the raw laser scans (manuscript/figures/fig10_laser_sections).

(a) shape of the deepest channel, scaled with its depth, against the threshold section;
(b) largest bed shear stress in each section (laminar flow in the measured section under the measured gradient);
(c) depth along pipes that were at rest during their growth, against the threshold pipe computed for the same state;
(d) largest depth against the discharge of the pipe at the section, with the pipe law (no fit).
Reads the outputs of laser_sections.py and laser_vs_theory.py.      usage: python make_figure_laser.py ../Data/Pol_4TU
"""
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipetheory import laserdata as ld
from pipetheory import poldata as pd
from pipetheory.aquifer import PipeResponse, pol_box
from pipetheory.coupled import solve_pipe_law, fold_law
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section
from laser_vs_theory import scan_state, pipe_head, read_csv, viscosity, water_temperature, D50

HERE = os.path.dirname(os.path.abspath(__file__))
TMP, FIG = os.path.join(HERE, "..", "tmp"), os.path.join(HERE, "..", "manuscript", "figures")
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "Data", "Pol_4TU")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "mathtext.fontset": "cm",
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "legend.fontsize": 6.5})
fnum = lambda v: float(v) if v not in ("", "nan", "None") else np.nan
out = []

secs = list(csv.DictReader(open(os.path.join(TMP, "paper1_laser_sections.csv"))))
rows = list(csv.DictReader(open(os.path.join(TMP, "paper1_laser_vs_theory.csv"))))
prof = np.load(os.path.join(TMP, "paper1_laser_profiles.npz"))
sands = {r["sand"]: r for r in read_csv("pol2022_sands.csv")}
table = {r["test"]: r for r in read_csv("pol2022_table3_2.csv")}
sec_th = {s: threshold_section(np.tan(np.radians(float(sands[s]["repose_deg"])))) for s in sands}

fig, axs = plt.subplots(2, 2, figsize=(5.1, 4.9))
DUMP = {"a": {}, "b": [], "c": [], "d": {}}                 # what is plotted, for make_figures_jfm.py

# ---------------------------------------------------------------- (a) shape of the deepest channel
ax = axs[0, 0]
xi = np.arange(0.0, 5.01, 0.125)
half = {"complete": [], "partial": []}
for s in secs:
    if s["good50"] != "True" or s["kind"] not in half:
        continue
    yc, dep = prof[f"{s['test']}|{s['file']}|{fnum(s['tape_cm'])}"][:, :2].T
    amax, ymax = fnum(s["amax"]), fnum(s["ymax"])
    for sign in (-1.0, 1.0):
        u = sign * (yc - ymax) / amax
        m = (u >= -0.01) & np.isfinite(dep)
        if m.sum() < 5:
            continue
        o = np.argsort(u[m])
        v = np.interp(xi, u[m][o], dep[m][o] / amax, right=np.nan)
        # stop at the first gap in the readings (points more than 3 mm apart)
        gap = np.where(np.diff(u[m][o]) * amax > 3.01)[0]
        if gap.size:
            v[xi > u[m][o][gap[0]]] = np.nan
        half[s["kind"]].append(v)
th = sec_th["B25"]
ax.plot(th["y"] / th["a_max"], th["a"] / th["a_max"], "k-", lw=1.4, label="threshold section", zorder=5)
for kind, col, ls, lab in (("complete", "0.55", "-", "pipes through the bed"), ("partial", "0.0", "--", "growing pipes")):
    H = np.array(half[kind])
    n = np.sum(np.isfinite(H), axis=0)
    ok = n >= max(8, 0.4 * len(H))
    med = np.nanmedian(H, axis=0)
    q1, q3 = np.nanpercentile(H, [25, 75], axis=0)
    DUMP["a"][kind] = dict(x=xi[ok].tolist(), med=med[ok].tolist(), q1=q1[ok].tolist(), q3=q3[ok].tolist(), n=len(H) // 2)
    if kind == "complete":
        ax.fill_between(xi[ok], q1[ok], q3[ok], color="0.85", lw=0)
    ax.plot(xi[ok], med[ok], ls=ls, color=col, lw=0.9, label=f"{lab} ({len(H) // 2})")
    out.append(f"(a) {kind}: {len(H)} half-profiles of {len(H) // 2} sections; median depth/a_max at |y|/a_max = 1, 2, 3, 4: "
               + ", ".join(f"{np.interp(v, xi, med):.2f}" for v in (1, 2, 3, 4)) + "; threshold section: "
               + ", ".join(f"{np.interp(v, th['y'] / th['a_max'], th['a'] / th['a_max'], right=0):.2f}" for v in (1, 2, 3, 4)))
DUMP["a"]["theory"] = dict(x=(th["y"] / th["a_max"]).tolist(), a=(th["a"] / th["a_max"]).tolist())
ax.set_xlim(0, 5); ax.set_ylim(1.05, 0)
ax.set_xlabel(r"distance from the deepest point / $a_{\max}$"); ax.set_ylabel(r"depth / $a_{\max}$")
ax.legend(frameon=False, loc="lower right", handlelength=1.8, borderaxespad=0.2)
ax.text(0.03, 0.92, "(a)", transform=ax.transAxes)

# ---------------------------------------------------------------- (b) bed shear stress at rest
ax = axs[0, 1]
groups = [("FPH", "complete"), ("B25", "complete"), ("B25", "partial"), ("FS35", "complete")]
rng = np.random.default_rng(1)
for i, (sand, kind) in enumerate(groups):
    v = np.array([fnum(r["tau_max"]) for r in rows if r["sand"] == sand and r["kind"] == kind])
    v = v[np.isfinite(v) & (v > 0)]
    v = v[v < 3.0]                                            # two sections with a spurious local gradient
    ax.plot(i + 0.22 * (rng.random(v.size) - 0.5), v, "o" if kind == "complete" else "s", ms=2.6,
            mfc="0.75" if kind == "complete" else "w", mec="0.25", mew=0.4)
    med, q1, q3 = np.median(v), *np.percentile(v, [25, 75])
    ax.plot([i - 0.3, i + 0.3], [med, med], "k-", lw=1.4)
    th = sec_th[sand]
    tc = float(sands[sand]["tau_eq_Pa"]) / (th["a_max"] * th["aavg_over_amax"])
    ax.plot([i - 0.38, i + 0.38], [tc, tc], "k:", lw=0.9)
    DUMP["b"].append(dict(sand=sand, kind=kind, v=v.tolist(), med=float(med), tau_c=float(tc)))
    theta = med / ((float({"FPH": 2610, "B25": 2650, "FS35": 2650}[sand]) - 1000.0) * 9.81 * D50[sand])
    out.append(f"(b) {sand} {kind}: n = {v.size}, largest bed shear stress median {med:.2f} Pa (quartiles {q1:.2f}-{q3:.2f}), "
               f"Shields number {theta:.3f}; value used so far (Pol's equilibrium stress / {th['a_max'] * th['aavg_over_amax']:.2f}) {tc:.2f} Pa")
ax.set_xticks(range(4))
ax.set_xticklabels(["0.185", "0.228", "0.228*", "0.422"])
ax.set_xlabel(r"$d_{50}$ (mm);  * growing pipes"); ax.set_ylabel(r"largest bed shear stress (Pa)")
ax.set_ylim(0, 1.45); ax.set_xlim(-0.6, 3.6)
ax.text(0.03, 0.92, "(b)", transform=ax.transAxes)

# ---------------------------------------------------------------- (c) depth along pipes at rest during growth
ax = axs[1, 0]
marks = {"B25_230": "o", "B25_236": "s", "B25_245": "^"}
k_mean = np.mean([float(r["k_m_per_s"]) for r in table.values() if r["sand"] == "B25"])
ratio = []
seen = set()
for sc in ld.list_scans(DATA):
    if sc["kind"] != "partial" or sc["test"] not in marks:
        continue
    test, sand = sc["test"], "B25"
    st = scan_state(DATA, sc)
    x_tip = pd.X_EXIT + 0.01 * (st["L_tape"] + 1.0)
    ph = pipe_head(st, x_tip)
    mine = [r for r in rows if r["test"] == test and r["file"] == sc["file"]]
    if not mine:
        continue
    s_m = np.array([fnum(r["s_tip"]) for r in mine]); a_m = np.array([fnum(r["amax"]) for r in mine])
    k = float(table[test]["k_m_per_s"]) if test in table else k_mean
    nu = viscosity(water_temperature(DATA, test))
    th = sec_th[sand]
    law = PipeLaw(th, float(sands[sand]["tau_eq_Pa"]) / (th["a_max"] * th["aavg_over_amax"]), nu=nu)
    R = PipeResponse(pol_box(), x_tip - pd.X_EXIT, rtol=1e-9)
    h = st["h"]
    H = h[15] + (h[15] - h[14]) / 0.02 * (pd.X_FILTER - pd.X_PORT[15]) - ph["h_exit"]
    Hf, sf = fold_law(R, law, k)
    sol = solve_pipe_law(R, law, max(H, 1.001 * Hf), k, sf["h"] * max(H, 1.001 * Hf) / Hf) or sf
    back = R.x_tip - R.x_st
    a_th = law.geometry(sol["S"])[1]
    m = back > 0.015
    ax.plot(100 * back[m], 1e3 * a_th[m], "-", color="0.6", lw=0.6, label="threshold pipe" if not seen else None)
    ax.plot(100 * s_m, 1e3 * a_m, marks[test], ms=3.5, mfc="w" if test not in table else "0.3", mec="k", mew=0.5,
            label=f"test {test[4:]}" if test not in seen else None)
    seen.add(test)
    DUMP["c"].append(dict(test=test, known_k=test in table, back_cm=(100 * back[m]).tolist(), a_th_mm=(1e3 * a_th[m]).tolist(),
                          s_cm=(100 * s_m).tolist(), a_mm=(1e3 * a_m).tolist()))
    r_ = a_m / np.interp(s_m, back[::-1], a_th[::-1])
    ratio.extend(r_)
    out.append(f"(c) {test} l = {100 * (x_tip - pd.X_EXIT):.1f} cm, H = {100 * H:.2f} cm (fold at {100 * Hf:.2f}), k = {k:.2e}{'' if test in table else ' (mean of the sand)'}: "
               f"measured/computed depth " + ", ".join(f"{v:.2f}" for v in r_))
ratio = np.array(ratio)
kap = k_mean * 1.0e-6 / 9.81
ss = np.linspace(0.015, 0.25, 50)
ax.plot(100 * ss, 1e3 * 10 * (kap * ss) ** (1 / 3), "k--", lw=0.8, label="regression, through pipes")
DUMP["c_regression"] = dict(s_cm=(100 * ss).tolist(), a_mm=(1e3 * 10 * (kap * ss) ** (1 / 3)).tolist())
out.append(f"(c) all {ratio.size} sections of pipes at rest during growth: measured / computed largest depth = {ratio.mean():.2f} +- {ratio.std():.2f} "
           f"(median {np.median(ratio):.2f}, range {ratio.min():.2f}-{ratio.max():.2f})")
ax.set_xlabel("distance from the tip (cm)"); ax.set_ylabel(r"largest depth $a_{\max}$ (mm)")
ax.set_xlim(0, 25); ax.set_ylim(0, 2.6)
ax.legend(frameon=False, loc="lower right", handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)
ax.text(0.03, 0.92, "(c)", transform=ax.transAxes)

# ---------------------------------------------------------------- (d) pipe law: largest depth against discharge
ax = axs[1, 1]
G_ = 9.81
qq = np.logspace(np.log10(6.0), np.log10(500.0), 50) / 6e7
allq, alla = [], []
for sand, mk in (("FPH", "s"), ("B25", "o"), ("FS35", "^")):
    th = sec_th[sand]
    tc = float(sands[sand]["tau_eq_Pa"]) / (th["a_max"] * th["aavg_over_amax"])
    A = 2.0 * tc / (1000.0 * G_)
    ratio_all = []
    for kind in ("complete", "partial"):
        v = np.array([[fnum(r["Q_pipe_x"]), fnum(r["amax"]), fnum(r["nu"]), fnum(r["Q_stokes"]), fnum(r["S"])] for r in rows
                      if r["sand"] == sand and r["kind"] == kind and 0 < fnum(r["tau_max"]) < 3.0])
        if not len(v):
            continue
        v = v[np.isfinite(v[:, :3]).all(axis=1) & (v[:, 0] > 0)]
        if not len(v):
            continue
        ax.loglog(6e7 * v[:, 0], 1e3 * v[:, 1], mk, ms=3, mfc="0.75" if kind == "complete" else "w", mec="0.2", mew=0.4,
                  label=f"$d_{{50}}={1e3 * D50[sand]:.3f}$ mm" if kind == "complete" else None)
        DUMP["d"].setdefault(sand, {})[kind] = dict(Q=(6e7 * v[:, 0]).tolist(), a=(1e3 * v[:, 1]).tolist())
        pred = th["a_max"] * (v[:, 2] * v[:, 0] / (th["chi"] * G_ * A)) ** (1.0 / 3.0)
        Lam = th["chi"] * G_ * A ** 4 / v[:, 2]
        ratio_all.extend(v[:, 1] / pred)
        allq.extend(v[:, 0]); alla.extend(v[:, 1])
        out.append(f"(d) {sand} {kind}: n = {len(v)}; largest depth / alpha (nu Q / chi g A)^(1/3): median {np.median(v[:, 1] / pred):.2f} "
                   f"(quartiles {np.percentile(v[:, 1] / pred, 25):.2f}-{np.percentile(v[:, 1] / pred, 75):.2f}); Q S^3 / Lambda median {np.median(v[:, 0] * v[:, 4] ** 3 / Lam):.2f}; "
                   f"flow in the measured section / discharge of the pipe median {np.nanmedian(v[:, 3] / v[:, 0]):.2f}")
    nu_s = 1.0e-6
    DUMP["d"].setdefault(sand, {})["law"] = dict(Q=(6e7 * qq).tolist(), a=(1e3 * th["a_max"] * (nu_s * qq / (th["chi"] * G_ * A)) ** (1.0 / 3.0)).tolist())
    ax.loglog(6e7 * qq, 1e3 * th["a_max"] * (nu_s * qq / (th["chi"] * G_ * A)) ** (1.0 / 3.0), "-", color="0.45", lw=0.7,
              label=r"pipe law, $\alpha(\nu Q/\chi gA)^{1/3}$" if sand == "FS35" else None)
allq, alla = np.array(allq), np.array(alla)
pf, cov = np.polyfit(np.log(allq), np.log(alla), 1, cov=True)
out.append(f"(d) all sands: n = {allq.size}; fit a_max ~ Q^n over Q = {6e7 * allq.min():.0f}-{6e7 * allq.max():.0f} mL/min: n = {pf[0]:.3f} +- {np.sqrt(cov[0, 0]):.3f} (theory 1/3)")
DUMP["d_fit"] = dict(n=float(pf[0]), sd=float(np.sqrt(cov[0, 0])), count=int(allq.size))
ax.set_xlim(6, 500); ax.set_ylim(0.5, 4.0)
ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5", "1", "2", "4"]); ax.minorticks_off()
ax.set_xlabel(r"discharge at the section (mL min$^{-1}$)"); ax.set_ylabel(r"largest depth $a_{\max}$ (mm)")
ax.legend(frameon=False, loc="lower right", handletextpad=0.3, borderaxespad=0.2)
ax.text(0.03, 0.92, "(d)", transform=ax.transAxes)

for ax in axs.ravel():
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(w_pad=1.6, h_pad=1.2)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig10_laser_sections.{ext}"), dpi=300)
import json
json.dump(DUMP, open(os.path.join(TMP, "paper1_laser_figdata.json"), "w"))
txt = "\n".join(out)
print(txt)
open(os.path.join(TMP, "paper1_laser_figure.txt"), "w").write(txt + "\n")
