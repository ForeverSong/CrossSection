"""Bifurcation diagram of the threshold pipe in the box of Pol (2022) and the map of the controlling criterion.

1. Both solution branches (pseudo-arclength continuation through the fold) for one pipe length.
2. Head profiles on the two branches and at the fold.
3. Which criterion controls: the bed criterion (fold) or the tip criterion.  With the tip criterion written as a
   critical gradient i_c over a distance x0 ahead of the tip, a sand is described by
       Pi_s = i_c sqrt(x0) (rho g / tau_c) (nu k / g)^(1/4)
   and the box by  Pi_s*(l) = (16 chi N_min)^(1/4) gamma_fold sqrt(x0),  gamma_fold = tip gradient per unit head
   at the fold.  Pi_s < Pi_s*: the tip criterion is already met at the fold (bed-controlled);  Pi_s > Pi_s*:
   the head must exceed H_min (tip-controlled).

usage:  python run_bifurcation.py [--out ../tmp] [--length 0.18]
"""
import argparse
import csv
import os
import numpy as np
from pipetheory import RHO_W, G, NU
from pipetheory.aquifer import pol_box, PipeResponse
from pipetheory.coupled import trace_branches, tip_gradient, fold, sand_number
from pipetheory.section import threshold_section

HERE = os.path.dirname(os.path.abspath(__file__))
X0 = 0.02


def read_csv(name):
    with open(os.path.join(HERE, "data", name)) as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    ap.add_argument("--length", type=float, default=0.18)
    a = ap.parse_args()
    box = pol_box(); L = box.seepage_length
    R = PipeResponse(box, a.length, with_fields=True)
    Nmin, br = trace_branches(R, N_start_factor=600.0)
    x = np.array([(b["N"] / Nmin) ** 0.25 for b in br])           # H / H_min
    drop = np.array([b["h"][-1] for b in br])                     # head loss along the pipe / H
    gam = np.array([tip_gradient(R, b, X0) for b in br])          # tip gradient per unit H [1/m]
    stable = np.array([b["branch"] == 0 for b in br])
    i_f = int(np.argmin(x))
    print(f"pipe length {R.length * 100:.1f} cm (l/L = {R.length / L:.2f}); fold at N_min = {Nmin:.4f} m2")
    print(f"at the fold: head loss along the pipe = {drop[i_f]:.3f} H; tip gradient over {X0 * 100:.0f} cm = {gam[i_f]:.2f} H/m; pipe discharge = {br[i_f]['Q'][0] * 1e3:.1f} mm x k H")
    print("\n  H/H_min | large-pipe branch: pipe head loss/H, tip gradient x L/H | pinched branch: pipe head loss/H, tip gradient x L/H")
    for xv in (1.0, 1.05, 1.2, 1.5, 2.0, 3.0, 4.0):
        row = []
        for m in (stable, ~stable):
            xs, ds_, gs = x[m], drop[m], gam[m]
            o = np.argsort(xs)
            row.append((np.interp(xv, xs[o], ds_[o]), np.interp(xv, xs[o], gs[o]) * L))
        print(f"  {xv:7.2f} | {row[0][0]:10.3f} {row[0][1]:10.2f}                         | {row[1][0]:10.3f} {row[1][1]:10.2f}")

    # regime map: fold and tip gradient at the fold for a range of pipe lengths
    sands = {r["sand"]: r for r in read_csv("pol2022_sands.csv")}
    sec = {s: threshold_section(np.tan(np.radians(float(r["repose_deg"]))), 16, 161, 41) for s, r in sands.items()}
    chi = sec["B25"]["chi"]
    ls, pis = [], []
    for l in np.arange(0.04, 0.321, 0.02):
        Rl = PipeResponse(box, l, with_fields=True)
        N, s = fold(Rl)
        ls.append(Rl.length); pis.append((16.0 * chi * N) ** 0.25 * tip_gradient(Rl, s, X0) * np.sqrt(X0))
    ls, pis = np.array(ls), np.array(pis)
    print("\nRegime boundary Pi_s*(l) for the box (chi of B25):")
    print("   " + "  ".join(f"l={l * 100:.0f}cm:{p:.2f}" for l, p in zip(ls, pis)))
    tests = [t for t in read_csv("pol2022_table3_2.csv") if t["i_tip_c"]]
    pts = []
    for t in tests:
        o = sec[t["sand"]]
        tau = float(sands[t["sand"]]["tau_eq_Pa"]) / (o["a_max"] * o["aavg_over_amax"])
        C = float(t["i_tip_c"]) * np.sqrt(X0)
        pts.append((t["test"], t["sand"], float(t["l_c_cm"]) * 1e-2, sand_number(C, tau, float(t["k_m_per_s"]), RHO_W, G, NU)))
    print("Sand number Pi_s of the tests with a measured critical tip gradient:")
    for sn in ("FPH", "B25", "FS35"):
        v = [p[3] for p in pts if p[1] == sn]
        print(f"   {sn:5s} {min(v):.2f} - {max(v):.2f}  (n = {len(v)})")
    print(f"   all tests lie above the boundary (max Pi_s* = {pis.max():.2f}): with the self-formed section the tip criterion controls.")
    print("   (With the pipe width fixed by the tip the bed criterion alone gives about 0.9 of the measured critical head and the")
    print("    two criteria are within about 10 % of each other - see run_pol_box.py.)")

    try:
        import plotstyle
        plt = plotstyle.apply()
    except Exception as e:
        print("no figure:", e); return
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for m, ls_, lab in ((stable, "-", "large pipe (stable)"), (~stable, "--", "pinched pipe (unstable)")):
        o = np.argsort(x[m]) if m is stable else slice(None)
        ax1.plot(x[m][o], drop[m][o], ls_, color=plotstyle.BLUE if m is stable else plotstyle.INK_2, label=lab)
    ax1.plot([1.0], [drop[i_f]], "o", color=plotstyle.ORANGE, ms=6)
    ax1.annotate("fold: H = H_min", (1.0, drop[i_f]), xytext=(1.5, drop[i_f] + 0.13), color=plotstyle.INK_2, fontsize=8,
                 arrowprops=dict(arrowstyle="-", color=plotstyle.INK_2, lw=0.6))
    ax1.axvspan(0.0, 1.0, color=plotstyle.GRID, alpha=0.6, lw=0)
    ax1.text(0.5, 0.05, "bed cannot be\nheld at the\nthreshold", ha="center", va="bottom", fontsize=8, color=plotstyle.INK_2)
    ax1.set_xlim(0, 4); ax1.set_ylim(0, 0.8)
    ax1.set_xlabel("head H / H_min"); ax1.set_ylabel("head loss along the pipe / H")
    ax1.set_title(f"Two threshold pipes above H_min (l/L = {R.length / L:.2f})"); ax1.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.42))

    xr = np.linspace(box.x_exit, box.Lx, 400)
    pick = [("fold", i_f, "-", plotstyle.ORANGE)]
    for m, ls_, lab in ((stable, "-", "large pipe, H = 1.5 H_min"), (~stable, "--", "pinched pipe, H = 1.5 H_min")):
        idx = np.where(m)[0]
        pick.append((lab, idx[np.argmin(np.abs(x[idx] - 1.5))], ls_, plotstyle.BLUE if m is stable else plotstyle.INK_2))
    for lab, i, ls_, col in pick:
        xc, roof = R.roof_head(br[i]["h"], 1.0)
        ax2.plot((xc - R.x_exit) / L, roof, ls_, color=col, label=lab if lab != "fold" else "at the fold")
    ax2.axvline(R.length / L, color=plotstyle.INK_2, lw=0.8, ls=":")
    ax2.text(R.length / L + 0.015, 0.03, "tip", color=plotstyle.INK_2, fontsize=8)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_xlabel("distance from the exit / L"); ax2.set_ylabel("head under the roof / H")
    ax2.set_title("Head along the pipe axis"); ax2.legend(frameon=False, fontsize=8, loc="lower right")

    ax3.plot(ls / L, pis, color=plotstyle.BLUE)
    ax3.fill_between(ls / L, 0, pis, color=plotstyle.BLUE, alpha=0.10, lw=0)
    mk = {"FPH": "s", "B25": "o", "FS35": "^"}
    for sn in mk:
        pp = [p for p in pts if p[1] == sn]
        ax3.plot([p[2] / L for p in pp], [p[3] for p in pp], mk[sn], color=plotstyle.ORANGE, ms=6, label=f"{sn} tests")
    ax3.text(0.5, 0.45, "bed criterion controls", ha="center", color=plotstyle.INK_2, fontsize=8)
    ax3.text(0.5, 6.3, "tip criterion controls", ha="center", color=plotstyle.INK_2, fontsize=8)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 7)
    ax3.set_xlabel("pipe length l / L"); ax3.set_ylabel("sand number Pi_s")
    ax3.set_title("Which criterion sets the critical head"); ax3.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    out = os.path.join(a.out, "paper1_bifurcation.png"); fig.savefig(out); print("\nfigure:", out)


if __name__ == "__main__":
    main()
