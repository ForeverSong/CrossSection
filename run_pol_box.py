"""Critical head of the small-scale hole-exit tests of Pol (2022) from the threshold-pipe theory.

Model: 3-D Darcy flow in the test box + a pipe whose bed is at the threshold of motion.  Two conditions for
the pipe to progress at length l:
    bed:  H >= H_min(l)   (fold: below it the pipe cannot keep its bed at the threshold)
    tip:  gradient over the 2 cm ahead of the tip >= the critical value measured in the same test
The critical head is the maximum over l of the head that satisfies both.
Inputs per test (none fitted): conductivity k, the equilibrium bed stress measured for the sand, its repose
angle (-> mu) and the measured critical tip gradient.

Variants (printed side by side)
    sink   base: pipe footprint 10 mm, 6 mm hole;   wide: eroded width assumed in the thesis (25 mm, 30 mm for FS35);
           lens: wide + an eroded lens of 15 mm radius around the hole (the radius is an assumption)
    law    self-formed threshold section everywhere;   or width fixed by the tip (W0 = 30 d50) where the
           threshold section would be narrower
    T      tortuosity of the pipe path (1 = straight)

usage:  python run_pol_box.py [--out ../tmp] [--cell 5] [--tau-as-flat] [--quick]
"""
import argparse
import csv
import os
import numpy as np
from pipetheory.aquifer import pol_box, PipeResponse
from pipetheory.coupled import fold, c3_from_N, fold_law, head_for_tip_gradient
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section

HERE = os.path.dirname(os.path.abspath(__file__))


def read_csv(name):
    with open(os.path.join(HERE, "data", name)) as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def sink_kw(sink, sand):
    if sink == "base":
        return {}
    kw = dict(pipe_width=0.030 if sand == "FS35" else 0.025)
    if sink == "lens":
        kw["lens_radius"] = 0.015
    return kw


def responses(cell, lengths, kw):
    return [PipeResponse(pol_box(d=cell, **kw), l, with_fields=True) for l in lengths]


def critical(Rs, law, k, ic):
    """Fold head and head satisfying both criteria for each length [m]."""
    out = []
    for R in Rs:
        Hf, sf = fold_law(R, law, k)
        Hb = head_for_tip_gradient(R, law, k, ic, H_fold=Hf, sol_fold=sf)[0] if ic else np.nan
        out.append((Hf, Hb))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    ap.add_argument("--cell", type=float, default=5.0, help="cell size [mm]")
    ap.add_argument("--tau-as-flat", action="store_true",
                    help="use the measured equilibrium stress directly as the flat-bed threshold (default: read it as the section average)")
    ap.add_argument("--quick", action="store_true", help="coarser scan over pipe length")
    a = ap.parse_args()
    cell = a.cell * 1e-3
    lengths = np.arange(0.04, 0.341, 0.04 if a.quick else 0.02)
    L = pol_box().seepage_length
    sands = {r["sand"]: r for r in read_csv("pol2022_sands.csv")}
    tests = read_csv("pol2022_table3_2.csv")
    sec = {}
    for s, r in sands.items():
        o = threshold_section(np.tan(np.radians(float(r["repose_deg"]))), 16, 161, 41)
        factor = o["a_max"] * o["aavg_over_amax"]           # section-averaged bed stress / flat-bed threshold
        tau = float(r["tau_eq_Pa"]) / (1.0 if a.tau_as_flat else factor)
        sec[s] = dict(o=o, factor=factor, tau_c=tau, d50=float(r["d50_mm"]) * 1e-3)
        print(f"   {s:5s} mu = {o['mu']:.2f}  chi = {o['chi']:.4f}  section-average/flat-bed stress = {factor:.3f}"
              f"  -> flat-bed tau_c = {tau:.3f} Pa, Lambda = {PipeLaw(o, tau).Lam:.3e} m3/s")

    print("\n1. Bed criterion alone, self-formed section: fold curve of the box (geometry only; sink 'base')")
    print("   l [cm]  l/L    N_min [m2]   c3 = Hmin/L / [(tau_c/rho g)(g/(nu k))^(1/4) L^(-1/2)]   (chi of B25)")
    R_base = responses(cell, lengths, {})
    chi = sec["B25"]["o"]["chi"]
    Nmin = np.array([fold(R)[0] for R in R_base])
    ls = np.array([R.length for R in R_base])
    c3 = c3_from_N(Nmin, chi, L)
    for l, N, c in zip(ls, Nmin, c3):
        print(f"   {l * 100:5.0f}  {l / L:5.2f}  {N:10.4f}   {c:6.2f}")
    print(f"   maximum c3 = {c3.max():.2f} at l/L = {ls[c3.argmax()] / L:.2f}")

    print("\n2. Predicted / measured critical head, mean +/- s.d. over tests  (bed only: 19 tests; bed + tip: the 11 tests with a measured tip gradient)")
    print(f"   {'sink':5s} {'pipe law':22s} {'T':>4s} | {'bed only':>14s} | {'bed + tip':>14s} | {'l_c predicted [cm]':>18s}")
    Rcache = {("base", "B25"): R_base, ("base", "FPH"): R_base, ("base", "FS35"): R_base}
    for sink in ("wide", "lens"):
        for sand in ("B25", "FS35"):
            Rcache[(sink, sand)] = responses(cell, lengths, sink_kw(sink, sand))
        Rcache[(sink, "FPH")] = Rcache[(sink, "B25")]
    results = {}
    cases = [("base", 0, 1.0), ("wide", 0, 1.0), ("lens", 0, 1.0), ("base", 30, 1.0), ("wide", 30, 1.0), ("lens", 30, 1.0),
             ("wide", 0, 1.2), ("wide", 30, 1.2)]
    for sink, w0, T in cases:
        rows = []
        for t in tests:
            s = sec[t["sand"]]; k = float(t["k_m_per_s"]); Hc = float(t["Hc_corr_cm"]); lc = float(t["l_c_cm"])
            ic = float(t["i_tip_c"]) if t["i_tip_c"] else None
            law = PipeLaw(s["o"], s["tau_c"], W0=w0 * s["d50"] if w0 else None, tortuosity=T)
            hh = critical(Rcache[(sink, t["sand"])], law, k, ic) * 100
            j = np.nanargmax(hh[:, 1]) if ic else None
            rows.append(dict(test=t["test"], sand=t["sand"], k=k, Hc=Hc, lc=lc, ic=ic if ic else np.nan, H_bed=hh[:, 0].max(),
                             l_bed=ls[hh[:, 0].argmax()] * 100, H_both=hh[j, 1] if ic else np.nan, l_both=ls[j] * 100 if ic else np.nan,
                             curve=hh))
        r1 = np.array([r["H_bed"] / r["Hc"] for r in rows])
        r2 = np.array([r["H_both"] / r["Hc"] for r in rows if np.isfinite(r["H_both"])])
        lp = np.array([r["l_both"] for r in rows if np.isfinite(r["H_both"])])
        results[(sink, w0, T)] = rows
        print(f"   {sink:5s} {('width from the tip, %g d50' % w0) if w0 else 'self-formed':22s} {T:4.1f} | {r1.mean():6.2f} +/- {r1.std(ddof=1):.2f} | {r2.mean():6.2f} +/- {r2.std(ddof=1):.2f} | {lp.mean():9.1f} +/- {lp.std(ddof=1):.1f}")
    lm = np.array([float(t["l_c_cm"]) for t in tests if t["i_tip_c"]])
    print(f"   measured critical pipe length (same 11 tests): {lm.mean():.1f} +/- {lm.std(ddof=1):.1f} cm")

    for key, title in ((("base", 0, 1.0), "sink 'base', self-formed section, straight pipe"), (("wide", 30, 1.0), "sink 'wide', width from the tip (30 d50), straight pipe")):
        print(f"\n3. Test by test, {title} (heads in cm)")
        print(f"   {'test':9s} {'k':>8s} {'Hc meas':>8s} {'l_c':>5s} {'i_tip,c':>7s} | {'bed only':>9s} {'ratio':>6s} | {'bed+tip':>9s} {'at l':>5s} {'ratio':>6s}")
        for r in results[key]:
            both = f"{r['H_both']:9.2f} {r['l_both']:5.0f} {r['H_both'] / r['Hc']:6.2f}" if np.isfinite(r["H_both"]) else f"{'-':>9s} {'-':>5s} {'-':>6s}"
            print(f"   {r['test']:9s} {r['k']:8.1e} {r['Hc']:8.1f} {r['lc']:5.1f} {r['ic']:7.2f} | {r['H_bed']:9.2f} {r['H_bed'] / r['Hc']:6.2f} | {both}")
    with open(os.path.join(a.out, "paper1_pol_box_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sink", "W0_in_d50", "tortuosity", "test", "sand", "k", "Hc_meas_cm", "lc_meas_cm", "i_tip_c", "H_bed_cm", "H_bed_tip_cm", "lc_pred_cm"])
        for (sink, w0, T), rows in results.items():
            for r in rows:
                w.writerow([sink, w0, T, r["test"], r["sand"], r["k"], r["Hc"], r["lc"], r["ic"], round(r["H_bed"], 3), round(r["H_both"], 3), r["l_both"]])
    try:
        import plotstyle
        plt = plotstyle.apply()
    except Exception as e:
        print("no figure:", e); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.9))
    ref = next(r for r in results[("base", 0, 1.0)] if r["test"] == "B25_232")
    alt = next(r for r in results[("wide", 30, 1.0)] if r["test"] == "B25_232")
    ax1.plot(ls * 100, ref["curve"][:, 0], color=plotstyle.BLUE, label="bed criterion, self-formed section")
    ax1.plot(ls * 100, alt["curve"][:, 0], color=plotstyle.BLUE, ls="--", label="bed criterion, width from the tip (30 d50)")
    ax1.plot(ls * 100, alt["curve"][:, 1], color=plotstyle.INK_2, ls=":", label="bed + tip criterion (i_tip,c = 0.64)")
    b25 = [r for r in results[("base", 0, 1.0)] if r["sand"] == "B25"]
    ax1.plot([r["lc"] for r in b25], [r["Hc"] for r in b25], "o", color=plotstyle.ORANGE, ms=5.5, label="measured (l_c, H_c), B25 tests")
    ax1.set_xlim(0, 35); ax1.set_ylim(0, 9)
    ax1.set_xlabel("pipe length l [cm]"); ax1.set_ylabel("head across the sample [cm]")
    ax1.set_title("Head needed for the pipe to progress (B25, k = 2.7e-4 m/s)"); ax1.legend(frameon=False, fontsize=7.5, loc="lower right")
    mk = {"FPH": "s", "B25": "o", "FS35": "^"}
    for sn in mk:
        rr = [r for r in results[("base", 0, 1.0)] if r["sand"] == sn]
        ax2.plot([r["Hc"] for r in rr], [r["H_bed"] for r in rr], mk[sn], color=plotstyle.BLUE, ms=6, mfc="none", label=f"{sn}: bed only, self-formed")
        rr = [r for r in results[("wide", 30, 1.0)] if r["sand"] == sn and np.isfinite(r["H_both"])]
        ax2.plot([r["Hc"] for r in rr], [r["H_both"] for r in rr], mk[sn], color=plotstyle.ORANGE, ms=6, label=f"{sn}: bed + tip, width from the tip")
    ax2.plot([0, 9], [0, 9], color=plotstyle.INK_2, lw=1.0)
    ax2.set_xlim(3, 8.5); ax2.set_ylim(3, 8.5); ax2.set_aspect("equal")
    ax2.set_xlabel("measured critical head [cm]"); ax2.set_ylabel("predicted [cm]")
    ax2.set_title("No fitted parameter"); ax2.legend(frameon=False, fontsize=7.5, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    out = os.path.join(a.out, "paper1_pol_box.png"); fig.savefig(out); print("\nfigure:", out)


if __name__ == "__main__":
    main()
