"""Small and medium scale with the same closures: equilibrium heads, pipe gradients and discharge.

Data: Van Beek et al. (2015), Table 5 - medium-scale tests Bms19 (Baskarp) and Ims20 (Itterbeck 330), pipes at rest
at several lengths, with measured head, mean pipe gradient and discharge; critical heads from their Table 4.
Box: 1.90 x 0.881 x 0.403 m, seepage length 1.385 m, exit hole 20.5 mm (the small-scale box scaled up four times).

Closures for the width of the pipe (bed at the threshold of motion in all of them):
    self     self-formed section                       (no input)
    tip      width >= 30 d50                           (tip width observed at small scale)
    seep     width = Q_pipe / (k l i*), >= 30 d50      (mean seepage gradient into the pipe bed = i*)
    cap      tip + depth never above c d50; the pipe widens instead   (c from the measured medium-scale pipe gradients)
tau_c: Shields number of Pol's sands interpolated in grain size, read as section average (see pipe_width_data.py).

usage:  python run_scales.py [--out ../tmp] [--istar 2.5] [--cap 7] [--pol-states ../tmp/paper1_pol_states.csv]
"""
import argparse
import csv
import os
import warnings
import numpy as np
from pipetheory import G
from pipetheory.aquifer import Box, pol_box, PipeResponse
from pipetheory.coupled import fold_law, solve_pipe_law, fold_seepage_width, solve_seepage_width
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section
from pipe_width_data import read_csv, shields_fit, DRHO

HERE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore", message="Mean of empty slice")


_cache = {}


def medium_R(l):
    key = round(l, 4)
    if key not in _cache:
        _cache[key] = PipeResponse(Box(Lx=1.90, Wy=0.88, D=0.40, x_exit=0.52, d=0.01, pipe_width=0.04, hole_radius=0.01), l, rtol=1.0e-8)
    return _cache[key]


def closures(sec, tau_c, d50, istar, cap):
    return {"self": dict(kind="law", law=PipeLaw(sec, tau_c)),
            "tip": dict(kind="law", law=PipeLaw(sec, tau_c, W0=30 * d50)),
            f"seep i*={istar:g}": dict(kind="seep", i=istar, Wmin=30 * d50),
            f"cap {cap:g} d50": dict(kind="law", law=PipeLaw(sec, tau_c, W0=30 * d50, a_cap=cap * d50))}


def fold_head(R, c, sec, tau_c, k):
    if c["kind"] == "law":
        return fold_law(R, c["law"], k)[0]
    return fold_seepage_width(R, sec, tau_c, k, c["i"], c["Wmin"])[0]


def state(R, c, sec, tau_c, k, H):
    if c["kind"] == "law":
        sol = solve_pipe_law(R, c["law"], H, k); W = np.nan
    else:
        o = solve_seepage_width(R, sec, tau_c, H, k, c["i"], c["Wmin"])
        sol, W = (o[0], o[1]) if o else (None, np.nan)
    if sol is None:
        return None
    return dict(p=np.polyfit(R.x_st, sol["h"], 1)[0], Q=R.upstream_flow(sol["h"], H, k), W=W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    ap.add_argument("--istar", type=float, default=2.5)
    ap.add_argument("--cap", type=float, default=7.0)
    ap.add_argument("--pol-states", default=None, help="csv written by compare_pol_heads_all.py: adds the small-scale resting states")
    a = ap.parse_args()
    sands = {r["sand"]: r for r in read_csv("pol2022_sands.csv")}
    theta, _ = shields_fit(sands)
    sec = threshold_section(0.77, 16, 161, 41)
    fac = sec["a_max"] * sec["aavg_over_amax"]
    tau = lambda d: theta(d) * DRHO * G * d / fac
    rows = [r for r in read_csv("vanbeek2015_table5.csv") if r["test"] in ("Bms19", "Ims20")]
    print("1. Medium-scale pipes at rest (Van Beek et al. 2015, Table 5): head needed by the bed criterion / measured head,")
    print("   and at the measured head: mean pipe gradient and discharge, model / measured (- : below the fold)")
    summary = {}
    for r in rows:
        d50, k, l, H, p, Q = float(r["d50_mm"]) * 1e-3, float(r["k_m_per_s"]), float(r["l_m"]), float(r["H_m"]), float(r["p"]), float(r["Q_m3_per_s"])
        R = medium_R(l)
        line = f"   {r['test']:6s} l = {l:5.3f} m  H = {H * 100:5.1f} cm |"
        for name, c in closures(sec, tau(d50), d50, a.istar, a.cap).items():
            Hf = fold_head(R, c, sec, tau(d50), k)
            s = state(R, c, sec, tau(d50), k, H) if H > Hf else None
            summary.setdefault((r["test"], name), []).append((Hf / H, s["p"] / p if s else np.nan, s["Q"] / Q if s else np.nan, s["W"] if s else np.nan))
            line += f" {name}: {Hf / H:4.2f}" + (f" (p {s['p'] / p:4.2f}, Q {s['Q'] / Q:4.2f})" if s else " (-)") + " |"
        print(line)
    print("\n   mean over states:   head ratio | pipe gradient ratio | discharge ratio | width [mm]")
    for (test, name), v in summary.items():
        v = np.array(v)
        print(f"   {test:6s} {name:13s}  {v[:, 0].mean():.2f} +/- {v[:, 0].std(ddof=1):.2f} | {np.nanmean(v[:, 1]):.2f} | {np.nanmean(v[:, 2]):.2f} | {np.nanmean(v[:, 3]) * 1e3:.0f}")

    print("\n2. Critical head (maximum of the bed criterion over pipe length), predicted / measured")
    small = [PipeResponse(pol_box(), l) for l in np.arange(0.06, 0.33, 0.04)]
    med = [medium_R(l) for l in (0.44, 0.69, 0.94, 1.135)]
    cases = [("B132 (small, Baskarp)", small, 0.132e-3, 9.3e-5, 0.065),
             ("B142 (small, Baskarp)", small, 0.132e-3, 6.2e-5, 0.080),
             ("Bms19 (medium, Baskarp)", med, 0.132e-3, 8.0e-5, 0.210),
             ("Ims20 (medium, Itterbeck)", med, 0.283e-3, 3.9e-4, 0.194)]
    out = []
    for label, Rs, d50, k, Hc in cases:
        box = Rs[0].box
        line = f"   {label:26s} Hc = {Hc * 100:5.1f} cm |"
        for name, c in closures(sec, tau(d50), d50, a.istar, a.cap).items():
            hh = np.array([fold_head(R, c, sec, tau(d50), k) for R in Rs])
            line += f" {name}: {hh.max() / Hc:4.2f} (l/L {Rs[hh.argmax()].length / box.seepage_length:4.2f}) |"
            out.append((label, name, hh.max(), Hc))
        print(line)
    print("\n3. Ratio of critical heads, medium / small scale, Baskarp sand (same closure, k as in each test)")
    for name in dict.fromkeys(o[1] for o in out):
        g = {o[0]: o[2] for o in out if o[1] == name}
        print(f"   {name:13s} Bms19 / B132: {g['Bms19 (medium, Baskarp)'] / g['B132 (small, Baskarp)']:.2f}   measured {0.210 / 0.065:.2f};"
              f"   Bms19 / B142: {g['Bms19 (medium, Baskarp)'] / g['B142 (small, Baskarp)']:.2f}   measured {0.210 / 0.080:.2f}")
    if a.pol_states:
        print(f"\n4. Small-scale resting states of Pol (2022): head needed by the bed criterion / head at rest, closure 'cap {a.cap:g} d50' vs 'tip'")
        kk = {r["test"]: float(r["k_m_per_s"]) for r in read_csv("pol2022_table3_2.csv")}
        with open(a.pol_states) as f:
            st = [r for r in csv.DictReader(f) if r["variant"] == "wide"]
        res = []
        for r in st:
            sd = sands[r["sand"]]; d50 = float(sd["d50_mm"]) * 1e-3
            so = threshold_section(np.tan(np.radians(float(sd["repose_deg"]))), 16, 161, 41)
            tc = float(sd["tau_eq_Pa"]) / (so["a_max"] * so["aavg_over_amax"])
            R = PipeResponse(pol_box(pipe_width=0.030 if r["sand"] == "FS35" else 0.025), float(r["l_cm"]) * 1e-2)
            Hn = float(r["H_net_cm"]) * 1e-2
            res.append((r["sand"], fold_law(R, PipeLaw(so, tc, W0=30 * d50), kk[r["test"]])[0] / Hn,
                        fold_law(R, PipeLaw(so, tc, W0=30 * d50, a_cap=a.cap * d50), kk[r["test"]])[0] / Hn))
        v = np.array([x[1:] for x in res])
        print(f"   n = {len(v)}:  tip {v[:, 0].mean():.2f} +/- {v[:, 0].std(ddof=1):.2f};   cap {v[:, 1].mean():.2f} +/- {v[:, 1].std(ddof=1):.2f}   "
              + "  ".join(f"{sn}: {np.mean([x[2] for x in res if x[0] == sn]):.2f}" for sn in ("B25", "FPH", "FS35")))
    with open(os.path.join(a.out, "paper1_scales.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "closure", "Hc_pred_m", "Hc_meas_m"]); w.writerows(out)


if __name__ == "__main__":
    main()
