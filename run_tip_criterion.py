"""Is there one tip criterion that holds pipes at rest at both scales?

The pipe is modelled with the closure that reproduces the measured pipe gradients at both scales (bed at the
threshold, width >= 30 d50, depth <= 7 d50).  For every pipe at rest (Pol 2022, small scale; Van Beek et al. 2015,
medium scale) the model is solved at the measured head and three measures of the loading of the tip are evaluated:
    I    inflow per unit length over the last 2 cm of the pipe / (k * 30 d50)   - mean seepage gradient into the tip
    G2   secant gradient between the tip and the roof 2 cm ahead of it         - what Pol measured
    K    (head 5 cm ahead - head at the tip) / sqrt(5 cm)   [m^1/2]             - intensity of a square-root field
If a measure is the criterion, it must take the same value in all states at rest of a given sand.  Each measure is
then calibrated on the small-scale states and used to predict the head of the medium-scale states.

usage:  python run_tip_criterion.py --pol-states ../tmp/paper1_pol_states.csv [--out ../tmp]
"""
import argparse
import csv
import os
import numpy as np
from pipetheory import G
from pipetheory.aquifer import Box, pol_box, PipeResponse
from pipetheory.coupled import fold_law, solve_pipe_law
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section
from pipe_width_data import read_csv, shields_fit, DRHO

HERE = os.path.dirname(os.path.abspath(__file__))
LAM = 0.02


def measures(R, sol, H, k, d50):
    n_l = max(1, int(round(LAM / R.box.d)))
    q_t = sol["Q"][R.n - n_l] / (n_l * R.box.d)
    x, roof = R.roof_head(sol["h"], H)
    ht = sol["h"][-1]
    g = lambda x0: (np.interp(R.x_tip + x0, x, roof) - ht) / x0
    return dict(I=q_t / (k * 30.0 * d50), G2=g(0.02), K=g(0.05) * np.sqrt(0.05), tip_head=ht / H)


def evaluate(R, law, H, k, d50):
    Hf = fold_law(R, law, k)[0]
    if H <= Hf * 1.005:
        return None, Hf
    sol = solve_pipe_law(R, law, H, k)
    return (measures(R, sol, H, k, d50) if sol else None), Hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pol-states", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    ap.add_argument("--cap", type=float, default=7.0)
    a = ap.parse_args()
    sands = {r["sand"]: r for r in read_csv("pol2022_sands.csv")}
    theta, _ = shields_fit(sands)
    kk = {r["test"]: float(r["k_m_per_s"]) for r in read_csv("pol2022_table3_2.csv")}
    rows = []
    with open(a.pol_states) as f:
        st = [r for r in csv.DictReader(f) if r["variant"] == "wide"]
    for r in st:
        sd = sands[r["sand"]]; d50 = float(sd["d50_mm"]) * 1e-3
        sec = threshold_section(np.tan(np.radians(float(sd["repose_deg"]))), 16, 161, 41)
        tau = float(sd["tau_eq_Pa"]) / (sec["a_max"] * sec["aavg_over_amax"])
        R = PipeResponse(pol_box(pipe_width=0.030 if r["sand"] == "FS35" else 0.025), float(r["l_cm"]) * 1e-2, with_fields=True)
        H, k = float(r["H_net_cm"]) * 1e-2, kk[r["test"]]
        m, Hf = evaluate(R, PipeLaw(sec, tau, W0=30 * d50, a_cap=a.cap * d50), H, k, d50)
        rows.append(dict(test=r["test"], scale="small", sand=r["sand"], l=R.length, H=H, fold_over_H=Hf / H, **(m or {})))
    sec = threshold_section(0.77, 16, 161, 41); fac = sec["a_max"] * sec["aavg_over_amax"]
    med = []
    for r in read_csv("vanbeek2015_table5.csv"):
        if r["test"] not in ("Bms19", "Ims20"):
            continue
        d50, k, H = float(r["d50_mm"]) * 1e-3, float(r["k_m_per_s"]), float(r["H_m"])
        law = PipeLaw(sec, theta(d50) * DRHO * G * d50 / fac, W0=30 * d50, a_cap=a.cap * d50)
        R = PipeResponse(Box(Lx=1.90, Wy=0.88, D=0.40, x_exit=0.52, d=0.01, pipe_width=0.04, hole_radius=0.01), float(r["l_m"]), with_fields=True, rtol=1.0e-8)
        m, Hf = evaluate(R, law, H, k, d50)
        H2 = max(0.85 * H, 1.02 * Hf)
        m2 = measures(R, solve_pipe_law(R, law, H2, k), H2, k, d50)
        rows.append(dict(test=r["test"], scale="medium", sand=r["sand"], l=R.length, H=H, fold_over_H=Hf / H, **m))
        med.append((r["test"], H, H2, m, m2))
    keys = ["test", "scale", "sand", "l", "H", "fold_over_H", "I", "G2", "K", "tip_head"]
    with open(os.path.join(a.out, "paper1_tip_criterion.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    print(f"Tip loading of pipes at rest (pipe closure: width >= 30 d50, depth <= {a.cap:g} d50); mean (coefficient of variation)")
    print(f"{'group':22s} {'n':>3s} {'above fold':>10s} | {'I':>12s} | {'G2':>12s} | {'K [m^1/2]':>13s} | {'tip head / H':>12s}")
    groups = [("small " + s, [r for r in rows if r["scale"] == "small" and r["sand"] == s]) for s in ("B25", "FPH", "FS35")]
    groups.append(("small, all", [r for r in rows if r["scale"] == "small"]))
    groups += [("medium " + t, [r for r in rows if r["test"] == t]) for t in ("Bms19", "Ims20")]
    stat = {}
    for name, rr in groups:
        ok = [r for r in rr if "I" in r]
        f = lambda key: (np.mean([r[key] for r in ok]), np.std([r[key] for r in ok], ddof=1) / np.mean([r[key] for r in ok]))
        stat[name] = {key: f(key) for key in ("I", "G2", "K", "tip_head")}
        print(f"{name:22s} {len(rr):3d} {len(ok):10d} | " + " | ".join(f"{stat[name][key][0]:6.3f} ({stat[name][key][1]:.2f})" for key in ("I", "G2", "K", "tip_head")))
    print("\nMedium-scale head predicted with each measure calibrated on the small-scale states, / measured head")
    for test in ("Bms19", "Ims20"):
        out = {}
        for key in ("I", "G2", "K"):
            target = stat["small, all"][key][0]
            out[key] = [(H + (target - m[key]) * (H - H2) / (m[key] - m2[key])) / H for t_, H, H2, m, m2 in med if t_ == test]
        print(f"   {test}: " + ";  ".join(f"{key}: {np.mean(v):.2f} +/- {np.std(v, ddof=1):.2f}" for key, v in out.items()))
    print("   (bed criterion alone, same closure: " + ", ".join(f"{t} {np.mean([r['fold_over_H'] for r in rows if r['test'] == t]):.2f}" for t in ("Bms19", "Ims20")) + ")")


if __name__ == "__main__":
    main()
