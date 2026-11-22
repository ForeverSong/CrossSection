"""Screening: over which distance ahead of the tip is the gradient the same at all scales?

For pipes at rest at three scales (seepage length 0.35, 1.4 and 7.2 m) the model is solved at the measured head with
the pipe closure that reproduces the measured pipe gradients, and the secant gradient between the tip and the roof a
distance x0 ahead of it is evaluated for several choices of x0: a fixed 5 cm (local criterion), a fraction of the
thickness D of the sand bed, a fraction of the seepage length L, and the whole distance to the upstream boundary.
A valid criterion must give the same value for one sand at two scales.  Same-sand pairs: FPH (0.35 and 7.2 m) and
Baskarp (0.34 and 1.4 m; at the small scale only the critical head is known and the pipe length is the model's).

usage:  python run_tip_nonlocal.py --pol-states ../tmp/paper1_pol_states.csv [--out ../tmp]
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
CAP = {"FPH": 4.0}                                   # largest pipe depth in d50 (7 for the other sands)
NAMES = ["5 cm", "D/4", "D/2", "D", "0.1 L", "0.2 L", "to upstream"]


def gradients(R, law, H, k, L_up_end):
    """Secant gradients ahead of the tip at head H; None if H is below the bed-criterion fold."""
    if H <= 1.005 * fold_law(R, law, k)[0]:
        return None
    sol = solve_pipe_law(R, law, H, k)
    if sol is None:
        return None
    x, roof = R.roof_head(sol["h"], H)
    ht, D, L = sol["h"][-1], R.box.D, R.box.seepage_length
    out = []
    for x0 in (0.05, D / 4, D / 2, D, 0.1 * L, 0.2 * L, L_up_end - R.x_tip):
        out.append((np.interp(R.x_tip + x0, x, roof) - ht) / x0 if R.x_tip + x0 <= L_up_end + 1e-9 else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pol-states", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    a = ap.parse_args()
    sands = {r["sand"]: r for r in read_csv("pol2022_sands.csv")}
    theta, _ = shields_fit(sands)
    kk = {r["test"]: float(r["k_m_per_s"]) for r in read_csv("pol2022_table3_2.csv")}
    res = {}

    def add(group, g):
        if g is not None:
            res.setdefault(group, []).append(g)

    secs = {}
    for sn, sd in sands.items():
        sec = threshold_section(np.tan(np.radians(float(sd["repose_deg"]))), 16, 161, 41); d50 = float(sd["d50_mm"]) * 1e-3
        secs[sn] = (sec, float(sd["tau_eq_Pa"]) / (sec["a_max"] * sec["aavg_over_amax"]), d50)
    with open(a.pol_states) as f:
        st = [r for r in csv.DictReader(f) if r["variant"] == "wide"]
    for r in st:
        sec, tau, d50 = secs[r["sand"]]
        R = PipeResponse(pol_box(pipe_width=0.030 if r["sand"] == "FS35" else 0.025), float(r["l_cm"]) * 1e-2, with_fields=True)
        law = PipeLaw(sec, tau, W0=30 * d50, a_cap=CAP.get(r["sand"], 7.0) * d50)
        add(f"{r['sand']} small", gradients(R, law, float(r["H_net_cm"]) * 1e-2, kk[r["test"]], 0.48))
    sec77 = threshold_section(0.77, 16, 161, 41); fac = sec77["a_max"] * sec77["aavg_over_amax"]
    lawB = PipeLaw(sec77, theta(0.132e-3) * DRHO * G * 0.132e-3 / fac, W0=30 * 0.132e-3, a_cap=7 * 0.132e-3)
    for k_s in (6.3e-5, 6.2e-5):                     # B118, B142 at their critical head; length where the tip loading is smallest
        add("Baskarp small (crit.)", gradients(PipeResponse(pol_box(), 0.14, with_fields=True), lawB, 0.080, k_s, 0.48))
    for r in read_csv("vanbeek2015_table5.csv"):
        if r["test"] not in ("Bms19", "Ims20"):
            continue
        d50 = float(r["d50_mm"]) * 1e-3
        law = PipeLaw(sec77, theta(d50) * DRHO * G * d50 / fac, W0=30 * d50, a_cap=7 * d50)
        R = PipeResponse(Box(Lx=1.90, Wy=0.88, D=0.40, x_exit=0.52, d=0.01, pipe_width=0.04, hole_radius=0.01), float(r["l_m"]), with_fields=True, rtol=1.0e-8)
        add("Baskarp medium" if r["test"] == "Bms19" else "Itterbeck medium", gradients(R, law, float(r["H_m"]), float(r["k_m_per_s"]), 1.90))
    sec, tau, d50 = secs["FPH"]
    R = PipeResponse(Box(Lx=9.5, Wy=1.2, D=1.0, x_exit=1.3, x_river=8.5, d=0.025, pipe_width=0.05, hole_radius=0.0065), 1.65, with_fields=True, rtol=1.0e-8)
    add("FPH large (crit.)", gradients(R, PipeLaw(sec, tau, W0=30 * d50, a_cap=4 * d50, nu=1.52e-6), 1.42, 9.0e-5, 8.5))

    print("Secant gradient between the tip and the roof a distance x0 ahead of it; mean over the pipes at rest of each group")
    print(f"{'group':22s} {'n':>2s} | " + " ".join(f"{n:>11s}" for n in NAMES))
    mean = {}
    for gname in ("B25 small", "FS35 small", "FPH small", "FPH large (crit.)", "Baskarp small (crit.)", "Baskarp medium", "Itterbeck medium"):
        v = np.array(res.get(gname, [[np.nan] * len(NAMES)]), float)
        mean[gname] = np.nanmean(v, axis=0)
        print(f"{gname:22s} {len(res.get(gname, [])):2d} | " + " ".join(f"{x:11.2f}" for x in mean[gname]))
    print("\nratio large scale / small scale for the same sand (1 = the measure is scale-independent)")
    print(f"{'':22s}    | " + " ".join(f"{n:>11s}" for n in NAMES))
    print(f"{'FPH, 7.2 m / 0.35 m':22s}    | " + " ".join(f"{x:11.2f}" for x in mean["FPH large (crit.)"] / mean["FPH small"]))
    print(f"{'Baskarp, 1.4 / 0.34 m':22s}    | " + " ".join(f"{x:11.2f}" for x in mean["Baskarp medium"] / mean["Baskarp small (crit.)"]))
    with open(os.path.join(a.out, "paper1_tip_nonlocal.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group"] + NAMES)
        for gname, rows in res.items():
            for row in rows:
                w.writerow([gname] + [f"{x:.4f}" for x in row])


if __name__ == "__main__":
    main()
