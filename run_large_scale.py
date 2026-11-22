"""Same sand at 20 times the scale: the large-scale test of Pol (2022, chapter 4) from small-scale calibration.

Sand: the fine sand of the test levee (d50 0.185 mm) is the sand FPH of the small-scale tests.  Large-scale test:
sand bed 9.5 x 1.2 x 1.0 m, exit hole 13 mm at 1.3 m from the downstream end, seepage length 7.2 m, in-situ
conductivity 9e-5 m/s at 5 deg C (nu = 1.52e-6 m2/s).  Measured: critical head 1.42 m (corrected for exit loss),
critical pipe length 1.65 m, pipe gradient about 0.12, gradient over the first 0.8 m ahead of the tip 0.31-0.41.

Everything in the model is taken from the small-scale FPH tests: bed stress at rest and repose angle (-> tau_c, mu),
tip width 30 d50, the largest pipe depth (--cap, in d50; small-scale pipes of this sand are 3.5-4.5 d50 deep) and
the tip strength K_c (mean K of the small-scale pipes at rest, same pipe closure).

usage:  python run_large_scale.py --pol-states ../tmp/paper1_pol_states.csv [--cap 4.0] [--out ../tmp]
"""
import argparse
import csv
import os
import numpy as np
from pipetheory.aquifer import Box, pol_box, PipeResponse
from pipetheory.coupled import solve_pipe_law
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section
from pipe_width_data import read_csv
from run_tip_criterion import evaluate, measures
from run_tip_same_sand import head_for_K

HERE = os.path.dirname(os.path.abspath(__file__))
D50, NU_COLD, K_FIELD = 0.185e-3, 1.52e-6, 9.0e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pol-states", required=True)
    ap.add_argument("--cap", type=float, default=4.0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    a = ap.parse_args()
    sd = next(r for r in read_csv("pol2022_sands.csv") if r["sand"] == "FPH")
    sec = threshold_section(np.tan(np.radians(float(sd["repose_deg"]))), 16, 161, 41)
    tau = float(sd["tau_eq_Pa"]) / (sec["a_max"] * sec["aavg_over_amax"])
    kk = {r["test"]: float(r["k_m_per_s"]) for r in read_csv("pol2022_table3_2.csv")}
    law = PipeLaw(sec, tau, W0=30 * D50, a_cap=a.cap * D50)
    print(f"pipe closure: bed at the threshold (tau_c = {tau:.3f} Pa), width >= 30 d50, depth <= {a.cap:g} d50  ->  pipe gradient not below {law.S_floor:.3f}")
    with open(a.pol_states) as f:
        st = [r for r in csv.DictReader(f) if r["variant"] == "wide" and r["sand"] == "FPH"]
    Ks, fr = [], []
    for r in st:
        R = PipeResponse(pol_box(pipe_width=0.025), float(r["l_cm"]) * 1e-2, with_fields=True)
        m, Hf = evaluate(R, law, float(r["H_net_cm"]) * 1e-2, kk[r["test"]], D50)
        fr.append(Hf / (float(r["H_net_cm"]) * 1e-2))
        if m:
            Ks.append(m["K"])
    Kc = float(np.mean(Ks))
    print(f"\n1. Small scale (FPH, {len(st)} pipes at rest): bed criterion / head at rest = {np.mean(fr):.2f} +/- {np.std(fr, ddof=1):.2f};"
          f"  K at rest = {Kc:.3f} +/- {np.std(Ks, ddof=1):.3f} m^1/2 ({len(Ks)} states above the fold)")

    cold = PipeLaw(sec, tau, W0=30 * D50, a_cap=a.cap * D50, nu=NU_COLD)
    print("\n2. Large scale: head required at each pipe length [m]")
    print("   l [m]   bed criterion   tip criterion (K_c from small scale)   required")
    rows = []
    for l in (0.8, 1.2, 1.65, 2.2, 3.0):
        R = PipeResponse(Box(Lx=9.5, Wy=1.2, D=1.0, x_exit=1.3, x_river=8.5, d=0.025, pipe_width=0.05, hole_radius=0.0065), l, with_fields=True, rtol=1.0e-8)
        Ht, Hf = head_for_K(R, cold, K_FIELD, Kc, 1.42)
        rows.append((R.length, Hf, Ht, max(Hf, Ht), R))
        print(f"   {R.length:5.2f} {Hf:12.2f} {Ht:22.2f} {max(Hf, Ht):28.2f}", flush=True)
    j = int(np.argmax([r[3] for r in rows]))
    print(f"   critical head: predicted {rows[j][3]:.2f} m at l = {rows[j][0]:.2f} m;  measured 1.42 m at l = 1.65 m;  ratio {rows[j][3] / 1.42:.2f}")
    print(f"   bed criterion alone: {max(r[1] for r in rows):.2f} m ({max(r[1] for r in rows) / 1.42:.2f} of the measured head)")

    R = rows[2][4]
    sol = solve_pipe_law(R, cold, 1.42, K_FIELD)
    print("\n3. State at the measured critical head and length (H = 1.42 m, l = 1.65 m)")
    if sol is None:
        print("   below the fold of the model"); return
    x, roof = R.roof_head(sol["h"], 1.42)
    ht = sol["h"][-1]
    m = measures(R, sol, 1.42, K_FIELD, D50)
    print(f"   mean pipe gradient: model {np.polyfit(R.x_st, sol['h'], 1)[0]:.3f}, measured about 0.12")
    print(f"   gradient over the first 0.8 m ahead of the tip: model {(np.interp(R.x_tip + 0.8, x, roof) - ht) / 0.8:.2f}, measured 0.31-0.41")
    print(f"   head at the tip / H: model {ht / 1.42:.2f};   K at this state: {m['K']:.3f} m^1/2 (K_c from small scale {Kc:.3f})")
    print(f"   discharge through the exit: model {R.upstream_flow(sol['h'], 1.42, K_FIELD) * 1e6:.1f} mL/s")


if __name__ == "__main__":
    main()
