"""One sand, two scales: is the tip criterion the same?  (Baskarp sand; Van Beek et al. 2015.)

Small scale (seepage length 0.344 m): only the critical head of each test is known.  Medium scale (1.385 m, test
Bms19): six pipes at rest and the critical head.  Pipe closure as in run_tip_criterion.py.  The loading of the tip
is measured by K = (head 5 cm ahead of the tip - head at the tip) / sqrt(5 cm), the intensity of a square-root
field (a critical secant gradient i_c(x) = K_c x^(-1/2), Robbins et al. 2018).
    1. K_c from the dense small-scale tests: the smallest K over pipe length at the measured critical head.
    2. With that K_c, the head at which each medium-scale pipe should come to rest, and the medium-scale critical head.
    3. The reverse: K at rest at medium scale -> critical head of the small-scale tests.
The head required at length l is the larger of the head that gives K = K_c and the head of the bed criterion.

usage:  python run_tip_same_sand.py [--out ../tmp]
"""
import argparse
import os
import numpy as np
from pipetheory import G
from pipetheory.aquifer import Box, pol_box, PipeResponse
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section
from pipe_width_data import read_csv, shields_fit, DRHO
from run_tip_criterion import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
D50 = 0.132e-3


def head_for_K(R, law, k, Kc, H0):
    """Head at which the tip loading equals Kc (secant iteration), not below the bed-criterion fold.  Returns (H, fold)."""
    m0, Hf = evaluate(R, law, max(H0, 1.03 * Hf0(R, law, k)), k, D50)
    Ha = max(H0, 1.03 * Hf); Hb = max(0.85 * Ha, 1.015 * Hf)
    Ka = evaluate(R, law, Ha, k, D50)[0]["K"]; Kb = evaluate(R, law, Hb, k, D50)[0]["K"]
    for _ in range(3):
        Hn = Ha + (Kc - Ka) * (Ha - Hb) / (Ka - Kb)
        if Hn <= 1.01 * Hf:
            return Hf, Hf
        Hb, Kb = Ha, Ka
        Ha, Ka = Hn, evaluate(R, law, Hn, k, D50)[0]["K"]
    return Ha, Hf


def Hf0(R, law, k):
    from pipetheory.coupled import fold_law
    if not hasattr(R, "_fold"):
        R._fold = {}
    key = (id(law), k)
    if key not in R._fold:
        R._fold[key] = fold_law(R, law, k)[0]
    return R._fold[key]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    ap.add_argument("--small-footprint", type=float, default=0.010, help="footprint of the pipe on the roof in the small-scale model [m]")
    a = ap.parse_args()
    theta, _ = shields_fit({r["sand"]: r for r in read_csv("pol2022_sands.csv")})
    sec = threshold_section(0.77, 16, 161, 41)
    law = PipeLaw(sec, theta(D50) * DRHO * G * D50 / (sec["a_max"] * sec["aavg_over_amax"]), W0=30 * D50, a_cap=7 * D50)
    small = [PipeResponse(pol_box(pipe_width=a.small_footprint), l, with_fields=True) for l in np.arange(0.06, 0.301, 0.04)]
    print(f"footprint of the pipe on the roof: small scale {a.small_footprint * 1e3:.0f} mm (cells 5 mm), medium scale 40 mm (cells 10 mm)")
    dense = (("B115", 5.4e-5, 0.080), ("B118", 6.3e-5, 0.080), ("B142", 6.2e-5, 0.080))
    print("1. Small scale, dense Baskarp (RD 0.89-0.91): smallest tip loading over pipe length at the measured critical head")
    Kc = []
    for name, k, Hc in dense:
        ms = [evaluate(R, law, Hc, k, D50)[0] for R in small]
        j = int(np.argmin([m["K"] if m else np.inf for m in ms]))
        Kc.append(ms[j]["K"])
        print(f"   {name}: K = {ms[j]['K']:.3f} m^1/2 at l = {small[j].length * 100:.0f} cm  (2 cm secant gradient there: {ms[j]['G2']:.2f})")
    Kc_small = float(np.mean(Kc))
    print(f"   K_c (small scale) = {Kc_small:.3f} m^1/2")

    rows = [r for r in read_csv("vanbeek2015_table5.csv") if r["test"] == "Bms19"]
    k = 8.0e-5
    print("\n2. Medium scale, Bms19 (RD 0.94): pipes at rest")
    print("   l [m]   H measured [cm]   K at rest   head for K_c(small) / measured   bed criterion / measured")
    Km, ratio, Hreq, med = [], [], [], []
    for r in rows:
        R = PipeResponse(Box(Lx=1.90, Wy=0.88, D=0.40, x_exit=0.52, d=0.01, pipe_width=0.04, hole_radius=0.01), float(r["l_m"]), with_fields=True, rtol=1.0e-8)
        H = float(r["H_m"])
        m, Hf = evaluate(R, law, H, k, D50)
        Hp, _ = head_for_K(R, law, k, Kc_small, H)
        Km.append(m["K"]); ratio.append(Hp / H); Hreq.append(Hp); med.append(R)
        print(f"   {R.length:5.3f} {H * 100:12.1f} {m['K']:13.3f} {Hp / H:22.2f} {Hf / H:28.2f}")
    print(f"   K at rest: {np.mean(Km):.3f} +/- {np.std(Km, ddof=1):.3f};  K_c(small) / K at rest = {Kc_small / np.mean(Km):.2f}")
    print(f"   head for K_c(small) / measured head: {np.mean(ratio):.2f} +/- {np.std(ratio, ddof=1):.2f}")
    print(f"   critical head (largest required head over the six lengths): predicted {max(Hreq) * 100:.1f} cm, measured 21.0 cm, ratio {max(Hreq) / 0.21:.2f}")

    Kc_med = float(np.mean(Km))
    print(f"\n3. Reverse: K_c = {Kc_med:.3f} m^1/2 (medium scale, at rest) -> critical head of the small-scale tests")
    for name, k_s, Hc in dense:
        hh = [head_for_K(R, law, k_s, Kc_med, Hc)[0] for R in small]
        print(f"   {name}: predicted {max(hh) * 100:.1f} cm at l = {small[int(np.argmax(hh))].length * 100:.0f} cm, measured {Hc * 100:.1f} cm, ratio {max(hh) / Hc:.2f}")


if __name__ == "__main__":
    main()
