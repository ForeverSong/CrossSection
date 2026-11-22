"""Threshold-pipe theory in the model-levee geometry of Okamura & Kusube (2025, Soils and Foundations 65, 101591).

Their analysis (Fig. 5): sand bed 380 x 120 x 60 mm; landside surface 110 mm (head 0), levee base L = 190 mm
(impermeable), riverside bed 80 mm and the riverside end wall at head H; k_h = 5 k_v; pipe from the landside toe,
l = 90 mm.  They take a rectangular pipe (width 8 x depth), choose the size that maximises the Shields number at
the tip ("optimum pipe size") and require that maximum to reach the critical Shields number.  Here the pipe has
the self-formed threshold section and the head for progression is the fold of the coupled problem - no pipe size
is assumed.  Sand data from their Table 2 (critical Shields number at 1 g, d70, k_v).

usage:  python run_okamura.py [--out ../tmp] [--cell 2.5] [--mu 0.7]
"""
import argparse
import os
import numpy as np
from pipetheory import RHO_W, G, NU
from pipetheory.aquifer import okamura_box, PipeResponse
from pipetheory.coupled import fold, head_from_N, fold_law
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section

HERE = os.path.dirname(os.path.abspath(__file__))
SANDS = {  # name: d70 [m], critical Shields number at 1 g, k_v [m/s], their predicted H/L, observed H/L (l/L)
    "#8": dict(d=0.15e-3, d50=0.12e-3, theta=0.11, kv=3.7e-5, pred=0.27, obs="0.32 (l/L = 0.47)", ip_pred=0.11, ip_obs="0.17-0.22"),
    "#5": dict(d=0.60e-3, d50=0.52e-3, theta=0.038, kv=5.7e-4, pred=0.24, obs="0.26 (0.49), 0.29 (0.43)", ip_pred=0.14, ip_obs="0.15-0.19"),
}
GS_MINUS_1 = 1.64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    ap.add_argument("--cell", type=float, default=2.5, help="cell size [mm]")
    ap.add_argument("--mu", type=float, default=0.7, help="friction coefficient of the sand (repose angle not reported)")
    a = ap.parse_args()
    d = a.cell * 1e-3
    sec = threshold_section(a.mu, 16, 161, 41)
    L = okamura_box(d=d).seepage_length
    print(f"geometry: L = {L * 1e3:.0f} mm, cell {a.cell} mm, k_h = 5 k_v, mu = {a.mu} (chi = {sec['chi']:.4f})")

    print("\n1. Fold of the geometry, N_min [m2] (k = k_h), for three footprints of the pipe on the roof")
    lengths = np.array([0.03, 0.06, 0.09, 0.12, 0.15])
    Nmin, sols, Rs = {}, {}, {}
    for w in (0.005, 0.010, 0.020):
        Nmin[w] = []
        for l in lengths:
            R = PipeResponse(okamura_box(d=d, pipe_width=w), l)
            N, s = fold(R)
            Nmin[w].append(N)
            if abs(l - 0.09) < 1e-9:
                sols[w], Rs[w] = s, R
        Nmin[w] = np.array(Nmin[w])
        print(f"   footprint {w * 1e3:4.0f} mm: " + "  ".join(f"l={l * 1e3:.0f}mm: {N:.3f}" for l, N in zip(lengths, Nmin[w])))

    print("\n2. Head for progression at l = 90 mm (l/L = 0.47), H/L")
    print(f"   {'sand':4s} {'tau_c [Pa]':>10s} {'k_h [m/s]':>10s} | {'this theory (footprint 5 / 10 / 20 mm)':>38s} | {'O&K predicted':>13s} | observed")
    out = {}
    for name, s in SANDS.items():
        tau = s["theta"] * GS_MINUS_1 * RHO_W * G * s["d"]
        law = PipeLaw(sec, tau)
        kh = 5.0 * s["kv"]
        HL = {w: head_from_N(Nmin[w][2], law.Lam, kh) / L for w in Nmin}
        out[name] = (tau, law, kh, HL)
        print(f"   {name:4s} {tau:10.3f} {kh:10.2e} | {HL[0.005]:12.3f} {HL[0.010]:8.3f} {HL[0.020]:8.3f}         | {s['pred']:13.2f} | {s['obs']}")

    print("\n2b. Same, with the pipe width fixed by the tip (W0 = 30 d50, as observed by Van Beek 2015 and Pol 2022) wherever the")
    print("    threshold section would be narrower; footprint 10 mm.  Also the pipe law of O&K (rectangle B = 8h, hydraulic")
    print("    diameter) put into this model, to separate the effect of the pipe law from the rest of their procedure.")
    R = Rs[0.010]
    for name, s in SANDS.items():
        tau, law, kh, HL = out[name]
        lawW = PipeLaw(sec, tau, W0=30.0 * s["d50"])
        Hw, sw = fold_law(R, lawW, kh)
        Ww, aw, _ = lawW.geometry(sw["S"])
        lam_ok = 1.266 * G * law.A ** 4 / NU                      # Q S^3 for B = 8h, k_p = g De^2/(32 nu), tau = rho g (De/4) S
        H_ok = head_from_N(Nmin[0.010][2], lam_ok, kh)
        print(f"   {name}: self-formed {HL[0.010]:.3f};  width from the tip (W0 = {30e3 * s['d50']:.1f} mm) {Hw / L:.3f}"
              f" (depth at the exit {aw[0] * 1e3:.2f} mm = {aw[0] / s['d50']:.1f} d50, mid-pipe {aw[R.n // 2] / s['d50']:.1f} d50);"
              f"  O&K pipe law in this model {H_ok / L:.3f};  O&K predicted {s['pred']}, observed {s['obs']}")

    print("\n3. The pipe at the fold, l = 90 mm, footprint 10 mm")
    for name, s in SANDS.items():
        tau, law, kh, HL = out[name]
        w = 0.010; R, sol = Rs[w], sols[w]
        H = HL[w] * L
        S = sol["S"] * H; Q = kh * H * sol["Q"]
        W, amax, aavg = law.geometry(S)
        ip = sol["h"][-1] * H / R.length
        Re = Q[0] / (W[0] * NU)
        print(f"   {name}: H = {H * 1e3:.1f} mm; mean pipe gradient {ip:.3f} = {ip / HL[w]:.2f} x H/L  (O&K: predicted {s['ip_pred']}, observed {s['ip_obs']});"
              f" head loss along the pipe {sol['h'][-1]:.2f} H")
        print(f"       section at the exit: width {W[0] * 1e3:.1f} mm, max depth {amax[0] * 1e3:.2f} mm ({amax[0] / s['d']:.1f} d70); mid-pipe: {W[R.n // 2] * 1e3:.1f} x {amax[R.n // 2] * 1e3:.2f} mm;"
              f" discharge {Q[0] * 6e7:.1f} mL/min, Re = Q/(W nu) = {Re:.0f}")

    print("\n4. Geometrically similar levees (same sand): H/L from the L^(-1/2) law, footprint 10 mm at model scale")
    print(f"   {'L [m]':>6s} | {'#8 this theory':>14s} {'O&K':>6s} | {'#5 this theory':>14s} {'O&K (1 g, laminar branch)':>26s}")
    ok8 = {0.19: 0.27, 1.9: 0.086, 19.0: 0.027, 38.0: 0.017}
    ok5 = {0.19: 0.24, 1.9: 0.07}
    for Lp in (0.19, 1.9, 19.0, 38.0):
        f = (Lp / L) ** -0.5
        v8, v5 = out["#8"][3][0.010] * f, out["#5"][3][0.010] * f
        print(f"   {Lp:6.2f} | {v8:14.3f} {ok8.get(Lp, float('nan')):6.3f} | {v5:14.3f} {ok5.get(Lp, float('nan')):26.3f}")
    print("   (the scale law is exact for the theory: N_min scales as L^2 in similar geometries, so H_min/L ~ L^(-1/2);")
    print("    at large L the pipe flow of coarse sand is no longer laminar - see Re above - and the laminar law does not apply)")


if __name__ == "__main__":
    main()
