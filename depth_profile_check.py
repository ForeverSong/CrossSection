"""Depth of the pipe against distance from the tip: theory against the regression of Pol (2022, eq. 3.1).
Pol fitted all laser scans of tests 232-250 (three sands): a_max = 10 (kappa s)^(1/3) and a_avg = 6.0 (kappa s)^(1/3), with kappa the
intrinsic permeability and s the distance from the tip.  Here: the depth along a threshold pipe computed for one state at rest
per sand (the longest one in tmp/paper1_pol_states.csv), with nothing fitted, for the self-formed law and for the law with the
width of the tip.          usage:  python depth_profile_check.py [--states ../tmp/paper1_pol_states.csv]
"""
import argparse
import csv
import numpy as np
from pipetheory.aquifer import PipeResponse, pol_box
from pipetheory.coupled import solve_pipe_law, fold_law
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section
from pipe_width_data import read_csv

ap = argparse.ArgumentParser(); ap.add_argument("--states", default="../tmp/paper1_pol_states.csv"); a = ap.parse_args()
sands = {r["sand"]: r for r in read_csv("pol2022_sands.csv")}
kk = {r["test"]: float(r["k_m_per_s"]) for r in read_csv("pol2022_table3_2.csv")}
best = {}
for r in csv.DictReader(open(a.states)):
    if r["test"] in kk and (r["sand"] not in best or float(r["l_cm"]) > float(best[r["sand"]]["l_cm"])):
        best[r["sand"]] = r
S_CM = (2, 4, 7, 10, 14)
print("depth [grain diameters] at distance s behind the tip; Pol's regression for the largest depth, 10 (kappa s)^(1/3), and theory")
print("sand  test      l[cm] H[cm] | s[cm]: " + "  ".join(f"{s:>5d}" for s in S_CM) + " | fitted power and coefficient of a_max = C (kappa s)^p over 2-14 cm")
for sand, r in best.items():
    sd = sands[sand]; d50 = float(sd["d50_mm"]) * 1e-3; k = kk[r["test"]]; kappa = k * 1e-6 / 9.81
    sec = threshold_section(np.tan(np.radians(float(sd["repose_deg"]))), 16, 161, 41)
    tau = float(sd["tau_eq_Pa"]) / (sec["a_max"] * sec["aavg_over_amax"])
    R = PipeResponse(pol_box(), float(r["l_cm"]) / 100, rtol=1e-9)
    back = R.x_tip - R.x_st
    pol = [10 * (kappa * s / 100) ** (1 / 3) / d50 for s in S_CM]
    print(f"{sand:5s} {r['test']:9s} {float(r['l_cm']):5.1f} {float(r['H_net_cm']):5.2f} | measured (fit)  " + "  ".join(f"{v:5.1f}" for v in pol))
    for name, law in (("self-formed", PipeLaw(sec, tau)), ("tip width", PipeLaw(sec, tau, W0=30 * d50))):
        Hf, sf = fold_law(R, law, k)
        H = max(float(r["H_net_cm"]) / 100, 1.001 * Hf)
        sol = solve_pipe_law(R, law, H, k, sf["h"] * H / Hf) or sf
        amax = law.geometry(sol["S"])[1]
        v = [np.interp(s / 100, back[::-1], amax[::-1]) / d50 for s in S_CM if s / 100 < back.max()]
        m = (back > 0.02) & (back < 0.14)
        p, c = np.polyfit(np.log(kappa * back[m]), np.log(amax[m]), 1)
        print(f"{'':27s}| {name:15s} " + "  ".join(f"{x:5.1f}" for x in v) + f"{'':{7 * (len(S_CM) - len(v))}s} | p = {p:.2f}, C = {np.exp(c + (p - 1 / 3) * np.log(kappa * 0.07)):.1f} (at s = 7 cm, for p = 1/3)")
