"""Convergence and uniqueness of the numerical solution of the threshold cross-section (manuscript appendix A).

1. Grid convergence: spline nodes, grid points across the half-section and across the gap; Richardson
   extrapolation of the two finest grids (the scheme is of second order).
2. The same with seepage (sigma = 0.4).
3. Uniqueness: the same solution is reached from starting guesses whose width and shape differ widely.
usage: python run_section_convergence.py [--mu 0.77] [--out ../tmp]        (about 6 minutes)
"""
import argparse
import os
import warnings
import numpy as np
from pipetheory.section import threshold_section

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, default=0.77)
ap.add_argument("--out", default="../tmp")
a = ap.parse_args()
path = os.path.join(a.out, "paper1_section_convergence.txt")
f = open(path, "w")


def say(line=""):
    print(line, flush=True)
    f.write(line + "\n"); f.flush()


say(f"mu = {a.mu}")
say("\n1. grid convergence (nodes, ns, nz): a_max/a_c, W/a_c, chi, residual of the threshold condition at the nodes")
grids = [(12, 121, 31), (16, 161, 41), (24, 241, 61), (32, 321, 81), (40, 401, 101)]
KEYS = (("a_max", 1.0), ("half_width", 2.0), ("chi", 1.0))
for sg in (0.0, 0.4):
    say(f"   sigma = {sg}")
    res = []
    for g in grids:
        o = threshold_section(a.mu, *g, sigma=sg)
        res.append(o)
        say(f"   {g[0]:3d} {g[1]:4d} {g[2]:4d}   {o['a_max']:.6f}  {2 * o['half_width']:.6f}  {o['chi']:.7f}   {o['max_residual']:.1e}")
    h2 = np.array([1.0 / (g[1] - 1) ** 2 for g in grids])
    ext = {k: res[-1][k] + (res[-1][k] - res[-2][k]) * h2[-1] / (h2[-2] - h2[-1]) for k, _ in KEYS}
    say("   extrapolated to zero cell size:   " + "  ".join(f"{c * ext[k]:.6f}" for k, c in KEYS))
    for g, o in zip(grids, res):
        say(f"      {g} relative to the extrapolation:  a_max {o['a_max'] / ext['a_max'] - 1:+.1e}  W {o['half_width'] / ext['half_width'] - 1:+.1e}  chi {o['chi'] / ext['chi'] - 1:+.1e}")
    d = [res[i + 1]["chi"] - res[i]["chi"] for i in range(len(res) - 1)]
    say("      observed order in chi from successive differences: " + ", ".join(
        f"{np.log(d[i] / d[i + 1]) / np.log(np.sqrt((h2[i] - h2[i + 1]) / (h2[i + 1] - h2[i + 2]))) * 1.0:.2f}" for i in range(len(d) - 1)) +
        "   (2 = differences in proportion to the differences of the squared cell size)")
say("\n3. starting guesses on the grid (16, 161, 41): factor on the half-width, exponent on the cosine profile -> a_max/a_c, W/a_c, chi")
vals, n_try = [], 0
for init in ((1.0, 1.0), (0.5, 1.0), (0.7, 1.0), (1.5, 1.0), (2.0, 1.0), (1.0, 0.5), (1.0, 2.0), (0.7, 1.6), (1.6, 0.6)):
    n_try += 1
    try:
        o = threshold_section(a.mu, 16, 161, 41, init=init, max_nfev=60 * 17)
        ok = o["max_residual"] < 1e-8
        if ok:
            vals.append([o["a_max"], 2 * o["half_width"], o["chi"]])
        say(f"   {init}:  {o['a_max']:.6f}  {2 * o['half_width']:.6f}  {o['chi']:.7f}   residual {o['max_residual']:.1e}" + ("" if ok else "   (iteration did not converge)"))
    except Exception as e:                                   # the iteration may fail from a poor guess; it must not find another solution
        say(f"   {init}:  no convergence ({type(e).__name__})")
vals = np.array(vals)
say(f"   largest relative difference between the converged solutions: {np.max(np.ptp(vals, axis=0) / vals.mean(axis=0)):.1e}  ({len(vals)} of {n_try} converged)")
f.close()
