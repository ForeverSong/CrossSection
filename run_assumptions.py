"""Orders of magnitude behind the assumptions of the theory (manuscript section 7.3), for the 29 states at rest
of Pol (2022) computed with the self-formed section.

For every station of every state:
  Re_D   Reynolds number on the hydraulic diameter, 4 area / wetted perimeter (the flow is taken as laminar)
  Re_a   Reynolds number on the largest depth and the mean velocity
  eps_l  Re_a * d a_max / d s: inertia over viscous stress in a slowly varying section (lubrication)
  dW/ds  rate at which the section widens along the pipe (slow variation)
  v_n/U  velocity of the seepage through the bed over the mean velocity in the pipe
  slip   sqrt(intrinsic permeability) / largest depth (slip at the surface of the sand, Beavers & Joseph)
  q/nu   inflow per unit length over the viscosity; in a self-formed pipe the inertial term of run_inertia.py is
         (4 beta chi / 3 m^2) q/nu times the friction slope, m a_c^2 being the area of the section
  a/d    largest depth in grain diameters
and the depth below which a pipe at the threshold remains laminar.
usage: python run_assumptions.py [--out ../tmp]         (needs tmp/paper1_pol_states.csv; about 1 minute)
"""
import argparse
import csv
import os
import warnings
import numpy as np
from pipetheory.aquifer import PipeResponse, pol_box
from pipetheory.coupled import solve_pipe_law, fold_law
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section, momentum_coefficient

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RHO, G, NU = 1000.0, 9.81, 1.0e-6


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def perimeter(sec):
    """Wetted perimeter (roof + bed) of the threshold section in units of a_c."""
    y, a = sec["y"], sec["a"]
    return 2.0 * (y[-1] + np.sum(np.hypot(np.diff(y), np.diff(a))))


def rng(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return f"median {np.median(v):.3g}, range {v.min():.3g}-{v.max():.3g}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../tmp")
    a = ap.parse_args()
    sands = {r["sand"]: r for r in read_csv(os.path.join(HERE, "data", "pol2022_sands.csv"))}
    table = {r["test"]: r for r in read_csv(os.path.join(HERE, "data", "pol2022_table3_2.csv"))}
    mu = {s: float(np.tan(np.radians(float(sands[s]["repose_deg"])))) for s in sands}
    sec = {s: threshold_section(mu[s]) for s in sands}
    tau_c = {s: float(sands[s]["tau_eq_Pa"]) / (sec[s]["a_max"] * sec[s]["aavg_over_amax"]) for s in sands}
    states = [r for r in read_csv(os.path.join(a.out, "paper1_pol_states.csv")) if r["variant"] == "base"]
    keys = ("Re_D", "Re_a", "eps_l", "dWds", "vn_U", "slip", "q_nu", "a_d", "a_d_tip")
    acc = {s: {k_: [] for k_ in keys} for s in list(sands) + ["all"]}
    cache = {}
    for st in states:
        sand, test = st["sand"], st["test"]
        l, H, k = 0.01 * float(st["l_cm"]), 0.01 * float(st["H_net_cm"]), float(table[test]["k_m_per_s"])
        if round(l, 4) not in cache:
            cache[round(l, 4)] = PipeResponse(pol_box(), l, rtol=1e-9)
        R = cache[round(l, 4)]
        law = PipeLaw(sec[sand], tau_c[sand])
        Hf, sf = fold_law(R, law, k)
        Hh = max(H, Hf * 1.0001)                                   # states below the computed fold are taken at the fold
        sol = solve_pipe_law(R, law, 2.5 * Hh, k, sf["h"] * 2.5 * Hh / Hf)
        Hn = 2.5 * Hh
        while sol is not None and Hn > Hh * 1.0001:                # walk down the stable branch to the head at rest
            H2 = max(Hh, Hn * 0.93)
            sol = solve_pipe_law(R, law, H2, k, sol["h"] * H2 / Hn)
            Hn = H2
        if sol is None:
            continue
        ds = float(np.mean(np.diff(R.s_st)))
        S, Q, q = sol["S"], sol["Q"], sol["q"] / ds
        a_c = law.A / S
        W, amax = 2.0 * sec[sand]["half_width"] * a_c, sec[sand]["a_max"] * a_c
        area, per = sec[sand]["area"] * a_c ** 2, perimeter(sec[sand]) * a_c
        U = Q / area
        d50 = 1e-3 * float(sands[sand]["d50_mm"])
        dads = np.abs(np.gradient(amax, R.s_st))
        val = dict(Re_D=U * 4.0 * area / per / NU, Re_a=U * amax / NU, eps_l=U * amax / NU * dads,
                   dWds=np.abs(np.gradient(W, R.s_st)), vn_U=q / W / U, slip=np.sqrt(k * NU / G) / amax, q_nu=q / NU, a_d=amax / d50)
        inner = slice(1, -1)                                        # the two end stations are not resolved
        for name in (sand, "all"):
            for k_ in keys[:-1]:
                acc[name][k_] += list(np.asarray(val[k_])[inner])
            acc[name]["a_d_tip"].append(float(amax[-2] / d50))
    L = ["Orders of magnitude for the self-formed pipe in the 29 states at rest of Pol (2022) (all stations but the two ends)", ""]
    names = dict(Re_D="Reynolds number on the hydraulic diameter", Re_a="Reynolds number on the largest depth",
                 eps_l="Re_a * d a_max/ds (inertia / viscous stress)", dWds="dW/ds", vn_U="seepage velocity through the bed / mean velocity in the pipe",
                 slip="sqrt(permeability) / largest depth", q_nu="inflow per unit length / viscosity", a_d="largest depth / d50", a_d_tip="largest depth / d50 at the station next to the tip")
    for name in list(sands) + ["all"]:
        L.append(f"{name}  (tau_c = {tau_c.get(name, float('nan')):.2f} Pa)" if name != "all" else "all sands")
        for k_ in keys:
            L.append(f"   {names[k_]:62s} {rng(acc[name][k_])}")
    L += ["", "Inertial term / friction slope in a self-formed pipe = c q/nu with c = 4 beta chi / (3 m^2): " + ", ".join(
        f"{s_} {4.0 * momentum_coefficient(sec[s_]) * sec[s_]['chi'] / (3.0 * sec[s_]['area'] ** 2):.4f}" for s_ in sands)]
    L += ["", "Depth below which a wide pipe at the threshold stays laminar.  Between a roof and a bed that both carry tau_c the",
          "mean velocity is tau_c a / (6 rho nu) and the Reynolds number on the hydraulic diameter 2a is tau_c a^2 / (3 rho nu^2);",
          "it reaches 2000 at a = sqrt(6000 rho nu^2 / tau_c):"]
    for t in (0.3, 0.5, 0.8):
        L.append(f"   tau_c = {t:.1f} Pa: a = {1e3 * np.sqrt(6000.0 * RHO * NU ** 2 / t):.1f} mm at 20 C (nu = 1e-6);  {1e3 * np.sqrt(6000.0 * RHO * (1.3e-6) ** 2 / t):.1f} mm at 10 C (nu = 1.3e-6)")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(a.out, "paper1_assumptions.txt"), "w").write(txt + "\n")
