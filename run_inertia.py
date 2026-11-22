"""Weak inertia in a threshold pipe (manuscript section 4 and its use in section 5).

1. Wide pipe: exact first-order coefficients (pipetheory.inertia.wide_pipe_coefficients) against the similarity solution
   of the Navier-Stokes equations for a uniform gap with injection through one wall; law of the wide threshold pipe.
2. Self-formed pipe in the shallow limit: exact first order (pipetheory.inertia.shallow_self_formed) against the
   momentum-integral closure, for several distributions of the inflow across the bed.
3. Self-formed section: S = S_f (1 + c q/nu), c = 4 beta chi/(3 m^2) (momentum integral, profile of the uniform flow).
   The 29 states at rest of Pol (2022) are recomputed without inertia, with this term, and with the term multiplied by
   the ratio exact/closure of the shallow limit for an inflow with the edge singularity of the actual section.
4. The coupled problem has one more dimensionless number, J = c k H/nu.  Head at the fold over its creeping value as a
   function of J0 = c k H0/nu (H0: head at the fold without inertia), for three pipe lengths in the small-scale box.
usage: python run_inertia.py [--out ../tmp]        (needs tmp/paper1_pol_states.csv; about 10 minutes)
"""
import argparse
import csv
import os
import warnings
import numpy as np
from pipetheory import poldata as pd
from pipetheory.aquifer import PipeResponse, pol_box
from pipetheory.coupled import solve_pipe_law, fold_law
from pipetheory.inertia import (wide_pipe_coefficients, wide_threshold_pipe, similarity_one_wall, self_formed_coefficient,
                                 shallow_self_formed, edge_exponent)
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section, momentum_coefficient
from pipetheory.seepage import SeepageLaw, solve_pipe_seepage, fold_seepage, _ds

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
NU = 1.0e-6


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def ms(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return f"{v.mean():.2f} +- {v.std():.2f} (n = {v.size})" if v.size else "n = 0"


def fold_with_inertia(R, sec_, mu_, tau_c_, k, b, shape_, H_ref=None):
    """Head at the fold without and with the inertial term (momentum coefficient b), and the two solutions there."""
    lawd = PipeLaw(sec_, tau_c_)
    Hd, sd = fold_law(R, lawd, k)
    Hs = 2.5 * max(Hd, H_ref or Hd)
    sol = solve_pipe_law(R, lawd, Hs, k, sd["h"] * Hs / Hd)
    law = SeepageLaw(mu_, tau_c_, k, strength=0.0, shape=shape_)
    h, ok = sol["h"], True
    for f in (0.25, 0.5, 0.75, 1.0):                             # switch the term on in steps
        law.inertia = f * b
        ss = solve_pipe_seepage(R, law, Hs, h)
        if ss is None:
            ok = False
            break
        h = ss["h"]
    Hf, sf = fold_seepage(R, law, Hs, h) if ok else (None, None)
    return Hd, sd, Hf, sf, law, Hs, h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../tmp")
    a = ap.parse_args()
    L = []

    # ------------------------------------------------------------------ 1. wide pipe
    c = wide_pipe_coefficients()
    w = wide_threshold_pipe()
    L += ["1. Wide pipe, first order in the inertia (exact; x in the direction of the flow, V = dQ/dx through the bed)",
          f"   head gradient  S = 12 nu Q/(g h^3) + ({c['S_V']}) Q V/(g h^2) + ({c['S_h']}) Q^2 h'/(g h^3)",
          f"   bed stress     tau_b/rho = 6 nu Q/h^2 + ({c['bed_V']}) Q V/h + ({c['bed_h']}) Q^2 h'/h^2",
          f"   roof stress    tau_r/rho = 6 nu Q/h^2 + ({c['roof_V']}) Q V/h + ({c['roof_h']}) Q^2 h'/h^2",
          f"   momentum balance, h S-terms - (bed + roof + d(6/5 Q^2/h)/dx): {c['S_V'] - (c['bed_V'] + c['roof_V'] + 2 * c['beta0'])}, "
          f"{c['S_h'] - (c['bed_h'] + c['roof_h'] - c['beta0'])}",
          f"   wide threshold pipe (h ~ Q^(1/2)):  h = h0 [1 + ({w['depth']}) R],  S = S0 [1 + ({w['slope']}) R],  R = V h0/nu",
          f"      of which at fixed depth ({w['slope_fixed_depth']}) R;  momentum integral with the parabolic profile: ({w['slope_closure']}) R;"
          f"  exact / closure = {w['ratio']} = {float(w['ratio']):.3f}",
          "", "   similarity solution of the Navier-Stokes equations, uniform gap, injection through the bed (R = V h/nu):",
          "      R     gradient/creeping  (first order)   bed stress  (first order)   roof stress  (first order)"]
    sim = []
    for R_ in (0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0):
        K, b_, r_ = similarity_one_wall(R_)
        sim.append((R_, K, b_, r_))
        L.append(f"   {R_:5.2f}        {K:.4f}        ({1 + float(c['S_V']) / 12 * R_:.4f})        {b_:.4f}     ({1 + float(c['bed_V']) / 6 * R_:.4f})"
                 f"        {r_:.4f}     ({1 + float(c['roof_V']) / 6 * R_:.4f})")
    L.append(f"   slopes at R = 0.01: {(sim[0][1] - 1) / 0.01:.5f}, {(sim[0][2] - 1) / 0.01:.5f}, {(sim[0][3] - 1) / 0.01:.5f};"
             f"  exact {float(c['S_V']) / 12:.5f}, {float(c['bed_V']) / 6:.5f}, {float(c['roof_V']) / 6:.5f}")

    # ------------------------------------------------------------------ 2. self-formed pipe, shallow limit
    L += ["", "2. Self-formed pipe in the shallow limit, first order: S = S0 [1 + s_hat mu q/(6 nu)], a_max = a_c [1 + d_hat mu q/(6 nu)]",
          "   inflow V ~ (1 - (2y/W)^2)^(-p): p = 1/2 seepage into a strip, p = 0 uniform; momentum-integral closure s_hat = 8/25",
          "      p      s_hat   exact/closure   d_hat    width change   momentum identity - 16/15"]
    for pw in (0.5, 0.45, 0.4, 0.368, 0.3, 0.2, 0.1, 0.0):
        r = shallow_self_formed("strip" if pw == 0.5 else pw, 1600, 200)
        L.append(f"   {pw:6.3f}   {r['s_hat']:.4f}      {r['ratio']:.4f}      {r['d_hat']:+.4f}     {r['width']:+.4f}        {r['check'] - 16.0 / 15.0:+.1e}")

    # ------------------------------------------------------------------ 3. self-formed section, 29 states
    sands = {r["sand"]: r for r in read_csv(os.path.join(HERE, "data", "pol2022_sands.csv"))}
    table = {r["test"]: r for r in read_csv(os.path.join(HERE, "data", "pol2022_table3_2.csv"))}
    mu = {s: float(np.tan(np.radians(float(sands[s]["repose_deg"])))) for s in sands}
    sec = {s: threshold_section(mu[s]) for s in sands}
    beta = {s: momentum_coefficient(sec[s]) for s in sands}
    cc = {s: self_formed_coefficient(sec[s], beta[s]) for s in sands}
    shape = {s: sec[s]["area"] / (2.0 * sec[s]["half_width"] * sec[s]["a_max"]) for s in sands}
    tau_c = {s: float(sands[s]["tau_eq_Pa"]) / (sec[s]["a_max"] * sec[s]["aavg_over_amax"]) for s in sands}
    lam = {s: shallow_self_formed(edge_exponent(mu[s]), 1600, 200)["ratio"] for s in sands}   # exact/closure, edge exponent of the section
    states = [r for r in read_csv(os.path.join(a.out, "paper1_pol_states.csv")) if r["variant"] == "base"]
    rows, cache = [], {}
    for st in states:
        test, sand = st["test"], st["sand"]
        l, H, k = 0.01 * float(st["l_cm"]), 0.01 * float(st["H_net_cm"]), float(table[test]["k_m_per_s"])
        if round(l, 4) not in cache:
            cache[round(l, 4)] = PipeResponse(pol_box(), l, rtol=1e-9)
        R = cache[round(l, 4)]
        ds = _ds(R)
        inpipe = [pd.X_PORT[n] for n in range(2, 14) if pd.X_PORT[n] <= pd.X_EXIT + l - 0.004]
        try:
            rise_meas = 0.01 * float(st["rise_pipe_meas_cm"])
        except ValueError:
            rise_meas = np.nan
        if len(inpipe) < 2 or not rise_meas > 0.002:
            rise_meas = np.nan
        Q_meas = float(st["Q_meas_mLmin"]) / 6e7
        row = dict(test=test, sand=sand, l=l, H=H, k=k)
        for tag, fac in (("0", 0.0), ("i", 1.0), ("w", lam[sand])):
            if fac == 0.0:
                lawd = PipeLaw(sec[sand], tau_c[sand])
                Hf, sf = fold_law(R, lawd, k)
                Hs = 2.5 * max(Hf, H)
                h = solve_pipe_law(R, lawd, Hs, k, sf["h"] * Hs / Hf)["h"]
                law = SeepageLaw(mu[sand], tau_c[sand], k, strength=0.0, shape=shape[sand])
                row["J0"] = cc[sand] * k * Hf / NU
            else:
                _, _, Hf, sf, law, Hs, h = fold_with_inertia(R, sec[sand], mu[sand], tau_c[sand], k, fac * beta[sand], shape[sand], H)
            row["fold_" + tag] = Hf / H if Hf else np.nan
            if not Hf:
                continue
            row["Hf_" + tag] = Hf
            if fac:
                row["share_fold_" + tag] = float(np.sum(sf["S_inertia"] * ds) / sf["h"][-1])
            if H > Hf:
                sH, Hh, hh = None, Hs, h
                while Hh > H * 1.0001:                           # walk down the branch to the head at rest
                    Hn = max(H, Hh * 0.93)
                    sH = solve_pipe_seepage(R, law, Hn, hh * Hn / Hh)
                    if sH is None:
                        break
                    Hh, hh = Hn, sH["h"]
                if sH is not None:
                    hp = np.interp([inpipe[0], inpipe[-1]], R.x_st, sH["h"]) if len(inpipe) > 1 else [np.nan, np.nan]
                    row["rise_" + tag] = float(hp[1] - hp[0]) / rise_meas
                    row["Q_" + tag] = float(sH["Q"][0]) / Q_meas
                    if fac:
                        row["share_" + tag] = float(np.sum(sH["S_inertia"] * ds) / sH["h"][-1])
                        row["local_" + tag] = float(np.max(sH["S_inertia"][1:-1] / (sH["S"][1:-1] - sH["S_inertia"][1:-1])))
                    else:
                        m = slice(1, -1)                         # injection Reynolds number on the mean depth, (q/W) a_mean/nu
                        row["R_inj"] = float(np.median((sH["q"][m] / sH["W"][m]) * shape[sand] * sH["a_max"][m] / NU))
                        row["q_nu"] = float(np.max(sH["q"][m]) / NU)
        rows.append(row)
        print(test, f"l = {l:.3f}", " ".join(f"{k_}={v:.3f}" for k_, v in row.items() if isinstance(v, float) and k_ not in ("l", "H", "k")), flush=True)

    def col(key, sand=None):
        return np.array([r.get(key, np.nan) for r in rows if sand in (None, r["sand"])], float)

    L += ["", "3. Self-formed section: S = S_f (1 + c q/nu); 29 states at rest of Pol (2022), no seepage force",
          "   beta: " + ", ".join(f"{s} {beta[s]:.3f}" for s in sands) + ";   c = 4 beta chi/(3 m^2): " + ", ".join(f"{s} {cc[s]:.4f}" for s in sands),
          "   edge exponent p = 1 - pi/(2 (pi - arctan mu)): " + ", ".join(f"{s} {edge_exponent(mu[s]):.3f}" for s in sands)
          + ";   factor (shallow limit, exact/closure at that p): " + ", ".join(f"{s} {lam[s]:.3f}" for s in sands),
          "", "                                                        all                    " + "   ".join(f"{s:>5s}" for s in sands)]
    for name, key in (("J0 = c k H0/nu (H0: head at the fold, creeping)", "J0"), ("largest q/nu along the pipe at rest", "q_nu"),
                      ("injection Reynolds number (q/W) a_mean/nu, median along the pipe", "R_inj"),
                      ("head at the fold / head at rest, creeping", "fold_0"), ("   with the inertial term", "fold_i"), ("   with the term times the shallow-limit factor", "fold_w"),
                      ("head rise along the pipe model/meas, creeping", "rise_0"), ("   with the inertial term", "rise_i"), ("   with the term times the shallow-limit factor", "rise_w"),
                      ("discharge model/meas, creeping", "Q_0"), ("   with the inertial term", "Q_i"),
                      ("share of inertia in the head of the tip, at rest", "share_i"), ("   at the fold", "share_fold_i"),
                      ("largest inertial term / friction slope at rest", "local_i")):
        L.append(f"   {name:64s} {ms(col(key)):26s} " + "   ".join(f"{np.nanmean(col(key, s)):5.2f}" for s in sands))
    for name, key in (("head at the fold with / without inertia", "Hf_i"), ("   with the term times the shallow-limit factor", "Hf_w")):
        L.append(f"   {name:64s} {ms(col(key) / col('Hf_0')):26s} " + "   ".join(f"{np.nanmean(col(key, s) / col('Hf_0', s)):5.2f}" for s in sands))

    # ------------------------------------------------------------------ 3. head at the fold against J0
    L += ["", "4. Head at the fold / its creeping value against J0 = c k H0/nu (sand with d50 = 0.228 mm; k varied)",
          "   l (m)   " + "  ".join(f"{'J0':>6s} {'F':>6s}" for _ in range(3))]
    curve = []
    s_ = "B25"
    for l in (0.10, 0.15, 0.20):
        R = cache.get(round(l, 4)) or PipeResponse(pol_box(), l, rtol=1e-9)
        out = []
        for k in np.logspace(-4.7, -2.4, 10):
            Hd, sd, Hf, sf, law, Hs, h = fold_with_inertia(R, sec[s_], mu[s_], tau_c[s_], k, beta[s_], shape[s_])
            if Hf:
                out.append((cc[s_] * k * Hd / NU, Hf / Hd))
                curve.append((l, k, out[-1][0], out[-1][1]))
        L.append(f"   {l:.2f}   " + "  ".join(f"{j:6.3f} {f:6.3f}" for j, f in out))
        print(L[-1], flush=True)
    cv = np.array(curve)
    small = cv[cv[:, 2] < 0.3]
    L.append(f"   initial slope (J0 < 0.3): F - 1 = {np.sum((small[:, 3] - 1) * small[:, 2]) / np.sum(small[:, 2] ** 2):.3f} J0")
    for lo, hi in ((0.0, 0.3), (0.3, 1.0), (1.0, 5.0)):
        sel = cv[(cv[:, 2] >= lo) & (cv[:, 2] < hi)]
        if len(sel):
            L.append(f"   J0 in [{lo}, {hi}): (F - 1)/J0 = {np.mean((sel[:, 3] - 1) / sel[:, 2]):.3f} (range {np.min((sel[:, 3] - 1) / sel[:, 2]):.3f}-{np.max((sel[:, 3] - 1) / sel[:, 2]):.3f}, n = {len(sel)})")

    txt = "\n".join(L)
    print(txt)
    open(os.path.join(a.out, "paper1_inertia.txt"), "w").write(txt + "\n")
    with open(os.path.join(a.out, "paper1_inertia.csv"), "w") as f:
        keys = ["test", "sand", "l", "H", "k", "J0", "q_nu", "R_inj", "fold_0", "fold_i", "fold_w", "Hf_0", "Hf_i", "Hf_w", "rise_0", "rise_i", "rise_w",
                "Q_0", "Q_i", "share_i", "share_fold_i", "local_i"]
        f.write(",".join(keys) + "\n")
        for r_ in rows:
            f.write(",".join(str(r_.get(k_, "")) if not isinstance(r_.get(k_), float) else f"{r_[k_]:.5g}" for k_ in keys) + "\n")
    np.savetxt(os.path.join(a.out, "paper1_inertia_curve.csv"), cv, delimiter=",", header="l_m,k_m_per_s,J0,F", comments="", fmt="%.6g")
    np.savetxt(os.path.join(a.out, "paper1_inertia_similarity.csv"), np.array([(R_,) + similarity_one_wall(R_) for R_ in np.linspace(0.0, 5.0, 26)]),
               delimiter=",", header="R,gradient,bed,roof", comments="", fmt="%.6g")
