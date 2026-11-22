"""Threshold pipe with seepage through its bed against the measurements of Pol (2022).

A. Section by section (no coupled model): with the measured gradient and the computed inflow at each scanned section, the
   section with seepage gives the width and the largest depth.  One constant per sand, the threshold stress without
   seepage tau_c0, is set so that the median depth of the pipes through the bed is reproduced; the widths, and the depths
   and widths of the growing pipes, then follow with nothing adjusted.
B. Coupled to the sand: the 29 states at rest of paper1_pol_states.csv recomputed with the local law of
   pipetheory/seepage.py; head at the fold, head rise along the pipe, discharge.

Both are done for several values of eps, the fraction of the gradient under the centre of the strip that acts on the
grains of the bed surface (1: full seepage force; 0: none).
Needs tmp/paper1_laser_vs_theory.csv and tmp/paper1_pol_states.csv.

    python run_seepage_coupled.py --out ../tmp          (about 4 minutes)
"""
import argparse
import csv
import os
import numpy as np

from pipetheory import poldata as pd
from pipetheory.aquifer import PipeResponse, pol_box
from pipetheory.coupled import solve_pipe_law, fold_law
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section
from pipetheory.seepage import SeepageLaw, solve_pipe_seepage, fold_seepage

HERE = os.path.dirname(os.path.abspath(__file__))
RHO, G = 1000.0, 9.81
RHO_S = {"FPH": 2610.0, "B25": 2650.0, "FS35": 2650.0}
D50 = {"FPH": 0.185e-3, "B25": 0.228e-3, "FS35": 0.422e-3}
I_F = {s: 0.60 * (RHO_S[s] / RHO - 1.0) for s in RHO_S}
EPS_LOCAL = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5708)
EPS_COUPLED = (0.0, 0.25, 0.5, 1.0)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def stat(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return "n=0"
    return f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}, {np.percentile(v, 75):.2f}] n={v.size}"


def local_predictions(rows, mu, tau_c0, c_sigma, sand):
    out = []
    for r in rows:
        law = SeepageLaw(mu, tau_c0, r["k"], I_F[sand], c_sigma, nu=r["nu"])
        s, W, a, Q = law.section_at(r["S"], r["q"])
        out.append((float(s), float(W), float(a), float(Q)))
    return np.array(out)


def calibrate(rows, mu, c_sigma, sand):
    """tau_c0 for which the median of predicted / measured largest depth over the given sections is one."""
    am = np.array([r["amax"] for r in rows])
    lo, hi = 0.2, 6.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        ratio = np.median(local_predictions(rows, mu, mid, c_sigma, sand)[:, 2] / am)
        lo, hi = (mid, hi) if ratio < 1.0 else (lo, mid)
    return np.sqrt(lo * hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../tmp")
    a = ap.parse_args()
    sands = {r["sand"]: r for r in read_csv(os.path.join(HERE, "data", "pol2022_sands.csv"))}
    table = {r["test"]: r for r in read_csv(os.path.join(HERE, "data", "pol2022_table3_2.csv"))}
    mu = {s: float(np.tan(np.radians(float(sands[s]["repose_deg"])))) for s in sands}
    sec = {s: threshold_section(mu[s]) for s in sands}
    tau_c = {s: float(sands[s]["tau_eq_Pa"]) / (sec[s]["a_max"] * sec[s]["aavg_over_amax"]) for s in sands}

    secs = []
    for r in read_csv(os.path.join(a.out, "paper1_laser_vs_theory.csv")):
        d = dict(test=r["test"], sand=r["sand"], kind=r["kind"], S=fnum(r["S"]), q=fnum(r["q_over_k"]) * fnum(r["k"]), k=fnum(r["k"]),
                 nu=fnum(r["nu"]), amax=fnum(r["amax"]), W_er=fnum(r["W_er"]), tau=fnum(r["tau_max"]), Qx=fnum(r["Q_pipe_x"]))
        if all(np.isfinite([d["S"], d["q"], d["k"], d["amax"], d["W_er"], d["tau"]])) and 0 < d["tau"] < 3 and d["S"] > 0.01:
            secs.append(d)

    L = ["A. Section by section: measured gradient and computed inflow -> section with seepage",
         "   eps = fraction of the gradient under the centre of the strip that acts on the grains of the bed surface",
         "   (eps = 1: full seepage force, gradient at the centre; pi/2: full force, mean gradient; eps = 0: no seepage)",
         "   tau_c0 is set per sand on the pipes through the bed (median depth reproduced); everything else follows", ""]
    tau_c0 = {}
    for eps in EPS_LOCAL:
        L.append(f"   eps = {eps:.2f}")
        for sand in ("FPH", "B25", "FS35"):
            thr = [r for r in secs if r["sand"] == sand and r["kind"] == "complete"]
            if len(thr) < 5:
                continue
            t0 = calibrate(thr, mu[sand], eps, sand)
            tau_c0[(sand, eps)] = t0
            theta = t0 / ((RHO_S[sand] - RHO) * G * D50[sand])
            p = local_predictions(thr, mu[sand], t0, eps, sand)
            am, We, Qx = (np.array([r[k_] for r in thr]) for k_ in ("amax", "W_er", "Qx"))
            L.append(f"   {sand:5s} tau_c0 = {t0:.2f} Pa (Shields {theta:.2f}) | through the bed: sigma_eff {stat(p[:, 0])} | "
                     f"width pred/eroded {stat(p[:, 1] / We)} | discharge pred/pipe {stat(p[:, 3] / Qx)}")
            gr = [r for r in secs if r["sand"] == sand and r["kind"] == "partial"]
            if gr:
                p = local_predictions(gr, mu[sand], t0, eps, sand)
                am, We, Qx = (np.array([r[k_] for r in gr]) for k_ in ("amax", "W_er", "Qx"))
                L.append(f"         growing ({len(gr)} sections): sigma_eff {stat(p[:, 0])} | depth pred/meas {stat(p[:, 2] / am)} | "
                         f"width pred/eroded {stat(p[:, 1] / We)} | discharge pred/pipe {stat(p[:, 3] / Qx)}")
        L.append("")

    # ------------------------------------------------------------------ B. coupled problem, 29 states at rest
    states = [r for r in read_csv(os.path.join(a.out, "paper1_pol_states.csv")) if r["variant"] == "base"]
    L += ["B. Coupled to the sand: the states at rest recomputed with the local law with seepage",
          "   per eps: head at the fold / head at rest; and at the head at rest: head rise along the pipe model/measured,",
          "   discharge model/measured, sigma_eff and width at mid-pipe, largest depth relative to eps = 0", ""]
    res = {eps: [] for eps in EPS_COUPLED}
    cache = {}
    for st in states:
        test, sand = st["test"], st["sand"]
        l, H, k = 0.01 * fnum(st["l_cm"]), 0.01 * fnum(st["H_net_cm"]), float(table[test]["k_m_per_s"])
        key = round(l, 4)
        if key not in cache:
            cache[key] = PipeResponse(pol_box(), l, rtol=1e-9)
        R = cache[key]
        # measured rise of the head along the pipe: between the first and the last port that the pipe has passed
        x_tip = pd.X_EXIT + l
        inpipe = [pd.X_PORT[n] for n in range(2, 14) if pd.X_PORT[n] <= x_tip - 0.004]
        rise_meas, Q_meas = 0.01 * fnum(st["rise_pipe_meas_cm"]), fnum(st["Q_meas_mLmin"]) / 6e7
        if len(inpipe) < 2 or not rise_meas > 0.002:
            rise_meas = np.nan
        a_ref = None
        for eps in EPS_COUPLED:
            if (sand, eps) not in tau_c0:
                continue
            law = SeepageLaw(mu[sand], tau_c0[(sand, eps)], k, I_F[sand], eps if eps > 0 else 1.0)
            lawd = PipeLaw(sec[sand], tau_c0[(sand, eps)])
            Hd, sd = fold_law(R, lawd, k)
            Hs = 2.5 * max(Hd, H)
            sol = solve_pipe_law(R, lawd, Hs, k, sd["h"] * Hs / Hd)
            h, ok = (sol["h"], True) if sol else (None, False)
            for strength in ((0.25, 0.5, 0.75, 1.0) if eps > 0 else (0.0,)):
                law.strength = strength
                ss = solve_pipe_seepage(R, law, Hs, h) if ok else None
                if ss is None:
                    ok = False
                    break
                h = ss["h"]
            Hf, sf = fold_seepage(R, law, Hs, h) if ok else (None, None)
            row = dict(test=test, sand=sand, l=l, H=H, ratio=Hf / H if Hf else np.nan)
            if Hf and H > Hf:
                sH, Hh, hh = None, Hs, h
                while Hh > H * 1.0001:                          # walk down to the head at rest
                    Hn = max(H, Hh * 0.93)
                    sH = solve_pipe_seepage(R, law, Hn, hh * Hn / Hh)
                    if sH is None:
                        break
                    Hh, hh = Hn, sH["h"]
                if sH is not None:
                    m = R.n // 2
                    if eps == 0.0:
                        a_ref = sH["a_max"].copy()
                    hp = np.interp([inpipe[0], inpipe[-1]], R.x_st, sH["h"]) if len(inpipe) > 1 else [np.nan, np.nan]
                    row.update(rise=float(hp[1] - hp[0]) / rise_meas, Q=float(sH["Q"][0]) / Q_meas, sig=float(sH["sigma"][m]), W=float(sH["W"][m]),
                               adep=float(np.median(sH["a_max"][2:-2] / a_ref[2:-2])) if a_ref is not None else np.nan)
            res[eps].append(row)
    paper = np.array([fnum(st["H_fold_self_cm"]) / fnum(st["H_net_cm"]) for st in states])
    L.append(f"   reference, no seepage with the threshold stress used in the paper: head at the fold / head at rest = {paper.mean():.2f} +- {paper.std():.2f} (n = {paper.size})")
    L.append("   eps   tau_c0 (FPH, B25, FS35) | fold/rest (mean +- sd; n; FPH, B25, FS35) | states above the fold | rise model/meas | Q model/meas | sigma_eff | width mm | depth / depth(eps=0)")
    for eps in EPS_COUPLED:
        rr = res[eps]
        col = lambda k_: np.array([r.get(k_, np.nan) for r in rr], float)
        ms = lambda v: f"{np.nanmean(v):.2f} +- {np.nanstd(v):.2f}"
        by = ", ".join(f"{np.nanmean([r['ratio'] for r in rr if r['sand'] == s]):.2f}" for s in ("FPH", "B25", "FS35"))
        L.append(f"   {eps:.2f}  {', '.join('%.2f' % tau_c0[(s, eps)] for s in ('FPH', 'B25', 'FS35') if (s, eps) in tau_c0)} | {ms(col('ratio'))} (n = {np.isfinite(col('ratio')).sum()}; {by}) | "
                 f"{np.isfinite(col('rise')).sum()} | {ms(col('rise'))} | {ms(col('Q'))} | {ms(col('sig'))} | {1e3 * np.nanmean(col('W')):.1f} +- {1e3 * np.nanstd(col('W')):.1f} | {ms(col('adep'))}")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(a.out, "paper1_seepage_coupled.txt"), "w").write(txt + "\n")
    with open(os.path.join(a.out, "paper1_seepage_coupled.csv"), "w") as f:
        f.write("eps,test,sand,l_m,H_m,fold_over_rest,rise_model_over_meas,Q_model_over_meas,sigma_mid,W_mid_m,depth_over_eps0\n")
        for eps in EPS_COUPLED:
            for r in res[eps]:
                f.write(f"{eps},{r['test']},{r['sand']},{r['l']:.4f},{r['H']:.5f}," + ",".join(f"{r.get(k_, float('nan')):.5g}" for k_ in ("ratio", "rise", "Q", "sig", "W", "adep")) + "\n")


if __name__ == "__main__":
    main()
