"""Every resting state of the guided tests of Pol (2022): measured heads and discharge vs the threshold-pipe theory.

A resting state is a run of readings in the observation sheet with constant head and constant pipe length
(>= 4 min).  For each one the model is given: the pipe length, the head at the filter face and at the exit
(both from the pressure ports, as in the thesis), the conductivity of the test (thesis, Table 3.2) and the sand
(equilibrium bed stress, repose angle).  Nothing is fitted.  Compared: total discharge, heads at the ports in
and ahead of the pipe, and the gradient across the tip.

Variants of the sink geometry (the head in all eroded parts equals the pipe head):
    base   pipe footprint 10 mm, 6 mm hole
    wide   footprint = total eroded width assumed in the thesis (25 mm; 30 mm for FS35)
    lens   wide + an eroded lens of 15 mm radius around the hole

usage:  python compare_pol_heads_all.py <path to Data/Pol_4TU> [--out ../tmp] [--tau-as-flat] [--w0 30]
"""
import argparse
import csv
import os
import warnings
import numpy as np
from pipetheory import poldata as pd
from pipetheory.aquifer import pol_box, PipeResponse
from pipetheory.coupled import solve_pipe_law, fold_law
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section

HERE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore", message="wmf image format")


def read_csv(name):
    with open(os.path.join(HERE, "data", name)) as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def variants(sand):
    wide = 0.030 if sand == "FS35" else 0.025
    return {"base": dict(), "wide": dict(pipe_width=wide), "lens": dict(pipe_width=wide, lens_radius=0.015)}


def model_state(box_kw, l, law, H, k, h_exit, law_self, law_tip):
    """Port heads [cm above the downstream level], total discharge, pipe solution; None below the fold."""
    R = PipeResponse(pol_box(**box_kw), l, with_fields=True)
    R.H_fold = fold_law(R, law, k)[0]                    # head needed to keep the bed of a pipe of this length at the threshold
    R.H_fold_self = R.H_fold if law is law_self else fold_law(R, law_self, k)[0]
    R.H_fold_tip = R.H_fold if law is law_tip else fold_law(R, law_tip, k)[0]
    sol = solve_pipe_law(R, law, H, k) if H > R.H_fold else None
    if sol is None:
        return R, None
    x, roof = R.roof_head(sol["h"], H)
    port = {n: (np.interp(xp, x, roof) + h_exit) * 100 for n, xp in pd.X_PORT.items()}
    return R, dict(port=port, Q=R.upstream_flow(sol["h"], H, k), sol=sol, x=x, roof=(roof + h_exit) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    ap.add_argument("--tau-as-flat", action="store_true")
    ap.add_argument("--w0", type=float, default=0.0, help="pipe width set by the tip, in grain diameters (0 = self-formed section everywhere)")
    ap.add_argument("--sinks", default="base,wide,lens", help="sink variants to run")
    ap.add_argument("--max-drift", type=float, default=0.15, help="discard states whose port heads drift more than this [cm]")
    a = ap.parse_args()
    sands = {r["sand"]: r for r in read_csv("pol2022_sands.csv")}
    laws, laws_self, laws_tip = {}, {}, {}
    for s, r in sands.items():
        sec = threshold_section(np.tan(np.radians(float(r["repose_deg"]))), 16, 161, 41)
        tau = float(r["tau_eq_Pa"]) / (1.0 if a.tau_as_flat else sec["a_max"] * sec["aavg_over_amax"])
        d50 = float(r["d50_mm"]) * 1e-3
        laws_self[s] = PipeLaw(sec, tau)
        laws_tip[s] = PipeLaw(sec, tau, W0=(a.w0 or 30.0) * d50)
        laws[s] = laws_tip[s] if a.w0 else laws_self[s]
    rows, profiles = [], []
    for tr in read_csv("pol2022_table3_2.csv"):
        test, sand, k = tr["test"], tr["sand"], float(tr["k_m_per_s"])
        try:
            sheet = pd.load_sheet(a.data, test); t, dP = pd.load_pressure(a.data, test)
        except FileNotFoundError:
            continue
        states = pd.resting_states(sheet)
        meas = [pd.heads(t, dP, s) for s in states]
        he_all = [m[3] for m in meas if np.isfinite(m[3])]
        for s, (h, drift, h_f, h_e) in zip(states, meas):
            if drift > a.max_drift:
                continue
            he_from = "regression"
            if not np.isfinite(h_e):
                if not he_all:
                    continue
                h_e, he_from = float(np.mean(he_all)), "test mean"
            H = (h_f - h_e) * 1e-2
            x_tip = pd.X_EXIT + s["l"]
            inpipe = [n for n in range(2, 14) if pd.X_PORT[n] <= x_tip - 0.004]
            ahead = [n for n in range(2, 14) if pd.X_PORT[n] >= x_tip + 0.006]
            for name, kw in variants(sand).items():
                if name not in a.sinks.split(","):
                    continue
                R, m = model_state(kw, s["l"], laws[sand], H, k, h_e * 1e-2, laws_self[sand], laws_tip[sand])
                both = dict(H_fold_self_cm=R.H_fold_self * 100, H_fold_tip_cm=R.H_fold_tip * 100)
                if m is None:
                    rows.append(dict(test=test, sand=sand, variant=name, l_cm=s["l"] * 100, H_net_cm=H * 100, below_fold=1,
                                     H_fold_cm=R.H_fold * 100, h_exit_cm=h_e, h_exit_from=he_from, **both))
                    continue
                pm = m["port"]
                n0 = max((n for n in range(2, 14) if pd.X_PORT[n] <= x_tip + 0.005), default=None)
                rows.append(dict(
                    test=test, sand=sand, variant=name, l_cm=s["l"] * 100, H_net_cm=H * 100, below_fold=0, H_fold_cm=R.H_fold * 100,
                    h_exit_cm=h_e, h_exit_from=he_from, Q_meas_mLmin=s["Q"] * 6e7, Q_model_mLmin=m["Q"] * 6e7,
                    Q_pipe_model_mLmin=m["sol"]["Q"][0] * 6e7,
                    bias_pipe_cm=np.mean([pm[n] - h[n] for n in inpipe]) if inpipe else np.nan,
                    bias_ahead_cm=np.mean([pm[n] - h[n] for n in ahead]) if ahead else np.nan,
                    rise_pipe_meas_cm=h[inpipe[-1]] - h[inpipe[0]] if len(inpipe) > 1 else np.nan,
                    rise_pipe_model_cm=pm[inpipe[-1]] - pm[inpipe[0]] if len(inpipe) > 1 else np.nan, **both,
                    i_tip_meas=(h[n0 + 1] - h[n0]) / 2.0 if n0 and n0 < 13 else np.nan,
                    i_tip_model=(pm[n0 + 1] - pm[n0]) / 2.0 if n0 and n0 < 13 else np.nan))
                if name == a.sinks.split(",")[-1]:
                    profiles.append((test, name, s, h, m, x_tip, H, h_e))
    keys = ["test", "sand", "variant", "l_cm", "H_net_cm", "H_fold_cm", "H_fold_self_cm", "H_fold_tip_cm", "below_fold", "h_exit_cm", "h_exit_from", "Q_meas_mLmin", "Q_model_mLmin",
            "Q_pipe_model_mLmin", "bias_pipe_cm", "bias_ahead_cm", "rise_pipe_meas_cm", "rise_pipe_model_cm", "i_tip_meas", "i_tip_model"]
    os.makedirs(a.out, exist_ok=True)
    tag = "" if not a.w0 else f"_w{a.w0:g}"
    with open(os.path.join(a.out, f"paper1_pol_states{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)

    print(f"pipe law: {'self-formed section everywhere' if not a.w0 else f'width set by the tip, {a.w0:g} d50'};"
          f" bed stress read as {'flat-bed threshold' if a.tau_as_flat else 'section average'}")
    print(f"\nPer state, sink variant '{a.sinks.split(',')[-1]}' (heads in cm; Q in mL/min)")
    print(f"{'test':9s} {'l':>5s} {'H_net':>6s} {'h_exit':>6s} | {'Q meas':>7s} {'model':>6s} | {'pipe rise meas':>14s} {'model':>6s} | {'bias pipe':>9s} {'ahead':>6s} | {'i_tip meas':>10s} {'model':>6s}")
    for r in rows:
        if r["variant"] != a.sinks.split(",")[-1]:
            continue
        if r["below_fold"]:
            print(f"{r['test']:9s} {r['l_cm']:5.1f} {r['H_net_cm']:6.2f}   below the fold ({r['H_fold_cm']:.2f} cm): no threshold solution at this head")
            continue
        print(f"{r['test']:9s} {r['l_cm']:5.1f} {r['H_net_cm']:6.2f} {r['h_exit_cm']:6.2f}{'*' if r['h_exit_from'] != 'regression' else ' '}| {r['Q_meas_mLmin']:7.1f} {r['Q_model_mLmin']:6.1f} |"
              f" {r['rise_pipe_meas_cm']:14.2f} {r['rise_pipe_model_cm']:6.2f} | {r['bias_pipe_cm']:+9.2f} {r['bias_ahead_cm']:+6.2f} | {r['i_tip_meas']:10.2f} {r['i_tip_model']:6.2f}")
    print("  (* exit head taken as the mean of the other states of the same test: the tip had not passed P5)")

    print("\nSummary over states (mean +/- s.d.); only states whose exit head comes from the regression")
    print(f"{'variant':8s} {'n':>3s} {'below fold':>10s} | {'H_fold/H_net':>13s} | {'Q model/meas':>13s} | {'pipe rise model/meas':>20s} | {'bias in pipe [cm]':>17s} | {'bias ahead [cm]':>15s} | {'tip gradient model/meas':>23s}")
    summary = {}
    for name in a.sinks.split(","):
        rr = [r for r in rows if r["variant"] == name and r.get("h_exit_from", "regression") == "regression"]
        ok = [r for r in rr if not r["below_fold"]]
        f = lambda v: f"{np.nanmean(v):.2f} +/- {np.nanstd(v, ddof=1):.2f}"
        q = [r["Q_model_mLmin"] / r["Q_meas_mLmin"] for r in ok]
        rise = [r["rise_pipe_model_cm"] / r["rise_pipe_meas_cm"] for r in ok if r["rise_pipe_meas_cm"] > 0.2]
        it = [r["i_tip_model"] / r["i_tip_meas"] for r in ok if r["i_tip_meas"] > 0.1]
        summary[name] = (np.nanmean(q), np.nanmean(rise), np.nanmean([r["bias_pipe_cm"] for r in ok]), np.nanmean([r["bias_ahead_cm"] for r in ok]), np.nanmean(it))
        print(f"{name:8s} {len(rr):3d} {len(rr) - len(ok):10d} | {f([r['H_fold_cm'] / r['H_net_cm'] for r in rr]):>13s} | {f(q):>13s} | {f(rise):>20s} | {f([r['bias_pipe_cm'] for r in ok]):>17s} | {f([r['bias_ahead_cm'] for r in ok]):>15s} | {f(it):>23s}")

    last = a.sinks.split(",")[-1]
    eq = [r for r in rows if r["variant"] == last]
    print(f"\nHead needed to keep the bed of the pipe at the threshold (fold) / head at which the pipe came to rest; sink '{last}', all {len(eq)} states")
    for key, lab in (("H_fold_self_cm", "self-formed section"), ("H_fold_tip_cm", f"width fixed by the tip, {a.w0 or 30:g} d50")):
        x = np.array([r[key] / r["H_net_cm"] for r in eq]); sd = np.array([r["sand"] for r in eq]); ll = np.array([r["l_cm"] for r in eq])
        print(f"   {lab:32s} {x.mean():.2f} +/- {x.std(ddof=1):.2f}   " + "  ".join(f"{sn} {x[sd == sn].mean():.2f}" for sn in ("B25", "FPH", "FS35"))
              + f"   l < 12 cm: {x[ll < 12].mean():.2f}, 12-20 cm: {x[(ll >= 12) & (ll < 20)].mean():.2f}, > 20 cm: {x[ll >= 20].mean():.2f}")
    try:
        import plotstyle
        plt = plotstyle.apply()
    except Exception as e:
        print("no figure:", e); return
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    mk = {"FPH": "s", "B25": "o", "FS35": "^"}
    for sn in mk:
        rr = [r for r in eq if r["sand"] == sn]
        ax.plot([r["H_net_cm"] for r in rr], [r["H_fold_self_cm"] for r in rr], mk[sn], color=plotstyle.BLUE, mfc="none", ms=6, label=f"{sn}: self-formed section")
        ax.plot([r["H_net_cm"] for r in rr], [r["H_fold_tip_cm"] for r in rr], mk[sn], color=plotstyle.ORANGE, ms=6, label=f"{sn}: width fixed by the tip ({a.w0 or 30:g} d50)")
    ax.plot([0, 8], [0, 8], color=plotstyle.INK_2, lw=1.0)
    ax.set_xlim(2.5, 7.5); ax.set_ylim(2.5, 7.5); ax.set_aspect("equal")
    ax.set_xlabel("head at which the pipe came to rest [cm]"); ax.set_ylabel("head needed by the bed criterion [cm]")
    ax.set_title("Resting pipes sit on the fold (no fitted parameter)"); ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"paper1_pol_equilibrium{tag}.png"))
    print("figure:", os.path.join(a.out, f"paper1_pol_equilibrium{tag}.png"))
    pick = [p for p in profiles if p[2]["l"] > 0.14]
    chosen = []
    for want in ("B25_245", "B25_243", "FPH_235", "FS35_239"):
        c = [p for p in pick if p[0] == want]
        if c:
            chosen.append(c[-1] if want != "FPH_235" else c[0])
    if not chosen:
        return
    fig, axs = plt.subplots(1, len(chosen), figsize=(3.6 * len(chosen), 3.5), squeeze=False)
    for ax, (test, name, s, h, m, x_tip, H, h_e) in zip(axs[0], chosen):
        ax.plot(m["x"] * 100, m["roof"], color=plotstyle.BLUE, label="theory")
        ports = sorted(pd.X_PORT)
        ax.plot([pd.X_PORT[n] * 100 for n in ports], [h[n] for n in ports], "o", color=plotstyle.ORANGE, ms=5, label="measured")
        ax.axvline(x_tip * 100, color=plotstyle.INK_2, lw=0.8, ls="--")
        ax.set_xlim(12, 48); ax.set_ylim(0, None)
        ax.set_xlabel("x [cm]"); ax.set_title(f"{test.replace('_', '-')}, l = {s['l'] * 100:.1f} cm")
    axs[0][0].set_ylabel("head above downstream level [cm]"); axs[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(a.out, f"paper1_pol_states{tag}.png"); fig.savefig(out); print("\nfigure:", out)


if __name__ == "__main__":
    main()
