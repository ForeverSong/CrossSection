"""What the measured pipe gradients say about the pipe cross-section, from small to medium scale.

For every state in which a pipe was at rest and both the discharge Q and the mean gradient S in the pipe were
measured (Pol 2022, small scale, from the 4TU data; Van Beek et al. 2015, Table 5, small and medium scale), assume
only that the bed is at the threshold of motion.  Then
    depth          a = A / S,   A = 2 tau_c / (rho g)
    self-formed?   tau_app / tau_c = (Q_rep S^3 / Lambda)^(1/4)   should be 1 if the section is the self-formed one
    width          W = 12 nu Q_rep S^2 / (g A^3)   if the bed is flat and wider than the self-formed section
    mean seepage gradient into the pipe bed   i_bed = Q / (k l W)
Q_rep = phi * Q is the discharge representative of the pipe (phi = 0.5; the measured Q is the total at the exit).
tau_c: for Pol's sands the measured equilibrium stress / 0.74 (section average -> flat bed); for the other sands the
same Shields numbers interpolated in grain size (power law through Pol's three sands).

usage:  python pipe_width_data.py [path to Data/Pol_4TU] [--phi 0.5] [--out ../tmp]
"""
import argparse
import csv
import os
import warnings
import numpy as np
from pipetheory import RHO_W, G, NU
from pipetheory.section import threshold_section

HERE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="wmf image format")
DRHO = 1650.0


def read_csv(name):
    with open(os.path.join(HERE, "data", name)) as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def shields_fit(sands):
    d = np.array([float(r["d50_mm"]) for r in sands.values()]) * 1e-3
    th = np.array([float(r["tau_eq_Pa"]) for r in sands.values()]) / (DRHO * G * d)
    m, c = np.polyfit(np.log(d), np.log(th), 1)
    return lambda dd: float(np.exp(c) * dd ** m), m


def analyse(st, sec, phi):
    """st: dict with d50 [m], tau_eq [Pa], k, l, Q, S."""
    fac = sec["a_max"] * sec["aavg_over_amax"]
    tau_c = st["tau_eq"] / fac
    A = 2.0 * tau_c / (RHO_W * G)
    Lam = sec["chi"] * G * A ** 4 / NU
    Qr = phi * st["Q"]
    a = A / st["S"]
    W_thr = 2.0 * sec["half_width"] * a
    W_flat = 12.0 * NU * Qr * st["S"] ** 2 / (G * A ** 3)
    W = max(W_flat, W_thr)
    return dict(tau_c=tau_c, a_over_d=a / st["d50"], tau_app=(Qr * st["S"] ** 3 / Lam) ** 0.25, W_thr_over_d=W_thr / st["d50"],
                W_flat_over_d=W_flat / st["d50"], W_mm=W * 1e3, W_over_a=W / a,
                i_bed=st["Q"] / (st["k"] * st["l"] * W) if st["k"] else np.nan,
                Pi=A ** 2 * G / (NU * st["k"]) if st["k"] else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=None)
    ap.add_argument("--phi", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    a = ap.parse_args()
    sands = {r["sand"]: r for r in read_csv("pol2022_sands.csv")}
    theta, m = shields_fit(sands)
    sec = threshold_section(0.77, 16, 161, 41)
    print(f"Shields number from Pol's three sands: theta = {theta(2e-4):.3f} at 0.2 mm, exponent {m:.2f} in grain size;  phi = {a.phi}")
    states = []
    for r in read_csv("vanbeek2015_table5.csv"):
        d = float(r["d50_mm"]) * 1e-3
        states.append(dict(test=r["test"], scale=r["scale"], sand=r["sand"], d50=d, tau_eq=theta(d) * DRHO * G * d,
                           k=float(r["k_m_per_s"]) if r["k_m_per_s"] else None, l=float(r["l_m"]), Q=float(r["Q_m3_per_s"]), S=float(r["p"]), H=float(r["H_m"])))
    if a.data:
        from pipetheory import poldata as pd
        for tr in read_csv("pol2022_table3_2.csv"):
            try:
                sheet = pd.load_sheet(a.data, tr["test"]); t, dP = pd.load_pressure(a.data, tr["test"])
            except FileNotFoundError:
                continue
            for s in pd.resting_states(sheet):
                h, drift, h_f, h_e = pd.heads(t, dP, s)
                x_tip = pd.X_EXIT + s["l"]
                ports = [n for n in range(2, 14) if pd.X_PORT[n] <= x_tip - 0.004]
                if drift > 0.15 or len(ports) < 4 or not np.isfinite(s["Q"]):
                    continue
                S = np.polyfit([pd.X_PORT[n] for n in ports], [h[n] for n in ports], 1)[0] * 1e-2
                sd = sands[tr["sand"]]
                states.append(dict(test=tr["test"], scale="small", sand=tr["sand"], d50=float(sd["d50_mm"]) * 1e-3, tau_eq=float(sd["tau_eq_Pa"]),
                                   k=float(tr["k_m_per_s"]), l=s["l"], Q=s["Q"], S=S, H=(h_f - (h_e if np.isfinite(h_e) else 0.0)) * 1e-2))
    print(f"\n{'test':11s} {'scale':6s} {'sand':12s} {'l [m]':>6s} {'Q [mL/min]':>10s} {'S':>6s} | {'depth/d50':>9s} {'tau_app/tau_c':>13s} | {'W_thr/d50':>9s} {'W_flat/d50':>10s} {'W [mm]':>7s} {'W/a':>5s} | {'i_bed':>5s} {'Pi':>5s}")
    rows = []
    for st in states:
        o = analyse(st, sec, a.phi)
        rows.append({**st, **o})
        print(f"{st['test']:11s} {st['scale']:6s} {st['sand']:12s} {st['l']:6.3f} {st['Q'] * 6e7:10.1f} {st['S']:6.3f} | {o['a_over_d']:9.1f} {o['tau_app']:13.2f} |"
              f" {o['W_thr_over_d']:9.0f} {o['W_flat_over_d']:10.0f} {o['W_mm']:7.1f} {o['W_over_a']:5.0f} | {o['i_bed']:5.2f} {o['Pi']:5.0f}")
    print("\nBy sand and scale (mean; n):")
    print(f"{'sand':12s} {'scale':6s} {'n':>2s} | {'depth/d50':>9s} {'tau_app/tau_c':>13s} {'W/d50':>6s} {'W/a':>5s} {'i_bed':>6s}")
    for key in sorted({(r["sand"], r["scale"]) for r in rows}):
        rr = [r for r in rows if (r["sand"], r["scale"]) == key]
        f = lambda k_: np.nanmean([r[k_] for r in rr])
        print(f"{key[0]:12s} {key[1]:6s} {len(rr):2d} | {f('a_over_d'):9.1f} {f('tau_app'):13.2f} {np.mean([r['W_mm'] * 1e-3 / r['d50'] for r in rr]):6.0f} {f('W_over_a'):5.0f} {f('i_bed'):6.2f}")
    os.makedirs(a.out, exist_ok=True)
    keys = ["test", "scale", "sand", "d50", "k", "l", "H", "Q", "S", "tau_c", "a_over_d", "tau_app", "W_thr_over_d", "W_flat_over_d", "W_mm", "W_over_a", "i_bed", "Pi"]
    with open(os.path.join(a.out, "paper1_pipe_width_data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    try:
        import plotstyle
        plt = plotstyle.apply()
    except Exception as e:
        print("no figure:", e); return
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.8))
    mk = {"B25": "o", "FPH": "s", "FS35": "^", "Enschede": "D", "SandA": "v", "Baskarp": "o", "Itterbeck330": "^"}
    for ax, key, lab in ((axs[0], "W_mm", "width of the conveying channel [mm]"), (axs[1], "a_over_d", "depth / d50"), (axs[2], "S", "mean gradient in the pipe")):
        for sn in mk:
            for scale, col in (("small", plotstyle.BLUE), ("medium", plotstyle.ORANGE)):
                rr = [r for r in rows if r["sand"] == sn and r["scale"] == scale]
                if rr:
                    ax.plot([r["l"] for r in rr], [r[key] for r in rr], mk[sn], color=col, ms=6, mfc="none" if scale == "small" else col,
                            label=f"{sn} ({scale})" if ax is axs[2] else None)
        ax.set_xscale("log"); ax.set_xlabel("pipe length l [m]"); ax.set_ylabel(lab)
        ax.set_xticks([0.15, 0.3, 0.6, 1.2]); ax.set_xticklabels(["0.15", "0.3", "0.6", "1.2"]); ax.minorticks_off()
    axs[0].set_yscale("log"); axs[0].set_yticks([5, 10, 20, 50]); axs[0].set_yticklabels(["5", "10", "20", "50"]); axs[0].minorticks_off()
    axs[1].set_ylim(0, 11); axs[2].set_ylim(0, 0.2)
    axs[0].set_title("Pipes get wider with scale ..."); axs[1].set_title("... while their depth stays a few grains"); axs[2].set_title("Measured gradient (input)")
    axs[2].legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "paper1_pipe_width_data.png")); print("figure:", os.path.join(a.out, "paper1_pipe_width_data.png"))


if __name__ == "__main__":
    main()
