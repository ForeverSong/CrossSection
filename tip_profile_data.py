"""Measured head profile around the pipe tip, in coordinates that move with the tip (small-scale tests of Pol 2022).

Model-free.  At every reading of the observation sheet the tip position is known (tape reading + 1 cm) and the roof
ports (2 cm apart) give the head along the pipe axis.  The head is interpolated linearly between ports and sampled
at fixed distances xi from the tip (negative: in the pipe).  Averaging over many readings, in which the tip sits at
different places between two ports, gives the mean profile with a resolution of about 2 cm.
Questions: (1) is the pipe steeper just behind the tip than further downstream (a pinched tip zone)?
           (2) how does the gradient decay ahead of the tip (square-root field: secant gradient ~ xi^(-1/2))?

usage:  python tip_profile_data.py <path to Data/Pol_4TU> [--out ../tmp]
"""
import argparse
import csv
import os
import warnings
import numpy as np
from pipetheory import poldata as pd

HERE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")
XI = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]) * 1e-2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    a = ap.parse_args()
    with open(os.path.join(HERE, "data", "pol2022_table3_2.csv")) as f:
        tests = list(csv.DictReader(r for r in f if not r.startswith("#")))
    ports = sorted(pd.X_PORT); xp = np.array([pd.X_PORT[n] for n in ports])
    prof = {}
    for tr in tests:
        try:
            sheet = pd.load_sheet(a.data, tr["test"]); t, dP = pd.load_pressure(a.data, tr["test"])
        except FileNotFoundError:
            continue
        for ti, Li, dHi in zip(sheet["t"], sheet["L"], sheet["dH"]):
            if not np.isfinite(Li) or not np.isfinite(dHi):
                continue
            x_tip = pd.X_EXIT + (Li + 1.0) * 1e-2
            if not 0.23 <= x_tip <= 0.39:
                continue
            w = (t >= 60 * ti - 15) & (t <= 60 * ti + 15)
            if not w.any():
                continue
            h = np.array([dHi - np.mean(dP[n][w]) for n in ports])
            p = np.interp(x_tip + XI, xp, h, left=np.nan, right=np.nan)
            prof.setdefault(tr["sand"], []).append(p - p[5])                 # head relative to the head at the tip [cm]
    print("Mean head relative to the tip [cm] at distance xi from the tip (negative: in the pipe); readings with the tip between x = 0.23 and 0.39 m")
    print(f"{'sand':5s} {'n':>4s} | " + " ".join(f"{x * 100:6.0f}" for x in XI) + "   (xi in cm)")
    out = {}
    for sn in ("B25", "FPH", "FS35"):
        v = np.array(prof.get(sn, []))
        m = np.nanmean(v, axis=0); out[sn] = (m, np.nanstd(v, axis=0), len(v))
        print(f"{sn:5s} {len(v):4d} | " + " ".join(f"{x:6.2f}" for x in m))
    print("\nGradients from the mean profiles")
    print(f"{'sand':5s} | pipe, 10-6 cm behind the tip | pipe, 4-2 cm behind | across the tip (-2..+2 cm) | ahead: 0-2 cm, 2-4, 4-8 | secant 0-2, 0-4, 0-8 cm | secant x sqrt(x) [m^1/2]")
    for sn, (m, sd, n) in out.items():
        g = lambda i, j: (m[j] - m[i]) / ((XI[j] - XI[i]) * 100)
        sec = [g(5, 6), g(5, 7), g(5, 9)]
        print(f"{sn:5s} | {g(0, 2):28.3f} | {g(3, 4):19.3f} | {g(4, 6):26.3f} | {g(5, 6):6.2f} {g(6, 7):5.2f} {g(7, 9):5.2f} | {sec[0]:6.2f} {sec[1]:5.2f} {sec[2]:5.2f}"
              f" | {sec[0] * 0.02 ** 0.5:.3f} {sec[1] * 0.04 ** 0.5:.3f} {sec[2] * 0.08 ** 0.5:.3f}")
    try:
        import plotstyle
        plt = plotstyle.apply()
    except Exception as e:
        print("no figure:", e); return
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for (sn, (m, sd, n)), mk, col in zip(out.items(), ("o", "s", "^"), (plotstyle.BLUE, plotstyle.ORANGE, plotstyle.INK_2)):
        ax.errorbar(XI * 100, m, yerr=sd / np.sqrt(n), marker=mk, ms=5, color=col, lw=1.2, capsize=2, label=f"{sn} (n = {n})")
    ax.axvline(0, color=plotstyle.INK_2, lw=0.8, ls="--"); ax.set_xlabel("distance from the tip [cm]  (negative: in the pipe)")
    ax.set_ylabel("head relative to the tip [cm]"); ax.set_title("Measured head around the tip, small scale"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "paper1_tip_profile_data.png")); print("figure:", os.path.join(a.out, "paper1_tip_profile_data.png"))


if __name__ == "__main__":
    main()
