"""Progression rate x pipe depth in the tests of Pol (2022): is the sand carried at a saturated rate?

If the pipe advances at speed v with a cross-section of depth a that keeps its shape, the sand discharge per unit
width of bed at the exit is about (1 - n) v a.  In the 'Parker regime' of an active laminar channel (Popovic et al.
2021) that flux cannot exceed a value fixed by the sand.  Here v comes from the pipe length recorded in the
observation sheets (from the last rest to the moment the pipe reaches the filter), the mean gradient S in the pipe
from the pressure ports during that phase, and a = A / S (threshold depth; the bed is slightly above threshold
while it carries sand, so a is an upper estimate).  Output is normalised by V_s d50 (Stokes settling speed x grain
size), the scale of the laminar transport law.

usage:  python progression_rate_data.py <path to Data/Pol_4TU> [--out ../tmp]
"""
import argparse
import csv
import os
import warnings
import numpy as np
from pipetheory import RHO_W, G, NU
from pipetheory import poldata as pd
from pipetheory.section import threshold_section

HERE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore", message="wmf image format")
POROSITY = 0.40


def read_csv(name):
    with open(os.path.join(HERE, "data", name)) as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    a = ap.parse_args()
    sands = {r["sand"]: r for r in read_csv("pol2022_sands.csv")}
    rows = []
    for tr in read_csv("pol2022_table3_2.csv"):
        try:
            sheet = pd.load_sheet(a.data, tr["test"]); t, dP = pd.load_pressure(a.data, tr["test"])
        except FileNotFoundError:
            continue
        sd = sands[tr["sand"]]; d50 = float(sd["d50_mm"]) * 1e-3
        sec = threshold_section(np.tan(np.radians(float(sd["repose_deg"]))), 16, 161, 41)
        tau_c = float(sd["tau_eq_Pa"]) / (sec["a_max"] * sec["aavg_over_amax"])
        A = 2.0 * tau_c / (RHO_W * G)
        L, tm = sheet["L"], sheet["t"]
        ok = np.isfinite(L)
        rest = pd.resting_states(sheet, min_minutes=3.0)
        t0 = max([s["t1"] for s in rest], default=tm[ok][0])           # end of the last rest
        ph = ok & (tm >= t0)
        if ph.sum() < 3 or L[ph].max() - L[ph].min() < 5.0:
            continue
        i_end = np.where(ph)[0][np.argmax(L[ph] >= min(33.0, L[ph].max()))]
        sel = ph & (tm <= tm[i_end])
        v = np.polyfit(tm[sel] * 60.0, L[sel] * 1e-2, 1)[0]                # m/s
        S = []
        for ti, Li, dHi in zip(tm[sel], L[sel], sheet["dH"][sel]):
            w = (t >= 60 * ti - 15) & (t <= 60 * ti + 15)
            ports = [n for n in range(2, 14) if pd.X_PORT[n] <= pd.X_EXIT + (Li + 1.0) * 1e-2 - 0.004]
            if w.sum() and len(ports) >= 4:
                S.append(-np.polyfit([pd.X_PORT[n] for n in ports], [np.mean(dP[n][w]) for n in ports], 1)[0] * 1e-2)
        if len(S) < 2:
            continue
        S = float(np.mean(S)); depth = A / S
        Vs = (2650.0 - RHO_W) * G * d50 ** 2 / (18.0 * RHO_W * NU)
        theta_t = tau_c / ((2650.0 - RHO_W) * G * d50)
        rows.append(dict(test=tr["test"], sand=tr["sand"], v_mm_s=v * 1e3, S=S, depth_mm=depth * 1e3, depth_over_d=depth / d50,
                         flux=(1 - POROSITY) * v * depth, phi=(1 - POROSITY) * v * depth / (Vs * d50), theta_t=theta_t, minutes=tm[i_end] - t0))
    print(f"{'test':9s} {'phase [min]':>11s} {'v [mm/s]':>9s} {'S in pipe':>9s} {'depth [mm]':>10s} {'depth/d50':>9s} {'(1-n) v a [mm2/s]':>17s} {'/ (V_s d50)':>11s}")
    for r in rows:
        print(f"{r['test']:9s} {r['minutes']:11.0f} {r['v_mm_s']:9.3f} {r['S']:9.3f} {r['depth_mm']:10.2f} {r['depth_over_d']:9.1f} {r['flux'] * 1e6:17.3f} {r['phi']:11.4f}")
    print("\nby sand: mean (coefficient of variation)")
    for sn in ("B25", "FPH", "FS35"):
        rr = [r for r in rows if r["sand"] == sn]
        if len(rr) < 2:
            continue
        f = lambda k: (np.mean([r[k] for r in rr]), np.std([r[k] for r in rr], ddof=1) / np.mean([r[k] for r in rr]))
        th = rr[0]["theta_t"]
        print(f"   {sn:5s} n = {len(rr)}:  v = {f('v_mm_s')[0]:.3f} mm/s ({f('v_mm_s')[1]:.2f});  depth/d50 = {f('depth_over_d')[0]:.1f} ({f('depth_over_d')[1]:.2f});"
              f"  (1-n) v a / (V_s d50) = {f('phi')[0]:.4f} ({f('phi')[1]:.2f});  saturated flux of Popovic et al. scaled to this sand: {1.84 * th * 0.09:.3f} - {1.84 * th * 0.22:.3f}")
    with open(os.path.join(a.out, "paper1_progression_rate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
