"""Large-scale test of Pol, Kanning & Jonkman: what the published records say about the pipe.

Data set: 'FPH piping experiment December 2018', 4TU.ResearchData, doi:10.4121/uuid:3b44f87b-cc0f-4c03-a192-e81078dec53d
(CC BY 4.0).  Geometry and transducer positions are from Pol's thesis, ch. 4: bed 9.5 x 1.2 x 1.0 m, exit hole at
x = 1.3 m, infiltration box over x = 8.5-9.5 m, transducers at the sand/clay interface on the centre line at
x = 0.5 (P1), 1.9 (P2), 3.2 (P3), 4.0 (P5), 4.8 (P7), 5.6 (P8), 6.4 (P9), 7.2 (P10), 8.0 (P12), 9.0 (P14, in the box);
P4, P6 beside P5 and P11, P13 beside P12 (lateral offsets not given); P15 levee core; P16 upstream basin.

Everything printed here comes from the records; nothing is fitted.  Heads are in m above the upstream basin floor:
head = dP/9.81 + 0.065 for the transducers in the bed (zeroed at a basin level of 0.065 m), dP/9.81 for P16
(this reproduces the upstream levels quoted in the thesis).

usage:  python large_scale_data.py <folder with raw_data/ and figures/> [--out ../tmp]
"""
import argparse
import csv
import datetime
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = datetime.datetime(2018, 12, 17)
XPOS = {1: 0.5, 2: 1.9, 3: 3.2, 5: 4.0, 7: 4.8, 8: 5.6, 9: 6.4, 10: 7.2, 12: 8.0, 14: 9.0}
X_EXIT = 1.3
NU, G = 1.52e-6, 9.81                         # water at 5 C
D50, RHO_S, POROSITY = 0.185e-3, 2610.0, 0.383
TAU_EQ = 0.339                                # Pa, measured at small scale with the mean depth of the section (Pol 2022)
TAU_C = TAU_EQ / 0.74                         # flat-bed threshold as used in the paper (section-mean / 0.74); the other reading is TAU_EQ itself
K_LAB, K_LEVEE, WIDTH, DEPTH = 8.0e-5, 9.0e-5, 1.2, 1.0


def hours(d):
    return (d - T0).total_seconds() / 3600.0


def load(folder):
    import openpyxl
    ws = openpyxl.load_workbook(os.path.join(folder, "raw_data", "81CR800_Pipingproef2_ENG.xlsx"), data_only=True, read_only=True).worksheets[0]
    t, d = [], []
    for r in ws.iter_rows(min_row=5, values_only=True):
        if isinstance(r[0], datetime.datetime):
            t.append(hours(r[0])); d.append([float(c) if isinstance(c, (int, float)) else np.nan for c in r[21:37]])
    t, raw = np.array(t), np.array(d) / G + 0.065
    raw[:, 15] -= 0.065
    ws = openpyxl.load_workbook(os.path.join(folder, "figures", "fph2_head.xlsx"), data_only=True, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    chan = [int(str(c)[3:-1]) for c in rows[0][2:]]
    tp = np.array([hours(r[1]) for r in rows[2:]]); proc = np.array([[float(c) for c in r[2:]] for r in rows[2:]])
    ws = openpyxl.load_workbook(os.path.join(folder, "figures", "fph2_pipeposition.xlsx"), data_only=True, read_only=True).worksheets[0]
    tip = np.array([[hours(r[0]), float(r[2])] for r in list(ws.iter_rows(values_only=True))[1:] if isinstance(r[0], datetime.datetime)])
    mass = []
    for r in csv.reader(open(os.path.join(folder, "raw_data", "geometry", "Mass_data.csv"), encoding="latin-1"), delimiter=";"):
        try:
            mass.append([(float(r[1].replace(",", ".")) - 43451.0) * 24.0, float(r[2].replace(",", "."))])
        except (ValueError, IndexError):
            pass
    ws = openpyxl.load_workbook(os.path.join(folder, "raw_data", "observation_sheet_fph2.xlsx"), data_only=True, read_only=True).worksheets[0]
    obs = [r for r in ws.iter_rows(min_row=20, values_only=True) if isinstance(r[0], (int, float))]
    weir = np.array([[float(r[3]) * 24 + 9.4, float(r[7]), float(r[8])] for r in obs if isinstance(r[7], (int, float)) and isinstance(r[8], (int, float))])
    return t, raw, tp, proc, chan, tip, np.array(mass), weir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    a = ap.parse_args()
    t, raw, tp, proc, chan, tip, mass, weir = load(a.folder)
    P = lambda n: raw[:, n - 1]
    Q = lambda n: proc[:, chan.index(n)]
    mean = lambda v, tt, lo, hi: float(np.nanmean(v[(tt > lo) & (tt < hi)]))
    tip_at = lambda h: float(np.interp(h, tip[:, 0], tip[:, 1]))
    mass_at = lambda h: float(np.interp(h, mass[:, 0], mass[:, 1]))
    t_pass = {x: h for h, x in tip}

    print(f"records: {len(t)} samples, {t[0]:.1f} to {t[-1]:.1f} h after 17 Dec 2018 00:00; processed file {len(tp)} samples (600 s moving average)")
    dev = {n: float(np.nanmax(np.abs(np.interp(tp[::20], t, P(n)) - Q(n)[::20]))) for n in chan if n != 16}
    print("largest difference between my conversion of the raw records and the authors' processed heads [m]: "
          + ", ".join(f"P{n} {v:.2f}" for n, v in dev.items()))
    print("  (P2 and P3: the authors removed 'sudden drops'; the raw P2 lies up to 0.14 m below their P2 after t = 25 h)")
    print("tip record of the authors [h: x]: " + ", ".join(f"{h:.1f}: {x:g}" for h, x in tip))

    print("\n1. Course of the test (means over the window; heads in m; P2 raw / as processed by the authors)")
    print("   window [h]   basin   box(P14)  P12    P1    P2 raw/proc    P3   H=basin-P1  tip (authors) [m]  sand [kg]")
    for lo, hi in ((13.5, 14.5), (15.0, 15.9), (16.2, 17.2), (18.3, 20.5), (21.0, 23.0), (23.5, 26.5), (27.2, 29.5), (30.0, 35.5), (36.0, 36.8),
                   (37.5, 40.3), (41.0, 45.5), (46.0, 47.5), (48.0, 51.0)):
        print(f"   {lo:4.1f}-{hi:4.1f}   {mean(P(16), t, lo, hi):6.2f}  {mean(P(14), t, lo, hi):6.2f}  {mean(P(12), t, lo, hi):6.2f} {mean(P(1), t, lo, hi):6.2f}"
              f"   {mean(P(2), t, lo, hi):5.2f}/{mean(Q(2), tp, lo, hi):4.2f}  {mean(Q(3), tp, lo, hi):6.2f}    {mean(P(16), t, lo, hi) - mean(P(1), t, lo, hi):5.2f}"
              f"       {tip_at(lo):4.2f}-{tip_at(hi):4.2f}       {mass_at(hi):5.2f}")

    print("\n2. Hydraulic gradient inside the pipe (pairs of transducers that the tip has passed; mean and s.d. up to t = 51 h)")
    for n1, n2, tag in ((1, 2, "exit zone: P1 (0.8 m downstream of the exit) to P2 (0.6 m into the pipe), authors' P2"), (2, 3, "authors' P2"), (3, 5, ""), (5, 7, ""), (7, 8, "")):
        lo = t_pass[XPOS[n2]] + 0.5
        w = (tp > lo) & (tp < 51.0)
        g = (Q(n2)[w] - Q(n1)[w]) / (XPOS[n2] - XPOS[n1] if n1 > 1 else XPOS[2] - X_EXIT)
        print(f"   P{n1}-P{n2}: {np.mean(g):6.3f} +- {np.std(g):.3f}   ({lo:.1f}-51 h) {tag}")
    w = (t > t_pass[3.2] + 0.5) & (t < 51.0)
    print(f"   P2-P3 with the raw P2: {np.mean((np.interp(t[w], tp, Q(3)) - P(2)[w]) / 1.3):6.3f}")
    lo, hi = 46.5, 51.0
    hp = {n: mean(Q(n), tp, lo, hi) - mean(Q(1), tp, lo, hi) for n in (2, 3, 5, 7)}
    print(f"   head above P1 at {lo}-{hi} h: " + ", ".join(f"P{n} ({XPOS[n] - X_EXIT:.1f} m from the exit) {v:.3f}" for n, v in hp.items()))
    S_pipe = (hp[7] - hp[2]) / (XPOS[7] - XPOS[2])
    print(f"   mean gradient P2 to P7: {S_pipe:.3f};   depth of a flat bed at the threshold under that gradient, 2 tau_c/(rho g S): {2 * TAU_C / (9810 * S_pipe) / D50:.1f} d50"
          f" ({2 * TAU_EQ / (9810 * S_pipe) / D50:.1f} d50 if the measured stress is read as the flat-bed threshold itself)")

    print("\n3. Head just ahead of the tip at the moment it passes a transducer (authors' passing times), and far upstream")
    print("   passing     basin-P1   gradient over the next 0.8 m   the 0.8 m after that   P10-P12")
    order = [3, 5, 7, 8, 9, 10, 12]
    for n in (3, 5, 7, 8):
        h = t_pass[XPOS[n]]
        v = {m: mean(Q(m), tp, h - 0.1, h + 0.1) for m in order + [1]}
        i = order.index(n)
        print(f"   P{n} {h:5.1f} h   {mean(P(16), t, h - 0.1, h + 0.1) - v[1]:5.2f}          {(v[order[i + 1]] - v[n]) / 0.8:5.2f}                     {(v[order[i + 2]] - v[order[i + 1]]) / 0.8:5.2f}"
              f"              {(v[12] - v[10]) / 0.8:5.2f}")
    lo, hi = 30.0, 35.5
    v = {m: mean(Q(m), tp, lo, hi) for m in order + [1, 2, 14]}
    print(f"   long stop of the tip, {lo}-{hi} h: head above P1: " + ", ".join(f"P{m} {v[m] - v[1]:.3f}" for m in [2] + order + [14]))
    print(f"      gradient P5-P8 {(v[8] - v[5]) / 1.6:.3f}, P8-P10 {(v[10] - v[8]) / 1.6:.3f}, P10-P12 {(v[12] - v[10]) / 0.8:.3f};  P12 to the box (P14) {v[14] - v[12]:.3f} m over 0.5 m")
    i_bed = (v[12] - v[5]) / 4.0
    print(f"      bed-average gradient P5-P12 {i_bed:.3f} -> discharge k i W D = {K_LAB * i_bed * WIDTH * DEPTH * 1e6:.0f} mL/s with k_lab, {K_LEVEE * i_bed * WIDTH * DEPTH * 1e6:.0f} with k_levee")
    print(f"      P3 minus P1: " + ", ".join(f"{h:g} h {mean(Q(3), tp, h - 0.3, h + 0.3) - mean(Q(1), tp, h - 0.3, h + 0.3):.2f}" for h in (18, 21, 23, 25, 30, 36, 42, 46, 50)))
    np.savez(os.path.join(a.out, "paper1_large_scale_standstill.npz"), x=np.array([XPOS.get(m, np.nan) for m in chan]),
             head=np.array([mean(Q(m), tp, lo, hi) for m in chan]))

    print("\n4. Across the bed (raw records): P4 | P5 (centre) | P6 at x = 4.0 m;  P11 | P12 (centre) | P13 at x = 8.0 m")
    for lo, hi, tag in ((11.0, 12.5, "before any pipe"), (30.0, 35.5, "tip stopped near 3 m"), (42.0, 45.0, "tip 0.2-0.7 m past P5"), (48.0, 51.0, "tip 1.7-2.4 m past P5")):
        print(f"   {lo:4.1f}-{hi:4.1f} h  {mean(P(4), t, lo, hi):5.2f} | {mean(P(5), t, lo, hi):5.2f} | {mean(P(6), t, lo, hi):5.2f}     "
              f"{mean(P(11), t, lo, hi):5.2f} | {mean(P(12), t, lo, hi):5.2f} | {mean(P(13), t, lo, hi):5.2f}    {tag}")

    print("\n5. Discharge at the exit (V-notch weir; the thesis calls it an order-of-magnitude reading)")
    w = weir[(weir[:, 0] > 21) & (weir[:, 0] < 41) & (weir[:, 2] > 0)]
    q_weir = float(np.median(w[:, 2]))
    print(f"   {len(w)} readings between 21 and 41 h: median {q_weir * 1e6:.0f} mL/s, range {w[:, 2].min() * 1e6:.0f}-{w[:, 2].max() * 1e6:.0f}")

    print("\n6. Eroded sand and what it implies for the pipe")
    bulk = RHO_S * (1 - POROSITY)
    print("   tip [m] (authors)   time [h]   sand [kg]   volume / pipe length [mm2]")
    for h, x in tip[1:8]:
        print(f"        {x:3.1f}            {h:5.1f}     {mass_at(h):6.2f}        {mass_at(h) / bulk / (x - X_EXIT) * 1e6:6.0f}")
    h1, h2 = t_pass[3.2], t_pass[6.4]
    V = (mass_at(h2) - mass_at(h1)) / bulk / 3.2
    print(f"   between the passing of P3 and of P9 (3.2 m of new pipe): {mass_at(h2) - mass_at(h1):.2f} kg -> {V * 1e6:.0f} mm2 per metre of new pipe (upper bound: the older pipe may also grow)")
    print("   a laminar slot that carries Q at gradient S and has cross-section V:  depth = sqrt(12 nu Q/(g S V)),  width = V/depth")
    for q, tag in ((K_LAB * i_bed * WIDTH * DEPTH, "Darcy, k_lab"), (q_weir, "weir")):
        for S in (0.10, 0.125):
            dep = np.sqrt(12 * NU * q / (G * S * V))
            print(f"      Q = {q * 1e6:4.1f} mL/s ({tag:12s}) S = {S:5.3f}: depth {dep * 1e3:.2f} mm = {dep / D50:.1f} d50, width {V / dep:.2f} m;"
                  f"  threshold depth {2 * TAU_C / (9810 * S) / D50:.1f} d50 (other reading {2 * TAU_EQ / (9810 * S) / D50:.1f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for n in (16, 14, 12, 10, 8, 7, 5, 3, 2, 1):
        ax[0].plot(tp[::30], Q(n)[::30] if n != 16 else np.interp(tp[::30], t, P(16)), lw=1, color="0.25" if n in (16, 1) else None)
        ax[0].text(51.6, mean(Q(n), tp, 50, 51) if n != 16 else mean(P(16), t, 50, 51), f"P{n}", fontsize=7, va="center")
    for h, x in tip[1:8]:
        ax[0].axvline(h, color="0.8", lw=0.6); ax[0].text(h, 1.95, f"{x:g}", fontsize=7, ha="center")
    ax[0].set_xlim(9, 54); ax[0].set_ylim(0, 2.02); ax[0].set_xlabel("hours after 17 Dec 2018 00:00"); ax[0].set_ylabel("head [m]")
    ax[0].set_title("(a) heads; grey lines: tip passes x [m]", fontsize=9)
    for lo, hi, lab in ((30.0, 35.5, "tip stopped near 3 m"), (t_pass[4.0] - 0.1, t_pass[4.0] + 0.1, "tip at 4.0 m"), (t_pass[5.6] - 0.1, t_pass[5.6] + 0.1, "tip at 5.6 m")):
        xs = [XPOS[n] for n in (2, 3, 5, 7, 8, 10, 12, 14)]
        ax[1].plot(xs, [mean(Q(n), tp, lo, hi) - mean(Q(1), tp, lo, hi) for n in (2, 3, 5, 7, 8, 10, 12, 14)], "o-", ms=3, lw=1, label=lab)
    xx = np.array([X_EXIT, 5.6]); ax[1].plot(xx, 0.126 * (xx - X_EXIT), "k--", lw=0.8, label="threshold pipe, depth 4 d50 (model)")
    ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("head above P1 [m]"); ax[1].legend(fontsize=7, frameon=False); ax[1].set_title("(b) head along the centre line (P9 left out)", fontsize=9)
    ax[2].plot(mass[:, 0], mass[:, 1], lw=1.2, color="0.2"); ax[2].set_xlim(9, 54); ax[2].set_xlabel("hours after 17 Dec 2018 00:00"); ax[2].set_ylabel("eroded dry sand [kg]")
    for h, x in tip[1:8]:
        ax[2].axvline(h, color="0.8", lw=0.6)
    ax[2].set_title("(c) sand collected at the exit", fontsize=9)
    for b in ax:
        b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "paper1_large_scale_data.png"), dpi=140)


if __name__ == "__main__":
    main()
