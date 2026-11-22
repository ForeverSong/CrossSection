"""Measured cross-sections of pipes at rest (laser scans, Pol 2022) against the threshold theory, section by section.

For every scan occasion: the hydraulic state at the time of the scan (head, discharge, heads at the ports, local
gradient of the pipe), the seepage into the pipe (three-dimensional Darcy flow with the measured pipe heads
prescribed; no pipe law), and for every resolved section
    * the laminar flow in the measured section under the measured gradient: largest bed shear stress and discharge;
    * the threshold section for the same gradient: depth and widths;
    * the upward seepage gradient under the pipe, sigma = 2 q / (pi k i_f W), q = inflow per unit length, W = eroded
      width, i_f = fluidisation gradient.
Nothing is fitted.  Needs the output of laser_sections.py.

    python laser_vs_theory.py ../Data/Pol_4TU --out ../tmp        (about 1 minute)
"""
import argparse
import csv
import os
import numpy as np

from pipetheory import laserdata as ld
from pipetheory import poldata as pd
from pipetheory.aquifer import pol_box, prescribed_pipe
from pipetheory.section import threshold_section, stokes_section

HERE = os.path.dirname(os.path.abspath(__file__))
RHO, G = 1000.0, 9.81
POROSITY = 0.40                       # all three sands were placed at n = 0.39-0.42 (thesis, Table 3.1)
RHO_S = {"FPH": 2610.0, "B25": 2650.0, "FS35": 2650.0}
D50 = {"FPH": 0.185e-3, "B25": 0.228e-3, "FS35": 0.422e-3}


def read_csv(name):
    with open(os.path.join(HERE, "data", name)) as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def viscosity(T):
    """Kinematic viscosity of water [m^2/s] at T [deg C] (Vogel-type fit, 0.5 % between 5 and 40 deg C)."""
    return 1.0e-6 * 1.787 / (1.0 + 0.0337 * T + 0.000221 * T ** 2)


def water_temperature(data, test):
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(data, "observations", f"observation_sheet_{test}.xlsx"), data_only=True)
    for r in wb["observations"].iter_rows(min_row=1, max_row=12, values_only=True):
        if r[0] == "Temp water" and isinstance(r[1], (int, float)):
            return float(r[1])
    return 20.0


def scan_state(data, sc):
    """Hydraulic state during a scan: head across the box, discharge, tape reading of the tip, heads at the ports."""
    test = sc["test"]
    sheet = pd.load_sheet(data, test)
    t, dP = pd.load_pressure_all(data, test)
    tA, tB = min(r[1] for r in sc["rows"]), max(r[2] for r in sc["rows"])
    if tB > t[-1] + 5.0:
        return None
    sel = (sheet["t"] >= tA / 60 - 1.5) & (sheet["t"] <= tB / 60 + 1.5) & np.isfinite(sheet["dH"])
    source = "during"
    if sel.sum() < 2:                                        # the sheet stops before the scan: last readings before it
        before = np.where((sheet["t"] <= tA / 60) & (sheet["t"] >= tA / 60 - 15.0) & np.isfinite(sheet["dH"]))[0]
        if before.size < 2:
            return None
        sel = np.zeros(sheet["t"].size, bool)
        sel[before[-4:]] = True
        source = "before"
    dH = float(np.median(sheet["dH"][sel]))
    if np.ptp(sheet["dH"][sel]) > 0.21:
        source += ", head changing"
    tt, mm = sheet["t"][sel], sheet["mass"][sel]
    ok = np.isfinite(mm)
    Q = (mm[ok][-1] - mm[ok][0]) / (tt[ok][-1] - tt[ok][0]) / 6.0e7 if ok.sum() >= 2 and tt[ok][-1] > tt[ok][0] else np.nan
    Lr = sheet["L"][sheet["t"] <= tB / 60 + 1.5]
    Lr = Lr[np.isfinite(Lr)]
    w = (t >= tA) & (t <= tB)
    h = {n: 0.01 * (dH - float(np.mean(dP[n][w]))) for n in dP}          # head above the downstream level [m]
    return dict(dH=0.01 * dH, Q=Q, L_tape=float(Lr[-1]) if Lr.size else np.nan, h=h, tA=tA, tB=tB, source=source)


def pipe_head(state, x_tip):
    """Head in the pipe as a function of x, from the ports the pipe has passed; local gradient."""
    ports = [n for n in range(2, 16) if n in state["h"] and pd.X_PORT[n] < x_tip - 0.004]
    xs = np.array([pd.X_PORT[n] for n in ports])
    hs = np.array([state["h"][n] for n in ports])
    if xs.size < 3:
        return None
    hsm = hs.copy()
    hsm[1:-1] = 0.25 * hs[:-2] + 0.5 * hs[1:-1] + 0.25 * hs[2:]
    lo = np.polyfit(xs[:4], hs[:4], 1)                     # towards the exit: regression on the first four ports
    hi = np.polyfit(xs[-3:], hs[-3:], 1)                   # towards the tip

    def head(x):
        x = np.asarray(x, float)
        out = np.interp(x, xs, hsm)
        out = np.where(x < xs[0], np.polyval(lo, x), out)
        return np.where(x > xs[-1], np.polyval(hi, x), out)

    def slope(x, half=0.035):
        m = np.abs(xs - x) <= half + 1e-9
        return float(np.polyfit(xs[m], hs[m], 1)[0]) if m.sum() >= 3 else np.nan

    return dict(head=head, slope=slope, xs=xs, hs=hs, h_exit=float(np.polyval(lo, pd.X_EXIT)),
                S_mean=float(np.polyfit(xs, hs, 1)[0]))


def roof_at(o, x, y):
    i = int(np.argmin(np.abs(o["xc"] - x)))
    j = int(np.argmin(np.abs(o["yc"] - y)))
    return float(o["roof"][i, j])


def seepage(state, ph, x_tip, complete, width, k, d=0.0025):
    """Darcy flow around the pipe with the measured pipe heads prescribed.  Returns the inflow per unit length over
    k [m] along the pipe, the head on the upstream face, computed heads at the side ports and the discharge."""
    box = pol_box(d=d)
    h = state["h"]
    if complete:
        # the head on the upstream face away from the mouth of the pipe is not measured: take the value for which
        # the computation gives the measured mean head at the two ports beside the pipe near the filter
        side = [h[n] for n in (16, 17) if n in h]
        if not side:
            return None
        target = float(np.mean(side))
        a = prescribed_pipe(box, x_tip, ph["head"], width, 0.0, ph["h_exit"], return_roof=True)
        b = prescribed_pipe(box, x_tip, ph["head"], width, 0.01, ph["h_exit"], return_roof=True)
        ra, rb = roof_at(a, pd.X_SIDE[16], pd.Y_SIDE), roof_at(b, pd.X_SIDE[16], pd.Y_SIDE)
        H = 0.01 * (target - ra) / (rb - ra)
    else:
        H = h[15] + (h[15] - h[14]) / (pd.X_PORT[15] - pd.X_PORT[14]) * (pd.X_FILTER - pd.X_PORT[15])
    o = prescribed_pipe(box, x_tip, ph["head"], width, H, ph["h_exit"], return_roof=True)
    side = {n: roof_at(o, pd.X_SIDE[n], pd.Y_SIDE) for n in (16, 17, 18, 19)}
    return dict(x=o["x"], q_over_k=o["q"], H=H, side=side, Q_darcy=k * o["Q_up"], Q_pipe=k * float(o["q"].sum() * box.d),
                Q_hole=k * o["Q_hole"])


def flow_in_section(yc, dep, y0, y1, ymax, S, nu):
    """Laminar flow in the measured section under the gradient S.  The section is the stretch of the profile
    around its deepest point (ymax) that has readings throughout, within the eroded zone [y0, y1] (mm).
    Returns the largest bed shear stress [Pa], the discharge [m^3/s] (nan unless the stretch spans the whole eroded
    zone) and |du/dn| at the deepest channel [m]."""
    m = (yc >= y0 - 1.0) & (yc <= y1 + 1.0)
    y, a = yc[m], dep[m]
    i0 = int(np.argmin(np.abs(y - ymax)))
    if not np.isfinite(a[i0]):
        return np.nan, np.nan, np.nan
    lo = hi = i0
    while lo > 0 and np.isfinite(a[lo - 1]):
        lo -= 1
    while hi < y.size - 1 and np.isfinite(a[hi + 1]):
        hi += 1
    whole = lo == 0 and hi == y.size - 1
    y, a = y[lo:hi + 1], np.clip(ld.smooth(a[lo:hi + 1], 3), 0.0, None)
    if y.size < 8:
        return np.nan, np.nan, np.nan
    yf = np.arange(y[0], y[-1] + 1e-9, 0.1)
    af = np.maximum(np.interp(yf, y, a), 2.0e-2)           # a film of 0.02 mm where the bed touches the roof
    if whole:
        af[0] = af[-1] = 0.0
    sol = stokes_section(yf, af, 41)
    near = np.abs(yf - ymax) <= 3.0
    dn = 1.0e-3 * float(np.max(sol["dudn"][near]))
    return RHO * G * S * dn, (G * S * 1.0e-12 * sol["q"] / nu if whole else np.nan), dn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="../tmp")
    a = ap.parse_args()
    table = {r["test"]: r for r in read_csv("pol2022_table3_2.csv")}
    sands = {r["sand"]: r for r in read_csv("pol2022_sands.csv")}
    secs = list(csv.DictReader(open(os.path.join(a.out, "paper1_laser_sections.csv"))))
    prof = np.load(os.path.join(a.out, "paper1_laser_profiles.npz"))
    sec_th = {s: threshold_section(np.tan(np.radians(float(sands[s]["repose_deg"])))) for s in sands}
    fnum = lambda v: float(v) if v not in ("", "nan", "None") else np.nan

    rows, scans = [], []
    for sc in ld.list_scans(a.data):
        test, sand = sc["test"], sc["test"].split("_")[0]
        if sc["kind"] == "other" or test == "FS35_242":
            continue
        try:
            st = scan_state(a.data, sc)
        except FileNotFoundError:
            continue
        if st is None:
            continue
        complete = sc["kind"] == "complete"
        x_tip = pd.X_FILTER if complete else pd.X_EXIT + 0.01 * (st["L_tape"] + 1.0)
        ph = pipe_head(st, x_tip)
        if ph is None:
            continue
        mine = [s for s in secs if s["test"] == test and s["file"] == sc["file"]]
        good = [s for s in mine if s["good50"] == "True" and fnum(s["x_m"]) < x_tip - 0.005]
        wer = np.array([[fnum(s["x_m"]), 1.0e-3 * fnum(s["W_er"])] for s in good]) if good else np.zeros((0, 2))
        w_med = float(np.median(wer[:, 1])) if len(wer) else np.nan
        width = (lambda x, w=wer: float(np.clip(np.interp(x, w[:, 0], w[:, 1]), 0.005, 0.035))) if len(wer) else 0.015
        k = float(table[test]["k_m_per_s"]) if test in table else np.nan
        nu = viscosity(water_temperature(a.data, test))
        i_f = (1.0 - POROSITY) * (RHO_S[sand] / RHO - 1.0)
        sp_ = seepage(st, ph, x_tip, complete, width, k) if np.isfinite(k) else None
        tau_eq = float(sands[sand]["tau_eq_Pa"])
        th = sec_th[sand]
        tau_c = tau_eq / (th["a_max"] * th["aavg_over_amax"])
        A = 2.0 * tau_c / (RHO * G)
        scan = dict(test=test, sand=sand, file=sc["file"], kind=sc["kind"], dH=st["dH"], Q=st["Q"], x_tip=x_tip,
                    l=x_tip - pd.X_EXIT, S_mean=ph["S_mean"], h_exit=ph["h_exit"], n_good=len(good), W_er=w_med, k=k,
                    nu=nu, source=st["source"], i_f=i_f)
        for n in (16, 17, 18, 19):
            scan[f"P{n}"] = st["h"].get(n, np.nan)
        scan["hp_024"], scan["hp_0434"] = float(ph["head"](0.24)), float(ph["head"](0.434))
        if sp_:
            scan.update(H=sp_["H"], Q_darcy=sp_["Q_darcy"], Q_pipe=sp_["Q_pipe"], Q_hole=sp_["Q_hole"],
                        **{f"M{n}": sp_["side"][n] for n in (16, 17, 18, 19)})
        scans.append(scan)
        for s in good:
            x = fnum(s["x_m"])
            S = ph["slope"](x)
            key = f"{test}|{sc['file']}|{fnum(s['tape_cm'])}"
            yc, dep = prof[key][:, 0], prof[key][:, 1]
            amax, W_er = 1.0e-3 * fnum(s["amax"]), 1.0e-3 * fnum(s["W_er"])
            r = dict(test=test, sand=sand, file=sc["file"], kind=sc["kind"], x=x, s_tip=x_tip - x, S=S, amax=amax,
                     w50=1.0e-3 * fnum(s["w50"]), w75=1.0e-3 * fnum(s["w75"]), w25=1.0e-3 * fnum(s["w25"]),
                     good25=s["good25"] == "True", W_er=W_er, A_er=1.0e-6 * fnum(s["A_er"]), k=k, nu=nu, d50=D50[sand],
                     tau_c=tau_c, i_f=i_f)
            if np.isfinite(S) and S > 0.01:
                tau, Qs, dn = flow_in_section(yc, dep, fnum(s["y_er0"]), fnum(s["y_er1"]), fnum(s["ymax"]), S, nu)
                r.update(tau_max=tau, Q_stokes=Qs, tau_flat=0.5 * RHO * G * amax * S,
                         amax_th=th["a_max"] * A / S, W_th=2.0 * th["half_width"] * A / S, Q_th=th["chi"] * G * A ** 4 / (nu * S ** 3))
            if sp_:
                qk = float(np.interp(x, sp_["x"], sp_["q_over_k"]))
                r.update(q_over_k=qk, sigma=2.0 * qk / (np.pi * i_f * W_er))
                # discharge in the pipe at x: inflow upstream of x (plus, for a complete pipe, what enters at the filter)
                up = sp_["x"] >= x
                Qx = k * float(np.sum(sp_["q_over_k"][up]) * 0.0025)
                if complete and np.isfinite(st["Q"]):
                    Qx += st["Q"] - sp_["Q_darcy"]
                r["Q_pipe_x"] = Qx
            rows.append(r)

    os.makedirs(a.out, exist_ok=True)
    keys = ["test", "sand", "file", "kind", "x", "s_tip", "S", "amax", "w75", "w50", "w25", "good25", "W_er", "A_er", "tau_flat",
            "tau_max", "Q_stokes", "amax_th", "W_th", "Q_th", "q_over_k", "sigma", "Q_pipe_x", "k", "nu", "d50", "tau_c", "i_f"]
    with open(os.path.join(a.out, "paper1_laser_vs_theory.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(("%.5g" % r[k_]) if isinstance(r.get(k_), float) else str(r.get(k_, "")) for k_ in keys) + "\n")
    skeys = ["test", "sand", "file", "kind", "source", "dH", "Q", "x_tip", "l", "S_mean", "h_exit", "n_good", "W_er", "k", "nu", "i_f",
             "H", "Q_darcy", "Q_pipe", "Q_hole", "hp_024", "P18", "P19", "M18", "hp_0434", "P16", "P17", "M16"]
    with open(os.path.join(a.out, "paper1_laser_scans.csv"), "w") as f:
        f.write(",".join(skeys) + "\n")
        for s in scans:
            f.write(",".join(("%.5g" % s[k_]) if isinstance(s.get(k_), float) else str(s.get(k_, "")) for k_ in skeys) + "\n")

    L = []
    L.append("Scan occasions with a hydraulic state (heads in cm, Q in mL/min, widths in mm)")
    L.append("test      kind     l[cm]  dH   Q    S_mean  W_er n | head beside the pipe at x=0.24: pipe, measured (P18,P19), computed | Q Darcy/meas")
    for s in scans:
        m18 = 100 * s["M18"] if "M18" in s else np.nan
        L.append(f"{s['test']:9s} {s['kind']:8s} {100 * s['l']:5.1f} {100 * s['dH']:5.2f} {6e7 * s['Q']:5.0f} {s['S_mean']:.3f}  {1e3 * s['W_er']:4.1f} {s['n_good']:2d} | "
                 f"{100 * s['hp_024']:5.2f}  {100 * s['P18']:5.2f} {100 * s['P19']:5.2f}  {m18:5.2f} | "
                 f"{(s.get('Q_darcy', np.nan) / s['Q']):.2f}   [{s['source']}]")
    open(os.path.join(a.out, "paper1_laser_vs_theory.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsections with a gradient: {sum(np.isfinite(r.get('S', np.nan)) for r in rows)} of {len(rows)}")


if __name__ == "__main__":
    main()
