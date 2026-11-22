"""Readers for the 4TU data set of Pol (2022), small-scale hole-exit tests (observation sheets and pressure records).

Conventions (checked against the thesis, sections 3.2 and 3.3.3)
    * dP_n in the pressure files is the head drop below the upstream level at port n [cm]; the head above the
      downstream level is dH - dP_n, with dH from the riser tubes (observation sheet);
    * centre-line ports P2..P13 are at x = 0.19 ... 0.41 m (0.02 m apart), P14 at 0.45 m, P15 at 0.47 m;
    * the clock of the pressure record and the stopwatch of the sheet start together (head steps in the record
      fall a few seconds after the sheet readings at 2, 7, 12 ... min);
    * "L [cm]" in the sheet is the tape reading; pipe length from the centre of the hole = L + 1 cm
      (Table 3.2 of the thesis gives l_c = L_crit + 1 cm);
    * head at the filter face: linear extrapolation of P14 and P15 to x = 0.48 m (as in the thesis);
    * exit loss: regression line through P2..P5 extended to the hole, once the tip has passed P5 (as in the thesis).
"""
import os
import numpy as np

X_EXIT, X_FILTER = 0.125, 0.48
X_PORT = {**{n: 0.19 + 0.02 * (n - 2) for n in range(2, 14)}, 14: 0.45, 15: 0.47}
# Ports beside the pipe, read from figure 3.2 of the thesis (to about 3 mm): P16 and P17 at x = 0.434 m, P18 and P19
# at x = 0.24 m, 0.0195 m on either side of the axis.
X_SIDE = {16: 0.434, 17: 0.434, 18: 0.24, 19: 0.24}
Y_SIDE = 0.0195


def load_sheet(data, test):
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(data, "observations", f"observation_sheet_{test}.xlsx"), data_only=True)
    rows = list(wb["observations"].iter_rows(values_only=True))
    hdr = next(i for i, r in enumerate(rows) if r[0] == "actual time")
    names = [str(c) if c is not None else "" for c in rows[hdr]]
    col = {n: i for i, n in enumerate(names) if n}
    num = lambda v: float(v) if isinstance(v, (int, float)) else np.nan
    body = [r for r in rows[hdr + 1:] if isinstance(r[col["t [min]"]], (int, float))]
    out = {k: np.array([num(r[col[c]]) for r in body]) for k, c in
           (("t", "t [min]"), ("dH", "dH [cm]"), ("mass", "cumulative mass water [g]"), ("L", "L [cm]"))}
    out["note"] = [str(r[0]) if isinstance(r[0], str) else "" for r in body]
    return out


def load_pressure(data, test):
    import scipy.io as sio
    m = sio.loadmat(os.path.join(data, "pressure", f"{test}_pressure.MAT"), squeeze_me=True, struct_as_record=False)
    ch = {m[k].SignalName: m[k.replace("_Header", "_Data")] for k in m if k.startswith("Channel_") and k.endswith("_Header")}
    t = ch[next(k for k in ch if k.startswith("Time"))]
    return t, {n: ch[f"dP_{n:03d}"] for n in range(1, 16)}


def load_pressure_all(data, test):
    """As load_pressure, with the ports beside the pipe (16-19) where they were recorded."""
    import scipy.io as sio
    m = sio.loadmat(os.path.join(data, "pressure", f"{test}_pressure.MAT"), squeeze_me=True, struct_as_record=False)
    ch = {m[k].SignalName: m[k.replace("_Header", "_Data")] for k in m if k.startswith("Channel_") and k.endswith("_Header")}
    t = ch[next(k for k in ch if k.startswith("Time"))]
    return t, {n: ch[f"dP_{n:03d}"] for n in range(1, 20) if f"dP_{n:03d}" in ch}


def resting_states(sheet, min_minutes=4.0, min_tape=5.5):
    """Runs of sheet rows with constant head and constant pipe length (the pipe is at rest), before the pipe
    reaches the filter.  Returns dicts with the time span [min], dH [cm], pipe length [m] and discharge [m^3/s]."""
    t, dH, L, mass = sheet["t"], sheet["dH"], sheet["L"], sheet["mass"]
    out, i, n = [], 0, len(t)
    while i < n:
        j = i
        while j + 1 < n and np.isfinite(L[j + 1]) and abs(L[j + 1] - L[i]) < 0.051 and abs(dH[j + 1] - dH[i]) < 0.051:
            j += 1
        if np.isfinite(L[i]) and L[i] >= min_tape and t[j] - t[i] >= min_minutes and L[i] < 33.0:
            later = L[j + 1:][np.isfinite(L[j + 1:])]
            Q = (mass[j] - mass[i]) / (t[j] - t[i]) / 6.0e7 if np.isfinite(mass[j] - mass[i]) else np.nan
            out.append(dict(t0=t[i], t1=t[j], dH=float(np.mean(dH[i:j + 1])), l=(L[i] + 1.0) * 1e-2, Q=Q,
                            grew_later=bool(later.size and later.max() > L[i] + 0.3)))
        i = j + 1
    return out


def heads(t, dP, state, skip=60.0):
    """Mean head above the downstream level [cm] at each port during a resting state, the drift within the window
    [cm], the head at the filter face and the exit head from the regression on P2..P5 (nan if the tip has not passed P5)."""
    a, b = 60.0 * state["t0"] + skip, 60.0 * state["t1"]
    w = (t >= a) & (t <= b)
    mid = 0.5 * (a + b)
    h = {n: state["dH"] - float(np.mean(dP[n][w])) for n in dP}
    drift = max(abs(float(np.mean(dP[n][w & (t > mid)]) - np.mean(dP[n][w & (t <= mid)]))) for n in range(2, 16))
    h_filter = h[15] + (h[15] - h[14]) / (X_PORT[15] - X_PORT[14]) * (X_FILTER - X_PORT[15])
    h_exit = np.nan
    if X_EXIT + state["l"] > X_PORT[5] + 0.005:
        xs = np.array([X_PORT[n] for n in range(2, 6)]); hs = np.array([h[n] for n in range(2, 6)])
        p = np.polyfit(xs, hs, 1)
        h_exit = float(np.polyval(p, X_EXIT))
    return h, drift, h_filter, h_exit
