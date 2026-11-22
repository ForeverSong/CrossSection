"""Raw laser scans of the pipe bed in the 4TU data set of Pol (2022), small-scale tests 225-250.

What the files contain (checked against the thesis, sections 3.2.2 and 3.3.5, and against its figure 3.10)
    * pressure/<test>_laser.MAT: 200 Hz record of the laser, three channels: time [s]; Laser_Depth [mA], the
      4-20 mA output of the distance sensor; Laser_Distance, labelled cm, the position of the sensor on its
      frame.  The frame is moved by hand across the box, several passes per transect.
    * laser/<test>/timing_transects_*.txt: for each transect the tape reading x [cm] and the time window.
      x in the box = 0.135 m + tape reading (the thesis gives 0.22-0.46 m for tape 8.5-32.5 cm).
    * The scans were made with the pipe at rest: after it had reached the upstream filter and the head had been
      lowered until the grains stopped ("complete pipe"), and in a few tests also during the growth of the pipe
      ("partial pipe").

Conversion (not documented in the data set; established here)
    * Laser_Distance is in cm as labelled: two channels 14 mm apart in figure 3.10 are 1.4 units apart.  Its
      origin differs from test to test (by up to 0.4 m).  The sand is seen through a window 250 mm wide in every
      scan; y is measured from the centre of that window, and the y of the centre (Y_WINDOW_CENTRE) is fixed by
      figure 3.10.  The absolute y is therefore good to a few millimetres; only differences of y are used.
    * A larger current is a larger distance, i.e. a deeper bed.  Readings taken through tubes and port fittings
      are too small; the bed is the upper envelope of the readings.
    * depth [mm] = MM_PER_MA * (current - current at the roof).  MM_PER_MA = 8.14 reproduces figure 3.10 of the
      thesis (test B25-245, x = 0.22 m) to 0.05 mm r.m.s. after a shift of the zero by 0.14 mm.  This is the
      conversion of the authors as far as it can be recovered from one figure; whether it includes a correction
      for refraction in the water is not stated in the thesis.
"""
import os
import re
import numpy as np

MM_PER_MA = 8.14
Y_WINDOW_CENTRE = 153.6   # y in the box [mm] of the centre of the 250 mm window through which the sand is seen
X_TAPE0 = 0.135        # x_box [m] = X_TAPE0 + tape [m]
Y_AXIS = 150.0         # centre line of the box [mm]
GUIDE_HALF = 17.5      # the guides are 35 mm apart


def load_laser_raw(data, test, variant=""):
    """Returns time [s], current [mA], Laser_Distance [cm]."""
    import scipy.io as sio
    f = os.path.join(data, "pressure", f"{test}_laser{variant}.MAT")
    m = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
    ch = {m[k].SignalName: m[k.replace("_Header", "_Data")] for k in m if k.startswith("Channel_") and k.endswith("_Header")}
    t = ch[next(k for k in ch if k.startswith("Time"))]
    return np.asarray(t, float), np.asarray(ch["Laser_Depth"], float), np.asarray(ch["Laser_Distance"], float)


def window_centre(t, cur, d, rows):
    """Centre of the window through which the sand is seen, in Laser_Distance [cm]: median over the transects of
    the mid-point of the positions at which the laser reads the sand surface (current above 7.6 mA)."""
    c, w = [], []
    for (_, t0, t1) in rows:
        m = (t >= t0) & (t <= t1) & (cur > 7.6)
        if m.sum() > 100:
            lo, hi = np.percentile(d[m], [0.5, 99.5])
            c.append(0.5 * (lo + hi)); w.append(hi - lo)
    if not c:
        return np.nan, np.nan
    return float(np.median(c)), float(np.median(w))


def load_scan(data, scan):
    """Laser record of one scan occasion (an entry of list_scans): time [s], current [mA], y in the box [mm]."""
    variant = "_error_in_distance" if scan["file"].endswith("FS35_242_15u10.txt") else ""
    t, cur, d = load_laser_raw(data, scan["test"], variant)
    c, w = window_centre(t, cur, d, scan["rows"])
    return t, cur, Y_WINDOW_CENTRE + 10.0 * (c - d), w


def read_timing(path):
    """Returns (comment, list of (tape [cm], t_start [s], t_end [s]))."""
    txt = open(path, encoding="latin-1").read()
    comment = ""
    for line in txt.splitlines():
        if line.startswith("%") and "equilibrium" in line:
            comment = line.strip("% ").strip()
    rows = []
    for mrow in re.finditer(r"\[?\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+NaN", txt):
        rows.append(tuple(float(v) for v in mrow.groups()))
    return comment, rows


def list_scans(data):
    """All scan occasions: dicts with test, file, comment, kind ('complete', 'partial', 'other'), rows."""
    out = []
    root = os.path.join(data, "laser")
    for test in sorted(os.listdir(root)):
        for f in sorted(os.listdir(os.path.join(root, test))):
            if not f.endswith(".txt") or f.endswith("_old.txt"):
                continue
            comment, rows = read_timing(os.path.join(root, test, f))
            c = comment.lower()
            kind = "partial" if ("partial" in c or "partly" in c) else ("other" if "widened" in c else "complete")
            out.append(dict(test=test, file=f, comment=comment, kind=kind, rows=rows))
    return out


def envelope(y, cur, y0, y1, dy=0.5, window=1.0, tol=0.04, nmin=4):
    """Upper envelope of the current on a grid of y.

    Over the water-filled pipe a large part of the readings fall short of the bed (tubes and fittings above the
    cover, reflection at the underside of the cover): such readings are scattered over several mA, whereas readings
    of the bed repeat to about 0.01 mA from pass to pass.  In each window of width `window` the bed is the deepest
    cluster of at least `nmin` readings within `tol` of each other; a single deeper reading is discarded.  Windows
    without such a cluster return nan."""
    yc = np.arange(y0 + dy / 2, y1, dy)
    prof = np.full(yc.size, np.nan)
    num = np.zeros(yc.size, int)
    order = np.argsort(y)
    ys, cs = y[order], cur[order]
    lo = np.searchsorted(ys, yc - window / 2, "left")
    hi = np.searchsorted(ys, yc + window / 2, "right")
    for i in range(yc.size):
        v = np.sort(cs[lo[i]:hi[i]])[::-1]
        for skip in (0, 1):
            if v.size - skip < nmin:
                break
            c = v[skip:][v[skip] - v[skip:] < tol]
            if c.size >= nmin:
                prof[i] = np.median(c)
                num[i] = c.size
                break
    return yc, prof, num


def transect(t, cur, y, t0, t1, half=32.0, dy=0.5):
    """Envelope profile [mA] of one transect over |y - Y_AXIS| < half."""
    w = (t >= t0) & (t <= t1) & (cur > 6.5)       # readings below 6.5 mA are on the frame, not on the sand
    return envelope(y[w], cur[w], Y_AXIS - half, Y_AXIS + half, dy=dy)


def despike(prof, reach=2, tol=0.025, passes=3):
    """Remove readings that are shallower than the bed on both sides of them by more than tol [mA].

    Over the water-filled pipe part of the readings fall short (tubes and fittings above the cover, reflection at
    the underside of the cover); the envelope removes most of them and leaves isolated bins that stand above
    their surroundings.  A bin is dropped if it lies above the deeper of its `reach` neighbours on each side."""
    out = prof.copy()
    n = prof.size
    for _ in range(passes):
        bad = np.zeros(n, bool)
        for i in range(n):
            if not np.isfinite(out[i]):
                continue
            l = out[max(0, i - reach):i]
            r = out[i + 1:i + 1 + reach]
            l, r = l[np.isfinite(l)], r[np.isfinite(r)]
            if l.size and r.size and out[i] < min(l.max(), r.max()) - tol:
                bad[i] = True
        if not bad.any():
            break
        out[bad] = np.nan
    return out


def fill_gaps(yc, dep, max_gap=3.0):
    """Linear interpolation across gaps no longer than max_gap [mm]."""
    out = dep.copy()
    ok = np.isfinite(dep)
    if ok.sum() < 2:
        return out
    idx = np.where(ok)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1 and yc[b] - yc[a] <= max_gap + 1e-9:
            out[a + 1:b] = np.interp(yc[a + 1:b], [yc[a], yc[b]], [dep[a], dep[b]])
    return out


def smooth(prof, win=3):
    """Running mean over win bins that ignores gaps."""
    out = np.full(prof.size, np.nan)
    h = win // 2
    for i in range(prof.size):
        seg = prof[max(0, i - h):min(prof.size, i + h + 1)]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def _lip(yw, pw, nmin=4, tol=0.012):
    """Shallowest cluster of at least nmin points within tol [mA] of each other: (mean y, median level)."""
    order = np.argsort(pw)
    ps, ys = pw[order], yw[order]
    for a in range(ps.size - nmin + 1):
        b = np.searchsorted(ps, ps[a] + tol, "right")
        if b - a >= nmin:
            return float(ys[a:b].mean()), float(np.median(ps[a:b]))
    return None


def roof_baseline(yc, prof, inner=10.0, outer=22.0, thr_mm=0.10):
    """Level of the roof across the pipe [mA]: the straight line from lip to lip.

    The undisturbed sand surface is not at one level across the box (the frame of the laser and the cover are not
    parallel, and in places the surface outside the guides lies up to 0.5 mm lower than between them), so the roof
    is taken locally.  On each side of the axis, between `inner` and `outer` mm from it (the guides are at
    17.5 mm), the lip is the shallowest level that at least four neighbouring readings share; the line joins the two
    lips.  Returns (coefficients of the line, y of the two edges of the eroded zone) or (None, None)."""
    ok = np.isfinite(prof)
    lw = ok & (yc >= Y_AXIS - outer) & (yc <= Y_AXIS - inner)
    rw = ok & (yc >= Y_AXIS + inner) & (yc <= Y_AXIS + outer)
    if lw.sum() < 6 or rw.sum() < 6:
        return None, None
    L, R = _lip(yc[lw], prof[lw]), _lip(yc[rw], prof[rw])
    if L is None or R is None:
        return None, None
    slope = (R[1] - L[1]) / (R[0] - L[0])
    line = np.array([slope, L[1] - slope * L[0]])
    dep = MM_PER_MA * (prof - np.polyval(line, yc))
    er = ok & (np.abs(yc - Y_AXIS) <= GUIDE_HALF + 1.0) & (dep > thr_mm)
    edges = (float(yc[er].min()), float(yc[er].max())) if er.sum() >= 2 else None
    return line, edges


def bed_profile(t, cur, y, t0, t1, half=30.0, dy=0.5):
    """Depth of the bed below the roof [mm] across one transect.
    Returns y [mm], depth [mm] (nan where there is no valid reading), edges of the eroded zone (or None)."""
    yc, prof, _ = transect(t, cur, y, t0, t1, half=half, dy=dy)
    prof = despike(prof)
    line, edges = roof_baseline(yc, prof)
    if line is None:
        return yc, np.full(yc.size, np.nan), None
    return yc, fill_gaps(yc, MM_PER_MA * (prof - np.polyval(line, yc))), edges


def channels(yc, depth, thr=0.15, min_width=1.5):
    """Contiguous stretches where the depth exceeds thr [mm].  Returns a list of dicts sorted by area (largest
    first): y0, y1, width, area [mm^2], amax [mm], y of the deepest point."""
    d = np.where(np.isfinite(depth), depth, 0.0)
    dy = yc[1] - yc[0]
    on = d > thr
    out, i, n = [], 0, on.size
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            width = (j - i + 1) * dy
            if width >= min_width:
                seg = d[i:j + 1]
                out.append(dict(y0=yc[i] - dy / 2, y1=yc[j] + dy / 2, width=width, area=float(seg.sum() * dy),
                                amax=float(seg.max()), ymax=float(yc[i + int(np.argmax(seg))])))
            i = j + 1
        else:
            i += 1
    return sorted(out, key=lambda c: -c["area"])
