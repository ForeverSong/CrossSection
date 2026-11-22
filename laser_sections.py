"""Cross-sections of the pipes of Pol (2022) from the raw laser records (tests 225-250).

For every scan occasion and every transect: the bed profile below the roof, the deepest channel and its shape
(widths at 25, 50 and 75 % of the largest depth, area, bank slopes), the whole eroded zone, and quality flags.
No model is used here; the comparison with the threshold section is in laser_vs_theory.py.

    python laser_sections.py ../Data/Pol_4TU --out ../tmp            (about 2 minutes)

Outputs (in --out)
    paper1_laser_sections.csv     one line per transect
    paper1_laser_profiles.npz     y [mm] and depth [mm] of every transect (keys <test>|<file>|<tape>)
    paper1_laser_sections.txt     summary
    laser_sheets/*.png            one sheet per scan occasion, for checking by eye
"""
import argparse
import os
import numpy as np

from pipetheory import laserdata as ld

SAND_D50 = {"FPH": 0.185, "B25": 0.228, "FS35": 0.422}     # mm
LEVELS = (0.25, 0.50, 0.75)


def crossing(yc, d, i0, level, step):
    """Walk from index i0 in direction step until the depth falls below level; linear interpolation of the
    crossing.  Returns (y of the crossing, index of the last point above the level) or (nan, index) if the
    profile ends or has a gap first."""
    i = i0
    n = yc.size
    while 0 <= i + step < n and np.isfinite(d[i + step]) and d[i + step] >= level:
        i += step
    j = i + step
    if not (0 <= j < n) or not np.isfinite(d[j]):
        return np.nan, i
    return float(yc[i] + (level - d[i]) / (d[j] - d[i]) * (yc[j] - yc[i])), i


def section_metrics(yc, dep, measured):
    """Shape of the deepest channel between the guides.  dep: gap-filled depth [mm]; measured: mask of bins with a
    reading (not interpolated)."""
    out = dict(amax=np.nan, ymax=np.nan)
    core = np.isfinite(dep) & (np.abs(yc - ld.Y_AXIS) <= ld.GUIDE_HALF + 1.0)
    if core.sum() < 10:
        return out
    ds = ld.smooth(dep, 3)                                   # 1.5 mm running mean for the position of the maximum
    i0 = int(np.where(core)[0][np.argmax(ds[core])])
    amax = float(ds[i0])
    out.update(amax=amax, ymax=float(yc[i0]), amax_raw=float(np.nanmax(dep[max(0, i0 - 2):i0 + 3])))
    dy = yc[1] - yc[0]
    for f in LEVELS:
        yl, il = crossing(yc, dep, i0, f * amax, -1)
        yr, ir = crossing(yc, dep, i0, f * amax, +1)
        tag = f"{int(100 * f)}"
        out["w" + tag] = yr - yl
        out["yl" + tag], out["yr" + tag] = yl, yr
        if np.isfinite(yl) and np.isfinite(yr):
            seg = slice(il, ir + 1)
            # area above the roof within the crossings (trapezoids, with the two end pieces)
            a = float(np.sum(dep[seg]) * dy) - 0.5 * dy * (dep[il] + dep[ir])
            a += 0.5 * (yc[il] - yl) * (dep[il] + f * amax) + 0.5 * (yr - yc[ir]) * (dep[ir] + f * amax)
            out["A" + tag] = a
            out["cov" + tag] = float(np.mean(measured[seg]))
        else:
            out["A" + tag] = np.nan
            out["cov" + tag] = 0.0
    # bank slopes between the 25 % and 75 % levels
    for side, a, b in (("L", "yl25", "yl75"), ("R", "yr75", "yr25")):
        run = abs(out[b] - out[a])
        out["slope" + side] = 0.5 * amax / run if np.isfinite(run) and run > 0 else np.nan
    # whole eroded zone between the guides
    er = core & (dep > 0.15)
    out["W_er"] = float(er.sum() * dy)
    out["A_er"] = float(np.sum(dep[er]) * dy)
    out["y_er0"] = float(yc[er].min()) if er.any() else np.nan
    out["y_er1"] = float(yc[er].max()) if er.any() else np.nan
    out["cov_core"] = float(np.mean(measured[np.abs(yc - ld.Y_AXIS) <= ld.GUIDE_HALF]))
    return out


def process(data, out_dir, sheets=True):
    rows_out, profiles = [], {}
    if sheets:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.join(out_dir, "laser_sheets"), exist_ok=True)
    for sc in ld.list_scans(data):
        try:
            t, cur, y, wwin = ld.load_scan(data, sc)
        except FileNotFoundError:
            continue
        if not np.isfinite(wwin) or wwin < 20.0:
            continue
        sand = sc["test"].split("_")[0]
        if sheets:
            n = len(sc["rows"])
            fig, axs = plt.subplots(n, 1, figsize=(8.5, 1.2 * n + 0.6), sharex=True)
            axs = np.atleast_1d(axs)
        for k, (tape, t0, t1) in enumerate(sc["rows"]):
            yc, prof, _ = ld.transect(t, cur, y, t0, t1, half=30.0)
            prof = ld.despike(prof)
            line, _ = ld.roof_baseline(yc, prof)
            rec = dict(test=sc["test"], sand=sand, file=sc["file"], kind=sc["kind"], tape_cm=tape,
                       x_m=ld.X_TAPE0 + 0.01 * tape, t0=t0, t1=t1, baseline=line is not None)
            if line is not None:
                raw = ld.MM_PER_MA * (prof - np.polyval(line, yc))
                dep = ld.fill_gaps(yc, raw)
                rec.update(section_metrics(yc, dep, np.isfinite(raw)))
                # roughness of the roof line: r.m.s. of the sand surface outside the eroded zone, between the guides
                flat = np.isfinite(raw) & (np.abs(yc - ld.Y_AXIS) <= ld.GUIDE_HALF + 4.0) & (raw < 0.15)
                rec["roof_rms"] = float(np.sqrt(np.mean(raw[flat] ** 2))) if flat.sum() > 5 else np.nan
                profiles[f"{sc['test']}|{sc['file']}|{tape}"] = np.c_[yc, dep, np.isfinite(raw)]
            rows_out.append(rec)
            if sheets:
                ax = axs[k]
                if line is not None:
                    ax.plot(yc, dep, "-", color="0.6", lw=0.7)
                    ax.plot(yc, raw, "k.", ms=2)
                    if np.isfinite(rec.get("w50", np.nan)):
                        ax.plot([rec["yl50"], rec["yr50"]], [0.5 * rec["amax"]] * 2, "r-", lw=1)
                    if np.isfinite(rec.get("w25", np.nan)):
                        ax.plot([rec["yl25"], rec["yr25"]], [0.25 * rec["amax"]] * 2, "b-", lw=1)
                ax.axhline(0, color="0.6", lw=0.5)
                ax.set_ylim(3.2, -0.4)
                lab = f"tape {tape:.1f} cm"
                if line is not None and np.isfinite(rec.get("amax", np.nan)):
                    lab += f"   a_max {rec['amax']:.2f}  w50 {rec['w50']:.1f}  cov {rec['cov50']:.2f}"
                ax.text(0.01, 0.12, lab, transform=ax.transAxes, fontsize=7)
                for g in (ld.Y_AXIS - ld.GUIDE_HALF, ld.Y_AXIS + ld.GUIDE_HALF):
                    ax.axvline(g, color="b", lw=0.4, ls=":")
        if sheets:
            axs[0].set_title(f"{sc['test']}  {sc['comment']}  [{sc['kind']}]", fontsize=8)
            axs[-1].set_xlabel("y [mm]")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "laser_sheets", sc["file"].replace("timing_transects_", "").replace(".txt", ".png")), dpi=70)
            plt.close(fig)
    return rows_out, profiles


def good(r, level="50", cov=0.8):
    """A transect whose deepest channel is resolved down to the given level."""
    return bool(r["baseline"]) and np.isfinite(r.get("w" + level, np.nan)) and r.get("cov" + level, 0) >= cov \
        and r["amax"] >= 0.4 and r.get("roof_rms", 1.0) < 0.12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="../tmp")
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args()
    rows, profiles = process(a.data, a.out, sheets=not a.no_sheets)
    keys = ["test", "sand", "file", "kind", "tape_cm", "x_m", "t0", "t1", "baseline", "amax", "amax_raw", "ymax",
            "w25", "w50", "w75", "A25", "A50", "A75", "cov25", "cov50", "cov75", "slopeL", "slopeR",
            "W_er", "A_er", "y_er0", "y_er1", "cov_core", "roof_rms", "good25", "good50"]
    for r in rows:
        r["good25"], r["good50"] = good(r, "25"), good(r, "50")
    with open(os.path.join(a.out, "paper1_laser_sections.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(("%.4g" % r[k] if isinstance(r.get(k), float) else str(r.get(k, ""))) for k in keys) + "\n")
    np.savez_compressed(os.path.join(a.out, "paper1_laser_profiles.npz"), **profiles)

    lines = [f"mm per mA = {ld.MM_PER_MA} (matched to figure 3.10 of the thesis); roof = line from lip to lip",
             f"scan occasions: {len(set((r['test'], r['file']) for r in rows))}; transects: {len(rows)}; "
             f"with a roof line: {sum(r['baseline'] for r in rows)}; deepest channel resolved to half depth: "
             f"{sum(r['good50'] for r in rows)}; to quarter depth: {sum(r['good25'] for r in rows)}", ""]
    lines.append("deepest channel, transects resolved to half depth (mean +- s.d.; n)")
    lines.append("sand  kind      n | a_max [mm]  a_max/d50 | w50/a_max   w75/a_max | w25/a_max (n) | A_er [mm2]  W_er [mm]")
    for sand in ("FPH", "B25", "FS35"):
        for kind in ("complete", "partial"):
            sel = [r for r in rows if r["sand"] == sand and r["kind"] == kind and r["good50"]]
            if not sel:
                continue
            g = lambda k, s=sel: np.array([r[k] for r in s], float)
            am = g("amax")
            s25 = [r for r in sel if r["good25"]]
            w25 = np.array([r["w25"] / r["amax"] for r in s25]) if s25 else np.array([np.nan])
            lines.append(f"{sand:5s} {kind:8s} {len(sel):3d} | {am.mean():.2f}+-{am.std():.2f}  {am.mean() / SAND_D50[sand]:5.1f}     | "
                         f"{(g('w50') / am).mean():.2f}+-{(g('w50') / am).std():.2f}  {np.nanmean(g('w75') / am):.2f}+-{np.nanstd(g('w75') / am):.2f} | "
                         f"{np.nanmean(w25):.2f}+-{np.nanstd(w25):.2f} ({len(s25)}) | {g('A_er').mean():5.1f}+-{g('A_er').std():.1f}  {g('W_er').mean():.1f}+-{g('W_er').std():.1f}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(a.out, "paper1_laser_sections.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
