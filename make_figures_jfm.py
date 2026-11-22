"""The seven figures of the manuscript, in one style (figstyle.py), at the width of the JFM text block.

    fig01_sketch    the configuration: long section, cross-section, oblique view (sand and water transparent)
    fig02_section   threshold cross-section: shapes, discharge coefficient, aspect ratio, a scanned bed
    fig03_fold      lumped model, the two branches and the fold, coefficient of the scale law
    fig04_inertia   wide gap with injection; head at the fold with inertia
    fig05_laser     sections of pipes at rest from the laser scans (data written by make_figure_laser.py)
    fig06_small     small-scale tests: heads at rest, critical head, depth along the pipe (computed at fixed distances
                    from the tip, against the regression of the scans)
    fig07_scales    larger scales: width and depth of the conveying channel, gradient against discharge

The figures show what the earlier scripts showed (make_paper_figures.py, make_figure_*.py, make_figures_extra.py); only
the presentation differs.  Curves that take time to compute are kept in tmp/paper1_figcache.pkl (delete it, or pass
--recompute, to compute them again).  Needs in tmp/: paper1_pol_states.csv, paper1_pol_states_w30.csv,
paper1_pol_box_predictions.csv, paper1_pipe_width_data.csv, paper1_inertia*.csv, paper1_laser_figdata.json.
usage:  python make_figures_jfm.py [--only 1,3] [--recompute]
"""
import argparse
import csv
import json
import os
import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.ticker import NullFormatter, FixedLocator
import figstyle as fs
from figstyle import INK, INK2, GREY, LIGHT, BLUE, ORANGE, AQUA, WATER, SAND, ROOF, SANDS

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, "..", "tmp")
W = fs.TEXTWIDTH
STIPPLE = "#b9a66f"
STATIONS = (0.02, 0.04, 0.07, 0.10, 0.14)      # distances from the tip at which the computed depth is shown [m]


def rd(name):
    with open(os.path.join(TMP, name)) as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------------------------------------------------
# computed curves
# ----------------------------------------------------------------------------------------------------------------------
def compute():
    from pipetheory.aquifer import pol_box, okamura_box, PipeResponse
    from pipetheory.coupled import trace_branches, fold, c3_from_N, solve_pipe_law, fold_law
    from pipetheory.pipelaw import PipeLaw
    from pipetheory.section import threshold_section
    from pipe_width_data import read_csv
    c = {}
    c["shapes"] = {mu: {k: np.asarray(v) for k, v in threshold_section(mu, 16, 161, 41).items() if k in ("y", "a")} for mu in (0.5, 0.7, 0.9)}
    mus = np.arange(0.4, 1.01, 0.1)
    o = [threshold_section(m, 16, 161, 41) for m in mus]
    c["mus"], c["chi"], c["aspect"] = mus, np.array([x["chi"] for x in o]), np.array([x["W_over_aavg"] for x in o])
    s77 = threshold_section(np.tan(np.radians(37.6)), 16, 161, 41)
    c["s77"] = dict(y=np.asarray(s77["y"]), a=np.asarray(s77["a"]))
    # the two branches (figure 3b)
    box = pol_box()
    R = PipeResponse(box, 0.18)
    Nmin, br = trace_branches(R, N_start_factor=600.0)
    x = np.array([(b["N"] / Nmin) ** 0.25 for b in br]); d = np.array([b["h"][-1] for b in br]); st = np.array([b["branch"] == 0 for b in br])
    o_ = np.argsort(x[st])
    c["fold"] = dict(x_large=x[st][o_], d_large=d[st][o_], x_pinch=x[~st], d_pinch=d[~st], d_fold=float(d[np.argmin(x)]))
    # coefficient of the scale law (figure 3c)
    chi = s77["chi"]
    c["c3"] = {}
    for name, bx in (("hole", pol_box()), ("toe", okamura_box())):
        L = bx.seepage_length
        res = []
        for frac in np.linspace(0.08, 0.92, 15):
            try:
                Rl = PipeResponse(bx, frac * L, rtol=1e-9)
                res.append((Rl.length / L, c3_from_N(fold(Rl)[0], chi, L)))
            except Exception:
                continue
        c["c3"][name] = np.array(res)
    # depth along the pipe, 29 states at rest (figure 6c)
    sands = {r["sand"]: r for r in read_csv("pol2022_sands.csv")}
    kk = {r["test"]: float(r["k_m_per_s"]) for r in read_csv("pol2022_table3_2.csv")}
    rows = [r for r in rd("paper1_pol_states.csv") if r["variant"] == "base" and r["test"] in kk]
    c["depth"] = []
    for r in rows:
        sd = sands[r["sand"]]; k = kk[r["test"]]; kappa = k * 1e-6 / 9.81
        sec = threshold_section(np.tan(np.radians(float(sd["repose_deg"]))), 16, 161, 41)
        law = PipeLaw(sec, float(sd["tau_eq_Pa"]) / (sec["a_max"] * sec["aavg_over_amax"]))
        Rs = PipeResponse(pol_box(), float(r["l_cm"]) / 100, rtol=1e-9)
        Hf, sf = fold_law(Rs, law, k)
        H = max(float(r["H_net_cm"]) / 100, 1.001 * Hf)
        sol = solve_pipe_law(Rs, law, H, k, sf["h"] * H / Hf) or sf
        back = Rs.x_tip - Rs.x_st; amax = law.geometry(sol["S"])[1]
        m = (back > 0.02) & (back < Rs.length - 0.01)
        c["depth"].append(dict(sand=r["sand"], ks=kappa * back[m], a_mm=amax[m] * 1e3, s=back[m], kappa=kappa))
    # regime diagram (figure 7c)
    sec = threshold_section(0.77, 16, 161, 41)
    pts = []
    for r in rd("paper1_pipe_width_data.csv"):
        law = PipeLaw(sec, float(r["tau_c"]), W0=30 * float(r["d50"]))
        pts.append((r["scale"], float(r["Q"]) / law.Q_x, float(r["S"]) / law.S_x, r["sand"]))
    fph = PipeLaw(sec, 0.339 / (sec["a_max"] * sec["aavg_over_amax"]), W0=30 * 0.185e-3, nu=1.52e-6)
    ref = PipeLaw(sec, 0.45, W0=30 * 0.2e-3)
    q = np.logspace(-2.5, 4.3, 300)
    c["regimes"] = dict(pts=pts, large=[(qq / fph.Q_x, 0.111 / fph.S_x) for qq in (18e-6, 24e-6)], q=q, s=ref.slope(q * ref.Q_x)[0] / ref.S_x,
                        plateaus={cc: 15.0 / (ref.beta * cc) for cc in (4, 7)})
    return c


def cache(recompute=False):
    path = os.path.join(TMP, "paper1_figcache.pkl")
    if os.path.exists(path) and not recompute:
        return pickle.load(open(path, "rb"))
    c = compute()
    pickle.dump(c, open(path, "wb"))
    return c


# ----------------------------------------------------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------------------------------------------------
def dim(ax, p0, p1, text, dx=0.0, dy=0.0, **kw):
    """Dimension line with arrowheads at both ends and a label at its middle."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<|-|>", mutation_scale=5.5, lw=0.6, color=INK, shrinkA=0, shrinkB=0, zorder=7))
    ax.text(0.5 * (p0[0] + p1[0]) + dx, 0.5 * (p0[1] + p1[1]) + dy, text, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"), zorder=8, **kw)


def flow(ax, p0, p1, rad=0.0, color=BLUE, lw=0.9, ms=6.5, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, **kw))


def stipple(ax, x0, x1, y0, y1, n, seed, keep=None):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)
    if keep is not None:
        m = keep(x, y); x, y = x[m], y[m]
    ax.plot(x, y, ".", ms=0.9, color=STIPPLE, mew=0, zorder=1.5)


def roof(ax, x0, x1, y0, y1):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=ROOF, ec=INK2, lw=0.0, hatch="//////", zorder=2))
    ax.plot([x0, x1], [y0, y0], color=INK, lw=1.4, zorder=3, solid_capstyle="butt")


# ----------------------------------------------------------------------------------------------------------------------
# figure 1: the configuration
# ----------------------------------------------------------------------------------------------------------------------
RED = "#d32f2f"                                                   # the cross-section marked in figure 1(a) and drawn in figure 1(c)


def oblique_view(a3, ga=False):
    """The sand bed, its roof and the pipe seen obliquely: x from the source to the exit, y across the bed (into the
    page), z upwards; sand and water are drawn transparent so that the pipe under the roof is seen.  This is figure
    1(a) or, with ga=True, the graphical abstract: seen from higher up, bolder and without text, since the journal shows
    it as a thumbnail 2.4 cm wide.  Returns the extent of the drawing (x0, x1, y0, y1)."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap
    if ga:
        Lx, B, D, xr, Hw, tr = 6.6, 5.2, 1.7, 0.95, 0.80, 0.20
        cy, sy = 0.35, 0.95
        x_e, x_t, wk, dk, mk, k = 5.75, 2.35, 1.9, 1.5, 1.6, 1.8
    else:
        Lx, B, D, xr, Hw, tr = 10.0, 4.2, 1.55, 1.15, 0.85, 0.15     # sand: length, width, depth; source: length, level; roof: thickness
        cy, sy = 0.42, 0.60                                           # how a unit of y is drawn
        x_e, x_t, wk, dk, mk, k = 8.0, 3.9, 1.0, 1.0, 1.0, 1.0       # pipe: exit and tip; factors on its width, depth and meander; on the line widths

    def P(x, y_, z):
        x, y_, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y_, float), np.asarray(z, float))
        return np.stack([x + cy * y_, z + sy * y_], axis=-1)

    def face(pts, fc, alpha, ec=INK2, lw=0.5, z=2, ls="-"):
        if alpha > 0:
            a3.add_patch(Polygon(P(*np.array(pts, float).T), closed=True, fc=fc, alpha=alpha, ec="none", zorder=z))
        if ec is not None:
            a3.add_patch(Polygon(P(*np.array(pts, float).T), closed=True, fc="none", ec=ec, lw=lw * k, ls=ls, zorder=z + 0.01, joinstyle="round"))

    def box(x0, x1, y0, y1, z0, z1):
        return dict(back=[(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], left=[(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
                    bottom=[(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], front=[(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
                    right=[(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], top=[(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])

    sand, src = box(0, Lx, 0, B, -D, 0), box(-xr, 0, 0, B, -D, Hw)
    for kf in ("back", "left", "bottom"):                      # far faces first
        face(src[kf], WATER, 0.28, ec=GREY, lw=0.5, z=1)
    for kf in ("back", "bottom"):
        face(sand[kf], SAND, 0.40, ec=GREY, lw=0.5, z=1.2)
    face(sand["left"], SAND, 0.0, ec=INK2, lw=0.6, z=1.3, ls=(0, (2.5, 1.5)))          # the face through which the water enters
    for kf in ("front", "right"):
        face(sand[kf], SAND, 0.60 if not ga else 0.85, z=2)
    rng = np.random.default_rng(11)
    nd = 260 if not ga else 170
    xd, zd = rng.uniform(0.1, Lx - 0.1, nd), rng.uniform(-D + 0.06, -0.06, nd)
    a3.plot(*P(xd, 0, zd).T, ".", ms=0.9 * k, color=STIPPLE, mew=0, zorder=2.1)
    face(sand["top"], SAND, 0.50, z=2.2)
    face(src["front"], WATER, 0.55 if not ga else 0.85, z=2); face(src["top"], WATER, 0.70 if not ga else 0.9, ec=BLUE, lw=0.6, z=2.3)
    face([(0, 0, 0), (0, B, 0), (0, B, Hw), (0, 0, Hw)], WATER, 0.28, ec=GREY, lw=0.5, z=1.9)

    # the pipe: a trough under the roof, as wide as its tip where its discharge is small, wider and deeper towards the exit
    t = np.linspace(0, 1, 90); eta = np.linspace(-1, 1, 21)
    xp = x_t + (x_e - x_t) * t
    yc = 0.5 * B + 0.17 * mk * np.sin(5.2 * t + 0.4) * (4 * t * (1 - t)) ** 0.7
    end = np.sqrt(np.clip(1 - (np.clip(t - 0.93, 0, None) / 0.07) ** 2, 0, 1))      # rounded end around the exit hole
    hw = wk * (0.085 + 0.40 * t ** 0.62) * end                      # half-width
    de = dk * (0.05 + 0.33 * t ** 0.40) * end                       # depth on the axis (exaggerated)
    X = xp[:, None] + 0 * eta[None, :]; Y = yc[:, None] + hw[:, None] * eta[None, :]; Z = -de[:, None] * np.cos(0.5 * np.pi * eta[None, :])
    Q2 = P(X, Y, Z)
    quads, shade = [], []
    for i in range(len(t) - 1):
        for j in range(len(eta) - 1):
            quads.append([Q2[i, j], Q2[i + 1, j], Q2[i + 1, j + 1], Q2[i, j + 1]])
            em = 0.5 * (eta[j] + eta[j + 1]); tm = 0.5 * (t[i] + t[i + 1])
            shade.append((0.22 + 0.78 * tm ** 0.75) * np.cos(0.5 * np.pi * em) ** 1.1 * (1.0 + 0.30 * em))      # deeper is darker; the far bank is lit
    cmap = LinearSegmentedColormap.from_list("pipe", ["#e6f1fe", "#bcd9f9", "#86b6ef", "#4a8fe0"] if not ga else ["#cfe3fb", "#93c0f2", "#5a9be6", "#2a78d6"])
    pc = PolyCollection(quads, array=np.clip(shade, 0, 1), cmap=cmap, edgecolors="face", linewidths=0.5, zorder=3)
    pc.set_clim(0, 1); a3.add_collection(pc)
    low = np.array([Q2[i, int(np.argmin(Q2[i, :, 1]))] for i in range(len(t))])                          # the bed as seen through the sand
    a3.plot(low[:, 0], low[:, 1], color="#2f6fc0", lw=0.6 * k, zorder=3.15)
    edge = np.r_[P(xp, yc - hw, 0), P(xp[::-1], (yc + hw)[::-1], 0)]
    a3.add_patch(Polygon(edge, closed=True, fc="none", ec=INK, lw=0.8 * k, zorder=3.2, joinstyle="round"))

    def arrow(p0, p1, lw=0.9, ms=6.0):                            # seepage on the plane under the roof
        a3.add_patch(FancyArrowPatch(tuple(P(p0[0], p0[1], 0.0)), tuple(P(p1[0], p1[1], 0.0)), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=BLUE,
                                     zorder=3.4, shrinkA=0, shrinkB=0))

    if ga:
        arrow((0.30, 0.5 * B), (x_t - 0.28, yc[0]), 2.0, 10)
        for sgn in (-1, 1):
            for kk in (24, 54):
                arrow((xp[kk] - 0.45, yc[kk] + sgn * (hw[kk] + 1.25)), (xp[kk], yc[kk] + sgn * (hw[kk] + 0.14)), 2.0, 10)
    else:                                                         # what the long section (b) cannot show: the seepage that enters from the sides
        for kk, sgn in ((30, -1), (46, -1), (12, 1), (28, 1)):
            arrow((xp[kk] - 0.18, yc[kk] + sgn * (hw[kk] + 0.66)), (xp[kk], yc[kk] + sgn * (hw[kk] + 0.07)))

    # roof: a transparent slab with a hatched edge, as in (b) and (c)
    slab = box(0, Lx, 0, B, 0, tr)
    face(slab["top"], "#ffffff", 0.20, ec=INK2, lw=0.5, z=4)
    for kf in ("front", "right"):
        a3.add_patch(Polygon(P(*np.array(slab[kf], float).T), closed=True, fc=ROOF, ec=INK2, lw=0.5 * k, hatch=None if ga else "//////", zorder=4.1))
    le = 1.2 * (1.35 if ga else 1.0)                              # the underside of the roof
    a3.plot(*P([0, Lx], 0, 0).T, color=INK, lw=le, zorder=4.2, solid_capstyle="butt"); a3.plot(*P(Lx, [0, B], 0).T, color=INK, lw=le, zorder=4.2, solid_capstyle="butt")

    if not ga:
        # the cross-section drawn in (c): its outline in red, the trace of the cutting plane under the roof, and the depth a(y)
        i_s = 60; ef = np.linspace(-1, 1, 61)
        sec = P(xp[i_s], yc[i_s] + hw[i_s] * ef, -de[i_s] * np.cos(0.5 * np.pi * ef))
        a3.add_patch(Polygon(sec, closed=True, fc=RED, alpha=0.32, ec="none", zorder=4.3))
        a3.add_patch(Polygon(sec, closed=True, fc="none", ec=RED, lw=1.3, zorder=4.4, joinstyle="round"))
        cut = P(xp[i_s], yc[i_s] - hw[i_s] - np.array([0.0, 0.62]), 0)             # carries the label, on the near side
        a3.plot(cut[:, 0], cut[:, 1], color=RED, lw=0.7, ls=(0, (3, 1.6)), zorder=4.35)
        a3.text(cut[1, 0] - 0.26, cut[1, 1] - 0.06, "cross-section ($c$)", fontsize=8, color=RED, ha="center", va="top", zorder=8)
        top_s, bot_s = P(xp[i_s], yc[i_s], 0), P(xp[i_s], yc[i_s], -de[i_s])
        a3.plot([top_s[0], bot_s[0]], [top_s[1], bot_s[1]], color=RED, lw=0.9, zorder=4.45)
        mid = 0.5 * (top_s + bot_s)
        a3.annotate("$a(y)$", xy=(mid[0], mid[1]), xytext=(mid[0] - 0.16, mid[1] + 0.93), fontsize=8.5, color=RED, ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=RED, shrinkA=1, shrinkB=0), zorder=8)

    # exit: a hole in the roof with a standpipe
    g = 1.5 if ga else 1.0
    th = np.linspace(0, 2 * np.pi, 80); r_e = 0.2 * g; ye = yc[-1]; xc = x_e - 0.24 * g; top_ = 0.92 * (1.2 if ga else 1.0); lev = 0.58 * (1.2 if ga else 1.0)
    ring = lambda z: P(xc + r_e * np.cos(th), ye + r_e * np.sin(th), z)
    lo, wl, hi = ring(tr), ring(lev), ring(top_)
    il, ir = int(np.argmin(lo[:, 0])), int(np.argmax(lo[:, 0]))
    a3.add_patch(Polygon(np.r_[lo[[il]], wl[[il]], wl[[ir]], lo[[ir]]], closed=True, fc=WATER, ec="none", alpha=0.85, zorder=5))
    a3.add_patch(Polygon(lo, closed=True, fc=WATER, ec=INK, lw=0.7 * k, zorder=4.9))
    a3.add_patch(Polygon(wl, closed=True, fc=WATER, ec=BLUE, lw=0.6 * k, zorder=5.2))
    for i_ in (il, ir):
        a3.plot([lo[i_, 0], hi[i_, 0]], [lo[i_, 1], hi[i_, 1]], color=INK, lw=0.7 * k, zorder=5.3)
    a3.add_patch(Polygon(hi, closed=True, fc="none", ec=INK, lw=0.7 * k, zorder=5.3))
    cq = P(xc, ye, top_ + 0.08)
    if ga:
        flow(a3, (cq[0], cq[1]), (cq[0], cq[1] + 1.15), 0.0, lw=2.2, ms=11, zorder=6)
    else:
        flow(a3, (cq[0], cq[1]), (cq[0], cq[1] + 0.85), 0.0, zorder=6); a3.text(cq[0] + 0.17, cq[1] + 0.62, "$Q$", va="center")
        a3.text(cq[0] + 0.42, cq[1] - 0.08, "exit, $h=0$", fontsize=8, ha="left", va="center", zorder=8)

    # source
    w0 = P(-0.5 * xr, 0.5 * B, Hw); a3.plot([w0[0]], [w0[1] + 0.12 * g], marker="v", ms=3.5 * k, color=BLUE, mew=0, zorder=6)
    lo_ = P(-xr, 0, -D)
    ext = (lo_[0], P(Lx, B, 0)[0], lo_[1], max(cq[1] + (1.25 if ga else 1.05), P(0, B, Hw)[1] + 0.15))
    if ga:
        return ext
    f0 = P(-0.5 * xr, 0, 0.5 * (Hw - D)); a3.text(f0[0], f0[1], "$h=H$", rotation=90, ha="center", va="center", fontsize=8, zorder=7)

    # lengths, on the front face; dotted lines carry the tip and the exit to the front edge
    for xx, y0 in ((x_t, yc[0] - hw[0]), (xc, ye - r_e)):
        a3.plot(*P(xx, [y0 - 0.04, 0.0], tr).T, color=INK2, lw=0.5, ls=(0, (1.5, 1.5)), zorder=4.3)
    for xx, zz in ((x_t, -1.02), (xc, -1.02), (0.0, -0.52)):
        a3.plot(*P(xx, 0, [-0.03, zz]).T, color=INK2, lw=0.5, ls=(0, (1.5, 1.5)), zorder=2.5)
    dim(a3, tuple(P(0.0, 0, -0.42)), tuple(P(xc, 0, -0.42)), "$L$", dx=-2.2, dy=-0.2)
    dim(a3, tuple(P(x_t, 0, -0.92)), tuple(P(xc, 0, -0.92)), "$l$", dy=-0.2)
    pt = P(x_t, yc[0], 0)
    a3.annotate("tip", xy=(pt[0] - 0.02, pt[1] + 0.05), xytext=(pt[0] - 0.35, pt[1] + 0.95), fontsize=8, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1, shrinkB=1), zorder=8)
    pw = P(x_t, yc[0] - hw[0], 0)
    a3.annotate("$W_0$", xy=(pw[0] + 0.02, pw[1] + 0.03), xytext=(pw[0] + 0.30, pw[1] - 0.62), fontsize=8.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1, shrinkB=1), zorder=8)
    pp = P(xp[44], yc[44] + hw[44], 0)
    a3.annotate("pipe", xy=(pp[0], pp[1] + 0.02), xytext=(pp[0] - 0.45, pp[1] + 0.72), fontsize=8, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1, shrinkB=1), zorder=8)
    pr = P(Lx - 0.22, 0.53, tr); a3.text(pr[0], pr[1], "impermeable\nroof", fontsize=8, ha="right", va="center", linespacing=0.95, zorder=8)
    ps = P(Lx - 0.25, 0, -D + 0.25); a3.text(ps[0], ps[1], "sand", fontsize=8, ha="right", va="center", zorder=8)
    a3.set_xlim(ext[0] - 0.75, ext[1] + 0.35); a3.set_ylim(ext[2] - 0.12, ext[3]); a3.set_aspect("equal"); a3.axis("off")
    return ext


def fig_graphical_abstract(c=None):
    """Graphical abstract: a single panel of 6 cm x 5 cm (aspect ratio 1.2:1) without text, as the journal asks; written
    as a vector PDF, a PNG and a JPEG at 600 dpi."""
    wcm, hcm = 6.0, 5.0
    fig = plt.figure(figsize=(wcm / 2.54, hcm / 2.54)); ax = fig.add_axes([0, 0, 1, 1])
    x0, x1, y0, y1 = oblique_view(ax, ga=True)
    m = 0.035 * (x1 - x0); x0, x1, y0, y1 = x0 - m, x1 + m, y0 - m, y1 + m
    w, h, r = x1 - x0, y1 - y0, wcm / hcm
    if w / h < r:
        x0, x1 = x0 - 0.5 * (r * h - w), x1 + 0.5 * (r * h - w)
    else:
        y0, y1 = y0 - 0.5 * (w / r - h), y1 + 0.5 * (w / r - h)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        for ext, kw in (("pdf", {}), ("png", dict(dpi=600)), ("jpg", dict(dpi=600, pil_kwargs=dict(quality=95)))):
            fig.savefig(os.path.join(fs.FIG, "graphical_abstract." + ext), facecolor="white", **kw)
    plt.close(fig)


def fig_sketch(c):
    fig = plt.figure(figsize=(W, 4.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0], width_ratios=[1.62, 1.0], left=0.008, right=0.992, top=0.99, bottom=0.0, hspace=0.07, wspace=0.03)
    a3, a1, a2 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])     # (a) oblique view; (b) long section; (c) cross-section

    # (b) long section: the source on the left, the exit on the right
    xe, xt = 8.0, 4.0
    xs = np.linspace(xt, xe, 80); dep = 0.36 * ((xs - xt) / (xe - xt)) ** 0.34
    a1.add_patch(Rectangle((0, -3), 10, 3, fc=SAND, ec="none", zorder=1))
    stipple(a1, 0.05, 9.95, -2.95, -0.05, 520, 3, keep=lambda x, y: ~((x > xt) & (x < xe) & (y > -0.36 * np.clip((x - xt) / (xe - xt), 0, 1) ** 0.34)))
    a1.add_patch(Rectangle((-0.8, -3), 0.8, 4.3, fc=WATER, ec="none", zorder=1))
    a1.plot([-0.8, 0], [1.3, 1.3], color=BLUE, lw=0.9); a1.plot([-0.4], [1.42], marker="v", ms=3.5, color=BLUE, mew=0)
    a1.plot([0, 0], [-3, 0], color=INK2, lw=0.7, ls=(0, (2.5, 1.5)), zorder=3)
    roof(a1, 0, 10, 0, 0.42)
    a1.add_patch(Rectangle((xe - 0.02, -0.01), 0.20, 0.68, fc=WATER, ec="none", zorder=4))
    a1.plot([xe - 0.02, xe - 0.02], [0, 0.67], color=INK, lw=0.8, zorder=5); a1.plot([xe + 0.18, xe + 0.18], [0, 0.67], color=INK, lw=0.8, zorder=5)
    a1.plot([xe - 0.02, xe + 0.18], [0.67, 0.67], color=BLUE, lw=0.9, zorder=5)
    a1.fill_between(xs, -dep, 0, color=WATER, lw=0, zorder=2.5); a1.plot(xs, -dep, color=INK, lw=0.9, zorder=4)
    a1.plot([-1.35, xe - 0.1], [0.67, 0.67], color=GREY, lw=0.5, ls=(0, (2, 2)), zorder=1)
    dim(a1, (-1.35, 0.67), (-1.35, 1.3), "$H$", dx=-0.24)
    for p0, p1, r in (((0.05, -0.55), (4.3, -0.20), 0.10), ((0.05, -1.30), (5.5, -0.32), 0.20), ((0.05, -2.30), (6.8, -0.38), 0.28)):
        flow(a1, p0, p1, r, zorder=3.5)
    flow(a1, (xe + 0.08, 0.8), (xe + 0.08, 1.62), 0.0, zorder=6); a1.text(xe + 0.27, 1.40, "$Q$", va="center")
    dim(a1, (xt, -1.0), (xe, -1.0), "$l$", dy=-0.28)
    dim(a1, (0.0, -2.05), (xe, -2.05), "$L$", dy=-0.28)
    a1.add_patch(FancyArrowPatch((xe - 0.35, 0.98), (xe - 1.55, 0.98), arrowstyle="-|>", mutation_scale=5.5, lw=0.6, color=INK)); a1.text(xe - 1.68, 0.98, "$s$", va="center", ha="right")
    a1.text(xe + 0.42, 0.96, "exit, $h=0$", ha="left", va="center", fontsize=8)
    a1.text(1.75, 0.21, "impermeable roof", ha="center", va="center", fontsize=8, zorder=6, bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))
    a1.annotate("tip", xy=(xt - 0.02, -0.02), xytext=(xt - 0.5, 1.0), fontsize=8, ha="right", va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1, shrinkB=1), zorder=8)
    a1.text(9.75, -2.72, "sand, conductivity $k$", fontsize=8, ha="right")
    a1.text(xe - 0.5, -0.68, "pipe", fontsize=8, ha="center")
    a1.text(-0.4, -1.5, "$h=H$", rotation=90, va="center", ha="center", fontsize=8)
    a1.text(0.50, -2.72, "seepage", fontsize=8, ha="left")
    a1.set_xlim(-1.8, 10.75); a1.set_ylim(-3.08, 1.72); a1.axis("off")

    # (c) cross-section, outlined in the red of its mark in (a)
    y = np.linspace(-1, 1, 201); sh = -np.cos(np.pi * y / 2) ** 0.9
    a2.add_patch(Rectangle((-1.85, -1.85), 3.85, 1.85, fc=SAND, ec="none", zorder=1))
    stipple(a2, -1.8, 1.95, -1.8, -0.03, 330, 5, keep=lambda x, yy: ~((np.abs(x) < 1) & (yy > -np.cos(np.pi * np.clip(x, -1, 1) / 2) ** 0.9)))
    a2.add_patch(Polygon(np.c_[y, sh], closed=True, fc=WATER, ec="none", zorder=2.5)); a2.plot(y, sh, color=RED, lw=1.2, zorder=4); a2.plot([-1, 1], [0, 0], color=RED, lw=1.2, zorder=4.5, solid_capstyle="butt")
    roof(a2, -1.85, 2.0, 0, 0.26)
    for yy in (-0.66, -0.3, 0.3, 0.66):
        j = int(np.argmin(np.abs(y - yy))); t = np.array([y[j + 1] - y[j - 1], sh[j + 1] - sh[j - 1]]); n = np.array([t[1], -t[0]]) / np.hypot(*t)
        p = np.array([y[j], sh[j]]); flow(a2, tuple(p + 0.52 * n), tuple(p + 0.06 * n), 0.0, ms=5.5, lw=0.8, zorder=3.5)
    dim(a2, (-1, 0.55), (1, 0.55), "$W$", dy=0.17)
    dim(a2, (0, 0), (0, -1), "$a(y)$", dx=0.1, dy=0.08, ha="left", fontsize=8.5)
    a2.text(-0.74, -0.9, r"$\varphi$", fontsize=9)
    a2.add_patch(FancyArrowPatch((1.2, -1.15), (1.78, -1.15), arrowstyle="-|>", mutation_scale=5.5, lw=0.6, color=INK)); a2.text(1.84, -1.15, "$y$", va="center")
    a2.add_patch(FancyArrowPatch((1.2, -1.15), (1.2, -1.72), arrowstyle="-|>", mutation_scale=5.5, lw=0.6, color=INK)); a2.text(1.3, -1.7, "$z$", va="center")
    a2.text(-1.75, -1.68, "seepage", color=INK, fontsize=8)
    a2.set_xlim(-1.95, 2.1); a2.set_ylim(-1.90, 1.06); a2.axis("off")          # roof and base level with those of (b)

    oblique_view(a3)                                              # (a) oblique view
    fs.panel(a3, "a", 0.0, 0.95); fs.panel(a1, "b", 0.0, 0.97); fs.panel(a2, "c", 0.0, 0.97)
    fs.save(fig, "fig01_sketch")


# ----------------------------------------------------------------------------------------------------------------------
# figure 2: the threshold cross-section
# ----------------------------------------------------------------------------------------------------------------------
def fig_section(c):
    fig = plt.figure(figsize=(W, 3.95))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.72], width_ratios=[1.45, 1.0, 1.0], left=0.085, right=0.985, top=0.945, bottom=0.095, hspace=0.50, wspace=0.52)
    a1, a2, a3, a4 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, :])
    # (a) shapes
    o = c["shapes"][0.7]
    a1.fill_between(np.r_[-o["y"][::-1], o["y"]], -np.r_[o["a"][::-1], o["a"]], 0, color=WATER, lw=0, zorder=1)
    yl = np.linspace(-np.pi / 1.4, np.pi / 1.4, 200)
    a1.plot(yl, -np.cos(0.7 * yl), color=GREY, lw=1.0, ls=(0, (1, 1.2)), label="shallow limit")
    for (mu, ls), col in zip(((0.5, (0, (4, 1.6))), (0.7, "-"), (0.9, (0, (1.5, 1.2, 4, 1.2)))), ("#5598e7", INK, "#104281")):
        s_ = c["shapes"][mu]
        a1.plot(np.r_[-s_["y"][::-1], s_["y"]], -np.r_[s_["a"][::-1], s_["a"]], color=col, ls=ls, lw=1.3, label=rf"$\mu={mu}$")
    a1.add_patch(Rectangle((-4.2, 0), 8.4, 0.16, fc=ROOF, ec=INK2, lw=0, hatch="//////", zorder=2)); a1.axhline(0, color=INK, lw=1.6, zorder=4)
    a1.set_xlabel(r"$y/a_c$"); a1.set_ylabel(r"$z/a_c$"); a1.set_xlim(-4.2, 4.2); a1.set_ylim(-1.3, 0.95); a1.set_yticks([-1, -0.5, 0]); a1.set_yticklabels(["1", "0.5", "0"])
    a1.yaxis.set_minor_locator(FixedLocator(np.arange(-1.2, 0.01, 0.1))); a1.set_xticks([-4, -2, 0, 2, 4])
    a1.legend(loc="upper center", ncol=2, columnspacing=1.4, handlelength=2.2, handletextpad=0.6, fontsize=7.5, borderaxespad=0.5)
    # (b), (c) discharge coefficient and aspect ratio
    mus = c["mus"]; mf = np.linspace(0.38, 1.02, 80)
    for ax, val, lim, lab, ylab, yl_ in ((a2, c["chi"], 1 / (9 * mf), r"$1/9\mu$", r"$\chi$", (0, 0.42)), (a3, c["aspect"], np.pi ** 2 / (2 * mf), r"$\pi^2/2\mu$", r"$W/\langle a\rangle$", (0, 14))):
        ax.axvspan(0.6, 0.8, color=SAND, lw=0, zorder=0)
        ax.plot(mus, val, "-o", color=INK, ms=4.2, mfc=BLUE, mec=INK, label="full problem")
        ax.plot(mf, lim, color=GREY, lw=1.0, ls=(0, (1, 1.2)), label=lab)
        ax.set_xlabel(r"$\mu$"); ax.set_ylabel(ylab); ax.set_xlim(0.36, 1.04); ax.set_ylim(*yl_); ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.text(0.7, yl_[1] * 0.045, "sands", ha="center", fontsize=7.5, color=INK2)
        ax.legend(loc="upper right", handlelength=1.8, fontsize=7.2, borderaxespad=0.4)
    # (d) a scanned bed
    d = np.loadtxt(os.path.join(HERE, "data", "pol2022_fig3_10_B25_245_x022.csv"), delimiter=",", skiprows=1)
    o_ = np.argsort(d[:, 0]); y, dep = d[o_, 0], np.maximum(d[o_, 1], 0.0)
    im = int(np.argmax(dep)); dm = dep[im]

    def edge(level, step):
        j = im
        while 0 < j < len(y) - 1 and dep[j] > level:
            j += step
        return y[j]

    y0 = 0.5 * (edge(0.5 * dm, -1) + edge(0.5 * dm, 1))
    ys, a = c["s77"]["y"], c["s77"]["a"]
    ys, a = np.concatenate([-ys[::-1], ys]), np.concatenate([a[::-1], a])
    sc = dm / a.max()
    a4.add_patch(Rectangle((-18.5, -2.75), 28, 2.75, fc=SAND, ec="none", zorder=0))
    a4.fill_between(y - y0, -dep, 0, color=WATER, lw=0, zorder=1, label="scanned bed")
    a4.plot(y - y0, -dep, color=BLUE, lw=0.7, zorder=2)
    a4.plot(ys * sc, -a * sc, color=INK, lw=1.4, zorder=3, label="threshold section of the same depth")
    a4.add_patch(Rectangle((-18.5, 0), 28, 0.42, fc=ROOF, ec=INK2, lw=0, hatch="//////", zorder=2)); a4.axhline(0, color=INK, lw=1.6, zorder=4)
    for xx, yy, txt in ((-13.3, -1.5, "channel, 1.1 mm"), (-7.6, -0.85, "shelf, 0.3 mm"), (0.0, -2.28, "channel, 1.9 mm")):
        a4.text(xx, yy, txt, ha="center", va="center", fontsize=7.5, color=INK2, linespacing=0.95)
    a4.set_xlabel("across the pipe (mm)"); a4.set_ylabel("depth (mm)"); a4.set_xlim(-17, 8); a4.set_ylim(-2.75, 0.42)
    a4.set_yticks([-2, -1, 0]); a4.set_yticklabels(["2", "1", "0"]); a4.tick_params(top=False, which="both")
    a4.legend(loc="lower left", handlelength=1.8, borderaxespad=0.5)
    fs.panel(a1, "a"); fs.panel(a2, "b"); fs.panel(a3, "c"); fs.panel(a4, "d")
    fs.save(fig, "fig02_section")


# ----------------------------------------------------------------------------------------------------------------------
# figure 3: the fold
# ----------------------------------------------------------------------------------------------------------------------
def fig_fold(c):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(W, 2.12))
    fig.subplots_adjust(left=0.078, right=0.988, top=0.90, bottom=0.185, wspace=0.50)
    # (a) lumped model
    eta = np.linspace(0, 1.25, 500)
    a1.axhline(1.0, color=GREY, lw=0.9)
    for h, ls, lw in ((0.9, (0, (1, 1.2)), 1.1), (1.0, "-", 1.4), (1.15, (0, (4, 1.6)), 1.1)):
        f = 256 / 27 * eta ** 3 * (h - eta)
        a1.plot(eta[f >= 0], f[f >= 0], color=INK, ls=ls, lw=lw)
    r = np.roots([1, -1.15, 0, 0, 27 / 256]); r = np.sort(r[np.isreal(r) & (r.real > 0)].real)
    a1.plot([r[0]], [1], "o", ms=5, mfc=BLUE, mec=INK, zorder=5); a1.plot([r[1]], [1], "D", ms=4.4, mfc=ORANGE, mec=INK, zorder=5)
    a1.plot([0.75], [1], "o", ms=4.6, mfc=INK, mec=INK, zorder=5)
    a1.text(1.17, 1.82, r"$H/H_{\min}=1.15$", fontsize=7.5, ha="right", va="bottom"); a1.text(0.75, 0.80, "1", fontsize=7.5, ha="center", va="center"); a1.text(0.68, 0.36, "0.9", fontsize=7.5, ha="center", va="center")
    a1.text(0.03, 0.94, "level set\nby the sand", fontsize=7, color=INK2, va="top", linespacing=0.95)
    a1.set_xlabel(r"$\gamma\Delta/H_{\min}$"); a1.set_ylabel(r"$\Delta^3(H-\gamma\Delta)\,/\,(\Lambda l^2/\mathcal{G}_0k)$", fontsize=8.5)
    a1.set_xlim(0, 1.2); a1.set_ylim(0, 2.0); a1.set_xticks([0, 0.4, 0.8, 1.2])
    # (b) the two branches
    f_ = c["fold"]
    a2.axvspan(0, 1, color=LIGHT, lw=0, zorder=0)
    a2.plot(f_["x_large"], f_["d_large"], color=BLUE, lw=1.5); a2.plot(f_["x_pinch"], f_["d_pinch"], color=ORANGE, lw=1.5, ls=(0, (4, 1.6)))
    a2.plot([1], [f_["d_fold"]], "o", ms=4.6, mfc=INK, mec=INK, zorder=5)
    a2.text(0.5, 0.40, "no threshold pipe", rotation=90, ha="center", va="center", fontsize=7.5, color=INK2)
    a2.text(1.75, 0.20, "large pipe", fontsize=7.5); a2.text(1.75, 0.495, "pinched pipe", fontsize=7.5); a2.text(1.12, 0.60, "fold", fontsize=7.5)
    a2.set_xlim(0, 4); a2.set_ylim(0, 0.8); a2.set_xlabel(r"$H/H_{\min}$"); a2.set_ylabel(r"$h(l)/H$")
    # (c) coefficient of the scale law
    for name, ls, lab, pos in (("hole", "-", "hole in the roof", (0.60, 1.95)), ("toe", (0, (4, 1.6)), "exit at the toe", (0.52, 5.35))):
        res = c["c3"][name]
        a3.plot(res[:, 0], res[:, 1], color=INK, ls=ls, lw=1.4)
        i = int(np.argmax(res[:, 1])); a3.plot([res[i, 0]], [res[i, 1]], "o", ms=4.2, mfc="white", mec=INK, zorder=5)
        a3.text(*pos, lab, fontsize=7.5, ha="center")
    a3.set_xlabel("$l/L$"); a3.set_ylabel("$C$"); a3.set_xlim(0, 1); a3.set_ylim(0, 6); a3.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); a3.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    fs.panel(a1, "a"); fs.panel(a2, "b"); fs.panel(a3, "c")
    fs.save(fig, "fig03_fold")


# ----------------------------------------------------------------------------------------------------------------------
# figure 4: inertia
# ----------------------------------------------------------------------------------------------------------------------
def fig_inertia(c):
    from pipetheory.inertia import wide_pipe_coefficients
    cw = wide_pipe_coefficients()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.5))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.915, bottom=0.16, wspace=0.30)
    sim = np.loadtxt(os.path.join(TMP, "paper1_inertia_similarity.csv"), delimiter=",", skiprows=1)
    R = sim[:, 0]
    for j, (lab, slope, dy) in enumerate((("head gradient", float(cw["S_V"]) / 12, 0.09), ("roof stress", float(cw["roof_V"]) / 6, 0.07), ("bed stress", float(cw["bed_V"]) / 6, 0.07))):
        col = (1, 3, 2)[j]
        a1.fill_between(R, sim[:, col], 1 + slope * R, color=LIGHT, lw=0, zorder=1)
        a1.plot(R, sim[:, col], color=INK, lw=1.5, zorder=3, label="Navier–Stokes" if j == 0 else None)
        a1.plot(R, 1 + slope * R, color=INK, lw=1.0, ls=(0, (4, 1.6)), zorder=3, label="first order" if j == 0 else None)
        k = int(np.argmin(np.abs(R - 3.6)))
        a1.text(3.6, sim[k, col] + dy, lab, ha="center", va="bottom", fontsize=7.5, rotation=np.degrees(np.arctan2((sim[k + 3, col] - sim[k - 3, col]) / (2.2 - 0.3) * 2.02, (R[k + 3] - R[k - 3]) / 5.0 * 2.25)))
    a1.set_xlim(0, 5); a1.set_ylim(0.3, 2.2); a1.set_xlabel(r"injection Reynolds number $Va/\nu$"); a1.set_ylabel("ratio to creeping flow")
    a1.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0))
    # inset: the gap
    ins = a1.inset_axes([0.035, 0.635, 0.44, 0.335]); ins.axis("off"); ins.set_xlim(0, 10); ins.set_ylim(-1.3, 2.75)
    ins.add_patch(Rectangle((0, -1.3), 10, 1.3, fc=SAND, ec="none")); ins.add_patch(Rectangle((0, 0), 10, 2, fc=WATER, ec="none"))
    ins.add_patch(Rectangle((0, 2), 10, 0.55, fc=ROOF, ec=INK2, lw=0, hatch="//////")); ins.plot([0, 10], [2, 2], color=INK, lw=1.2); ins.plot([0, 10], [0, 0], color=INK, lw=0.8)
    for xx in (0.9, 2.3, 3.7, 5.1):
        ins.add_patch(FancyArrowPatch((xx, -1.1), (xx, -0.05), arrowstyle="-|>", mutation_scale=4.5, lw=0.7, color=BLUE, shrinkA=0, shrinkB=0))
    zz = np.linspace(0, 2, 9)[1:-1]
    for z in zz:
        ins.add_patch(FancyArrowPatch((6.3, z), (6.3 + 2.9 * z * (2 - z), z), arrowstyle="-|>", mutation_scale=3.6, lw=0.6, color=INK, shrinkA=0, shrinkB=0))
    zc = np.linspace(0, 2, 40); ins.plot(6.3 + 2.9 * zc * (2 - zc), zc, color=INK, lw=0.6); ins.plot([6.3, 6.3], [0, 2], color=INK, lw=0.6)
    ins.text(3.0, -0.62, "$V$", fontsize=7.5, ha="center", va="center"); ins.text(5.75, 1.0, "$u$", fontsize=7.5, ha="right", va="center")
    ins.add_patch(FancyArrowPatch((1.6, 0.0), (1.6, 2.0), arrowstyle="<|-|>", mutation_scale=4, lw=0.5, color=INK, shrinkA=0, shrinkB=0)); ins.text(1.95, 1.0, "$a$", fontsize=7.5, va="center")
    # (b) head at the fold
    cv = np.loadtxt(os.path.join(TMP, "paper1_inertia_curve.csv"), delimiter=",", skiprows=1)
    jj = np.logspace(np.log10(0.03), np.log10(0.6), 30)
    a2.plot(jj, 1 + 0.22 * jj, color=GREY, lw=0.9, ls=(0, (1, 1.2)), label=r"$1+0.22\,J_0$")
    for l, ls in ((0.10, (0, (1, 1.2))), (0.15, "-"), (0.20, (0, (4, 1.6)))):
        s = cv[np.isclose(cv[:, 0], l)]
        a2.plot(s[:, 2], s[:, 3], color=INK, ls=ls, lw=1.2, label=rf"$l={100 * l:.0f}$ cm")
    rows = rd("paper1_inertia.csv")
    for sand in ("FPH", "B25", "FS35"):
        rr = [r for r in rows if r["sand"] == sand and r["Hf_i"] not in ("", "nan")]
        a2.plot([float(r["J0"]) for r in rr], [float(r["Hf_i"]) / float(r["Hf_0"]) for r in rr], **fs.marker(sand), label=SANDS[sand]["label"], zorder=5)
    a2.set_xscale("log"); a2.set_xlim(0.03, 3.0); a2.set_ylim(0.98, 1.5)
    a2.set_xlabel(r"$J_0=c\,kH_0/\nu$"); a2.set_ylabel(r"$H_{\min}/H_0$")
    a2.legend(loc="upper left", handlelength=2.2)
    fs.panel(a1, "a"); fs.panel(a2, "b")
    fs.save(fig, "fig04_inertia")


# ----------------------------------------------------------------------------------------------------------------------
# figure 5: sections of pipes at rest
# ----------------------------------------------------------------------------------------------------------------------
def fig_laser(c):
    D = json.load(open(os.path.join(TMP, "paper1_laser_figdata.json")))
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.75))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.955, bottom=0.085, wspace=0.30, hspace=0.36)
    # (a) shape
    ax = axs[0, 0]; A = D["a"]
    th = A["theory"]
    ax.fill_between(th["x"], th["a"], 0, color=WATER, lw=0, zorder=1)
    ax.fill_between(A["complete"]["x"], A["complete"]["q1"], A["complete"]["q3"], color=LIGHT, lw=0, alpha=0.85, zorder=2)
    ax.plot(A["complete"]["x"], A["complete"]["med"], color=GREY, lw=1.4, zorder=3, label=f"pipes through the bed ({A['complete']['n']})")
    ax.plot(A["partial"]["x"], A["partial"]["med"], color=INK2, lw=1.2, ls=(0, (4, 1.6)), zorder=3, label=f"growing pipes ({A['partial']['n']})")
    ax.plot(th["x"], th["a"], color=INK, lw=1.8, zorder=4, label="threshold section")
    ax.set_xlim(0, 5); ax.set_ylim(1.06, 0)
    ax.set_xlabel(r"distance from the deepest point / $a_{\max}$"); ax.set_ylabel(r"depth / $a_{\max}$")
    h, l = ax.get_legend_handles_labels(); ax.legend([h[2], h[0], h[1]], [l[2], l[0], l[1]], loc="lower right", handlelength=2.0)
    # (b) stress at rest
    ax = axs[0, 1]; rng = np.random.default_rng(1)
    for i, g in enumerate(D["b"]):
        v = np.array(g["v"]); x = i + 0.30 * (rng.random(v.size) - 0.5)
        ax.plot(x, v, **fs.marker(g["sand"], filled=g["kind"] == "complete", ms=3.4), alpha=0.9, zorder=3)
        ax.plot([i - 0.36, i + 0.36], [g["med"], g["med"]], color=INK, lw=1.8, zorder=5, solid_capstyle="butt", label="median" if i == 0 else None)
        ax.plot([i - 0.42, i + 0.42], [g["tau_c"], g["tau_c"]], color=INK2, lw=1.0, ls=(0, (1, 1.2)), zorder=4, label=r"$\tau_c$ used" if i == 0 else None)
    ax.set_xticks(range(4)); ax.set_xticklabels(["0.185", "0.228", "0.228*", "0.422"]); ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_xlabel(r"$d_{50}$ (mm);  * growing pipes"); ax.set_ylabel("largest bed shear stress (Pa)"); ax.set_ylim(0, 1.45); ax.set_xlim(-0.6, 3.6)
    ax.legend(loc="upper right", handlelength=1.6)
    # (c) depth along growing pipes
    ax = axs[1, 0]; seen = set()
    for n, g in enumerate(D["c"]):
        ax.plot(g["back_cm"], g["a_th_mm"], color=GREY, lw=0.8, zorder=2, label="threshold pipe" if n == 0 else None)
        ax.plot(g["s_cm"], g["a_mm"], **fs.marker("B25", filled=bool(g["known_k"]), ms=4.6), zorder=4,
                label=None if g["known_k"] in seen else ("scans, $k$ measured" if g["known_k"] else "scans, mean $k$ of the sand"))
        seen.add(g["known_k"])
    ax.plot(D["c_regression"]["s_cm"], D["c_regression"]["a_mm"], color=INK, lw=1.0, ls=(0, (4, 1.6)), zorder=3, label="through pipes, regression")
    ax.set_xlabel("distance from the tip (cm)"); ax.set_ylabel(r"largest depth $a_{\max}$ (mm)"); ax.set_xlim(0, 25); ax.set_ylim(0, 2.6)
    h, l = ax.get_legend_handles_labels(); o_ = [l.index(v) for v in ("scans, $k$ measured", "scans, mean $k$ of the sand", "threshold pipe", "through pipes, regression")]
    ax.legend([h[i] for i in o_], [l[i] for i in o_], loc="lower right", handlelength=1.8, borderaxespad=0.5, fontsize=7.2)
    # (d) pipe law
    ax = axs[1, 1]
    pref = [np.exp(np.mean(np.log(np.array(D["d"][s_]["law"]["a"]) / np.array(D["d"][s_]["law"]["Q"]) ** (1 / 3)))) for s_ in ("FPH", "B25", "FS35")]
    pm = float(np.exp(np.mean(np.log(pref))))                     # the laws of the three sands differ only by this prefactor
    print("figure 5(d): pipe law of each sand over their geometric mean:", np.round(np.array(pref) / pm, 3))
    Ql = np.array([6.0, 500.0]); ax.loglog(Ql, pm * Ql ** (1 / 3), color=GREY, lw=1.3, ls=(0, (4, 1.6)), zorder=2, label="pipe law")
    for sand in ("FPH", "B25", "FS35"):
        g = D["d"][sand]
        for kind in ("complete", "partial"):
            if kind in g:
                ax.loglog(g[kind]["Q"], g[kind]["a"], **fs.marker(sand, filled=kind == "complete", ms=4.2), zorder=4, label=SANDS[sand]["label"] if kind == "complete" else None)
    x0, x1 = 110.0, 330.0; y0 = 0.82
    ax.plot([x0, x1, x1, x0], [y0, y0, y0 * (x1 / x0) ** (1 / 3), y0], color=INK, lw=0.8)
    ax.text(np.sqrt(x0 * x1), y0 * 0.93, "3", fontsize=7.5, ha="center", va="top"); ax.text(x1 * 1.08, y0 * 1.19, "1", fontsize=7.5, ha="left", va="center")
    ax.text(0.965, 0.955, rf"fit: $a_{{\max}}\propto Q^{{\,{D['d_fit']['n']:.2f}\pm{D['d_fit']['sd']:.2f}}}$", transform=ax.transAxes, fontsize=8, va="top", ha="right")
    ax.set_xlim(6, 500); ax.set_ylim(0.5, 6.5); ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5", "1", "2", "4"]); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"discharge at the section (mL min$^{-1}$)"); ax.set_ylabel(r"largest depth $a_{\max}$ (mm)")
    ax.legend(loc="upper left", handletextpad=0.4, borderaxespad=0.4)
    for a_, s_ in zip(axs.ravel(), "abcd"):
        fs.panel(a_, s_)
    fs.save(fig, "fig05_laser")


# ----------------------------------------------------------------------------------------------------------------------
# figure 6: small-scale tests
# ----------------------------------------------------------------------------------------------------------------------
def fig_small(c):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(W, 1.98))
    fig.subplots_adjust(left=0.07, right=0.988, top=0.895, bottom=0.20, wspace=0.40)
    neutral = lambda filled: dict(marker="o", ls="none", ms=4.2, mew=0.5 if filled else 0.9, mfc=INK2 if filled else "white", mec=INK if filled else INK2)
    st = [r for r in rd("paper1_pol_states_w30.csv") if r["variant"] == "wide"]
    xx = np.array([2.5, 8.0])
    a1.plot(xx, xx, color=INK, lw=0.8); a1.plot(xx, 0.70 * xx, color=GREY, lw=0.8, ls=(0, (1, 1.2)))
    a1.text(7.75, 5.05, "0.70", fontsize=7, color=INK2, ha="right", rotation=35)
    for sand in ("FPH", "B25", "FS35"):
        rr = [r for r in st if r["sand"] == sand]
        a1.plot([float(r["H_net_cm"]) for r in rr], [float(r["H_fold_self_cm"]) for r in rr], **fs.marker(sand, filled=False, ms=4.2), zorder=3)
        a1.plot([float(r["H_net_cm"]) for r in rr], [float(r["H_fold_tip_cm"]) for r in rr], **fs.marker(sand, ms=4.2), zorder=4)
    a1.plot([], [], label="self-formed", **neutral(False)); a1.plot([], [], label="tip width", **neutral(True))
    a1.set_xlabel("head at rest (cm)"); a1.set_ylabel(r"$H_{\min}(l)$ (cm)"); a1.legend(loc="upper left", handletextpad=0.1, fontsize=7, borderaxespad=0.3)
    pr = [r for r in rd("paper1_pol_box_predictions.csv") if r["sink"] == "wide" and r["W0_in_d50"] == "30" and r["tortuosity"] == "1.0"]
    a2.plot(xx, xx, color=INK, lw=0.8)
    for sand in ("FPH", "B25", "FS35"):
        rr = [r for r in pr if r["sand"] == sand]
        a2.plot([float(r["Hc_meas_cm"]) for r in rr], [float(r["H_bed_cm"]) for r in rr], **fs.marker(sand, ms=4.2), zorder=4)
        rr = [r for r in rr if r["H_bed_tip_cm"] not in ("", "nan")]
        a2.plot([float(r["Hc_meas_cm"]) for r in rr], [float(r["H_bed_tip_cm"]) for r in rr], **fs.marker(sand, filled=False, ms=4.2), zorder=3)
    a2.plot([], [], label="bed", **neutral(True)); a2.plot([], [], label="bed and tip", **neutral(False))
    l1 = a2.legend(loc="upper left", handletextpad=0.1, fontsize=7, borderaxespad=0.3); a2.add_artist(l1)
    hs = [a2.plot([], [], **fs.marker(sd, ms=4.2))[0] for sd in ("FPH", "B25", "FS35")]
    a2.legend(hs, [SANDS[sd]["label"] for sd in ("FPH", "B25", "FS35")], loc="lower right", handletextpad=0.1, fontsize=7, borderaxespad=0.3)
    a2.set_xlabel(r"measured $H_c$ (cm)"); a2.set_ylabel(r"computed $H_c$ (cm)")
    for ax in (a1, a2):
        ax.set_xlim(2.5, 8); ax.set_ylim(2.5, 8); ax.set_xticks([3, 4, 5, 6, 7, 8]); ax.set_yticks([3, 4, 5, 6, 7, 8])
    for ax in (a1, a2, a3):
        ax.set_box_aspect(1)
    # (c) depth along the pipe: the computed pipes at fixed distances from the tip, and the regression of the scans
    x = np.logspace(-13.3, -10.5, 50)
    a3.loglog(x, 10 * x ** (1 / 3) * 1e3, color=INK, lw=1.3, zorder=2, label=r"scans, $10\,(\kappa x)^{1/3}$")
    for g in c["depth"]:
        o_ = np.argsort(g["s"]); sg, ag = g["s"][o_], g["a_mm"][o_]
        st_ = np.array([v for v in STATIONS if sg[0] <= v <= sg[-1]])
        a3.loglog(g["kappa"] * st_, np.interp(st_, sg, ag), **fs.marker(g["sand"], ms=3.4), alpha=0.9, zorder=4)
    a3.plot([], [], label="threshold pipe", **neutral(True))
    x0, x1 = 1.3e-12, 1.04e-11; y0 = 0.52
    a3.plot([x0, x1, x1, x0], [y0, y0, y0 * (x1 / x0) ** (1 / 3), y0], color=INK, lw=0.8)
    a3.text(np.sqrt(x0 * x1), y0 * 0.94, "3", fontsize=7, ha="center", va="top"); a3.text(x1 * 1.12, y0 * 1.38, "1", fontsize=7, ha="left", va="center")
    a3.set_xlabel(r"$\kappa x$ (m$^3$)"); a3.set_ylabel(r"largest depth $a_{\max}$ (mm)"); a3.set_xlim(6e-14, 3e-11); a3.set_ylim(0.35, 3.2)
    a3.set_yticks([0.5, 1, 2]); a3.set_yticklabels(["0.5", "1", "2"]); a3.yaxis.set_minor_formatter(NullFormatter())
    h, l = a3.get_legend_handles_labels(); a3.legend(h[::-1], l[::-1], loc="upper left", fontsize=7, handlelength=1.6, borderaxespad=0.4, handletextpad=0.5)
    fs.panel(a1, "a"); fs.panel(a2, "b"); fs.panel(a3, "c")
    fs.save(fig, "fig06_small")


# ----------------------------------------------------------------------------------------------------------------------
# figure 7: larger scales
# ----------------------------------------------------------------------------------------------------------------------
def fig_scales(c):
    fig = plt.figure(figsize=(W, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.32], left=0.075, right=0.988, top=0.90, bottom=0.175, wspace=0.42)
    a1, a2, a3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    rows = rd("paper1_pipe_width_data.csv")
    mk = {"B25": "o", "FPH": "s", "FS35": "^", "Enschede": "D", "SandA": "v", "Baskarp": "o", "Itterbeck330": "^"}
    kw = {"small": dict(mfc="white", mec=INK2, mew=0.8, ms=4.2), "medium": dict(mfc=INK2, mec=INK, mew=0.5, ms=4.6)}
    a2.axhspan(3, 8, color=LIGHT, lw=0, zorder=0); a2.text(0.118, 8.25, "3–8 grain diameters", fontsize=7, color=INK2, va="bottom")
    for sn, m in mk.items():
        for sc in ("small", "medium"):
            rr = [r for r in rows if r["sand"] == sn and r["scale"] == sc]
            a1.plot([float(r["l"]) for r in rr], [float(r["W_mm"]) * 1e-3 / float(r["d50"]) for r in rr], m, ls="none", zorder=3, **kw[sc])
            a2.plot([float(r["l"]) for r in rr], [float(r["a_over_d"]) for r in rr], m, ls="none", zorder=3, **kw[sc])
    for ax in (a1, a2):
        ax.set_xscale("log"); ax.set_xlim(0.1, 1.5); ax.set_xticks([0.15, 0.3, 0.6, 1.2]); ax.set_xticklabels(["0.15", "0.3", "0.6", "1.2"])
        ax.xaxis.set_minor_formatter(NullFormatter()); ax.tick_params(axis="x", which="minor", bottom=False, top=False); ax.set_xlabel(r"$l$ (m)")
    a1.set_yscale("log"); a1.set_ylim(10, 420); a1.set_yticks([10, 30, 100, 300]); a1.set_yticklabels(["10", "30", "100", "300"]); a1.yaxis.set_minor_formatter(NullFormatter()); a1.tick_params(axis="y", which="minor", left=False, right=False)
    a1.set_ylabel(r"$W/d_{50}$"); a2.set_ylabel(r"$a/d_{50}$"); a2.set_ylim(0, 11)
    a1.plot([], [], "o", ls="none", label="small scale", **kw["small"]); a1.plot([], [], "o", ls="none", label="medium scale", **kw["medium"])
    a1.legend(loc="upper left", handletextpad=0.1, fontsize=7, borderaxespad=0.3)
    # (c) gradient against discharge
    g = c["regimes"]; q, s = g["q"], g["s"]
    a3.axvspan(q[0], 1.0, color=SAND, lw=0, zorder=0)
    a3.loglog(q, s, color=INK, lw=1.5, zorder=3)
    a3.loglog(q[q < 1], q[q < 1] ** (-1 / 3), color=INK, lw=0.9, ls=(0, (1, 1.2)), zorder=3)
    for cc, y in g["plateaus"].items():
        a3.loglog([y ** -3, q[-1]], [y, y], color=GREY, lw=0.9, ls=(0, (4, 1.6)), zorder=2); a3.text(q[-1] * 0.8, y * 1.06, rf"$a={cc}\,d_{{50}}$", ha="right", va="bottom", fontsize=7)
    for sc, a_, b_, sn in g["pts"]:
        a3.loglog([a_], [b_], mk[sn], ls="none", zorder=4, **kw[sc])
    a3.loglog([x_ for x_, _ in g["large"]], [y_ for _, y_ in g["large"]], color=INK, lw=1.2, zorder=5)
    a3.loglog([np.sqrt(g["large"][0][0] * g["large"][1][0])], [g["large"][0][1]], "*", ms=9, mfc=INK, mec=INK, mew=0.4, ls="none", zorder=6, label="large scale")
    a3.set_xlabel(r"$Q/Q_\times$"); a3.set_ylabel(r"$S/S_\times$"); a3.set_xlim(q[0], q[-1]); a3.set_ylim(0.08, 6)
    a3.text(0.0045, 0.105, "limited by the\nwidth of the tip", fontsize=7, color=INK2, linespacing=0.95); a3.text(1.4e4, 0.30, "self-formed,\n" + r"$QS^3=\Lambda$", fontsize=7, color=INK2, linespacing=0.95, ha="right")
    a3.legend(loc="upper right", fontsize=7, handlelength=1.4, borderaxespad=0.3)
    fs.panel(a1, "a"); fs.panel(a2, "b"); fs.panel(a3, "c")
    fs.save(fig, "fig07_scales")


FIGS = {1: fig_sketch, 2: fig_section, 3: fig_fold, 4: fig_inertia, 5: fig_laser, 6: fig_small, 7: fig_scales, "ga": fig_graphical_abstract}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    fs.use()
    c = cache(a.recompute)
    for i in ([int(v) if v.isdigit() else v for v in a.only.split(",")] if a.only else list(FIGS)):      # --only 1,3 or --only ga
        FIGS[i](c)
        print("written figure", i)
