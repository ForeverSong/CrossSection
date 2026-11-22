"""Figures of the manuscript (vector PDF, JFM column width).  Writes manuscript/figures/fig*.pdf.

Figures 4 and 5 read the tables written by compare_pol_heads_all.py (--w0 30 --sinks wide), run_pol_box.py and
pipe_width_data.py in tmp/; run those first.      usage:  python make_paper_figures.py [--tmp ../tmp]
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipetheory.aquifer import pol_box, PipeResponse
from pipetheory.coupled import trace_branches, fold, c3_from_N
from pipetheory.section import threshold_section, lubrication

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "manuscript", "figures")
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "legend.frameon": False,
                     "legend.fontsize": 7.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
K, G1, G2 = "k", "0.45", "0.7"


def label(ax, s, x=-0.2, y=1.02):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=9, va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=200); plt.close(fig)


def fig_sketch():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.3, 1.75), gridspec_kw=dict(width_ratios=[1.5, 1]))
    a1.fill_between([0, 10], -3, 0, color="0.9", lw=0); a1.plot([0, 10], [0, 0], K, lw=1.6)
    xs = np.linspace(2, 6, 60); a1.fill_between(xs, -0.32 * (1 - (xs - 2) / 4) ** 0.34, 0, color="w", lw=0)
    a1.plot(xs, -0.32 * (1 - (xs - 2) / 4) ** 0.34, K, lw=0.8); a1.plot([1.85, 1.85, 2.0, 2.0], [0.9, 0, 0, 0.9], K, lw=0.8)
    a1.annotate("", (2, -0.75), (6, -0.75), arrowprops=dict(arrowstyle="<->", lw=0.6)); a1.text(4, -1.25, "$l$", ha="center")
    a1.annotate("", (2, -1.9), (10, -1.9), arrowprops=dict(arrowstyle="<->", lw=0.6)); a1.text(6, -2.45, "$L$", ha="center")
    a1.text(1.2, 0.35, "exit, $h=0$", ha="right", fontsize=8); a1.text(10.15, -1.5, "$h=H$", rotation=90, va="center", fontsize=8)
    a1.text(6.1, -0.38, "tip", fontsize=8); a1.text(7.4, 0.3, "impermeable roof", ha="center", fontsize=8); a1.text(0.2, -2.7, "sand, conductivity $k$", fontsize=8)
    a1.annotate("", (3.5, 0.3), (2.3, 0.3), arrowprops=dict(arrowstyle="->", lw=0.6)); a1.text(3.65, 0.3, "$s$", fontsize=8, va="center")
    a1.set_xlim(-0.2, 10.6); a1.set_ylim(-3, 1.1); a1.axis("off")
    y = np.linspace(-1, 1, 200); sh = -np.cos(np.pi * y / 2) ** 0.9
    a2.fill_between([-1.6, 1.75], -1.8, 0, color="0.9", lw=0); a2.fill_between(y, sh, 0, color="w", lw=0)
    a2.plot(y, sh, K, lw=0.8); a2.plot([-1.6, 1.75], [0, 0], K, lw=1.6)
    a2.annotate("", (0, -1.0), (0, 0), arrowprops=dict(arrowstyle="<->", lw=0.6)); a2.text(0.08, -0.55, "$a(y)$", fontsize=8)
    a2.annotate("", (-1, 0.22), (1, 0.22), arrowprops=dict(arrowstyle="<->", lw=0.6)); a2.text(0, 0.3, "$W$", ha="center", fontsize=8)
    a2.annotate("", (1.5, -1.0), (1.05, -1.0), arrowprops=dict(arrowstyle="->", lw=0.6)); a2.text(1.54, -1.0, "$y$", fontsize=8, va="center")
    a2.annotate("", (1.05, -1.5), (1.05, -1.0), arrowprops=dict(arrowstyle="->", lw=0.6)); a2.text(1.05, -1.56, "$z$", fontsize=8, ha="center", va="top")
    a2.text(-0.72, -0.86, r"$\varphi$", fontsize=8); a2.set_xlim(-1.7, 1.8); a2.set_ylim(-1.8, 0.6); a2.axis("off")
    label(a1, "a", 0.0, 0.92); label(a2, "b", 0.0, 0.92); save(fig, "fig1_sketch")


def fig_section():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.3, 2.1))
    for mu, ls in ((0.5, ":"), (0.7, "-"), (0.9, "--")):
        o = threshold_section(mu, 16, 161, 41)
        a1.plot(np.r_[-o["y"][::-1], o["y"]], -np.r_[o["a"][::-1], o["a"]], K, ls=ls, label=rf"$\mu={mu}$")
    yl = np.linspace(-np.pi / 1.4, np.pi / 1.4, 200); a1.plot(yl, -np.cos(0.7 * yl), color=G2, label=r"lubrication, $\mu=0.7$")
    a1.axhline(0, color=K, lw=1.4); a1.set_xlabel(r"$y/a_c$"); a1.set_ylabel(r"$z/a_c$"); a1.set_ylim(-1.25, 0.95); a1.set_yticks([-1, -0.5, 0]); a1.legend(loc="upper center", ncol=2, columnspacing=1.0)
    mus = np.arange(0.4, 1.01, 0.1); o = [threshold_section(m, 16, 161, 41) for m in mus]
    a2.plot(mus, [x["chi"] for x in o], "o-", color=K, ms=3, label=r"$\chi$"); a2.plot(mus, 1 / (9 * mus), color=K, ls="--", label=r"$1/9\mu$")
    a2.set_xlabel(r"$\mu$"); a2.set_ylabel(r"$\chi$"); a2.set_ylim(0, 0.42)
    b = a2.twinx(); b.plot(mus, [x["W_over_aavg"] for x in o], "s-", color=G1, ms=3); b.plot(mus, np.pi ** 2 / (2 * mus), color=G1, ls="--")
    b.set_ylabel(r"$W/\langle a\rangle$", color=G1); b.tick_params(axis="y", colors=G1); b.set_ylim(0, 14)
    a2.legend(loc="lower left"); label(a1, "a"); label(a2, "b"); fig.subplots_adjust(wspace=0.45); save(fig, "fig2_section")


def fig_fold():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.3, 2.1))
    box = pol_box(); L = box.seepage_length
    R = PipeResponse(box, 0.18); Nmin, br = trace_branches(R, N_start_factor=600.0)
    x = np.array([(b["N"] / Nmin) ** 0.25 for b in br]); d = np.array([b["h"][-1] for b in br]); st = np.array([b["branch"] == 0 for b in br])
    o = np.argsort(x[st]); a1.plot(x[st][o], d[st][o], K, label="large pipe"); a1.plot(x[~st], d[~st], K, ls="--", label="pinched pipe")
    a1.plot([1], [d[np.argmin(x)]], "o", color=K, ms=3.5); a1.axvspan(0, 1, color="0.9", lw=0); a1.text(0.5, 0.05, "no\nthreshold\npipe", ha="center", fontsize=6.5)
    a1.set_xlim(0, 4); a1.set_ylim(0, 0.8); a1.set_xlabel(r"$H/H_{\min}$"); a1.set_ylabel(r"$h(l)/H$"); a1.legend(loc="upper right")
    ls = np.arange(0.04, 0.341, 0.02); N = []; lr = []
    for l in ls:
        Rl = PipeResponse(box, l); N.append(fold(Rl)[0]); lr.append(Rl.length)
    chi = threshold_section(0.77, 16, 161, 41)["chi"]
    a2.plot(np.array(lr) / L, c3_from_N(np.array(N), chi, L), "o-", color=K, ms=3); a2.set_xlim(0, 1); a2.set_ylim(0, 3.6)
    a2.set_xlabel(r"$l/L$"); a2.set_ylabel(r"$c_3$"); label(a1, "a"); label(a2, "b"); fig.subplots_adjust(wspace=0.38); save(fig, "fig3_fold")


def rd(path):
    with open(path) as f:
        return list(csv.DictReader(f))


D50_LABEL = {"FPH": r"$d_{50}=0.185$ mm", "B25": r"$d_{50}=0.228$ mm", "FS35": r"$d_{50}=0.422$ mm"}


def fig_small(tmp):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.3, 2.45))
    mk = {"FPH": "s", "B25": "o", "FS35": "^"}
    st = [r for r in rd(os.path.join(tmp, "paper1_pol_states_w30.csv")) if r["variant"] == "wide"]
    for sn, m in mk.items():
        rr = [r for r in st if r["sand"] == sn]
        a1.plot([float(r["H_net_cm"]) for r in rr], [float(r["H_fold_self_cm"]) for r in rr], m, color=G1, mfc="none", ms=4)
        a1.plot([float(r["H_net_cm"]) for r in rr], [float(r["H_fold_tip_cm"]) for r in rr], m, color=K, ms=4, label=D50_LABEL[sn])
    a1.set_xlabel("head at rest (cm)"); a1.set_ylabel(r"$H_{\min}(l)$ (cm)"); a1.legend(loc="upper left", handletextpad=0.2, fontsize=6.5, borderpad=0.2)
    pr = [r for r in rd(os.path.join(tmp, "paper1_pol_box_predictions.csv")) if r["sink"] == "wide" and r["W0_in_d50"] == "30" and r["tortuosity"] == "1.0"]
    for sn, m in mk.items():
        rr = [r for r in pr if r["sand"] == sn]
        a2.plot([float(r["Hc_meas_cm"]) for r in rr], [float(r["H_bed_cm"]) for r in rr], m, color=K, ms=4)
        rr = [r for r in rr if r["H_bed_tip_cm"] not in ("", "nan")]
        a2.plot([float(r["Hc_meas_cm"]) for r in rr], [float(r["H_bed_tip_cm"]) for r in rr], m, color=G1, mfc="none", ms=4)
    a2.set_xlabel(r"measured $H_c$ (cm)"); a2.set_ylabel(r"predicted $H_c$ (cm)")
    for ax in (a1, a2):
        ax.plot([0, 9], [0, 9], color=K, lw=0.6); ax.set_xlim(2.5, 8); ax.set_ylim(2.5, 8); ax.set_aspect("equal")
    label(a1, "a"); label(a2, "b"); fig.subplots_adjust(wspace=0.35); save(fig, "fig4_small_scale")


def fig_scales(tmp):
    rows = rd(os.path.join(tmp, "paper1_pipe_width_data.csv"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.3, 2.1))
    mk = {"B25": "o", "FPH": "s", "FS35": "^", "Enschede": "D", "SandA": "v", "Baskarp": "o", "Itterbeck330": "^"}
    for sn, m in mk.items():
        for sc in ("small", "medium"):
            rr = [r for r in rows if r["sand"] == sn and r["scale"] == sc]
            kw = dict(color=K, ms=4, mfc="none" if sc == "small" else K)
            a1.plot([float(r["l"]) for r in rr], [float(r["W_mm"]) * 1e-3 / float(r["d50"]) for r in rr], m, **kw)
            a2.plot([float(r["l"]) for r in rr], [float(r["a_over_d"]) for r in rr], m, **kw)
    for ax in (a1, a2):
        ax.set_xscale("log"); ax.set_xticks([0.15, 0.3, 0.6, 1.2]); ax.set_xticklabels(["0.15", "0.3", "0.6", "1.2"]); ax.minorticks_off(); ax.set_xlabel(r"$l$ (m)")
    a1.set_yscale("log"); a1.set_yticks([10, 30, 100, 300]); a1.set_yticklabels(["10", "30", "100", "300"]); a1.minorticks_off()
    a1.set_ylabel(r"$W/d_{50}$"); a2.set_ylabel(r"$a/d_{50}$"); a2.set_ylim(0, 11)
    label(a1, "a"); label(a2, "b"); fig.subplots_adjust(wspace=0.32); save(fig, "fig5_scales")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tmp", default=os.path.join(HERE, "..", "tmp")); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fig_sketch(); fig_section(); fig_fold(); fig_small(a.tmp); fig_scales(a.tmp)
    print("figures written to", os.path.normpath(OUT))
