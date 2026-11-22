"""Robbins-type BEP analysis: B realizations advanced together on the GPU.

Examples
    python code/run_baseline.py                       # one homogeneous run
    python code/run_baseline.py --B 256 --cov-k 1.0 --theta 0.5
    python code/run_baseline.py --B 4400 --cov-k 3 --cov-icr 0.5 --theta 5 --rho 0

Writes tmp/baseline_<tag>.csv (one (H/L)_cr per realization) and tmp/baseline_<tag>.png.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bepflow import pick_device                                    # noqa: E402
from bepflow.darcy_pipe import BEPConfig, lognormal_fields, run_bep  # noqa: E402
import plotstyle                                                   # noqa: E402

TMP = Path(__file__).resolve().parents[1] / "tmp"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=1, help="number of Monte-Carlo realizations")
    ap.add_argument("--nx", type=int, default=80)
    ap.add_argument("--ny", type=int, default=80)
    ap.add_argument("--mean-k", type=float, default=1.0e-5)
    ap.add_argument("--cov-k", type=float, default=0.0, help="sigma_k / mu_k")
    ap.add_argument("--mean-icr", type=float, default=0.3)
    ap.add_argument("--cov-icr", type=float, default=0.0)
    ap.add_argument("--theta", type=float, default=1.0, help="correlation length [m]")
    ap.add_argument("--rho", type=float, default=0.0, help="correlation of ln k and ln i_cr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pipe-factor", type=float, default=1.0,
                    help="multiplier on the laminar pipe conductance (eq. 4)")
    ap.add_argument("--free-initiation", action="store_true",
                    help="initiate along the whole downstream boundary (paper section 4.3)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    dev = pick_device("cpu" if args.cpu else "cuda")
    cfg = BEPConfig(nx=args.nx, ny=args.ny, pipe_factor=args.pipe_factor)
    print(f"torch {torch.__version__} on {dev}; B = {args.B}, grid {cfg.ny} x {cfg.nx}")
    k, icr = lognormal_fields(args.B, cfg, args.mean_k, args.cov_k, args.mean_icr, args.cov_icr,
                              args.theta, args.rho, args.seed, dev)
    t0 = time.time()
    res = run_bep(k, icr, cfg, init_rows=list(range(cfg.ny)) if args.free_initiation else None,
                  record=True, verbose=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    hl = res["HL_cr"].cpu().numpy()
    ok = np.isfinite(hl)
    print(f"done in {wall:.1f} s: {res['solves']} batched head solves, {res['cg_iters']} CG iterations")
    print(f"(H/L)_cr: mean {hl[ok].mean():.4f}  std {hl[ok].std():.4f}  "
          f"5% quantile {np.quantile(hl[ok], 0.05):.4f}  ({int(ok.sum())}/{args.B} breached)")

    TMP.mkdir(exist_ok=True)
    np.savetxt(TMP / f"baseline_{args.tag}.csv", hl, header="HL_cr", comments="")

    plt = plotstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    ext = (0, cfg.L, 0, cfg.W)
    im = ax.imshow(np.log10(k[0].cpu().numpy()), origin="lower", extent=ext, cmap=plotstyle.SEQ_BLUE)
    xc = (np.arange(cfg.nx) + 0.5) * cfg.dx
    yc = (np.arange(cfg.ny) + 0.5) * cfg.dy
    ax.contour(xc, yc, res["head"][0].cpu().numpy(), levels=12, colors=plotstyle.INK, linewidths=0.5, alpha=0.6)
    pj, pi = np.nonzero(res["pipe"][0].cpu().numpy())
    ax.scatter(xc[pi], yc[pj], s=7, marker="s", color=plotstyle.ORANGE, linewidths=0, label="erosion pipe")
    ax.set(title="Realization 0: log10 k, head contours, pipe path", xlabel="x [m]  (upstream at left)", ylabel="y [m]")
    ax.grid(False)
    ax.legend(loc="upper left", frameon=True, facecolor=plotstyle.SURFACE, edgecolor="none", markerscale=2)
    fig.colorbar(im, ax=ax, shrink=0.85, label="log10 k [m/s]")

    ax = axes[1]
    curve = np.array(res["curve"])
    ax.step(curve[:, 0], curve[:, 1], where="post", color=plotstyle.BLUE)
    ax.set(title="Realization 0: pipe progression curve", xlabel="pipe penetration l / L", ylabel="H / L",
           xlim=(0, 1), ylim=(0, None))

    ax = axes[2]
    if ok.sum() > 1:
        srt = np.sort(hl[ok])
        ax.step(np.concatenate([srt[:1], srt]), np.arange(0, srt.size + 1) / srt.size, where="post",
                color=plotstyle.BLUE)
        ax.set(title=f"CDF of (H/L)cr, {int(ok.sum())} realizations", xlabel="(H/L)cr", ylabel="probability of failure",
               ylim=(0, 1))
    else:
        depth = res["depth"][0].cpu().numpy() * 1e3
        prof = depth.max(axis=0)
        ax.plot(xc, prof, color=plotstyle.BLUE)
        ax.set(title="Realization 0: pipe depth along x", xlabel="x [m]", ylabel="pipe depth a [mm]", ylim=(0, None))
    fig.tight_layout()
    fig.savefig(TMP / f"baseline_{args.tag}.png")
    print(f"wrote {TMP / f'baseline_{args.tag}.csv'} and .png")


if __name__ == "__main__":
    main()
