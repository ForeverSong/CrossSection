"""Sediment-laden laminar flow in a 2-D erosion-pipe segment (Euler-Lagrange).

Grains are released uniformly in a developed Poiseuille flow driven by the
hydraulic gradient i_h and settle onto the bed while being carried downstream.

Examples
    python code/run_pipe.py                           # 20 000 grains
    python code/run_pipe.py --n 2000000 --float32     # GPU-sized run
    python code/run_pipe.py --one-way --bed reflect   # NTLP icase=0 behaviour

Writes tmp/pipe_<tag>.png and prints timing in particle-steps per second.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bepflow import pick_device                                  # noqa: E402
from bepflow.pipe_el2d import Flow, Particles, PipeConfig, advance  # noqa: E402
import plotstyle                                                 # noqa: E402

TMP = Path(__file__).resolve().parents[1] / "tmp"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20000, help="computational grains")
    ap.add_argument("--phi", type=float, default=5.0e-3, help="mean solid volume fraction")
    ap.add_argument("--Lx", type=float, default=0.03)
    ap.add_argument("--a", type=float, default=3.0e-3, help="pipe depth [m]")
    ap.add_argument("--nx", type=int, default=192)
    ap.add_argument("--nz", type=int, default=24)
    ap.add_argument("--i-h", type=float, default=0.02, help="hydraulic gradient along the pipe")
    ap.add_argument("--d-p", type=float, default=1.0e-4, help="grain diameter [m]")
    ap.add_argument("--t-end", type=float, default=0.2, help="simulated time [s]")
    ap.add_argument("--bed", choices=["deposit", "reflect"], default="deposit")
    ap.add_argument("--one-way", action="store_true")
    ap.add_argument("--float32", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    dev = pick_device("cpu" if args.cpu else "cuda")
    dtype = torch.float32 if args.float32 else torch.float64
    cfg = PipeConfig(Lx=args.Lx, a=args.a, nx=args.nx, nz=args.nz, i_h=args.i_h, d_p=args.d_p,
                     bed=args.bed, two_way=not args.one_way)
    flow = Flow(cfg, device=dev, dtype=dtype)
    flow.set_poiseuille()
    parts = Particles(cfg, args.n, flow, phi=args.phi)
    ub = flow.bulk_velocity()                    # clear-water steady state of the discrete equations
    tau_bed = 0.5 * cfg.rho_f * cfg.g * cfg.a * cfg.i_h
    shields = tau_bed / ((cfg.rho_p - cfg.rho_f) * cfg.g * cfg.d_p)
    print(f"torch {torch.__version__} on {dev} ({'float32' if args.float32 else 'float64'}); "
          f"grid {cfg.nz} x {cfg.nx}, {args.n} grains (each stands for {parts.mult:.3g} real ones)")
    print(f"bulk velocity {ub:.3f} m/s, Re = {ub * 2 * cfg.a / cfg.nu:.0f}, bed shear {tau_bed:.3f} Pa, "
          f"Shields {shields:.3f}, grain response time {cfg.tau_p * 1e3:.2f} ms, d_p/dz = {cfg.d_p / cfg.dz:.2f}")
    if cfg.d_p > cfg.dz:
        print("WARNING: grain larger than the grid spacing -- outside the point-particle assumption")

    dt = flow.stable_dt()
    nsteps = int(round(args.t_end / dt))
    hist = []
    t0 = time.time()
    for it in range(nsteps):
        advance(flow, parts, dt)
        if it % max(1, nsteps // 200) == 0 or it == nsteps - 1:
            hist.append((flow.time, parts.deposited_fraction(), flow.bulk_velocity() / ub))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    print(f"{nsteps} steps (dt = {dt * 1e6:.0f} us) in {wall:.1f} s: "
          f"{args.n * nsteps / wall:.3g} particle-steps/s, {cfg.nx * cfg.nz * nsteps / wall:.3g} cell-steps/s")
    dep = parts.deposited
    ws = cfg.settling_velocity()
    travel = parts.travel[dep].mean().item() * 100 if dep.any().item() else float("nan")
    print(f"deposited fraction {parts.deposited_fraction():.3f}; mean travel of the deposited grains {travel:.2f} cm "
          f"(settling velocity {ws * 1e3:.2f} mm/s, deposition length U a / 2 w_s = {ub * cfg.a / (2 * ws) * 100:.2f} cm); "
          f"max |div u| {flow.divergence().abs().amax().item():.2e} 1/s")

    hist = np.array(hist)
    plt = plotstyle.apply()
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(flow.u.cpu().numpy(), origin="lower", aspect="auto", cmap=plotstyle.SEQ_BLUE,
                   extent=(0, cfg.Lx * 1e3, 0, cfg.a * 1e3))
    show = min(args.n, 6000)
    x = parts.x[:show].cpu().numpy() * 1e3
    z = parts.z[:show].cpu().numpy() * 1e3
    d = dep[:show].cpu().numpy()
    ax.scatter(x[~d], z[~d], s=5, color=plotstyle.ORANGE, linewidths=0, label="in suspension")
    ax.scatter(x[d], z[d], s=5, color=plotstyle.INK, linewidths=0, label="deposited on the bed")
    ax.set(title=f"Streamwise velocity and grains at t = {flow.time:.2f} s (vertical scale stretched)",
           xlabel="x [mm]", ylabel="z [mm]")
    ax.grid(False)
    ax.legend(loc="upper right", ncol=2, frameon=True, facecolor=plotstyle.SURFACE, edgecolor="none", markerscale=3)
    fig.colorbar(im, ax=ax, pad=0.01, label="u [m/s]")
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(hist[:, 0], hist[:, 1], color=plotstyle.BLUE)
    ax.set(title="Fraction of grains deposited", xlabel="time [s]", ylabel="fraction", ylim=(0, 1.02))
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(hist[:, 0], hist[:, 2], color=plotstyle.BLUE)
    ax.axhline(1.0, color=plotstyle.INK_2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set(title="Bulk velocity relative to clear water (effect of the grains)", xlabel="time [s]", ylabel="U / U_clear")
    fig.tight_layout()
    TMP.mkdir(exist_ok=True)
    fig.savefig(TMP / f"pipe_{args.tag}.png")
    print(f"wrote {TMP / f'pipe_{args.tag}.png'}")


if __name__ == "__main__":
    main()
