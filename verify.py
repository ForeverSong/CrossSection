"""Verification of the bepflow prototypes against closed-form results.

    python code/verify.py            # CPU or GPU, whatever torch finds
    python code/verify.py --quick    # skip the slower progression run

Every check prints the computed number next to the exact one.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bepflow import pick_device                                        # noqa: E402
from bepflow.darcy_pipe import (BEPConfig, cell_gradient, conductances,   # noqa: E402
                                lognormal_fields, run_bep, solve_head)
from bepflow.pipe_el2d import Flow, Particles, PipeConfig, advance      # noqa: E402

RESULTS = []


def check(name, value, exact, rtol):
    err = abs(value - exact) / max(abs(exact), 1e-300)
    ok = err <= rtol
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.6e}  exact {exact:.6e}  rel.err {err:.1e} (tol {rtol:.0e})")


def check_small(name, value, tol):
    ok = abs(value) <= tol
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.3e}  (tol {tol:.0e})")


# ------------------------------------------------------------------ Darcy + pipe
def test_darcy_uniform(dev):
    cfg = BEPConfig(nx=40, ny=20)
    T = torch.full((2, cfg.ny, cfg.nx), 1.0e-5, dtype=torch.float64, device=dev)
    H = torch.tensor([1.0, 2.5], dtype=torch.float64, device=dev)
    Cx, Cy = conductances(T, cfg)
    h, _ = solve_head(Cx, Cy, H, torch.zeros_like(T), 1e-12, 5000)
    grad = cell_gradient(h, T, H, cfg)
    check("Darcy uniform soil, gradient = H/L", grad[1].amax().item(), 2.5 / cfg.L, 1e-8)
    q = (Cx[:, :, -1] * h[:, :, -1]).sum(dim=1)[1].item()
    check("Darcy uniform soil, discharge = k b W H/L", q, 1.0e-5 * cfg.W * 2.5 / cfg.L, 1e-8)


def test_pipe_cubic_law(dev):
    """A straight pipe row across the whole domain, soil made impermeable-ish:
    the discharge must follow q = rho g a^3 /(12 mu) * i * dy (eq. 3/4 of the paper)."""
    cfg = BEPConfig(nx=40, ny=9)
    a = 1.0e-3
    T = torch.full((1, cfg.ny, cfg.nx), 1.0e-14, dtype=torch.float64, device=dev)
    T[:, 4, :] = cfg.c_pipe * a ** 3
    H = torch.tensor([0.5], dtype=torch.float64, device=dev)
    Cx, Cy = conductances(T, cfg)
    h, _ = solve_head(Cx, Cy, H, torch.zeros_like(T), 1e-13, 20000)
    q = (Cx[:, 4, -1] * h[:, 4, -1]).item()
    exact = cfg.rho_w * cfg.g * a ** 3 / (12 * cfg.mu) * (0.5 / cfg.L) * cfg.dy
    check("pipe row discharge = cubic law", q, exact, 1e-6)
    tau = 0.5 * cfg.rho_w * cfg.g * a * cell_gradient(h, T, H, cfg)[0, 4, 20].item()
    check("pipe wall shear stress = rho g a i / 2", tau, 0.5 * cfg.rho_w * cfg.g * a * 0.5 / cfg.L, 1e-6)


def test_random_field(dev):
    cfg = BEPConfig()
    k, icr = lognormal_fields(400, cfg, mean_k=1e-5, cov_k=1.0, mean_icr=0.3, cov_icr=0.5,
                              theta=1.0, rho=1.0, seed=3, device=dev)
    check("lognormal k: sample mean", k.mean().item(), 1e-5, 0.03)
    cov = ((k - k.mean()) ** 2).mean().item() ** 0.5 / k.mean().item()
    check("lognormal k: sample COV", cov, 1.0, 0.06)
    g = torch.log(k)
    g = (g - g.mean()) / torch.sqrt(((g - g.mean()) ** 2).mean())
    lag = 4                                           # 4 cells = 0.5 m, rho = exp(-2*0.5/1)
    corr = (g[:, :, :-lag] * g[:, :, lag:]).mean().item()
    check("Markov correlation at r = 0.5 m", corr, math.exp(-1.0), 0.03)
    gi = torch.log(icr)
    gi = (gi - gi.mean()) / torch.sqrt(((gi - gi.mean()) ** 2).mean())
    check("cross-correlation of ln k and ln i_cr (rho = 1)", (g * gi).mean().item(), 1.0, 1e-6)


def test_progression(dev):
    """Homogeneous soil.  Robbins et al. Fig. 11/12 put the nearly homogeneous
    case (sigma_k/mu_k = 0.25, i_cr = 0.3) at (H/L)_cr of roughly 0.17-0.20."""
    cfg = BEPConfig()
    k, icr = lognormal_fields(1, cfg, cov_k=0.0, cov_icr=0.0, device=dev)
    t0 = time.time()
    res = run_bep(k, icr, cfg)
    hl = res["HL_cr"][0].item()
    ok = 0.12 < hl < 0.30
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] homogeneous progression: (H/L)_cr = {hl:.4f} "
          f"(paper, near-homogeneous: ~0.17-0.20); {res['solves']} solves, "
          f"{res['cg_iters']} CG its, {time.time() - t0:.1f} s")
    pipe = res["pipe"][0]
    sym = (pipe.to(torch.float64).sum(dim=1) > 0).to(torch.float64)
    print(f"       pipe cells {int(pipe.sum().item())}, rows touched {int(sym.sum().item())}, "
          f"max depth {res['depth'].amax().item() * 1e3:.2f} mm")


# ------------------------------------------------------------------ pipe Euler-Lagrange
def test_poiseuille(dev):
    cfg = PipeConfig(Lx=0.004, a=1.0e-3, nx=16, nz=32, i_h=0.05)
    flow = Flow(cfg, device=dev)
    flow.set_poiseuille(factor=0.5)                    # start at half the steady profile
    dt = flow.stable_dt()
    for _ in range(int(1.2 / dt)):                     # ~12 viscous times a^2/(pi^2 nu)
        flow.step(dt)
    # The scheme is second order: its steady bulk velocity is the exact one times (1 + 2/nz^2),
    # and the wall stress balances the body force exactly.
    discrete_bulk = cfg.poiseuille_bulk() * (1.0 + 2.0 / cfg.nz ** 2)
    check("Poiseuille bulk velocity = g i a^2/(12 nu) (1 + 2/nz^2)", flow.bulk_velocity(), discrete_bulk, 1e-4)
    check("Poiseuille bed shear stress = rho g a i / 2", flow.bed_shear_stress(),
          0.5 * cfg.rho_f * cfg.g * cfg.a * cfg.i_h, 1e-4)
    check_small("max |div u| after projection [1/s]", flow.divergence().abs().amax().item(), 1e-8)
    flow.set_poiseuille()
    u0 = flow.u.clone()
    for _ in range(50):
        flow.step(dt)
    check_small("discrete Poiseuille profile stays steady: max relative drift",
                ((flow.u - u0).abs().amax() / u0.amax()).item(), 1e-10)


def test_projection_random(dev):
    cfg = PipeConfig(Lx=0.02, a=2.0e-3, nx=64, nz=24)
    flow = Flow(cfg, device=dev)
    gen = torch.Generator(device=flow.dev).manual_seed(1)
    flow.u = torch.randn(cfg.nz, cfg.nx, generator=gen, device=flow.dev, dtype=flow.dtype)
    flow.w[1:-1] = torch.randn(cfg.nz - 1, cfg.nx, generator=gen, device=flow.dev, dtype=flow.dtype)
    before = flow.divergence().abs().amax().item()
    flow.project(1.0)
    after = flow.divergence().abs().amax().item()
    check_small(f"projection of a random field: max|div| {before:.2e} ->", after / before, 1e-9)


def test_settling(dev):
    cfg = PipeConfig(Lx=0.004, a=0.02, nx=8, nz=64, i_h=0.0, two_way=False, bed="deposit")
    flow = Flow(cfg, device=dev)
    parts = Particles(cfg, 200, flow, seed=5)
    parts.z = torch.full_like(parts.z, 0.019)
    dt = 2.0e-4
    for _ in range(int(0.05 / dt)):                    # ~30 response times
        parts.update(flow, dt)
    ws = -parts.w.mean().item()
    check("grain terminal velocity (Schiller-Naumann)", ws, cfg.settling_velocity(), 1e-6)


def test_deposition_length(dev):
    """One-way coupled grains released at their settling velocity in Poiseuille flow:
    the distance travelled before reaching the bed is (1/w_s) * integral of u dz over
    the fall height; its mean is about U_bulk a / (2 w_s), the deposition length."""
    cfg = PipeConfig(Lx=0.004, a=3.0e-3, nx=8, nz=48, i_h=0.02, two_way=False, c_m=0.0, bed="deposit")
    flow = Flow(cfg, device=dev)
    flow.set_poiseuille()
    parts = Particles(cfg, 4000, flow, seed=9)
    ws, r, c = cfg.settling_velocity(), 0.5 * cfg.d_p, cfg.g * cfg.i_h / (2 * cfg.nu)
    parts.w = torch.full_like(parts.w, -ws)
    z0 = parts.z.clone()
    for _ in range(int(0.6 / 1.0e-4)):
        parts.update(flow, 1.0e-4)
    check("all grains deposited", parts.deposited_fraction(), 1.0, 1e-12)
    q = 0.25 * cfg.dz ** 2                               # discrete steady profile: z (a - z) + dz^2/4
    prim_r = c * (cfg.a * r ** 2 / 2 - r ** 3 / 3 + q * r)
    kinematic = (c * (cfg.a * z0 ** 2 / 2 - z0 ** 3 / 3 + q * z0) - prim_r) / ws
    # first-order inertia: the grain keeps the faster velocity of the layer it came from
    lag = cfg.tau_p / (1 + 0.15 * (cfg.d_p * ws / cfg.nu) ** 0.687)
    inertial = lag * (c * z0 * (cfg.a - z0) - c * r * (cfg.a - r))
    exact = (kinematic + inertial).mean().item()
    check("mean deposition length of grains [m]", parts.travel.mean().item(), exact, 2e-3)
    print(f"       (simple estimate U_bulk a / (2 w_s) = {cfg.poiseuille_bulk() * cfg.a / (2 * ws) * 100:.2f} cm)")


def test_coupling_conservation(dev):
    cfg = PipeConfig(Lx=0.02, a=3.0e-3, nx=64, nz=32, i_h=0.02, bed="reflect")
    flow = Flow(cfg, device=dev)
    flow.set_poiseuille()
    parts = Particles(cfg, 5000, flow, phi=2e-3, seed=7)
    parts.u = 0.5 * parts.u                            # make the drag non-trivial
    dt = 0.5 * flow.stable_dt()
    parts.update(flow, dt)
    parts.project(flow)
    dv = cfg.dx * cfg.dz * cfg.width
    fx_grid = (flow.src_u.sum() * cfg.rho_f * dv).item()
    fz_grid = (flow.src_w.sum() * cfg.rho_f * dv).item()
    check("two-way coupling, x-momentum: grid source = -sum(drag)", fx_grid,
          -(parts.drag_x.sum() * parts.mult).item(), 1e-10)
    check("two-way coupling, z-momentum: grid source = -sum(drag)", fz_grid,
          -(parts.drag_z.sum() * parts.mult).item(), 1e-10)
    solid = parts.mult * parts.x.numel() * math.pi / 6 * cfg.d_p ** 3 / (cfg.Lx * cfg.a * cfg.width)
    check("solid volume fraction carried by the multiplicity", solid, 2e-3, 1e-12)


def test_suspension_weight(dev):
    """Steady settling in still water: the grains hand their submerged weight to the fluid."""
    cfg = PipeConfig(Lx=0.004, a=0.02, nx=8, nz=64, i_h=0.0, bed="reflect")
    flow = Flow(cfg, device=dev)
    parts = Particles(cfg, 500, flow, phi=1e-3, seed=2)
    parts.z = 0.01 + 0.002 * (parts.z / cfg.a)
    for _ in range(200):
        parts.update(flow, 2.0e-4)
    parts.project(flow)
    dv = cfg.dx * cfg.dz * cfg.width
    fz = (flow.src_w.sum() * cfg.rho_f * dv).item()
    weight = parts.mult * parts.x.numel() * parts.mass * (1 - cfg.rho_f / cfg.rho_p) * cfg.g
    check("settling suspension: force on fluid = -submerged weight", fz, -weight, 1e-6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    dev = pick_device("cpu" if args.cpu else "cuda")
    print(f"torch {torch.__version__} on {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    test_darcy_uniform(dev)
    test_pipe_cubic_law(dev)
    test_random_field(dev)
    test_poiseuille(dev)
    test_projection_random(dev)
    test_settling(dev)
    test_deposition_length(dev)
    test_coupling_conservation(dev)
    test_suspension_weight(dev)
    if not args.quick:
        test_progression(dev)
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)
