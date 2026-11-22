"""Robbins-type steady backward-erosion-piping (BEP) model, batched on the GPU.

Re-implementation of the plan-view model of
    Robbins, Griffiths & Fenton (2021), Comput. Geotech. 138, 104322
on a regular grid, written so that B Monte-Carlo realizations advance together
as one tensor of shape [B, ny, nx].  All equation numbers below refer to that paper.

Differences from the paper's program (rbep2d) -- read before comparing numbers:
  * cell-centred finite volumes with harmonic face transmissivities instead of
    4-node finite elements;
  * random fields by FFT circulant embedding of the Markov correlation
    rho(r) = exp(-2 r / theta) (point values at cell centres) instead of
    Local Average Subdivision, i.e. no local-averaging variance reduction;
  * the hydraulic gradient "in a cell" is taken from transmissivity-weighted
    face heads; like the paper's element gradient it is defined over one cell
    and therefore mesh dependent;
  * the upstream head is raised geometrically (H <- H (1 + dH_rel)); the paper
    says "increments of 1 percent of the assigned value".

Layout: x is the piping direction, index i = 0 is UPSTREAM (h = H),
i = nx-1 is DOWNSTREAM (h = 0); y = 0 and y = W are no-flow walls.
"""
from dataclasses import dataclass
import math

import torch


@dataclass
class BEPConfig:
    L: float = 10.0            # domain length in the piping direction [m]
    W: float = 10.0            # domain width [m]
    nx: int = 80
    ny: int = 80
    thickness: float = 1.0     # sand thickness represented by the plan view [m]
    d50: float = 2.0e-4        # median grain size [m]
    tau_c: float = 0.33        # critical bed shear stress [Pa]
    rho_w: float = 1000.0
    mu: float = 1.0e-3
    g: float = 9.81
    H0: float = 0.1            # first upstream head [m]
    dH_rel: float = 0.01       # relative head increment per equilibrium
    H_max: float = 20.0        # give up above this head [m]
    a0_factor: float = 3.0     # initial pipe depth = a0_factor * d50
    da_factor: float = 0.5     # depth increment  = da_factor * d50
    pipe_factor: float = 1.0   # multiplier on the laminar pipe conductance (1 = eq. 4);
                               # < 1 mimics extra resistance (turbulence, roughness, bed load)
    cg_tol: float = 1.0e-8     # relative residual of the head solve
    cg_maxit: int = 20000
    max_steps: int = 200000    # hard stop on outer iterations

    @property
    def dx(self):
        return self.L / self.nx

    @property
    def dy(self):
        return self.W / self.ny

    @property
    def c_pipe(self):
        """Pipe transmissivity is c_pipe * a**3  (eq. 4, flow per unit width)."""
        return self.pipe_factor * self.rho_w * self.g / (12.0 * self.mu)


# --------------------------------------------------------------------------
# random fields
# --------------------------------------------------------------------------
def gaussian_fields(B, cfg, theta, generator, device, dtype):
    """Two independent unit Gaussian fields [B, ny, nx] with the isotropic
    Markov correlation rho(r) = exp(-2 r / theta)."""
    ny, nx = cfg.ny, cfg.nx
    npx = nx + int(math.ceil(3.0 * theta / cfg.dx))
    npy = ny + int(math.ceil(3.0 * theta / cfg.dy))
    npx += npx % 2
    npy += npy % 2
    ix = torch.arange(npx, device=device, dtype=dtype)
    iy = torch.arange(npy, device=device, dtype=dtype)
    rx = torch.minimum(ix, npx - ix) * cfg.dx        # periodic lag
    ry = torch.minimum(iy, npy - iy) * cfg.dy
    r = torch.sqrt(ry[:, None] ** 2 + rx[None, :] ** 2)
    cov = torch.exp(-2.0 * r / theta)
    lam = torch.fft.fft2(cov, dim=(-2, -1)).real.clamp(min=0.0)
    amp = torch.sqrt(lam / (npx * npy))
    noise = torch.complex(
        torch.randn(B, npy, npx, generator=generator, device=device, dtype=dtype),
        torch.randn(B, npy, npx, generator=generator, device=device, dtype=dtype),
    )
    z = torch.fft.fft2(amp[None] * noise, dim=(-2, -1))
    return z.real[:, :ny, :nx].clone(), z.imag[:, :ny, :nx].clone()


def lognormal_fields(B, cfg, mean_k=1.0e-5, cov_k=0.0, mean_icr=0.3, cov_icr=0.0,
                     theta=1.0, rho=0.0, seed=0, device="cpu", dtype=torch.float64):
    """Cross-correlated lognormal k [m/s] and i_cr fields, each [B, ny, nx].

    rho is the correlation of the two underlying Gaussian fields
    (covariance-matrix decomposition for two fields).
    """
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)
    g1, g2 = gaussian_fields(B, cfg, theta, gen, dev, dtype)
    g2 = rho * g1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * g2

    def to_lognormal(gfield, mean, cov):
        s2 = math.log(1.0 + cov * cov)
        return torch.exp(math.log(mean) - 0.5 * s2 + math.sqrt(s2) * gfield)

    return to_lognormal(g1, mean_k, cov_k), to_lognormal(g2, mean_icr, cov_icr)


# --------------------------------------------------------------------------
# head solve  (eqs 1, 5, 6 with the pipe treated through eq. 4)
# --------------------------------------------------------------------------
def conductances(T, cfg):
    """Face conductances.  Cx: [B, ny, nx+1], Cy: [B, ny+1, nx]."""
    B, ny, nx = T.shape
    ax = cfg.dy / cfg.dx
    ay = cfg.dx / cfg.dy
    TL, TR = T[:, :, :-1], T[:, :, 1:]
    cx_in = 2.0 * TL * TR / (TL + TR) * ax
    Cx = torch.cat([2.0 * ax * T[:, :, :1], cx_in, 2.0 * ax * T[:, :, -1:]], dim=2)
    TS, TN = T[:, :-1, :], T[:, 1:, :]
    cy_in = 2.0 * TS * TN / (TS + TN) * ay
    wall = torch.zeros(B, 1, nx, dtype=T.dtype, device=T.device)
    Cy = torch.cat([wall, cy_in, wall], dim=1)
    return Cx, Cy


def apply_A(h, Cx, Cy):
    """Matrix-free SPD operator of the discrete -div(T grad h)."""
    out = torch.zeros_like(h)
    out[:, :, 0] += Cx[:, :, 0] * h[:, :, 0]
    out[:, :, -1] += Cx[:, :, -1] * h[:, :, -1]
    fx = Cx[:, :, 1:-1] * (h[:, :, 1:] - h[:, :, :-1])
    out[:, :, :-1] -= fx
    out[:, :, 1:] += fx
    fy = Cy[:, 1:-1, :] * (h[:, 1:, :] - h[:, :-1, :])
    out[:, :-1, :] -= fy
    out[:, 1:, :] += fy
    return out


def _dot(a, b):
    return (a * b).flatten(1).sum(dim=1)


def solve_head(Cx, Cy, H, h0, tol, maxit, check_every=10):
    """Batched Jacobi-preconditioned CG.  Returns (h, iterations)."""
    b = torch.zeros_like(h0)
    b[:, :, 0] = Cx[:, :, 0] * H[:, None]
    diag = Cx[:, :, :-1] + Cx[:, :, 1:] + Cy[:, :-1, :] + Cy[:, 1:, :]
    tiny = 1.0e-300 if h0.dtype == torch.float64 else 1.0e-30
    x = h0.clone()
    r = b - apply_A(x, Cx, Cy)
    z = r / diag
    p = z.clone()
    rz = _dot(r, z)
    target = (tol * tol) * _dot(b, b).clamp(min=tiny)
    zero = torch.zeros_like(rz)
    it = 0
    while it < maxit:
        live = _dot(r, r) > target
        if it % check_every == 0 and not live.any().item():
            break
        Ap = apply_A(p, Cx, Cy)
        alpha = torch.where(live, rz / _dot(p, Ap).clamp(min=tiny), zero)
        x += alpha[:, None, None] * p
        r -= alpha[:, None, None] * Ap
        z = r / diag
        rz_new = _dot(r, z)
        beta = torch.where(live, rz_new / rz.clamp(min=tiny), zero)
        p = z + beta[:, None, None] * p
        rz = rz_new
        it += 1
    return x, it


def cell_gradient(h, T, H, cfg):
    """|grad h| inside every cell, from transmissivity-weighted face heads."""
    B, ny, nx = h.shape
    TL, TR = T[:, :, :-1], T[:, :, 1:]
    hf = (TL * h[:, :, :-1] + TR * h[:, :, 1:]) / (TL + TR)
    up = H[:, None, None] * torch.ones(B, ny, 1, dtype=h.dtype, device=h.device)
    down = torch.zeros(B, ny, 1, dtype=h.dtype, device=h.device)
    hfx = torch.cat([up, hf, down], dim=2)
    gx = (hfx[:, :, 1:] - hfx[:, :, :-1]) / cfg.dx
    TS, TN = T[:, :-1, :], T[:, 1:, :]
    hg = (TS * h[:, :-1, :] + TN * h[:, 1:, :]) / (TS + TN)
    hfy = torch.cat([h[:, :1, :], hg, h[:, -1:, :]], dim=1)   # no-flow walls
    gy = (hfy[:, 1:, :] - hfy[:, :-1, :]) / cfg.dy
    return torch.sqrt(gx * gx + gy * gy)


def dilate8(mask):
    """Cells sharing a node with a True cell (8-neighbourhood), incl. itself."""
    B, ny, nx = mask.shape
    pad = torch.zeros(B, ny + 2, nx + 2, dtype=torch.bool, device=mask.device)
    pad[:, 1:-1, 1:-1] = mask
    out = torch.zeros_like(mask)
    for dj in (0, 1, 2):
        for di in (0, 1, 2):
            out = out | pad[:, dj:dj + ny, di:di + nx]
    return out


# --------------------------------------------------------------------------
# progression algorithm (paper Fig. 4), vectorized over realizations
# --------------------------------------------------------------------------
@torch.no_grad()
def run_bep(k, icr, cfg, init_rows=None, record=False, verbose=False):
    """Advance B realizations until the pipe breaches the upstream boundary.

    k, icr    : [B, ny, nx] hydraulic conductivity [m/s] and critical gradient
    init_rows : rows of the downstream column where piping is initiated
                (default: the centre row, as in most of the paper's analyses)
    record    : keep the progression curve of realization 0

    Returns a dict with
      'HL_cr'   [B]  critical average gradient (NaN if no breach below H_max)
      'pipe'    [B, ny, nx] bool, final pipe cells
      'depth'   [B, ny, nx] final pipe depth a [m]
      'head'    [B, ny, nx] final head [m]
      'solves'  number of batched head solves, 'cg_iters' total CG iterations
      'curve'   list of (pipe length / L, H / L) for realization 0 if record
    """
    B0, ny, nx = k.shape
    dev, dt = k.device, k.dtype
    a0 = cfg.a0_factor * cfg.d50
    da = cfg.da_factor * cfg.d50
    if init_rows is None:
        init_rows = [ny // 2]

    ids = torch.arange(B0, device=dev)
    pipe = torch.zeros(B0, ny, nx, dtype=torch.bool, device=dev)
    for j in init_rows:
        pipe[:, j, nx - 1] = True
    depth = torch.where(pipe, torch.full((B0, ny, nx), a0, dtype=dt, device=dev),
                        torch.zeros(B0, ny, nx, dtype=dt, device=dev))
    H = torch.full((B0,), cfg.H0, dtype=dt, device=dev)
    xc = (torch.arange(nx, device=dev, dtype=dt) + 0.5) * cfg.dx
    h = H[:, None, None] * (1.0 - xc / cfg.L)[None, None, :] * torch.ones(B0, ny, 1, dtype=dt, device=dev)
    done = torch.zeros(B0, dtype=torch.bool, device=dev)

    out = {
        "HL_cr": torch.full((B0,), float("nan"), dtype=dt, device=dev),
        "pipe": torch.zeros(B0, ny, nx, dtype=torch.bool, device=dev),
        "depth": torch.zeros(B0, ny, nx, dtype=dt, device=dev),
        "head": torch.zeros(B0, ny, nx, dtype=dt, device=dev),
    }
    curve = []
    solves = 0
    cg_iters = 0
    neg_inf = torch.full((1,), -float("inf"), dtype=dt, device=dev)
    half_rho_g = 0.5 * cfg.rho_w * cfg.g

    def store(sel):
        out["pipe"][ids[sel]] = pipe[sel]
        out["depth"][ids[sel]] = depth[sel]
        out["head"][ids[sel]] = h[sel]

    for step in range(cfg.max_steps):
        B = k.shape[0]
        T = torch.where(pipe, cfg.c_pipe * depth ** 3, k * cfg.thickness)
        Cx, Cy = conductances(T, cfg)
        h, nit = solve_head(Cx, Cy, H, h, cfg.cg_tol, cfg.cg_maxit)
        solves += 1
        cg_iters += nit
        grad = cell_gradient(h, T, H, cfg)
        live = ~done

        # (1) Picard iteration on the pipe depth: deepen where tau > tau_c  (eqs 7, 8)
        tau = half_rho_g * depth * grad
        deepen = pipe & (tau > cfg.tau_c) & live[:, None, None]
        deepened = deepen.flatten(1).any(dim=1)
        depth = torch.where(deepen, depth + da, depth)

        # (2) pipe progression: neighbour with the largest (i_h - i_cr) > 0  (eq. 9)
        cand = dilate8(pipe) & ~pipe
        excess = torch.where(cand, grad - icr, neg_inf).flatten(1)
        best = excess.argmax(dim=1)
        progress = live & ~deepened & (excess.amax(dim=1) > 0.0)
        rows = torch.arange(B, device=dev)[progress]
        jj = (best // nx)[progress]
        ii = (best % nx)[progress]
        pipe[rows, jj, ii] = True
        depth[rows, jj, ii] = a0

        # (3) breach, or raise the upstream head once in equilibrium
        breached = live & pipe[:, :, 0].any(dim=1)
        out["HL_cr"][ids[breached]] = H[breached] / cfg.L
        equil = live & ~deepened & ~progress
        gave_up = equil & (H * (1.0 + cfg.dH_rel) > cfg.H_max)
        newly_done = breached | gave_up
        store(newly_done)
        done = done | newly_done
        if record and bool((ids == 0).any().item()):
            m = pipe[0]
            reach = (nx - torch.arange(nx, device=dev)[m.any(dim=0)].amin()).item() / nx
            curve.append((reach, H[0].item() / cfg.L))
        scale = torch.where(equil & ~done, torch.full_like(H, 1.0 + cfg.dH_rel), torch.ones_like(H))
        H = H * scale
        h = h * scale[:, None, None]          # exact new solution: problem is linear in H

        n_done = int(done.sum().item())
        if verbose and step % 200 == 0:
            print(f"  step {step:6d}  alive {B - n_done:5d}/{B0}  "
                  f"H/L(max) {H.amax().item() / cfg.L:.3f}  CG its {nit}", flush=True)
        if n_done == B:
            break
        if 2 * n_done >= B:                   # compact the batch
            keep = ~done
            k, icr, pipe, depth = k[keep], icr[keep], pipe[keep], depth[keep]
            H, h, ids, done = H[keep], h[keep], ids[keep], done[keep]
    else:
        store(~done)

    out.update(solves=solves, cg_iters=cg_iters, curve=curve)
    return out
