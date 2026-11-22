"""2-D (x-z) Euler-Lagrange solver for sediment-laden flow in an erosion-pipe segment.

A single-GPU PyTorch re-write of the core algorithm of RichterLab/NTLP, reduced
to two dimensions and to what a water-sand pipe flow needs.  The mapping to the
NTLP Fortran routines is given next to each method so the two can be read side
by side (see ../NTLP_notes.md).

Flow (class Flow)
  * incompressible Navier-Stokes, x periodic and Fourier pseudo-spectral,
    z between two no-slip walls with second-order staggered differences
    (u, p at cell centres, w at faces) -- the NTLP discretization with y dropped;
  * low-storage RK3 with a pressure projection every stage (NTLP comp1/comp_p/comp2);
  * pressure Poisson: FFT in x, then one pre-inverted nz-by-nz system per
    wavenumber (NTLP: fft2d_mpi + solve_trid, with MPI transposes in between);
  * driven by a uniform body force g*i_h, i.e. a mean hydraulic gradient i_h.

Particles (class Particles)
  * point particles, struct-of-arrays tensors instead of NTLP's linked list;
  * bilinear interpolation of the fluid velocity (NTLP uf_interp_lin);
  * Stokes drag with the Schiller-Naumann correction, backward-Euler update
    (NTLP particle_update_BE);
  * two-way coupling by projecting the drag reaction onto the 4 surrounding
    nodes (NTLP particle_coupling_update), each computational particle
    standing for `mult` real ones (NTLP part%mult);
  * water-sand additions that NTLP (droplets in air) does not need:
    buoyancy-reduced gravity, added mass, deposition on the bed.

Validity: the point-particle closure assumes d_p smaller than the grid spacing
and a dilute suspension.  Grains of a real erosion pipe (d50/a ~ 0.05-0.3, dense
bed load with contacts) are outside that range -- see ../README.md, "What this
prototype is not".
"""
from dataclasses import dataclass
import math

import torch

RK_GAMMA = (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)       # NTLP: gama(1:3)
RK_ZETA = (0.0, -17.0 / 60.0, -5.0 / 12.0)           # NTLP: zetas(1:3)


@dataclass
class PipeConfig:
    Lx: float = 0.03           # length of the periodic pipe segment [m]
    a: float = 3.0e-3          # pipe depth [m]
    nx: int = 192
    nz: int = 24
    nu: float = 1.0e-6         # kinematic viscosity [m2/s]
    rho_f: float = 1000.0
    g: float = 9.81
    i_h: float = 0.02          # mean hydraulic gradient along the pipe [-]
    d_p: float = 1.0e-4        # grain diameter [m]
    rho_p: float = 2650.0
    c_m: float = 0.5           # added-mass coefficient (0 switches it off)
    two_way: bool = True
    bed: str = "deposit"       # "deposit" (stick to the bed) or "reflect" (NTLP icase=0)
    width: float = 1.0         # out-of-plane width represented by the 2-D slice [m]

    @property
    def dx(self):
        return self.Lx / self.nx

    @property
    def dz(self):
        return self.a / self.nz

    @property
    def tau_p(self):
        """Stokes response time of a grain [s]."""
        return self.rho_p * self.d_p ** 2 / (18.0 * self.rho_f * self.nu)

    def settling_velocity(self):
        """Terminal velocity in still water with the Schiller-Naumann correction [m/s]."""
        lo, hi = 1.0e-9, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            drag = (1.0 + 0.15 * (self.d_p * mid / self.nu) ** 0.687) * mid / self.tau_p
            lo, hi = (mid, hi) if drag < (1.0 - self.rho_f / self.rho_p) * self.g else (lo, mid)
        return 0.5 * (lo + hi)

    def poiseuille_bulk(self):
        """Mean velocity of laminar flow between plates = eq. 3 of Robbins et al."""
        return self.g * self.i_h * self.a ** 2 / (12.0 * self.nu)


class Flow:
    def __init__(self, cfg, device="cpu", dtype=torch.float64):
        self.cfg = cfg
        self.dev = torch.device(device)
        self.dtype = dtype
        self.cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        nx, nz = cfg.nx, cfg.nz
        dev = self.dev

        m = torch.arange(nx // 2 + 1, device=dev, dtype=dtype)
        kx = 2.0 * math.pi * m / cfg.Lx
        if nx % 2 == 0:
            kx[-1] = 0.0                       # no odd derivative of the Nyquist mode
        self.kx = kx
        self.ikx = torch.complex(torch.zeros_like(kx), kx)
        self.kx2 = kx * kx
        self.dealias = (m <= nx // 3).to(dtype)            # 2/3 rule
        self.zc = (torch.arange(nz, device=dev, dtype=dtype) + 0.5) * cfg.dz

        # Poisson operator per wavenumber, inverted once (float64, then cast)
        idx = torch.arange(nz, device=dev)
        d2 = torch.zeros(nz, nz, dtype=torch.float64, device=dev)
        d2[idx, idx] = -2.0 / cfg.dz ** 2
        d2[idx[:-1], idx[1:]] = 1.0 / cfg.dz ** 2
        d2[idx[1:], idx[:-1]] = 1.0 / cfg.dz ** 2
        d2[0, 0] = -1.0 / cfg.dz ** 2          # Neumann walls
        d2[-1, -1] = -1.0 / cfg.dz ** 2
        eye = torch.eye(nz, dtype=torch.float64, device=dev)
        mats = d2[None] - self.kx2.to(torch.float64)[:, None, None] * eye[None]
        self.singular = self.kx2 == 0.0        # kx = 0 and Nyquist: pin phi at k = 0
        mats[self.singular, 0, :] = 0.0
        mats[self.singular, 0, 0] = 1.0
        self.minv = torch.linalg.inv(mats).to(self.cdtype)

        self.u = torch.zeros(nz, nx, dtype=dtype, device=dev)
        self.w = torch.zeros(nz + 1, nx, dtype=dtype, device=dev)
        self.ru_old = torch.zeros_like(self.u)
        self.rw_old = torch.zeros_like(self.w)
        self.src_u = torch.zeros_like(self.u)   # particle feedback (NTLP partsrc)
        self.src_w = torch.zeros_like(self.w)
        self.time = 0.0

    # ----- helpers --------------------------------------------------------
    def set_poiseuille(self, factor=1.0):
        """Steady laminar profile of the DISCRETE equations: the parabola plus dz^2/4,
        so that an undisturbed flow stays exactly steady (bulk = exact * (1 + 2/nz^2))."""
        cfg = self.cfg
        prof = cfg.g * cfg.i_h / (2.0 * cfg.nu) * (self.zc * (cfg.a - self.zc) + 0.25 * cfg.dz ** 2)
        self.u = factor * prof[:, None] * torch.ones(1, cfg.nx, dtype=self.dtype, device=self.dev)
        self.w.zero_()

    def ddx(self, f, dealias=False):
        fh = torch.fft.rfft(f, dim=-1) * self.ikx
        if dealias:
            fh = fh * self.dealias
        return torch.fft.irfft(fh, n=self.cfg.nx, dim=-1)

    def d2dx2(self, f):
        return torch.fft.irfft(-self.kx2 * torch.fft.rfft(f, dim=-1), n=self.cfg.nx, dim=-1)

    def u_ext(self):
        """u with one ghost row per wall so that u = 0 on the wall (NTLP fill_ext)."""
        return torch.cat([-self.u[:1], self.u, -self.u[-1:]], dim=0)

    # ----- right-hand side (NTLP comp1 -> rhs_uvw_DNS) --------------------------
    def rhs(self):
        cfg = self.cfg
        dz = cfg.dz
        u, w = self.u, self.w
        ue = self.u_ext()
        u_face = 0.5 * (ue[:-1] + ue[1:])                  # [nz+1, nx], zero on walls
        uw = u_face * w
        ru = -self.ddx(u * u, dealias=True) - (uw[1:] - uw[:-1]) / dz
        ru = ru + cfg.nu * (self.d2dx2(u) + (ue[2:] - 2.0 * ue[1:-1] + ue[:-2]) / dz ** 2)
        ru = ru + cfg.g * cfg.i_h + self.src_u

        wc = 0.5 * (w[:-1] + w[1:])                        # w at centres [nz, nx]
        ww = wc * wc
        rw = torch.zeros_like(w)
        rw[1:-1] = (-self.ddx(uw, dealias=True)[1:-1] - (ww[1:] - ww[:-1]) / dz
                    + cfg.nu * (self.d2dx2(w)[1:-1] + (w[2:] - 2.0 * w[1:-1] + w[:-2]) / dz ** 2)
                    + self.src_w[1:-1])
        return ru, rw

    # ----- pressure projection (NTLP comp_p -> pressure/solve_trid, comp2) ------
    def divergence(self):
        return self.ddx(self.u) + (self.w[1:] - self.w[:-1]) / self.cfg.dz

    def project(self, cdt):
        cfg = self.cfg
        rh = torch.fft.rfft(self.divergence() / cdt, dim=-1)      # [nz, nk]
        rh = rh.permute(1, 0).unsqueeze(-1).clone()                # [nk, nz, 1]
        rh[self.singular, 0, 0] = 0.0
        ph = torch.matmul(self.minv, rh)[:, :, 0].permute(1, 0)    # [nz, nk]
        self.u = self.u - cdt * torch.fft.irfft(ph * self.ikx, n=cfg.nx, dim=-1)
        phi = torch.fft.irfft(ph, n=cfg.nx, dim=-1)
        self.w[1:-1] -= cdt * (phi[1:] - phi[:-1]) / cfg.dz

    # ----- one time step: three RK stages (NTLP main loop) --------------------
    @torch.no_grad()
    def step(self, dt):
        for s in range(3):
            ru, rw = self.rhs()
            self.u = self.u + dt * (RK_GAMMA[s] * ru + RK_ZETA[s] * self.ru_old)
            self.w = self.w + dt * (RK_GAMMA[s] * rw + RK_ZETA[s] * self.rw_old)
            self.project(dt * (RK_GAMMA[s] + RK_ZETA[s]))
            self.ru_old, self.rw_old = ru, rw
        self.time += dt

    def stable_dt(self, cfl=0.4):
        cfg = self.cfg
        lam = cfg.nu * (4.0 / cfg.dz ** 2 + (math.pi / cfg.dx) ** 2)
        umax = max(self.u.abs().amax().item(), 1.0e-12)
        wmax = max(self.w.abs().amax().item(), 1.0e-12)
        return min(2.0 / lam, cfl * cfg.dx / umax, cfl * cfg.dz / wmax)

    # ----- diagnostics ----------------------------------------------------------
    def bulk_velocity(self):
        return self.u.mean().item()

    def bed_shear_stress(self):
        """Mean wall shear stress on the lower wall [Pa]."""
        cfg = self.cfg
        return cfg.rho_f * cfg.nu * self.u[0].mean().item() / (0.5 * cfg.dz)


class Particles:
    def __init__(self, cfg, n, flow, phi=1.0e-3, seed=0):
        """n computational grains, uniformly seeded; phi = mean solid volume fraction."""
        self.cfg = cfg
        dev, dtype = flow.dev, flow.dtype
        self.dev, self.dtype = dev, dtype
        gen = torch.Generator(device=dev).manual_seed(seed)
        r = 0.5 * cfg.d_p
        self.x = torch.rand(n, generator=gen, device=dev, dtype=dtype) * cfg.Lx
        self.z = r + torch.rand(n, generator=gen, device=dev, dtype=dtype) * (cfg.a - 2.0 * r)
        self.deposited = torch.zeros(n, dtype=torch.bool, device=dev)
        self.u, self.w = self.interp(flow)      # start at the fluid velocity (NTLP it <= 1)
        vol_p = math.pi / 6.0 * cfg.d_p ** 3
        self.mass = cfg.rho_p * vol_p
        self.mult = phi * cfg.Lx * cfg.a * cfg.width / (n * vol_p)
        self.drag_x = torch.zeros(n, dtype=dtype, device=dev)   # drag force per grain [N]
        self.drag_z = torch.zeros(n, dtype=dtype, device=dev)
        self.travel = torch.zeros(n, dtype=dtype, device=dev)   # streamwise distance travelled [m]

    # ----- grid <-> particle weights (NTLP uf_interp_lin) ------------------------
    def _weights(self):
        cfg = self.cfg
        fx = self.x / cfg.dx
        i0f = torch.floor(fx)
        wx = fx - i0f
        i0 = i0f.long() % cfg.nx
        i1 = (i0 + 1) % cfg.nx
        fz = self.z / cfg.dz + 0.5                 # row in the ghost-extended u array
        k0f = torch.floor(fz).clamp(min=0.0, max=float(cfg.nz))
        wz = fz - k0f
        gz = self.z / cfg.dz                       # row in the w array
        m0f = torch.floor(gz).clamp(min=0.0, max=float(cfg.nz - 1))
        vz = gz - m0f
        return i0, i1, wx, k0f.long(), wz, m0f.long(), vz

    def interp(self, flow):
        i0, i1, wx, k0, wz, m0, vz = self._weights()
        ue = flow.u_ext()
        w = flow.w
        uf = ((1.0 - wz) * ((1.0 - wx) * ue[k0, i0] + wx * ue[k0, i1])
              + wz * ((1.0 - wx) * ue[k0 + 1, i0] + wx * ue[k0 + 1, i1]))
        wf = ((1.0 - vz) * ((1.0 - wx) * w[m0, i0] + wx * w[m0, i1])
              + vz * ((1.0 - wx) * w[m0 + 1, i0] + wx * w[m0 + 1, i1]))
        return uf, wf

    # ----- advance (NTLP particle_update_BE + particle_bcs_*) -----------------------
    @torch.no_grad()
    def update(self, flow, dt):
        cfg = self.cfg
        uf, wf = self.interp(flow)
        rel = torch.sqrt((self.u - uf) ** 2 + (self.w - wf) ** 2)
        corr = 1.0 + 0.15 * (cfg.d_p * rel / cfg.nu) ** 0.687       # Schiller-Naumann
        beta = 1.0 + cfg.c_m * cfg.rho_f / cfg.rho_p                # added-mass inertia
        rate = corr / (cfg.tau_p * beta)
        g_red = (1.0 - cfg.rho_f / cfg.rho_p) * cfg.g               # buoyancy-reduced gravity
        mobile = ~self.deposited
        zero = torch.zeros_like(self.u)

        x_new = self.x + dt * self.u
        z_new = self.z + dt * self.w
        u_new = (self.u + dt * rate * uf) / (1.0 + dt * rate)
        w_new = (self.w + dt * rate * wf - dt * g_red / beta) / (1.0 + dt * rate)

        # drag on each grain = m (beta dv/dt - g_red e_g); its reaction goes to the fluid
        self.drag_x = torch.where(mobile, self.mass * beta * (u_new - self.u) / dt, zero)
        self.drag_z = torch.where(mobile, self.mass * (beta * (w_new - self.w) / dt + g_red), zero)

        self.travel = self.travel + torch.where(mobile, dt * self.u, zero)
        self.x = torch.where(mobile, torch.remainder(x_new, cfg.Lx), self.x)
        self.z = torch.where(mobile, z_new, self.z)
        self.u = torch.where(mobile, u_new, zero)
        self.w = torch.where(mobile, w_new, zero)

        r = 0.5 * cfg.d_p
        top = cfg.a - r
        hit_top = self.z > top                                      # elastic roof
        self.z = torch.where(hit_top, 2.0 * top - self.z, self.z)
        self.w = torch.where(hit_top, -self.w, self.w)
        hit_bed = self.z < r
        if cfg.bed == "deposit":
            self.deposited = self.deposited | hit_bed
            self.u = torch.where(hit_bed, zero, self.u)
            self.w = torch.where(hit_bed, zero, self.w)
            self.z = torch.where(hit_bed, torch.full_like(self.z, r), self.z)
        else:
            self.z = torch.where(hit_bed, 2.0 * r - self.z, self.z)
            self.w = torch.where(hit_bed, -self.w, self.w)
        self.z = self.z.clamp(min=r, max=top)

    # ----- particle -> grid momentum source (NTLP particle_coupling_update) ------------
    @torch.no_grad()
    def project(self, flow):
        cfg = self.cfg
        nx, nz = cfg.nx, cfg.nz
        i0, i1, wx, k0, wz, m0, vz = self._weights()
        scale = -self.mult / (cfg.rho_f * cfg.dx * cfg.dz * cfg.width)
        su = torch.zeros(nz * nx, dtype=self.dtype, device=self.dev)
        sw = torch.zeros((nz + 1) * nx, dtype=self.dtype, device=self.dev)
        fx = scale * self.drag_x
        fz = scale * self.drag_z
        ka = (k0 - 1).clamp(min=0, max=nz - 1)     # ghost rows fold onto the wall cell
        kb = k0.clamp(min=0, max=nz - 1)
        for row, wr in ((ka, 1.0 - wz), (kb, wz)):
            su.index_add_(0, row * nx + i0, fx * wr * (1.0 - wx))
            su.index_add_(0, row * nx + i1, fx * wr * wx)
        for row, wr in ((m0, 1.0 - vz), (m0 + 1, vz)):
            sw.index_add_(0, row * nx + i0, fz * wr * (1.0 - wx))
            sw.index_add_(0, row * nx + i1, fz * wr * wx)
        flow.src_u = su.reshape(nz, nx)
        flow.src_w = sw.reshape(nz + 1, nx)

    # ----- diagnostics ------------------------------------------------------------
    def deposited_fraction(self):
        return self.deposited.to(self.dtype).mean().item()


def advance(flow, parts, dt):
    """One coupled step: flow (3 RK stages), then particles, then feedback."""
    flow.step(dt)
    if parts is not None:
        parts.update(flow, dt)
        if parts.cfg.two_way:
            parts.project(flow)
