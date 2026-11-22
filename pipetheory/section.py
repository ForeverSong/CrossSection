"""Cross-section of a laminar erosion pipe under a flat rigid roof, bed at the threshold of motion.

Scales.  S is the hydraulic gradient along the pipe, tau_c the critical shear stress of a flat bed.
    length    a_c = 2 tau_c / (rho g S)      (gap of a parallel-plate flow whose wall stress is tau_c)
    velocity  g S a_c^2 / nu
In these units the streamwise velocity u(y, z) in the section {-a(y) < z < 0} obeys
    u_yy + u_zz = -1,   u = 0 on the roof z = 0 and on the bed z = -a(y),
the bed shear stress is  tau/tau_c = 2 |du/dn|,  and the discharge is  Q = chi * g S a_c^4 / nu.

Threshold of a grain on a bank inclined at angle phi across the stream (Coulomb friction mu;
streamwise drag and the down-slope component of the weight are perpendicular):
    (mu tau/tau_c)^2 + sin^2(phi) = mu^2 cos^2(phi)
i.e.  tau/tau_c = cos(phi) sqrt(1 - tan^2(phi)/mu^2),   tan(phi) = |a'(y)|.

Lubrication limit (mu -> 0, tau/tau_c = a):  a = cos(mu y),  half-width pi/(2 mu),  chi = 1/(9 mu).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")      # NumPy 2 / NumPy 1


def lubrication(mu):
    """Closed-form shallow (small mu) solution, in units of a_c."""
    b = np.pi / (2.0 * mu)
    return dict(mu=mu, a_max=1.0, half_width=b, area=2.0 / mu, chi=1.0 / (9.0 * mu),
                W_over_amax=np.pi / mu, aavg_over_amax=2.0 / np.pi, W_over_aavg=np.pi ** 2 / (2.0 * mu),
                conveyance_vs_rectangle=np.pi ** 2 / 6.0)


def stokes_roofed(a, b, nz=61):
    """Unidirectional Stokes flow in the half-section 0 <= y <= b, -a(y) <= z <= 0 (symmetric about y = 0).

    a : gap on a uniform grid s = y/b in [0, 1]; a[-1] may be 0 (bank meets the roof) or > 0 (side wall).
    Mapped coordinates (s, zeta) with z = -a(s) zeta; the equation is multiplied by a^2 so that all
    coefficients stay bounded where a -> 0.
    Returns dict: u (ns, nz), tau_bed (ns), tau_roof (ns) [both = 2|du/dn|], q (full section), area (full).
    """
    a = np.asarray(a, float)
    ns = a.size
    hs, hz = 1.0 / (ns - 1), 1.0 / (nz - 1)
    a_s = np.gradient(a, hs, edge_order=2)
    a_s[0] = 0.0
    a_ss = np.empty(ns)
    a_ss[1:-1] = (a[2:] - 2.0 * a[1:-1] + a[:-2]) / hs ** 2
    a_ss[0] = 2.0 * (a[1] - a[0]) / hs ** 2
    a_ss[-1] = a_ss[-2]
    zeta = np.linspace(0.0, 1.0, nz)
    ni, nj = ns - 1, nz - 2                       # unknowns: i = 0..ns-2, j = 1..nz-2
    idx = -np.ones((ns, nz), np.int64)
    idx[:ni, 1:nz - 1] = np.arange(ni * nj).reshape(ni, nj)
    I, J = np.meshgrid(np.arange(ni), np.arange(1, nz - 1), indexing="ij")
    A_ = (a[I] / b) ** 2
    B_ = -2.0 * zeta[J] * a[I] * a_s[I] / b ** 2
    C_ = (zeta[J] * a_s[I] / b) ** 2 + 1.0
    D_ = zeta[J] * (2.0 * a_s[I] ** 2 - a[I] * a_ss[I]) / b ** 2
    rows, cols, vals = [], [], []
    me = idx[I, J].ravel()

    def add(di, dj, coef):
        ii, jj = I + di, J + dj
        ii = np.where(ii < 0, -ii, ii)            # symmetry about s = 0
        k = idx[ii, jj].ravel()
        ok = k >= 0                               # neighbours on Dirichlet boundaries drop out
        rows.append(me[ok]); cols.append(k[ok]); vals.append(coef.ravel()[ok])

    add(0, 0, -2.0 * A_ / hs ** 2 - 2.0 * C_ / hz ** 2)
    add(1, 0, A_ / hs ** 2); add(-1, 0, A_ / hs ** 2)
    add(0, 1, C_ / hz ** 2 + D_ / (2.0 * hz)); add(0, -1, C_ / hz ** 2 - D_ / (2.0 * hz))
    cross = B_ / (4.0 * hs * hz)
    add(1, 1, cross); add(-1, -1, cross); add(1, -1, -cross); add(-1, 1, -cross)
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ni * nj, ni * nj))
    u = np.zeros((ns, nz))
    u[:ni, 1:nz - 1] = spla.spsolve(M.tocsc(), (-a[I] ** 2).ravel()).reshape(ni, nj)
    with np.errstate(divide="ignore", invalid="ignore"):
        dudz_bed = (-4.0 * u[:, -2] + u[:, -3]) / (2.0 * hz)          # du/dzeta at zeta = 1 (u = 0 there)
        dudz_roof = (4.0 * u[:, 1] - u[:, 2]) / (2.0 * hz)            # du/dzeta at zeta = 0
        tau_bed = 2.0 * np.abs(dudz_bed) * np.sqrt(1.0 + (a_s / b) ** 2) / a
        tau_roof = 2.0 * np.abs(dudz_roof) / a
    s = np.linspace(0.0, 1.0, ns)
    q = 2.0 * b * trapz(a * trapz(u, zeta, axis=1), s)
    area = 2.0 * b * trapz(a, s)
    return dict(u=u, tau_bed=tau_bed, tau_roof=tau_roof, q=q, area=area, s=s, a_s=a_s)


def momentum_coefficient(sec, nz=61):
    """Momentum coefficient beta = area * integral(u^2 dA) / Q^2 of the unidirectional flow in a section returned by
    threshold_section (1.2 between parallel plates; 1.44 for the shallow cosine section)."""
    a, b = sec["a"], sec["half_width"]
    sol = stokes_roofed(a, b, nz)
    s, zeta = np.linspace(0.0, 1.0, a.size), np.linspace(0.0, 1.0, nz)
    m2 = 2.0 * b * trapz(a * trapz(sol["u"] ** 2, zeta, axis=1), s)
    return float(sol["area"] * m2 / sol["q"] ** 2)


def stokes_section(y, a, nz=41):
    """Unidirectional Stokes flow under a flat roof in a section of arbitrary shape (no symmetry assumed).

    y : uniform grid across the section; a : gap at each y (>= 0); the gap may vanish at the ends (banks meeting
    the roof) or be positive there (side walls).  Solves u_yy + u_zz = -1 with u = 0 on the roof, the bed and the
    ends, in the mapped coordinates z = -a(y) zeta, with the equation multiplied by a^2.  Units: those of y and a
    for lengths; u in (length)^2.  In SI the velocity is g S u / nu, the discharge g S q / nu and the bed shear
    stress rho g S t, with t = |du/dn| returned as `dudn`.
    Returns dict: u (ny, nz), dudn (ny) on the bed, q, area.
    """
    y = np.asarray(y, float)
    a = np.asarray(a, float)
    ns = a.size
    hy, hz = y[1] - y[0], 1.0 / (nz - 1)
    a_s = np.gradient(a, hy, edge_order=2)
    a_ss = np.empty(ns)
    a_ss[1:-1] = (a[2:] - 2.0 * a[1:-1] + a[:-2]) / hy ** 2
    a_ss[0], a_ss[-1] = a_ss[1], a_ss[-2]
    zeta = np.linspace(0.0, 1.0, nz)
    ni, nj = ns - 2, nz - 2                        # unknowns: i = 1..ns-2, j = 1..nz-2
    idx = -np.ones((ns, nz), np.int64)
    idx[1:ns - 1, 1:nz - 1] = np.arange(ni * nj).reshape(ni, nj)
    I, J = np.meshgrid(np.arange(1, ns - 1), np.arange(1, nz - 1), indexing="ij")
    A_ = a[I] ** 2
    B_ = -2.0 * zeta[J] * a[I] * a_s[I]
    C_ = (zeta[J] * a_s[I]) ** 2 + 1.0
    D_ = zeta[J] * (2.0 * a_s[I] ** 2 - a[I] * a_ss[I])
    rows, cols, vals = [], [], []
    me = idx[I, J].ravel()

    def add(di, dj, coef):
        k = idx[I + di, J + dj].ravel()
        ok = k >= 0
        rows.append(me[ok]); cols.append(k[ok]); vals.append(coef.ravel()[ok])

    add(0, 0, -2.0 * A_ / hy ** 2 - 2.0 * C_ / hz ** 2)
    add(1, 0, A_ / hy ** 2); add(-1, 0, A_ / hy ** 2)
    add(0, 1, C_ / hz ** 2 + D_ / (2.0 * hz)); add(0, -1, C_ / hz ** 2 - D_ / (2.0 * hz))
    cross = B_ / (4.0 * hy * hz)
    add(1, 1, cross); add(-1, -1, cross); add(1, -1, -cross); add(-1, 1, -cross)
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ni * nj, ni * nj))
    u = np.zeros((ns, nz))
    u[1:ns - 1, 1:nz - 1] = spla.spsolve(M.tocsc(), (-a[I] ** 2).ravel()).reshape(ni, nj)
    with np.errstate(divide="ignore", invalid="ignore"):
        dudzeta = (-4.0 * u[:, -2] + u[:, -3]) / (2.0 * hz)           # at the bed, zeta = 1
        dudn = np.abs(dudzeta) * np.sqrt(1.0 + a_s ** 2) / a
    dudn[~np.isfinite(dudn)] = 0.0
    q = trapz(a * trapz(u, zeta, axis=1), y)
    return dict(u=u, dudn=dudn, q=q, area=trapz(a, y))


def edge_slope(mu, sigma=0.0):
    """Slope |a'| of the bank where it meets the roof (no shear stress there): cos(phi) - sigma = sin(phi) / mu."""
    if sigma <= 0.0:
        return mu
    from scipy.optimize import brentq
    return brentq(lambda t: 1.0 - sigma * np.sqrt(1.0 + t * t) - t / mu, 0.0, mu)


def threshold_section(mu, n_nodes=24, ns=241, nz=61, sigma=0.0, init=(1.0, 1.0), max_nfev=None):
    """Roofed threshold cross-section for friction coefficient mu (full Stokes flow, exact Coulomb condition).

    sigma (default 0): uniform seepage out of the bed, as the ratio of the hydraulic gradient normal to the bed to
    the gradient that fluidises the sand.  The seepage force acts along the normal to the bed and reduces the
    normal force on a grain, so that the threshold condition becomes
        (tau/tau_c)^2 + sin^2(phi)/mu^2 = (cos(phi) - sigma)^2 ;
    on a flat bed tau = tau_c (1 - sigma) (Cheng & Chiew 1999).  Lengths stay in units of a_c = 2 tau_c/(rho g S),
    with tau_c the threshold stress of a flat bed without seepage.

    Unknowns: the aspect ratio r = half-width / a_max and the relative depth f = a/a_max at n_nodes - 1
    interior nodes (cubic spline, f(0) = 1, f'(0) = 0, f(1) = 0).  Equations: the threshold condition at
    the interior nodes, written for the bed shear normalised by its value at the centre (which is at
    threshold by construction), and the angle of repose where the bank meets the roof.  The overall size
    follows from tau(centre) = tau_c.  Solved with Levenberg-Marquardt from the lubrication shape.
    Returns a dict with the shape and its integral properties in units of a_c = 2 tau_c/(rho g S).
    """
    from scipy.interpolate import CubicSpline
    from scipy.optimize import least_squares
    M = n_nodes
    sc = np.linspace(0.0, 1.0, M + 1)
    s = np.linspace(0.0, 1.0, ns)

    def spline(p):
        return CubicSpline(sc, np.concatenate([[1.0], p[1:], [0.0]]), bc_type=((1, 0.0), (2, 0.0)))

    def resid(p):
        r, cs = p[0], spline(p)
        f = cs(s)
        if r <= 0.0 or f[1:-1].min() <= 0.0:
            return np.full(M, 1.0e3)
        f[-1] = 0.0
        sol = stokes_roofed(f, r, nz)
        tn = np.interp(sc[1:M], s, sol["tau_bed"] / sol["tau_bed"][0])
        t2 = (cs(sc[1:M], 1) / r) ** 2
        if sigma:
            cosp = 1.0 / np.sqrt(1.0 + t2)
            need = np.sqrt(np.clip((cosp - sigma) ** 2 - t2 * cosp ** 2 / mu ** 2, 0.0, None)) / (1.0 - sigma)
        else:
            need = np.sqrt(np.clip(1.0 - t2 / mu ** 2, 0.0, None) / (1.0 + t2))
        return np.concatenate([tn - need, [abs(cs(1.0, 1)) / r - mu_edge]])

    mu_edge = edge_slope(mu, sigma)
    # starting guess: the shallow-limit profile; init = (factor on the half-width, exponent on the profile) perturbs it
    p0 = np.concatenate([[init[0] * np.pi / (2.0 * mu_edge)], np.cos(0.5 * np.pi * sc[1:-1]) ** init[1]])
    out = least_squares(resid, p0, method="lm", xtol=1.0e-11, ftol=1.0e-11, max_nfev=max_nfev)
    r, cs = out.x[0], spline(out.x)
    f = cs(s)
    f[-1] = 0.0
    sol = stokes_roofed(f, r, nz)
    lam = (1.0 - sigma) / sol["tau_bed"][0]             # scale so that the centre is exactly at threshold
    a_max, b = lam, r * lam
    area, chi = sol["area"] * lam ** 2, sol["q"] * lam ** 4
    a_avg = area / (2.0 * b)
    return dict(mu=mu, sigma=sigma, y=s * b, a=f * lam, a_max=a_max, half_width=b, area=area, chi=chi,
                W_over_amax=2.0 * b / a_max, aavg_over_amax=a_avg / a_max, W_over_aavg=2.0 * b / a_avg,
                conveyance_vs_rectangle=chi / (2.0 * b * a_avg ** 3 / 12.0),
                tau_bed=sol["tau_bed"] * lam, tau_roof=sol["tau_roof"] * lam,
                max_residual=float(np.abs(out.fun).max()), converged=bool(out.success))


def rectangle_flow_rate(w, h, nterms=200):
    """Exact flow rate of u_yy + u_zz = -1 in a w x h rectangle, u = 0 on all four walls."""
    n = np.arange(1, 2 * nterms, 2)
    lo, hi = (h, w) if h <= w else (w, h)
    return lo ** 3 * hi / 12.0 * (1.0 - np.sum(192.0 * lo / (np.pi ** 5 * n ** 5 * hi) * np.tanh(n * np.pi * hi / (2.0 * lo))))


def dimensional(sec, tau_c, S, rho=1000.0, g=9.81, nu=1.0e-6):
    """Convert a dimensionless section (from threshold_section or lubrication) to SI for given tau_c and slope S."""
    a_c = 2.0 * tau_c / (rho * g * S)
    return dict(a_c=a_c, a_max=sec["a_max"] * a_c, width=2.0 * sec["half_width"] * a_c, area=sec["area"] * a_c ** 2,
                Q=sec["chi"] * g * S * a_c ** 4 / nu)


def pipe_constant(sec, tau_c, rho=1000.0, g=9.81, nu=1.0e-6):
    """Lambda in the pipe law  Q S^3 = Lambda  that follows from the threshold section."""
    return 16.0 * sec["chi"] * tau_c ** 4 / (nu * rho ** 4 * g ** 3)
