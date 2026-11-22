"""Weak inertia in a threshold pipe (manuscript section 4).

The flow in the pipe gains discharge along its length (inflow q per unit length through the bed) and its section
grows.  With eps = Re * depth / length of variation small, the first correction to the creeping flow is computed here

1. exactly for a wide pipe (two-dimensional gap of depth h(x), flux Q(x) per unit width, injection velocity V = dQ/dx
   through the bed, impermeable roof).  Leading order u0 = 6 Q f(eta)/h, f = eta - eta^2, eta = z/h measured from the
   bed.  At first order
        nu d2u1/dz2 = (1/rho) dp1/dx + u0 du0/dx + w0 du0/dz,     u1 = 0 on both walls,   int u1 dz = 0,
   which gives, x being the direction of the flow,
        head gradient   S = 12 nu Q/(g h^3) + (81/35) Q V/(g h^2) - (54/35) Q^2 h'/(g h^3)
        bed stress      tau_b/rho = 6 nu Q/h^2 - (19/35) Q V/h - (6/35) Q^2 h'/h^2
        roof stress     tau_r/rho = 6 nu Q/h^2 + (16/35) Q V/h - (6/35) Q^2 h'/h^2
   (wide_pipe_coefficients; exact rational arithmetic).  The injection terms are checked against the similarity
   solution of the Navier-Stokes equations for a uniform gap with injection through one wall (similarity_one_wall).
   For a wide pipe whose bed is at the threshold, 6 rho nu Q/h0^2 = tau_c, so that h0'/h0 = V/2Q and
        h = h0 [1 - (11/210) R],     S = S0 [1 + (2/7) R],     R = V h0/nu          (wide_threshold_pipe).

2. for the self-formed section, by the momentum integral with the velocity profile of the uniform flow:
        S = S_f [1 + c q/nu],    c = 4 beta chi / (3 m^2),
   beta being the momentum coefficient and m a_c^2 the area of the section (self_formed_coefficient).  The same
   closure applied to the wide threshold pipe gives (3/20) R where the exact result is (2/7) R: it leaves out the
   change of the stress on the bed.
"""
from fractions import Fraction as Fr
import numpy as np


# ------------------------------------------------------------------ exact polynomial arithmetic on [0, 1]
def _mul(p, q):
    r = [Fr(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] += a * b
    return r


def _add(p, q, s=1):
    n = max(len(p), len(q))
    return [(p[i] if i < len(p) else Fr(0)) + s * (q[i] if i < len(q) else Fr(0)) for i in range(n)]


def _int01(p):
    return sum(a / (i + 1) for i, a in enumerate(p))


def _der(p):
    return [i * a for i, a in enumerate(p)][1:]


def wide_pipe_coefficients():
    """Exact first-order coefficients for the wide pipe (see the module docstring).  Returns a dict of Fractions:
    S_V, S_h (head gradient: + S_V Q V/(g h^2) + S_h Q^2 h'/(g h^3)), bed_V, bed_h, roof_V, roof_h (stresses over rho:
    coefficients of Q V/h and Q^2 h'/h^2), and beta0 = 6/5."""
    eta = [Fr(0), Fr(1)]
    f = [Fr(0), Fr(1), Fr(-1)]                                   # eta - eta^2
    F = [Fr(0), Fr(0), Fr(1, 2), Fr(-1, 3)]                      # integral of f
    one_m6F = _add([Fr(1)], [6 * a for a in F], -1)
    c1 = _add([6 * a for a in _mul(f, f)], _mul(one_m6F, _der(f)))   # convective term / (6 Q V/h^2)
    c2 = _mul(f, f)                                              # convective term / (-36 Q^2 h'/h^3)
    one_m_eta = [Fr(1), Fr(-1)]
    I1, I2 = _int01(_mul(c1, f)), _int01(_mul(c2, f))
    # p1x/rho = -6 int C f  with C = (6 Q V/h^2) c1 - (36 Q^2 h'/h^3) c2 ;  S = -p_x/(rho g)
    S_V, S_h = 36 * I1, -216 * I2
    bed_V = 18 * I1 - 6 * _int01(_mul(one_m_eta, c1))
    bed_h = -108 * I2 + 36 * _int01(_mul(one_m_eta, c2))
    roof_V = 18 * I1 - 6 * _int01(_mul(eta, c1))
    roof_h = -108 * I2 + 36 * _int01(_mul(eta, c2))
    return dict(S_V=S_V, S_h=S_h, bed_V=bed_V, bed_h=bed_h, roof_V=roof_V, roof_h=roof_h, beta0=36 * _int01(c2))


def wide_threshold_pipe(n=Fr(1, 2)):
    """First-order law of a wide pipe whose bed is at the threshold and whose depth grows as Q^n (n = 1/2 when the
    threshold stress is uniform): h = h0 (1 + depth R), S = S0 (1 + slope R), R = V h0/nu.  Also the slope coefficient
    at fixed depth and the one given by the momentum integral with the parabolic profile."""
    c = wide_pipe_coefficients()
    # tau_b = tau_c:  6 nu Q/h^2 (1 + [bed_V + n bed_h] R/6) = tau_c/rho   ->  h = h0 (1 + [bed_V + n bed_h] R/12)
    depth = (c["bed_V"] + n * c["bed_h"]) / 12
    fixed = (c["S_V"] + n * c["S_h"]) / 12                       # (S - S0)/S0 at the depth h0
    slope = fixed - 3 * depth
    closure = c["beta0"] * (2 - n) / 12                          # d(beta Q^2/h)/dx / (g h) over S0
    return dict(depth=depth, slope=slope, slope_fixed_depth=fixed, slope_closure=closure, ratio=slope / closure)


def similarity_one_wall(R):
    """Navier-Stokes similarity solution for a uniform gap with injection velocity V through the wall eta = 0 and an
    impermeable wall at eta = 1: u = (Q/h) g(eta), w = V wt(eta), Q = Q0 + V x, R = V h/nu,
        g'' + K = R (g^2 + wt g'),   wt' = -g,   g(0) = g(1) = 0,   wt(0) = 1,   wt(1) = 0.
    Returns K/12 (head gradient over its creeping value), g'(0)/6 and -g'(1)/6 (bed and roof stress likewise)."""
    from scipy.integrate import solve_bvp

    def rhs(x, y, p):
        g, gp, w = y
        return np.vstack([gp, -p[0] + R * (g * g + w * gp), -g])

    def bc(ya, yb, p):
        return np.array([ya[0], yb[0], ya[2] - 1.0, yb[2]])

    x = np.linspace(0.0, 1.0, 201)
    y0 = np.vstack([6 * (x - x ** 2), 6 * (1 - 2 * x), 1 - 3 * x ** 2 + 2 * x ** 3])
    sol = solve_bvp(rhs, bc, x, y0, p=[12.0], tol=1e-10, max_nodes=20000)
    if not sol.success:
        return np.nan, np.nan, np.nan
    return sol.p[0] / 12.0, sol.y[1, 0] / 6.0, -sol.y[1, -1] / 6.0


def self_formed_coefficient(sec, beta=None):
    """c in S = S_f (1 + c q/nu) for the self-formed section sec (threshold_section): c = 4 beta chi / (3 m^2)."""
    from .section import momentum_coefficient
    beta = momentum_coefficient(sec) if beta is None else beta
    return 4.0 * beta * sec["chi"] / (3.0 * sec["area"] ** 2)


def shallow_self_formed(inflow="strip", n=400, m=96):
    """First-order inertia in the self-formed threshold pipe in the shallow limit (mu -> 0), where every column of
    the section is a wide gap.  With theta = mu y/a_c, the creeping pipe has a0 = a_c cos(theta), flux per unit width
    Qh0 = (g A/12 nu) a_c^2 cos^3(theta) and a_c ~ Q^(1/3).  The seepage enters with the velocity V(y); what a column
    does not need is carried across the section by the flux q_y = integral of (V - dQh0/dx), which vanishes at the
    edge.  For a flow with flux vector (Qh, q_y) in a gap a(x, y) the convective term of the axial momentum is
        C = (36/a^2) T f^2 + (6 Qh V/a^2) (1 - 6F) f',     T = Qh dQh/dx + q_y dQh/dy - (Qh/a)(Qh da/dx + q_y da/dy),
    so that, column by column, the flux loses (9 a/140 nu)(2T + Qh V) and the bed stress (rho/10 a)(6T + 11 Qh V).
    The head gradient is uniform over the section.  The threshold condition, linearised about the creeping section,
    and the given discharge then determine the change of the section and
        S = S0 [1 + s_hat eps],   a_max = a_c [1 + d_hat eps],   eps = mu q/(6 nu),
    S0 and a_c being those of the creeping pipe with the same discharge.

    inflow: "strip" (seepage into a strip at constant head on an impermeable plane, V ~ 1/sqrt((W/2)^2 - y^2)),
    "uniform", or a number p for V ~ (1 - (2y/W)^2)^(-p) (p = 0 uniform, p = 1/2 strip).  Returns s_hat, d_hat, the relative change of the width per unit eps, the same closure coefficient as in
    self_formed_coefficient (8/25), and check = integral of cos^2 (T + V) (momentum identity, must equal 16/15).
    """
    xg, wg = np.polynomial.legendre.leggauss(n)
    phi = 0.25 * np.pi * (xg + 1.0)                              # theta = (pi/2) sin(phi) removes the edge singularity
    wphi = 0.25 * np.pi * wg
    th = 0.5 * np.pi * np.sin(phi)
    wth = wphi * 0.5 * np.pi * np.cos(phi)

    def fields(t):
        t = np.asarray(t, float)
        c, s = np.cos(t), np.sin(t)
        bulk = 0.25 * (3.0 * s - s ** 3 - t * c ** 3)            # integral of dQh0/dx over the half-section, over q
        if inflow == "strip":
            V = (1.0 / np.pi) / np.sqrt(np.clip(0.25 * np.pi ** 2 - t ** 2, 1e-300, None))
            qy = np.arcsin(np.clip(2.0 * t / np.pi, -1.0, 1.0)) / np.pi - bulk
        elif inflow == "uniform":
            V = np.full_like(t, 1.0 / np.pi)
            qy = t / np.pi - bulk
        else:                                                    # V ~ (1 - (2y/W)^2)^(-p), p = inflow (0: uniform, 1/2: strip)
            from scipy.special import beta as _beta, hyp2f1
            pw = float(inflow)
            u = np.clip(2.0 * t / np.pi, -1.0, 1.0)
            norm = 2.0 / (np.pi * _beta(0.5, 1.0 - pw))
            V = norm * np.clip(1.0 - u ** 2, 1e-300, None) ** (-pw)
            qy = norm * 0.5 * np.pi * u * hyp2f1(0.5, pw, 1.5, u ** 2) - bulk
        T = 0.25 * c ** 2 * (c + 2.0 * t * s) - 2.0 * qy * s / np.clip(c, 1e-300, None)
        return T, V, c ** 2 * (6.0 * T + 11.0 * V) / 10.0        # T, V, stress taken off the bed (all per unit eps)

    T, V, that = fields(th)
    c = np.cos(th)
    check = 2.0 * np.sum(wth * c ** 2 * (T + V))
    I_c = 2.0 * np.sum(wth * c ** 4 * (2.0 * T + V))
    t0 = float(fields(np.array([0.0]))[2][0])
    # a_t(theta) = t0 - sin(theta) * integral_0^theta cos (that - t0)/sin^2
    xm, wm = np.polynomial.legendre.leggauss(m)
    tt = 0.5 * th[:, None] * (xm[None, :] + 1.0)
    g = np.cos(tt) * (fields(tt)[2] - t0) / np.sin(tt) ** 2
    a_t = t0 - np.sin(th) * 0.5 * th * np.sum(wm[None, :] * g, axis=1)
    I_a = 2.0 * np.sum(wth * c ** 2 * a_t)
    s_hat = (3.0 * I_a - 27.0 / 70.0 * I_c) / 4.0
    te = np.array([0.5 * np.pi * (1.0 - 1e-9)])
    tte = 0.5 * te[:, None] * (xm[None, :] + 1.0)
    ge = np.cos(tte) * (fields(tte)[2] - t0) / np.sin(tte) ** 2
    a_edge = -s_hat * 0.5 * np.pi + t0 - 0.5 * te[0] * np.sum(wm * ge[0])
    return dict(s_hat=float(s_hat), d_hat=float(t0 - s_hat), width=float(2.0 * a_edge / np.pi), closure=8.0 / 25.0,
                ratio=float(s_hat / (8.0 / 25.0)), check=float(check), t0=t0, I_c=float(I_c), I_a=float(I_a))


def edge_exponent(mu):
    """Exponent p of the seepage velocity V ~ (distance from the edge)^(-p) where a bank at the angle of repose meets
    the roof: the sand is a wedge of angle pi - arctan(mu) between the impermeable roof and the bed at constant head,
    in which the head grows as r^(pi/(2 (pi - arctan mu))).  p -> 1/2 (strip) as mu -> 0."""
    return 1.0 - np.pi / (2.0 * (np.pi - np.arctan(mu)))
