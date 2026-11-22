"""Threshold pipe with seepage through its bed: section family, local law, and the problem coupled to the sand.

Threshold condition with seepage (section.threshold_section, sigma):
    (tau/tau_c0)^2 + sin^2(phi)/mu^2 = (cos(phi) - sigma)^2,
sigma = hydraulic gradient normal to the bed / gradient that fluidises the sand, tau_c0 = threshold stress of a flat
bed without seepage.  For uniform sigma the section is, in units of a_c = A0/S with A0 = 2 tau_c0/(rho g),
    largest depth  alpha(sigma),   width  w(sigma),   discharge coefficient  chi(sigma)  (Q = chi g S a_c^4 / nu).
As sigma -> 1 the section becomes shallow and the shallow-limit solution becomes exact for any mu:
    alpha -> 1 - sigma,   w -> pi/mu,   chi -> (1 - sigma)^3 / (9 mu).

Closure for sigma.  Seen from the sand the pipe is a strip of width W at constant head; if it draws q per unit length
the gradient under its centre is 2 q / (pi k W) and the mean over the strip is q / (k W).  We take the uniform sigma
of the section as
    sigma = c_sigma * 2 q / (pi k i_f W),       W = w(sigma) A0 / S,
with c_sigma = 1 (gradient at the centre) or pi/2 (mean over the strip).  The solution for the non-uniform seepage
of a strip is not available; the two values bracket it.

Local law.  Given the discharge Q and the inflow q at a station, sigma solves
    sigma w(sigma) (chi(0)/chi(sigma))^(1/3) = Sigma,    Sigma = c_sigma (2 q / (pi k i_f A0)) S0,   S0 = (chi(0) g A0^4/(nu Q))^(1/3),
(the left-hand side increases from 0 without bound), and then S = S0 (chi(sigma)/chi(0))^(1/3).
"""
import os
import numpy as np
from scipy.optimize import root

from .section import threshold_section

HERE = os.path.dirname(os.path.abspath(__file__))
SIGMAS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def section_family(mu, cache=True):
    """Table of the section against sigma for friction coefficient mu: columns sigma, alpha, w, chi.
    Cached in code/data/section_seepage_mu<mu>.csv (ten solutions of the free-boundary problem, about a minute)."""
    path = os.path.join(HERE, "..", "data", f"section_seepage_mu{mu:.4f}.csv")
    if cache and os.path.exists(path):
        return np.loadtxt(path, delimiter=",", skiprows=1)
    rows = []
    for sg in SIGMAS:
        o = threshold_section(mu, sigma=float(sg))
        rows.append([sg, o["a_max"], 2.0 * o["half_width"], o["chi"]])
    tab = np.array(rows)
    if cache:
        np.savetxt(path, tab, delimiter=",", header="sigma,a_max_over_ac,W_over_ac,chi", comments="", fmt="%.8g")
    return tab


class SeepageLaw:
    """Local law of a threshold pipe with seepage through its bed (see the module docstring)."""

    def __init__(self, mu, tau_c0, k, i_f=0.99, c_sigma=1.0, rho=1000.0, g=9.81, nu=1.0e-6, strength=1.0,
                 inertia=0.0, shape=2.0 / np.pi):
        self.mu, self.tau_c0, self.k, self.i_f, self.c_sigma = mu, tau_c0, k, i_f, c_sigma
        self.rho, self.g, self.nu, self.strength = rho, g, nu, strength
        self.inertia, self.shape = inertia, shape              # momentum coefficient (0: creeping flow); area / (W a_max)
        self.A0 = 2.0 * tau_c0 / (rho * g)
        tab = section_family(mu)
        s = np.append(tab[:, 0], 1.0)
        # smooth functions with their exact limits at sigma = 1
        self._s = s
        self._ah = np.append(tab[:, 1] / (1.0 - tab[:, 0]), 1.0)
        self._w = np.append(tab[:, 2], np.pi / mu)
        self._ch = np.append(tab[:, 3] / (1.0 - tab[:, 0]) ** 3, 1.0 / (9.0 * mu))
        self.chi0 = tab[0, 3]
        self.Lam0 = self.chi0 * g * self.A0 ** 4 / nu

    def alpha(self, s):
        return (1.0 - s) * np.interp(s, self._s, self._ah)

    def w(self, s):
        return np.interp(s, self._s, self._w)

    def chi(self, s):
        return (1.0 - s) ** 3 * np.interp(s, self._s, self._ch)

    def _lhs(self, s):
        return s * self.w(s) * (self.chi0 / np.maximum(self.chi(s), 1e-300)) ** (1.0 / 3.0)

    def sigma(self, Q, q):
        """sigma at stations with discharge Q [m^3/s] and inflow per unit length q [m^2/s]."""
        Q, q = np.asarray(Q, float), np.asarray(q, float)
        S0 = (self.Lam0 / Q) ** (1.0 / 3.0)
        target = self.strength * self.c_sigma * 2.0 * np.maximum(q, 0.0) / (np.pi * self.k * self.i_f * self.A0) * S0
        lo, hi = np.zeros_like(target), np.full_like(target, 1.0 - 1.0e-9)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            up = self._lhs(mid) < target
            lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
        return 0.5 * (lo + hi)

    def local(self, Q, q):
        """Head gradient, sigma, width and largest depth at stations with discharge Q and inflow per unit length q.
        The section follows from the friction slope; with inertia > 0 the head gradient also contains the momentum
        that the water gains along the pipe (see inertial_slope)."""
        s = self.sigma(Q, q)
        S = (self.Lam0 / np.asarray(Q, float)) ** (1.0 / 3.0) * (self.chi(s) / self.chi0) ** (1.0 / 3.0)
        W, a = self.w(s) * self.A0 / S, self.alpha(s) * self.A0 / S
        if self.inertia:
            S = S + self.inertial_slope(Q, q, W, a)
        return S, s, W, a

    def inertial_slope(self, Q, q, W, a):
        """Head gradient that balances the growth of the momentum flux beta Q^2 / area along the pipe (momentum
        integral with the velocity profile of the uniform flow).  The area of a threshold pipe grows as Q^(2/3),
        so that d(beta Q^2/area)/ds = (4/3) beta Q q / area and the gradient is (4/3) beta Q q / (g area^2)."""
        area = self.shape * np.asarray(W, float) * np.asarray(a, float)
        return 4.0 / 3.0 * self.inertia * np.asarray(Q, float) * np.maximum(np.asarray(q, float), 0.0) / (self.g * area ** 2)

    def section_at(self, S, q):
        """Section for a given slope S and inflow q (no discharge needed): sigma solves
        sigma w(sigma) = c_sigma 2 q S / (pi k i_f A0).  Returns sigma, width, largest depth, discharge."""
        S, q = np.asarray(S, float), np.asarray(q, float)
        target = self.strength * self.c_sigma * 2.0 * np.maximum(q, 0.0) * S / (np.pi * self.k * self.i_f * self.A0)
        lo, hi = np.zeros_like(target), np.full_like(target, 1.0 - 1.0e-9)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            up = mid * self.w(mid) < target
            lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
        s = 0.5 * (lo + hi)
        return s, self.w(s) * self.A0 / S, self.alpha(s) * self.A0 / S, self.chi(s) * self.g * self.A0 ** 4 / (self.nu * S ** 3)


def _ds(R):
    ds = getattr(R, "ds", None)
    return np.full(R.n, R.box.d) if ds is None else ds


def solve_pipe_seepage(R, law, H, h0=None, tol=1.0e-11):
    """Heads in a threshold pipe with seepage through its bed, coupled to the sand through the response matrices R,
    for a head H [m] across the sand.  Returns dict(h, q [per unit length], Q, S, sigma, W, a_max) or None."""
    ds = _ds(R)

    def state(h):
        qs = law.k * (R.G @ h + R.g * H)                    # inflow per station (a roof cell of length d)
        Q = np.cumsum(qs[::-1])[::-1]
        return qs / R.box.d, Q

    def F(h):
        q, Q = state(h)
        if Q.min() <= 0.0 or not np.isfinite(Q).all():
            return np.full(R.n, 1.0e3)
        S = law.local(Q, q)[0]
        return h - np.cumsum(ds * S)

    h = np.zeros(R.n) if h0 is None else np.asarray(h0, float).copy()
    out = root(F, h, method="hybr", tol=1.0e-13)
    h = out.x
    q, Q = state(h)
    if Q.min() <= 0.0 or np.abs(F(h)).max() > tol * max(H, 1e-12) * 1e3:
        return None
    S, s, W, a = law.local(Q, q)
    return dict(h=h, q=q, Q=Q, S=S, sigma=s, W=W, a_max=a, S_inertia=law.inertial_slope(Q, q, W, a) if law.inertia else 0.0 * S)


def fold_seepage(R, law, H_start, h_start=None, shrink=0.97, rtol=2.0e-4):
    """Smallest head at which the pipe with seepage keeps its bed at the threshold, found by lowering the head from
    H_start along the branch of the larger pipe.  Returns (H_min, solution at H_min) or (None, None)."""
    sol = solve_pipe_seepage(R, law, H_start, h_start)
    if sol is None:
        return None, None
    H = H_start
    while True:                                             # step down until the branch is lost
        Hn = H * shrink
        sn = solve_pipe_seepage(R, law, Hn, sol["h"] * Hn / H)
        if sn is None or sn["h"][-1] > 0.9999 * Hn:
            break
        H, sol = Hn, sn
        if H < 1.0e-4 * H_start:
            return None, None
    lo, hi = H * shrink, H                                   # no solution at lo, solution at hi
    while hi / lo - 1.0 > rtol:
        mid = np.sqrt(lo * hi)
        sm = solve_pipe_seepage(R, law, mid, sol["h"] * mid / hi)
        if sm is None or sm["h"][-1] > 0.9999 * mid:
            lo = mid
        else:
            hi, sol = mid, sm
    return hi, sol
