"""Erosion pipe at the threshold of motion coupled to the aquifer.

Pipe law (from the threshold cross-section, section.py):   Q S^3 = Lambda,
    Lambda = 16 chi tau_c^4 / (nu rho^4 g^3),     S = -dh/ds along the pipe (s from the exit),
so the pipe head rises from the exit as   h(s) = int_0^s (Lambda / Q)^(1/3) ds',
while the aquifer feeds the pipe:          Q(s) = sum of the inflows upstream of s,   q = k (G h + g H).

For a given pipe length these equations have two solutions when H exceeds a minimum H_min and none
below it (a fold).  Below H_min the pipe cannot keep its whole bed at the threshold: sand supplied by
the tip cannot be carried to the exit.  H_min(l) is therefore the head needed for a pipe of length l to
progress by bed transport; its maximum over l is the critical head of the bed criterion.

With H as the unit of head and k H as the unit of discharge per unit length, the problem contains one number,
    N = H^4 k / Lambda      [m^2],
so that  H_min = (N_min Lambda / k)^(1/4)  with N_min a property of the geometry alone.

A pipe that follows a winding path of length T times the straight distance loses T times more head per
unit distance: replace Lambda by T^3 Lambda (so N_min is unchanged and H_min grows as T^(3/4)).
"""
import numpy as np


def _ds(R):
    ds = getattr(R, "ds", None)
    return np.full(R.n, R.box.d) if ds is None else ds


def _state(h, R):
    q = R.G @ h + R.g                              # inflow per station (units k H)
    Q = np.cumsum(q[::-1])[::-1]                   # flow through the face on the exit side of each station
    return q, Q


def _residual(h, lnN, R, C, ds):
    """F = h - cumsum(ds S), its Jacobian with respect to h, and dF/d(ln N).  None if some Q <= 0."""
    q, Q = _state(h, R)
    if Q.min() <= 0.0 or not np.isfinite(Q).all():
        return None
    S = np.exp(-lnN / 3.0) * Q ** (-1.0 / 3.0)
    F = h - np.cumsum(ds * S)
    dS = (-1.0 / 3.0) * S / Q
    J = np.eye(R.n) - np.cumsum((ds * dS)[:, None] * C, axis=0)
    FN = np.cumsum(ds * S) / 3.0
    return F, J, FN, q, Q, S


def solve_pipe(R, N, h0=None, tol=1.0e-12, maxit=60):
    """Newton solution for the pipe heads (in units of H) at given N = H^4 k / Lambda.  None if it fails."""
    h = np.zeros(R.n) if h0 is None else h0.copy()
    C = np.cumsum(R.G[::-1], axis=0)[::-1]         # dQ/dh
    ds, lnN = _ds(R), np.log(N)
    for _ in range(maxit):
        out = _residual(h, lnN, R, C, ds)
        if out is None:
            return None
        F, J, _, q, Q, S = out
        if np.abs(F).max() < tol:
            return dict(h=h, q=q, Q=Q, S=S)
        try:
            step = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return None
        t = 1.0
        while t > 1.0e-4:                           # keep Q positive
            if _state(h + t * step, R)[1].min() > 0.0:
                break
            t *= 0.5
        else:
            return None
        h = h + t * step
    return None


def fold(R, rtol=1.0e-6):
    """Smallest N for which the threshold pipe exists, and the solution there.

    Continuation from large N (nearly equipotential pipe) downwards, then bisection.
    """
    N = 1.0
    sol = solve_pipe(R, N)
    while sol is None:                              # find a starting point on the stable branch
        N *= 10.0
        sol = solve_pipe(R, N)
        if N > 1.0e30:
            raise RuntimeError("no solution found")
    lo, hi, sol_hi = None, N, sol
    while lo is None:
        trial = hi * 0.7
        s = solve_pipe(R, trial, sol_hi["h"])
        if s is None:
            lo = trial
        else:
            hi, sol_hi = trial, s
            if hi < 1.0e-30:
                raise RuntimeError("no fold found")
    while hi / lo - 1.0 > rtol:
        mid = np.sqrt(lo * hi)
        s = solve_pipe(R, mid, sol_hi["h"])
        if s is None:
            lo = mid
        else:
            hi, sol_hi = mid, s
    return hi, sol_hi


def trace_branches(R, N_start_factor=300.0, ds0=0.05, max_steps=4000, tol=1.0e-11):
    """Both solution branches by pseudo-arclength continuation in (h, ln N).

    Starts on the large-pipe branch at N = N_start_factor * N_min, passes the fold and follows the
    small-pipe branch back up to the same N.  Returns N_min and a list of dicts (N, h, q, Q, S, branch),
    branch = 0 before the fold (large, gently sloping pipe) and 1 after it (small, steep pipe).
    """
    Nmin, _ = fold(R)
    C = np.cumsum(R.G[::-1], axis=0)[::-1]
    ds, n = _ds(R), R.n
    lam_top = np.log(N_start_factor * Nmin)
    s0 = solve_pipe(R, np.exp(lam_top))
    if s0 is None:
        raise RuntimeError("no starting solution")
    u = np.concatenate([s0["h"], [lam_top]])
    t = np.zeros(n + 1); t[-1] = -1.0                # first move towards smaller N
    out = [dict(N=np.exp(u[-1]), branch=0, **s0)]
    step, passed, lam_min = ds0, False, lam_top
    for _ in range(max_steps):
        F, J, FN = _residual(u[:n], u[-1], R, C, ds)[:3]
        M = np.zeros((n + 1, n + 1)); M[:n, :n] = J; M[:n, -1] = FN; M[-1] = t
        rhs = np.zeros(n + 1); rhs[-1] = 1.0
        tn = np.linalg.solve(M, rhs); tn /= np.linalg.norm(tn)
        while True:                                  # predictor-corrector with step halving
            v = u + step * tn
            ok = False
            for it in range(25):
                r = _residual(v[:n], v[-1], R, C, ds)
                if r is None:
                    break
                F, J, FN = r[:3]
                g_ = np.concatenate([F, [tn @ (v - u) - step]])
                if np.abs(g_).max() < tol:
                    ok = True
                    break
                M[:n, :n] = J; M[:n, -1] = FN; M[-1] = tn
                v = v - np.linalg.solve(M, g_)
            if ok:
                break
            step *= 0.5
            if step < 1.0e-9:
                return Nmin, out
        if v[-1] > u[-1] and not passed:
            passed = True
        u, t = v, tn
        lam_min = min(lam_min, u[-1])
        q, Q = _state(u[:n], R)
        out.append(dict(N=np.exp(u[-1]), h=u[:n].copy(), q=q, Q=Q, S=np.exp(-u[-1] / 3.0) * Q ** (-1.0 / 3.0), branch=int(passed)))
        if it < 6:
            step = min(step * 1.5, 0.25)
        if passed and u[-1] >= lam_top:
            break
    return Nmin, out


def head_from_N(N, Lam, k):
    return (N * Lam / k) ** 0.25


def c3_from_N(N, chi, L):
    """Coefficient in  Hc/L = c3 (tau_c/rho g) (g/(nu k))^(1/4) L^(-1/2)  implied by N (geometry) and chi (section)."""
    return (16.0 * chi * N) ** 0.25 / np.sqrt(L)


def tip_gradient(R, sol, dist=0.02):
    """Secant gradient (per unit H) between the pipe tip and the roof `dist` ahead of it.

    R must have been built with with_fields=True.  The tip is the upstream face of the last station.
    """
    x, hroof = R.roof_head(sol["h"], 1.0)
    return (np.interp(R.x_tip + dist, x, hroof) - sol["h"][-1]) / dist


def stable_branch_table(R, n=40, span=1.0e4, dist=0.02):
    """N, tip gradient per unit H, pipe-tip head per unit H, exit discharge per (k H) from the fold upwards."""
    Nmin, s0 = fold(R)
    Ns = Nmin * np.logspace(0.0, np.log10(span), n)
    rows, h0 = [], None
    for N in Ns[::-1]:
        s = solve_pipe(R, N, h0)
        if s is None:
            continue
        h0 = s["h"]
        rows.append((N, tip_gradient(R, s, dist), s["h"][-1], s["Q"][0]))
    rows.append((Nmin, tip_gradient(R, s0, dist), s0["h"][-1], s0["Q"][0]))
    return Nmin, np.array(rows[::-1])


def sand_number(C_tip, tau_c, k, rho=1000.0, g=9.81, nu=1.0e-6):
    """Pi_s = C (rho g / tau_c) (nu k / g)^(1/4): tip-erosion strength relative to bed-erosion strength.

    C_tip [m^(1/2)] is the constant of the tip criterion  i_c(x) = C x^(-1/2)  (Robbins et al.).
    """
    return C_tip * (rho * g / tau_c) * (nu * k / g) ** 0.25


def regime_curve(table, chi, dist=0.02):
    """From a stable-branch table: Pi_s at which the tip criterion is just met, as a function of H / H_min.

    Returns (H/H_min, Pi_s).  The first entry is the fold: sands with Pi_s below that value are controlled
    by the bed criterion; above it the tip criterion requires H/H_min > 1.
    """
    N, gam = table[:, 0], table[:, 1]
    return (N / N[0]) ** 0.25, (16.0 * chi * N) ** 0.25 * gam * np.sqrt(dist)


# ---------------------------------------------------------------------------------------------------
# dimensional solver for a general slope-discharge law (pipelaw.PipeLaw)

def solve_pipe_law(R, law, H, k, h0=None, tol=1.0e-13, maxit=80):
    """Pipe heads [m] for upstream head H [m], conductivity k [m/s] and a PipeLaw.  None if no solution is found."""
    ds = _ds(R)
    h = np.zeros(R.n) if h0 is None else h0.copy()
    C = k * np.cumsum(R.G[::-1], axis=0)[::-1]

    def flows(hh):
        q = k * (R.G @ hh + R.g * H)
        return q, np.cumsum(q[::-1])[::-1]

    for _ in range(maxit):
        q, Q = flows(h)
        if Q.min() <= 0.0 or not np.isfinite(Q).all():
            return None
        S, dS = law.slope(Q)
        F = h - np.cumsum(ds * S)
        if np.abs(F).max() < tol * max(H, 1.0e-6):
            return dict(h=h, q=q, Q=Q, S=S)
        J = np.eye(R.n) - np.cumsum((ds * dS)[:, None] * C, axis=0)
        try:
            step = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return None
        t = 1.0
        while t > 1.0e-4:
            if flows(h + t * step)[1].min() > 0.0:
                break
            t *= 0.5
        else:
            return None
        h = h + t * step
    return None


def fold_law(R, law, k, rtol=1.0e-5):
    """Smallest upstream head [m] at which the pipe can keep its bed at the threshold, and the solution there."""
    H = 0.01
    sol = solve_pipe_law(R, law, H, k)
    while sol is None:
        H *= 2.0
        sol = solve_pipe_law(R, law, H, k)
        if H > 1.0e4:
            raise RuntimeError("no solution found")
    lo, hi, sol_hi = None, H, sol
    while lo is None:
        trial = hi * 0.9
        s = solve_pipe_law(R, law, trial, k, sol_hi["h"] * 0.9)
        if s is None:
            s = solve_pipe_law(R, law, trial, k, sol_hi["h"])
        if s is None:
            lo = trial
        else:
            hi, sol_hi = trial, s
            if hi < 1.0e-9:
                raise RuntimeError("no fold found")
    while hi / lo - 1.0 > rtol:
        mid = np.sqrt(lo * hi)
        s = solve_pipe_law(R, law, mid, k, sol_hi["h"])
        if s is None:
            lo = mid
        else:
            hi, sol_hi = mid, s
    return hi, sol_hi


def head_for_tip_gradient(R, law, k, i_c, dist=0.02, H_fold=None, sol_fold=None):
    """Head [m] at which the tip gradient over `dist` reaches i_c on the large-pipe branch (>= the fold head)."""
    if H_fold is None:
        H_fold, sol_fold = fold_law(R, law, k)

    def itip(H, h0):
        s = solve_pipe_law(R, law, H, k, h0)
        if s is None:
            s = solve_pipe_law(R, law, H, k)
        x, hr = R.roof_head(s["h"], H)
        return (np.interp(R.x_tip + dist, x, hr) - s["h"][-1]) / dist, s

    i0, _ = itip(H_fold * (1.0 + 1.0e-4), sol_fold["h"])
    if i0 >= i_c:
        return H_fold, sol_fold, False
    lo, hi, s = H_fold, H_fold * 1.3, sol_fold
    i_hi, s = itip(hi, s["h"])
    while i_hi < i_c:
        lo, hi = hi, hi * 1.3
        i_hi, s = itip(hi, s["h"])
        if hi > 1.0e3 * H_fold:
            raise RuntimeError("tip gradient never reached")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        i_m, s = itip(mid, s["h"])
        if i_m < i_c:
            lo = mid
        else:
            hi = mid
    return hi, s, True


# ---------------------------------------------------------------------------------------------------
# pipe whose width follows the seepage it collects

def solve_seepage_width(R, sec, tau_c, H, k, i_star, W_min=None, tortuosity=1.0, h0=None, rtol=1.0e-4, maxit=80):
    """Threshold pipe of uniform width  W = Q_pipe / (k l i_star):  the mean hydraulic gradient of the seepage entering
    through the bed of the pipe equals i_star.  W is not allowed below W_min (the width made by the tip), and where the
    self-formed section would be wider than W the pipe is self-formed.  Returns (solution, W) or None."""
    from .pipelaw import PipeLaw
    W = W_min
    for _ in range(maxit):
        law = PipeLaw(sec, tau_c, W0=W, tortuosity=tortuosity)
        sol = solve_pipe_law(R, law, H, k, h0)
        if sol is None and h0 is not None:
            sol = solve_pipe_law(R, law, H, k)
        if sol is None:
            return None
        W_new = max(W_min or 0.0, sol["Q"][0] / (k * R.length * i_star))
        if W is not None and abs(W_new - W) <= rtol * W_new:
            return sol, W_new, law
        W = W_new if W is None else 0.5 * (W + W_new)
        h0 = sol["h"]
    return None


def fold_seepage_width(R, sec, tau_c, k, i_star, W_min=None, tortuosity=1.0, rtol=2.0e-4):
    """Smallest head at which the seepage-width pipe can keep its bed at the threshold; returns (H, solution, W)."""
    hi, out = 0.005, None
    while out is None:
        hi *= 1.5
        out = solve_seepage_width(R, sec, tau_c, hi, k, i_star, W_min, tortuosity)
        if hi > 1.0e3:
            raise RuntimeError("no solution found")
    lo = hi / 1.5
    while hi / lo - 1.0 > rtol:
        mid = np.sqrt(lo * hi)
        o = solve_seepage_width(R, sec, tau_c, mid, k, i_star, W_min, tortuosity, h0=out[0]["h"])
        if o is None:
            lo = mid
        else:
            hi, out = mid, o
    return hi, out[0], out[1]
