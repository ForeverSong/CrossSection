"""3-D Darcy flow in a rectangular sand bed with a straight erosion pipe under its roof.

Coordinates
    x  along the seepage path, 0 <= x <= Lx; the face x = Lx is at the upstream head H
    y  across, half domain 0 <= y <= Wy/2 (the pipe runs along the symmetry plane y = 0)
    z  downward from the top of the sand, 0 <= z <= D
Two exit configurations
    exit = "hole"  (Van Beek 2015, Pol 2022): the whole top is an impermeable roof; exit hole on the
                   roof at x = x_exit (head 0); the pipe runs from the hole to x_exit + l
    exit = "toe"   (levee models, Okamura et al.): the sand surface is free (head 0) for x < x_exit,
                   the roof covers x_exit < x < x_river, and the surface is at head H for x > x_river;
                   the pipe starts at the toe x = x_exit
All other boundaries are impermeable.  Conductivity k_h horizontally and k_v = kv_over_kh * k_h
vertically.  Cell-centred finite volumes on cubes of side d; pipe (and hole) are top-layer cells with
prescribed head.

The problem is linear in the prescribed heads, so the inflow into the pipe is
    q_i = k_h * ( sum_j G_ij h_j + g_i H ),      i, j = pipe stations (one per cell length along x),
where G and g (units of length) depend on geometry only.  `PipeResponse` computes them once per pipe length.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cg(A, b, rtol, M):
    """Conjugate gradients; the tolerance keyword was renamed in SciPy 1.12."""
    try:
        return spla.cg(A, b, rtol=rtol, maxiter=40000, M=M)
    except TypeError:
        return spla.cg(A, b, tol=rtol, maxiter=40000, M=M)


def _fft_preconditioner(shape, rv, free):
    """Exact inverse of the same operator without pipe, hole or surface patches (cosine transforms), used as a
    preconditioner: impermeable walls (DCT-II) except the upstream face at fixed head (DCT-IV along x)."""
    nx, ny, nz = shape
    lam = ((2.0 - 2.0 * np.cos((np.arange(nx) + 0.5) * np.pi / nx))[:, None, None]
           + (2.0 - 2.0 * np.cos(np.arange(ny) * np.pi / ny))[None, :, None]
           + rv * (2.0 - 2.0 * np.cos(np.arange(nz) * np.pi / nz))[None, None, :])
    types = (4, 2, 2)

    def apply(r):
        f = np.zeros(shape)
        f[free] = r
        for ax, ty in enumerate(types):
            f = sfft.dct(f, type=ty, axis=ax, norm="ortho")
        f /= lam
        for ax, ty in enumerate(types):
            f = sfft.idct(f, type=ty, axis=ax, norm="ortho")
        return f[free]

    n = int(free.sum())
    return spla.LinearOperator((n, n), matvec=apply, dtype=float)


@dataclass
class Box:
    Lx: float = 0.48
    Wy: float = 0.30
    D: float = 0.10
    x_exit: float = 0.125
    d: float = 0.005                    # cell size
    pipe_width: float = 0.010           # footprint of the pipe on the roof (enters only logarithmically)
    hole_radius: float = 0.003
    lens_radius: float = 0.0            # radius of an eroded lens around the hole (head 0); 0 = none
    exit: str = "hole"                  # "hole" or "toe"
    x_river: Optional[float] = None     # start of the riverside bed (head H on the surface); None = roof up to x = Lx
    kv_over_kh: float = 1.0

    @property
    def shape(self):
        return (int(round(self.Lx / self.d)), int(round(0.5 * self.Wy / self.d)), int(round(self.D / self.d)))

    @property
    def seepage_length(self):
        return (self.Lx if self.x_river is None else self.x_river) - self.x_exit


def pol_box(d=0.005, **kw):
    """Small-scale box of Pol (2022): 0.48 x 0.30 x 0.10 m, hole exit at x = 0.125 m, filter at x = 0.48 m."""
    return Box(d=d, **kw)


def okamura_box(d=0.0025, **kw):
    """Model levee of Okamura & Kusube (2025), Fig. 5: sand bed 0.38 x 0.12 x 0.06 m; landside surface 0.11 m,
    levee L = 0.19 m, riverside bed 0.08 m; k_h = 5 k_v."""
    base = dict(Lx=0.38, Wy=0.12, D=0.06, x_exit=0.11, x_river=0.30, exit="toe", kv_over_kh=0.2, pipe_width=0.005)
    base.update(kw)
    return Box(d=d, **base)


class PipeResponse:
    """Response matrices of the aquifer for a pipe of length l (rounded to whole cells)."""

    def __init__(self, box: Box, length: float, rtol: float = 1.0e-10, with_fields: bool = False, width=None):
        self.box = box
        d, rv = box.d, box.kv_over_kh
        nx, ny, nz = box.shape
        self.xc = (np.arange(nx) + 0.5) * d
        self.yc = (np.arange(ny) + 0.5) * d
        jw = max(1, int(round(0.5 * box.pipe_width / d)))          # half-width of the pipe in cells
        self.jw = jw
        fixed = np.zeros((nx, ny, nz), bool)
        ie = int(round(box.x_exit / d))                            # the exit lies on the cell edge x = ie * d
        self.x_exit = ie * d
        if box.exit == "hole":
            # hole (and lens): top cells whose centre lies within the radius, and always the cells that touch the centre
            r = max(box.hole_radius, box.lens_radius)
            dist = np.hypot(self.xc[:, None] - self.x_exit, self.yc[None, :])
            disc = dist <= r
            disc[ie - 1:ie + 1, 0] = True
            fixed[:, :, 0] = disc
            nc = int(disc[ie:, 0].sum())                           # centre-line cells of the hole upstream of its centre
            n_st = max(1, int(round(length / d)) - nc)
            self.stations = ie + nc + np.arange(n_st)
            self.exit_edge = (ie + nc) * d                         # the pipe starts at the edge of the hole or lens
        elif box.exit == "toe":
            n_st = max(1, int(round(length / d)))
            self.stations = ie + np.arange(n_st)                   # first cell under the roof
            self.exit_edge = self.x_exit
        else:
            raise ValueError("exit must be 'hole' or 'toe'")
        self.n = n_st
        self.ds = np.full(n_st, d); self.ds[0] = 0.5 * d           # spacing along the pipe: exit edge -> station 1 -> ...
        i_last_roof = nx - 1 if box.x_river is None else int(round(box.x_river / d)) - 1
        if self.stations[-1] >= i_last_roof:
            raise ValueError("pipe reaches the upstream boundary")
        self.x_st = self.xc[self.stations]
        self.s_st = self.x_st - self.exit_edge                     # distance along the pipe from its downstream end
        self.x_tip = self.x_st[-1] + 0.5 * d
        self.length = self.x_tip - self.x_exit                     # from the centre of the hole (or the toe) to the tip
        # footprint of the pipe on the roof: box.pipe_width everywhere, or `width` = full width [m] at each station
        # (an array, or a function of the distance from the downstream end of the pipe), not narrower than two cells
        if width is None:
            jws = np.full(n_st, jw)
        else:
            w = np.asarray(width(self.s_st) if callable(width) else width, float) * np.ones(n_st)
            jws = np.clip(np.rint(0.5 * w / d).astype(int), 1, ny)
        self.jws = jws
        for i, j in zip(self.stations, jws):
            fixed[i, :j, 0] = True
        self.fixed = fixed
        free = ~fixed
        idx = -np.ones((nx, ny, nz), np.int64)
        n = int(free.sum())
        idx[free] = np.arange(n)
        self.idx, self.free = idx, free
        rows, cols, vals = [], [], []
        diag = np.zeros(n)
        me = idx[free]
        self._nb = []                                              # (free cell, x-index of fixed neighbour, weight)
        ixg = np.broadcast_to(np.arange(nx)[:, None, None], (nx, ny, nz))
        for axis, w in ((0, 1.0), (1, 1.0), (2, rv)):
            for shift in (-1, 1):
                nb_idx = np.roll(idx, -shift, axis=axis)
                nb_fix = np.roll(fixed, -shift, axis=axis)
                nb_ix = np.roll(ixg, -shift, axis=axis)
                valid = np.ones((nx, ny, nz), bool)
                sl = [slice(None)] * 3
                sl[axis] = -1 if shift == 1 else 0
                valid[tuple(sl)] = False                           # impermeable walls and the symmetry plane
                v = valid[free]
                diag[me[v]] += w
                isfree = (~nb_fix[free]) & v
                rows.append(me[isfree]); cols.append(nb_idx[free][isfree]); vals.append(np.full(int(isfree.sum()), -w))
                isfix = nb_fix[free] & v
                self._nb.append((me[isfix], nb_ix[free][isfix], w))
        rhs_H = np.zeros(n)
        up = idx[nx - 1][free[nx - 1]]                             # upstream face: half-cell distance
        diag[up] += 2.0
        rhs_H[up] += 2.0
        up_w = np.zeros(n)                                         # conductance to the boundaries held at H
        up_w[up] += 2.0
        top_free = free[:, :, 0]
        if box.exit == "toe":                                      # free landside surface at head 0
            m = (self.xc[:, None] < box.x_exit) & top_free
            diag[idx[:, :, 0][m]] += 2.0 * rv
        if box.x_river is not None:                                # riverside bed at head H
            m = (self.xc[:, None] > box.x_river) & top_free
            c = idx[:, :, 0][m]
            diag[c] += 2.0 * rv
            rhs_H[c] += 2.0 * rv
            up_w[c] += 2.0 * rv
        r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        self.A = (sp.csr_matrix((v, (r, c)), shape=(n, n)) + sp.diags(diag)).tocsr()
        self.Minv = _fft_preconditioner((nx, ny, nz), rv, free)
        # one solve per unit load: station j at head 1 (all else 0), and H = 1 (pipe and exit at 0)
        G = np.zeros((n_st, n_st))
        g = np.zeros(n_st)
        up_G = np.zeros(n_st)
        fields = []
        for j in range(n_st + 1):
            if j < n_st:
                rhs = np.zeros(n)
                for cells, ix, w in self._nb:
                    m = ix == self.stations[j]
                    np.add.at(rhs, cells[m], w)
            else:
                rhs = rhs_H
            h, info = _cg(self.A, rhs, rtol, self.Minv)
            if info != 0:
                raise RuntimeError("CG did not converge")
            col = self._inflow(h, unit_station=j if j < n_st else None)
            if j < n_st:
                G[:, j] = col
                up_G[j] = -(up_w @ h)
            else:
                g[:] = col
                up_g = up_w @ (1.0 - h)
            if with_fields:
                full = np.zeros((nx, ny, nz))
                full[free] = h
                if j < n_st:
                    full[self.stations[j], :self.jws[j], 0] = 1.0
                fields.append(full[:, :, 0].copy())                # head in the top layer of cells
        self.G = 2.0 * d * G                                       # both halves; conductance per face = k_h d
        self.g = 2.0 * d * g
        self.up_G = 2.0 * d * up_G                                 # flow entering through the boundaries at head H
        self.up_g = 2.0 * d * up_g
        self.top_fields = np.array(fields) if with_fields else None

    def _inflow(self, h, unit_station=None):
        """Flow (per unit k_h, per unit cell size, half domain) from free cells into each pipe station."""
        out = np.zeros(self.n)
        for cells, ix, w in self._nb:
            k = ix - self.stations[0]
            m = (k >= 0) & (k < self.n)
            hp = np.zeros(cells.size)
            if unit_station is not None:
                hp[ix == self.stations[unit_station]] = 1.0
            np.add.at(out, k[m], w * (h[cells] - hp)[m])
        return out

    def inflow(self, h_pipe, H, k=1.0):
        """Inflow into each station [m^3/s] for pipe heads h_pipe [m], upstream head H [m], k = k_h."""
        return k * (self.G @ np.asarray(h_pipe, float) + self.g * H)

    def upstream_flow(self, h_pipe, H, k=1.0):
        """Flow entering the sand through the boundaries held at H [m^3/s].  With a hole exit this is the total
        discharge leaving through the hole (pipe flow plus direct seepage into the hole)."""
        return k * (self.up_G @ np.asarray(h_pipe, float) + self.up_g * H)

    def roof_field(self, h_pipe, H):
        """Head in the top layer of cells, array (nx, ny) (needs with_fields=True)."""
        w = np.concatenate([np.asarray(h_pipe, float), [H]])
        return np.tensordot(w, self.top_fields, axes=1)

    def roof_head(self, h_pipe, H, y=0.0):
        """Head under the roof along the line y = const (default: the pipe axis)."""
        f = self.roof_field(h_pipe, H)
        j = int(np.argmin(np.abs(self.yc - abs(y)))) if y else 0
        return self.xc, f[:, j]


def prescribed_pipe(box: Box, x_tip: float, head, width, H, h_exit=0.0, rtol=1.0e-9, return_roof=False):
    """Seepage into a pipe whose head is known (hole exit only).  One solve; no pipe law.

    The pipe runs along the axis from the hole to x_tip, which may be the upstream face of the sand (a pipe that has
    reached the filter).  head(x) is the head in the pipe [m], width(x) the width over which it drains the roof
    [m] (not narrower than two cells), H the head on the upstream face, h_exit the head in the hole.
    Returns dict: x [m] of the stations, q [m^2/s per unit conductivity] = inflow per unit length of pipe divided
    by k, w [m] = the width used, Q_hole and Q_up [m^3/s per unit conductivity] = flow from the sand into the hole
    and flow entering the sand through the upstream face."""
    d = box.d
    nx, ny, nz = box.shape
    xc = (np.arange(nx) + 0.5) * d
    yc = (np.arange(ny) + 0.5) * d
    ie = int(round(box.x_exit / d))
    fixed = np.zeros((nx, ny, nz), bool)
    hfix = np.zeros((nx, ny, nz))
    r = max(box.hole_radius, box.lens_radius)
    disc = np.hypot(xc[:, None] - ie * d, yc[None, :]) <= r
    disc[ie - 1:ie + 1, 0] = True
    fixed[:, :, 0] = disc
    hfix[:, :, 0][disc] = h_exit
    ishole = fixed.copy()
    i0 = ie + int(disc[ie:, 0].sum())
    i1 = min(nx, int(round(x_tip / d)))
    st = np.arange(i0, i1)
    wv = np.asarray([width(x) for x in xc[st]], float) if callable(width) else np.full(st.size, float(width))
    jws = np.clip(np.rint(0.5 * wv / d).astype(int), 1, ny)
    for i, j in zip(st, jws):
        fixed[i, :j, 0] = True
        hfix[i, :j, 0] = head(xc[i])
    free = ~fixed
    n = int(free.sum())
    idx = -np.ones((nx, ny, nz), np.int64)
    idx[free] = np.arange(n)
    me = idx[free]
    rows, cols, vals, nb = [], [], [], []
    diag, rhs = np.zeros(n), np.zeros(n)
    ixg = np.broadcast_to(np.arange(nx)[:, None, None], (nx, ny, nz))
    rv = box.kv_over_kh
    for axis, w in ((0, 1.0), (1, 1.0), (2, rv)):
        for shift in (-1, 1):
            nb_idx = np.roll(idx, -shift, axis=axis)
            nb_fix = np.roll(fixed, -shift, axis=axis)
            nb_h = np.roll(hfix, -shift, axis=axis)
            nb_hole = np.roll(ishole, -shift, axis=axis)
            nb_ix = np.roll(ixg, -shift, axis=axis)
            valid = np.ones((nx, ny, nz), bool)
            sl = [slice(None)] * 3
            sl[axis] = -1 if shift == 1 else 0
            valid[tuple(sl)] = False
            v = valid[free]
            diag[me[v]] += w
            isfree = (~nb_fix[free]) & v
            rows.append(me[isfree]); cols.append(nb_idx[free][isfree]); vals.append(np.full(int(isfree.sum()), -w))
            isfix = nb_fix[free] & v
            np.add.at(rhs, me[isfix], w * nb_h[free][isfix])
            nb.append((me[isfix], nb_ix[free][isfix], nb_h[free][isfix], nb_hole[free][isfix], w))
    up = idx[nx - 1][free[nx - 1]]
    diag[up] += 2.0
    rhs[up] += 2.0 * H
    A = (sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)) + sp.diags(diag)).tocsr()
    h, info = _cg(A, rhs, rtol, _fft_preconditioner((nx, ny, nz), rv, free))
    if info != 0:
        raise RuntimeError("CG did not converge")
    q = np.zeros(nx)
    q_hole = 0.0
    for cells, ix, hn, hole, w in nb:
        flow = w * (h[cells] - hn)
        np.add.at(q, ix[~hole], flow[~hole])
        q_hole += float(flow[hole].sum())
    out = dict(x=xc[st], q=2.0 * q[st], w=2.0 * jws * d, Q_hole=2.0 * d * q_hole, Q_up=2.0 * d * float(np.sum(2.0 * (H - h[up]))))
    if return_roof:
        full = hfix.copy()
        full[free] = h
        out["roof"] = full[:, :, 0]
        out["xc"], out["yc"] = xc, yc
    return out
