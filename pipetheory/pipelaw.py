"""Slope-discharge law of an erosion pipe whose bed is at the threshold of motion.

Two regimes, set by the width W0 that tip erosion gives the pipe (about 30 grain diameters in the
experiments of Van Beek et al. 2015 and Pol 2022):

  self-formed  (W_thr >= W0, large discharge)   the roofed threshold section of section.py
        Q = chi g A^4 / (nu S^3),      A = 2 tau_c / (rho g),      W_thr = 2 beta A / S
  width-limited (W_thr < W0, small discharge)    two threshold banks and a flat bed at tau_c in between
        Q = g A^3 / (12 nu) * [ W0 / S^2 - (2 beta - 12 chi) A / S^3 ]
The two expressions coincide at S_x = 2 beta A / W0.  beta and chi come from the cross-section solution.
The centre of the width-limited section is treated as plane Poiseuille flow of gap a_c = A / S.

Optional depth cap (a_cap): the gap between roof and bed is not allowed to exceed a_cap.  Where the threshold depth
A / S would be larger the pipe keeps the depth a_cap and widens instead, so the slope cannot fall below A / a_cap
(a flat bed at the threshold carries the same discharge per unit width everywhere: W = 12 nu Q S^2 / (g A^3)).
This is an empirical closure suggested by measured pipe gradients, not a derived result.

A winding path of length T times the straight distance multiplies the head loss per unit distance by T.
"""
import numpy as np


class PipeLaw:
    def __init__(self, sec, tau_c, W0=None, tortuosity=1.0, a_cap=None, rho=1000.0, g=9.81, nu=1.0e-6):
        self.tau_c, self.rho, self.g, self.nu, self.T = tau_c, rho, g, nu, tortuosity
        self.A = 2.0 * tau_c / (rho * g)
        self.chi, self.beta, self.alpha = sec["chi"], sec["half_width"], sec["a_max"]
        self.mean_factor = sec["a_max"] * sec["aavg_over_amax"]
        self.Lam = self.chi * g * self.A ** 4 / nu                 # self-formed: Q S^3 = Lam
        self.W0 = W0
        self.S_floor = self.A / a_cap if a_cap else 0.0
        self.e = 2.0 * self.beta - 12.0 * self.chi
        if W0:
            self.S_x = 2.0 * self.beta * self.A / W0
            self.Q_x = self.Lam / self.S_x ** 3
            self._S = np.logspace(np.log10(self.S_x), np.log10(self.S_x) + 4.0, 800)
            self._Q = self._Qwide(self._S)
        else:
            self.S_x, self.Q_x = np.inf, 0.0

    def _Qwide(self, S):
        return self.g * self.A ** 3 / (12.0 * self.nu) * (self.W0 / S ** 2 - self.e * self.A / S ** 3)

    def slope(self, Q):
        """Head loss per unit straight distance and its derivative with respect to Q."""
        Q = np.asarray(Q, float)
        S = (self.Lam / Q) ** (1.0 / 3.0)
        dS = -S / (3.0 * Q)
        if self.W0:
            w = Q < self.Q_x
            if w.any():
                Sw = np.exp(np.interp(np.log(Q[w]), np.log(self._Q[::-1]), np.log(self._S[::-1])))
                for _ in range(3):                                  # polish with Newton
                    f = self._Qwide(Sw) - Q[w]
                    df = self.g * self.A ** 3 / (12.0 * self.nu) * (-2.0 * self.W0 / Sw ** 3 + 3.0 * self.e * self.A / Sw ** 4)
                    Sw = Sw - f / df
                S[w] = Sw
                dS[w] = 1.0 / (self.g * self.A ** 3 / (12.0 * self.nu) * (-2.0 * self.W0 / Sw ** 3 + 3.0 * self.e * self.A / Sw ** 4))
        if self.S_floor:
            cap = S < self.S_floor
            S = np.where(cap, self.S_floor, S)
            dS = np.where(cap, 0.0, dS)
        return self.T * S, self.T * dS

    def geometry(self, S_straight):
        """Width, maximum depth and mean depth of the section for a head loss S per unit straight distance."""
        S = np.asarray(S_straight, float) / self.T
        a_c = self.A / S
        W_thr = 2.0 * self.beta * a_c
        if not self.W0:
            return W_thr, self.alpha * a_c, self.mean_factor * a_c
        wide = W_thr < self.W0
        W = np.where(wide, self.W0, W_thr)
        a_max = np.where(wide, a_c, self.alpha * a_c)
        area = np.where(wide, self.mean_factor * a_c * W_thr + (self.W0 - W_thr) * a_c, self.mean_factor * a_c * W_thr)
        return W, a_max, area / W
