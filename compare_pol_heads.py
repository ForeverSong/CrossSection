"""Head profile along the pipe: threshold-pipe theory vs pressure measurements (Pol 2022, test B25_245).

No fitted parameters.  Inputs: hydraulic conductivity of the test, the equilibrium bed stress Pol measured
for this sand (read as the section average unless --tau-as-flat), mu from the repose angle of the sand, and
the head across the sample and the pipe length at the two moments chosen (from the observation sheet; the
pipe was at rest at these heads).

usage:  python compare_pol_heads.py <path to Data/Pol_4TU>  [--tau-as-flat] [--out figure.png]
"""
import argparse
import os
import numpy as np
import scipy.io as sio
from pipetheory import RHO_W, G
from pipetheory.aquifer import Box, PipeResponse
from pipetheory.coupled import solve_pipe, fold, tip_gradient
from pipetheory.section import threshold_section, pipe_constant

X_PORT = {**{n: 0.19 + 0.02 * (n - 2) for n in range(2, 14)}, 14: 0.45, 15: 0.47, 1: 0.05}   # m, centre-line ports
STATES = [  # label, DAQ window [s], total head difference dH [cm], pipe length l [m] (tape reading + 1 cm)
    ("regressive phase, pipe at rest, l = 12.9 cm", (2400.0, 2520.0), 5.3, 0.129),
    ("just below the critical head, l = 18.5 cm", (4700.0, 4920.0), 5.7, 0.185),
]


def measured(path, window):
    m = sio.loadmat(os.path.join(path, "pressure", "B25_245_pressure.MAT"), squeeze_me=True, struct_as_record=False)
    ch = {m[f"Channel_{c}_Header"].SignalName: m[f"Channel_{c}_Data"] for c in range(1, 26)}
    t = ch["Time__2_-_default_sample_rate"]
    s = (t >= window[0]) & (t <= window[1])
    return {n: float(np.mean(ch[f"dP_{n:03d}"][s])) for n in range(1, 16)}       # head drop below upstream [cm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--tau-as-flat", action="store_true",
                    help="use Pol's equilibrium stress directly as the flat-bed threshold (default: read it as the section average)")
    ap.add_argument("--mu", type=float, default=float(np.tan(np.radians(37.6))), help="friction coefficient (default: repose angle of B25 sand)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    k, tau_c = 3.1e-4, 0.377
    sec = threshold_section(a.mu, 16, 161, 41)
    if not a.tau_as_flat:
        tau_c /= sec["a_max"] * sec["aavg_over_amax"]
    Lam = pipe_constant(sec, tau_c)
    box = Box()
    results = []
    for label, win, dH, l in STATES:
        dP = measured(a.data, win)
        h = {n: dH - v for n, v in dP.items()}                                    # head above the downstream level [cm]
        slope = (h[15] - h[14]) / (X_PORT[15] - X_PORT[14])
        H = (h[15] + slope * (box.Lx - X_PORT[15])) * 1e-2                         # head at the filter face [m]
        R = PipeResponse(box, l, with_fields=True)
        N = H ** 4 * k / Lam
        Nmin, _ = fold(R)
        sol = solve_pipe(R, N)
        print(f"\n{label}:  dH = {dH} cm, head at the filter face = {H * 100:.2f} cm, N = {N:.2f} m2 (fold at {Nmin:.2f})")
        if sol is None:
            print("   no threshold solution at this head (below the fold): the model says this pipe cannot move sand")
            continue
        x, roof = R.roof_head(sol["h"], 1.0)
        roof = roof * H * 100
        x_tip = R.x_st[-1] + 0.5 * box.d
        print("   port   x [m]   measured [cm]   theory [cm]")
        for n in range(2, 16):
            print(f"   P{n:<3d}  {X_PORT[n]:.2f}   {h[n]:8.2f}      {np.interp(X_PORT[n], x, roof):8.2f}" + ("   (in the pipe)" if X_PORT[n] <= x_tip else ""))
        ip_m = [n for n in range(2, 14) if X_PORT[n] <= x_tip]
        if len(ip_m) > 2:
            pm = np.polyfit([X_PORT[n] for n in ip_m], [h[n] for n in ip_m], 1)[0] / 100
            pt = np.polyfit(R.x_st, sol["h"] * H, 1)[0]
            print(f"   mean pipe gradient: measured {pm:.3f}, theory {pt:.3f}")
        a_c = 2 * tau_c / (RHO_W * G * sol["S"] * H)
        print(f"   tip gradient over 2 cm: theory {tip_gradient(R, sol) * H:.2f};   theory pipe discharge at the exit {k * H * sol['Q'][0] * 6e7:.0f} mL/min")
        print(f"   theory max depth: {sec['a_max'] * a_c[0] * 1e3:.2f} mm at the exit, {sec['a_max'] * a_c[R.n // 2] * 1e3:.2f} mm mid-pipe; width {2 * sec['half_width'] * a_c[0] * 1e3:.1f} / {2 * sec['half_width'] * a_c[R.n // 2] * 1e3:.1f} mm")
        results.append((label, x, roof, h, x_tip, R, sol, H))
    if a.out and results:
        import plotstyle
        plt = plotstyle.apply()
        fig, axs = plt.subplots(1, len(results), figsize=(5.0 * len(results), 3.7), squeeze=False)
        for ax, (label, x, roof, h, x_tip, R, sol, H) in zip(axs[0], results):
            ax.plot(x * 100, roof, color=plotstyle.BLUE, label="theory (no fitted parameter)")
            ports = sorted(n for n in h if n >= 2)
            ax.plot([X_PORT[n] * 100 for n in ports], [h[n] for n in ports], "o", color=plotstyle.ORANGE, ms=5.5, label="measured (pressure ports)")
            ax.axvline(x_tip * 100, color=plotstyle.INK_2, lw=0.8, ls="--")
            ax.text(x_tip * 100 - 0.5, 0.97 * H * 100, "pipe tip", ha="right", va="top", color=plotstyle.INK_2, fontsize=8)
            ax.set_xlim(12, 48); ax.set_ylim(0, None)
            ax.set_xlabel("x [cm]"); ax.set_ylabel("head above downstream level [cm]")
            ax.set_title(label.capitalize()); ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(a.out)
        print("\nfigure:", a.out)

if __name__ == "__main__":
    main()
