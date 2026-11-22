"""Cross-section of a laminar erosion pipe under a rigid roof at the threshold of motion.

Prints the shape coefficients as a function of the friction coefficient mu (full Stokes solution and the
shallow/lubrication limit) and compares the predicted shape with a measured section
(Pol 2022, Fig. 3.10, digitised: code/data/pol2022_fig3_10_B25_245_x022.csv).

usage:  python run_section.py [--out ../tmp]
"""
import argparse
import os
import numpy as np
from pipetheory.section import threshold_section, lubrication, trapz

HERE = os.path.dirname(os.path.abspath(__file__))


def measured_section():
    y, z = np.loadtxt(os.path.join(HERE, "data", "pol2022_fig3_10_B25_245_x022.csv"), delimiter=",", skiprows=1, unpack=True)
    s = (y >= 149.7) & (y <= 158.8)                         # main channel, limits as marked by Pol
    y, z = y[s], z[s]
    W = y.max() - y.min(); A = trapz(z, y)
    return dict(y=y, a=z, W=W, a_max=z.max(), a_avg=A / W, area=A, y_peak=y[np.argmax(z)],
                conveyance=trapz(z ** 3, y) / (W * (A / W) ** 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    a = ap.parse_args()
    print("Threshold cross-section under a rigid roof (lengths in units of a_c = 2 tau_c / (rho g S))")
    print(f"{'mu':>5s} {'repose':>7s} | {'a_max':>6s} {'W/a_max':>8s} {'a_avg/a_max':>11s} {'W/a_avg':>8s} {'chi':>8s} {'Q/Q_rect':>8s}"
          f" | shallow limit: {'W/a_max':>8s} {'W/a_avg':>8s} {'chi':>8s}")
    secs = {}
    for mu in (0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        o = threshold_section(mu, 16, 161, 41); l = lubrication(mu); secs[mu] = o
        print(f"{mu:5.2f} {np.degrees(np.arctan(mu)):6.1f}d | {o['a_max']:6.3f} {o['W_over_amax']:8.2f} {o['aavg_over_amax']:11.3f} "
              f"{o['W_over_aavg']:8.2f} {o['chi']:8.4f} {o['conveyance_vs_rectangle']:8.2f} |                {l['W_over_amax']:8.2f} "
              f"{l['W_over_aavg']:8.2f} {l['chi']:8.4f}")
    print("chi: Q = chi g S a_c^4 / nu.   Q/Q_rect: discharge relative to a rectangular duct of the same width and mean depth (cubic law).")
    m = measured_section()
    mu_s = np.tan(np.radians(37.6))                         # repose angle of this sand (sand-boil slope, Pol Table 3.1)
    t = threshold_section(mu_s, 16, 161, 41)
    print(f"\nMeasured main channel (Pol 2022 Fig. 3.10, B25 sand): W = {m['W']:.1f} mm, a_max = {m['a_max']:.2f} mm, a_avg = {m['a_avg']:.2f} mm")
    print(f"{'':26s} {'W/a_max':>8s} {'a_avg/a_max':>11s} {'W/a_avg':>8s} {'mean(a^3)/a_avg^3':>18s}")
    print(f"{'measured':26s} {m['W'] / m['a_max']:8.2f} {m['a_avg'] / m['a_max']:11.2f} {m['W'] / m['a_avg']:8.2f} {m['conveyance']:18.2f}")
    ta = np.concatenate([t['a'][::-1], t['a'][1:]]); ty = np.concatenate([-t['y'][::-1], t['y'][1:]])
    conv_t = trapz(ta ** 3, ty) / (2 * t['half_width'] * (t['area'] / (2 * t['half_width'])) ** 3)
    print(f"{'theory, mu = tan(37.6 deg)':26s} {t['W_over_amax']:8.2f} {t['aavg_over_amax']:11.2f} {t['W_over_aavg']:8.2f} {conv_t:18.2f}")
    print(f"{'rectangle':26s} {'-':>8s} {1.0:11.2f} {'-':>8s} {1.0:18.2f}")
    try:
        import plotstyle
        plt = plotstyle.apply()
    except Exception as e:                                  # matplotlib missing: tables only
        print("no figure:", e); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.4))
    for mu, lw, al in ((0.5, 1.2, 0.6), (0.7, 2.2, 1.0), (0.9, 1.2, 0.6)):
        o = secs[mu]
        ax1.plot(np.concatenate([-o['y'][::-1], o['y'][1:]]), -np.concatenate([o['a'][::-1], o['a'][1:]]), color=plotstyle.BLUE, lw=lw, alpha=al)
        ax1.text(o['half_width'], 0.05, f"{mu}", ha="center", va="bottom", fontsize=8, color=plotstyle.INK_2)
    ax1.text(1.55, 0.05, "$\\mu$ =", ha="right", va="bottom", fontsize=8, color=plotstyle.INK_2)
    lb = lubrication(0.7); yy = np.linspace(-lb['half_width'], lb['half_width'], 200)
    ax1.plot(yy, -np.cos(0.7 * yy), color=plotstyle.INK_2, lw=1.0, ls="--")
    ax1.text(-lb['half_width'], 0.05, "shallow limit, 0.7", ha="center", va="bottom", fontsize=8, color=plotstyle.INK_2)
    ax1.axhline(0.0, color=plotstyle.INK, lw=1.5)
    ax1.set_ylim(-1.25, 0.28); ax1.set_xlim(-4.2, 4.2)
    ax1.set_xlabel("y / a_c"); ax1.set_ylabel("z / a_c"); ax1.set_title("Threshold section under a rigid roof (not to scale)", pad=10)
    scale = m['a_max'] / t['a_max']                         # same maximum depth; the width is then a prediction
    ax2.plot(ty * scale, -ta * scale, color=plotstyle.BLUE, label="theory, same maximum depth")
    ax2.plot(m['y'] - m['y_peak'], -m['a'], color=plotstyle.ORANGE, lw=1.6, label="measured main channel (Pol 2022, Fig. 3.10)")
    ax2.axhline(0.0, color=plotstyle.INK, lw=1.5)
    ax2.set_ylim(-2.3, 0.75); ax2.set_xlim(-6.5, 6.5)
    ax2.set_xlabel("y [mm]"); ax2.set_ylabel("depth below roof [mm]"); ax2.set_title("Predicted and measured shape (not to scale)", pad=10)
    ax2.legend(frameon=False, fontsize=8, loc="upper center", ncol=1)
    fig.tight_layout()
    out = os.path.join(a.out, "paper1_section.png"); fig.savefig(out); print("figure:", out)


if __name__ == "__main__":
    main()
