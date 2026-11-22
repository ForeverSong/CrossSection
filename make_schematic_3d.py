"""Three-dimensional schematic of the problem (manuscript/figures/schematic_3d.png): sand bed under a roof with an exit
hole, the pipe on top of the sand and the seepage that feeds it; and the cross-section of the pipe.  The box has the
proportions of the small-scale test (0.48 x 0.30 x 0.10 m, hole at x = 0.125 m); the pipe is drawn about three times
wider and ten times deeper than in reality so that it can be seen.        usage:  python make_schematic_3d.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "manuscript", "figures")
Lx, Wy, D, X_EXIT, LEN, TOP = 0.48, 0.30, 0.10, 0.125, 0.20, 0.012


def box(ax, x0, x1, y0, y1, z0, z1, color, alpha):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    ax.add_collection3d(Poly3DCollection([v[i] for i in f], facecolor=color, edgecolor="k", linewidths=0.5, alpha=alpha))


fig = plt.figure(figsize=(12, 5.4))
ax = fig.add_subplot(1, 2, 1, projection="3d")
box(ax, 0, Lx, -Wy / 2, Wy / 2, -D, 0, "#d9c79a", 0.30)                      # sand bed
box(ax, 0, Lx, -Wy / 2, Wy / 2, 0, TOP, "#9ecae1", 0.15)                     # transparent roof
box(ax, Lx, Lx + 0.03, -Wy / 2, Wy / 2, -D, TOP, "#4292c6", 0.25)            # upstream filter, head H
s = np.linspace(0, 1, 60); xs = X_EXIT + LEN * s
half = 0.5 * (0.034 - 0.020 * s)                                             # half-width of the pipe (exaggerated)
depth = 0.012 * (1 - s) ** (1 / 3) + 0.002                                   # depth (exaggerated), grows as distance^(1/3)
t = np.linspace(-1, 1, 21)
X = np.repeat(xs[:, None], len(t), 1); Y = half[:, None] * t[None, :]; Z = -depth[:, None] * np.cos(0.5 * np.pi * t[None, :])
ax.plot_surface(X, Y, Z, color="#fbfbfb", edgecolor="0.45", linewidth=0.2, alpha=1.0, shade=True)
th = np.linspace(0, 2 * np.pi, 50); r = 0.012
for z in (0.0, TOP):
    ax.plot(X_EXIT + r * np.cos(th), r * np.sin(th), z + 0 * th, "k-", lw=1.1)
for y0 in (-0.11, -0.055, 0.055, 0.11):
    ax.quiver(Lx - 0.015, y0, -0.05, -(Lx - 0.04 - X_EXIT - 0.7 * LEN), -0.8 * y0, 0.04, color="#08519c", arrow_length_ratio=0.1, lw=1.1)
ax.quiver(Lx - 0.015, 0, -0.05, -(Lx - 0.03 - X_EXIT - LEN), 0, 0.045, color="#08519c", arrow_length_ratio=0.1, lw=1.1)
ax.plot([X_EXIT, Lx], [Wy / 2 + 0.02] * 2, [0, 0], "k-", lw=0.8); ax.text((X_EXIT + Lx) / 2, Wy / 2 + 0.03, 0, "seepage length L", ha="center", fontsize=9)
ax.plot([X_EXIT, X_EXIT + LEN], [-Wy / 2 - 0.02] * 2, [0, 0], "k-", lw=0.8); ax.text(X_EXIT + LEN / 2, -Wy / 2 - 0.045, 0, "pipe length l", ha="center", fontsize=9)
ax.text(X_EXIT, 0, 0.05, "exit hole", ha="center", fontsize=9); ax.text(X_EXIT + LEN + 0.01, 0.02, 0.02, "tip", fontsize=9)
ax.text(0.02, -0.13, -0.085, "sand", fontsize=9); ax.text(0.02, 0.10, 0.03, "roof (impermeable cover)", fontsize=9)
ax.text(Lx + 0.015, 0, 0.045, "head H", ha="center", fontsize=9); ax.text(0.30, -0.09, -0.07, "seepage", color="#08519c", fontsize=9)
ax.set_xlim(0, Lx + 0.03); ax.set_ylim(-Wy / 2, Wy / 2); ax.set_zlim(-D, 0.03)
ax.set_box_aspect((Lx + 0.03, Wy, D + 0.03)); ax.view_init(elev=27, azim=-60); ax.set_axis_off()
ax.set_title("(a) pipe beneath the roof, fed by seepage (pipe enlarged for clarity)", fontsize=10)

b = fig.add_subplot(1, 2, 2)
W, a = 5.2, 1.0                                                               # true proportions of the threshold section
y = np.linspace(-W / 2, W / 2, 200); bed = -a * np.cos(np.pi * y / W)
b.fill_between([-W, W], -2.2, 0, color="#d9c79a", lw=0); b.fill_between(y, bed, 0, color="white", lw=0)
b.fill_between([-W, W], 0, 0.35, color="#9ecae1", lw=0); b.plot(y, bed, "k-", lw=1.4); b.plot([-W, -W / 2], [0, 0], "k-", lw=1.4); b.plot([W / 2, W], [0, 0], "k-", lw=1.4)
b.annotate("", xy=(-W / 2, 0.75), xytext=(W / 2, 0.75), arrowprops=dict(arrowstyle="<->", lw=0.8)); b.text(0, 0.85, "width W  (about 5 x depth; 20-35 grain diameters)", ha="center", fontsize=9)
b.annotate("", xy=(0, 0), xytext=(0, -a), arrowprops=dict(arrowstyle="<->", lw=0.8)); b.text(0.15, -0.55, "depth a\n(4-7 grain\ndiameters)", fontsize=9, va="center")
b.text(-W * 0.78, 0.12, "roof", fontsize=9); b.text(-W * 0.78, -1.6, "sand", fontsize=9); b.text(-1.9, -0.25, "water", fontsize=9, color="#08519c")
b.annotate("bed everywhere at the\nthreshold of grain motion", xy=(1.6, -0.55), xytext=(2.4, -1.75), fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.7))
for yy in (-2.2, -1.1, 1.1, 2.2):
    b.annotate("", xy=(yy * 0.55, -0.95 * np.cos(np.pi * yy * 0.55 / W)), xytext=(yy, -2.0), arrowprops=dict(arrowstyle="->", lw=0.8, color="#08519c"))
b.set_xlim(-W, W); b.set_ylim(-2.2, 1.3); b.set_aspect("equal"); b.axis("off")
b.set_title("(b) cross-section of the pipe (true proportions)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "schematic_3d.png"), dpi=200)
