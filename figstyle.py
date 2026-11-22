"""Common style of the manuscript figures (make_figures_jfm.py).

Framed axes with ticks pointing inwards on all four sides, a Times-like face (STIX) at the size of the JFM text, and one
colour code used throughout:
    theory                      black lines
    the three small-scale sands blue squares (d50 = 0.185 mm), orange circles (0.228 mm), green triangles (0.422 mm)
    the two branches of the fold blue (large pipe) and orange (pinched pipe)
    water, sand, roof           light blue, sand, grey (sketches and filled sections)
The three series colours are the first three slots of a palette checked for colour-vision deficiency on all pairs
(deuteranopia Delta E 9.2, normal vision 24.0); every series also differs by marker or line style, so that the figures
read in grey.  Figures are drawn at the width of the JFM text block, 32 pc = 5.33 in, so that type is at its true size.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

TEXTWIDTH = 5.33
INK, INK2, GREY, LIGHT = "#0b0b0b", "#52514e", "#8a8985", "#dddcd7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WATER, SAND, ROOF = "#cde2fb", "#efe6d0", "#b4b3ae"
SANDS = {"FPH": dict(color=BLUE, marker="s", label=r"$d_{50}=0.185$ mm"),
         "B25": dict(color=ORANGE, marker="o", label=r"$d_{50}=0.228$ mm"),
         "FS35": dict(color=AQUA, marker="^", label=r"$d_{50}=0.422$ mm")}
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "manuscript", "figures")


def use():
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
        "axes.linewidth": 0.8, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "axes.labelpad": 2.5,
        "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.4, "ytick.major.size": 3.4, "xtick.minor.size": 1.9, "ytick.minor.size": 1.9,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.major.pad": 3.0, "ytick.major.pad": 3.0,
        "lines.linewidth": 1.3, "lines.markersize": 4.6, "lines.markeredgewidth": 0.5, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 7.5, "legend.handlelength": 2.0, "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.5, "legend.labelspacing": 0.3, "hatch.linewidth": 0.5,
        "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150})


def panel(ax, letter, x=0.0, y=1.035, **kw):
    """Panel label in the JFM manner, (a), italic, above the top left corner of the frame."""
    ax.text(x, y, rf"$({letter})$", transform=ax.transAxes, fontsize=10, va="bottom", ha="left", **kw)


def marker(sand, filled=True, ms=4.6):
    """Keyword arguments of a marker of one of the three small-scale sands: filled, or open with a coloured edge."""
    s = SANDS[sand]
    if filled:
        return dict(marker=s["marker"], ls="none", ms=ms, mfc=s["color"], mec=INK, mew=0.5)
    return dict(marker=s["marker"], ls="none", ms=ms, mfc="white", mec=s["color"], mew=1.0)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
