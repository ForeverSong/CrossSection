"""Threshold cross-section with uniform seepage through the bed (manuscript table 2, section 4.3).

Threshold condition:  (tau/tau_c)^2 + sin^2(phi)/mu^2 = (cos(phi) - sigma)^2,  sigma = normal gradient / fluidisation gradient.
Prints, for a friction coefficient mu, the section in units of a_c (defined with the threshold stress without seepage) and,
for a given discharge, the slope, largest depth and width relative to the section without seepage; and the widths at
25, 50 and 75 % of the largest depth for comparison with the scanned sections.     usage: python run_section_seepage.py [--mu 0.77] [--out ../tmp]
"""
import argparse
import os
import numpy as np
from pipetheory.section import threshold_section, edge_slope

ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, default=0.77)
ap.add_argument("--out", default="../tmp")
a = ap.parse_args()
o0 = threshold_section(a.mu)
L = [f"mu = {a.mu}",
     "sigma  a_max/a_c  W/a_c  W/a_max  chi      edge slope | given Q: slope  depth  width | widths at 25, 50, 75 % of a_max (in a_max)"]
for sg in (0.0, 0.1, 0.2, 0.3, 0.37, 0.4, 0.5, 0.6):
    o = threshold_section(a.mu, sigma=sg)
    Sr = (o["chi"] / o0["chi"]) ** (1.0 / 3.0)                  # Q S^3 = chi g A^4 / nu at fixed Q
    y, d = o["y"] / o["a_max"], o["a"] / o["a_max"]
    w = [2.0 * np.interp(-f, -d, y) for f in (0.25, 0.5, 0.75)]
    L.append(f"{sg:4.2f}   {o['a_max']:.3f}      {2 * o['half_width']:.3f}  {o['W_over_amax']:5.2f}    {o['chi']:.4f}   {edge_slope(a.mu, sg):.3f}      |"
             f"          {Sr:.3f}  {o['a_max'] / o0['a_max'] / Sr:.3f}  {o['half_width'] / o0['half_width'] / Sr:.3f} | {w[0]:.2f} {w[1]:.2f} {w[2]:.2f}")
L.append("shallow limit (mu -> 0): a = (1 - sigma) cos(mu y); width unchanged; chi = (1 - sigma)^3 / (9 mu); at a given discharge "
         "slope x (1 - sigma), depth unchanged, width / (1 - sigma)")
txt = "\n".join(L)
print(txt)
open(os.path.join(a.out, "paper1_section_seepage.txt"), "w").write(txt + "\n")
