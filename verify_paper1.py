"""Checks for the pipetheory package (NumPy/SciPy only).   usage: python verify_paper1.py"""
import sys
import time
import types
import numpy as np
from pipetheory.section import stokes_roofed, stokes_section, rectangle_flow_rate, threshold_section, lubrication, edge_slope
from pipetheory.aquifer import pol_box, okamura_box, PipeResponse, prescribed_pipe
from pipetheory.coupled import fold, solve_pipe, trace_branches, fold_law, head_from_N, solve_seepage_width, solve_pipe_law
from pipetheory.pipelaw import PipeLaw

results = []


def check(name, value, target, tol, relative=True):
    err = abs(value / target - 1.0) if relative else abs(value - target)
    ok = err <= tol
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.6g} (expected {target:.6g}, {'rel.' if relative else 'abs.'} error {err:.2e}, tol {tol:.0e})")


t0 = time.time()
print("-- flow in a duct under a flat roof (mapped finite differences) --")
for w, h in ((8.0, 1.0), (2.0, 1.0)):
    sol = stokes_roofed(np.full(201, h), w / 2, 81)
    check(f"rectangle {w:g} x {h:g}: flow rate vs series solution", sol["q"], rectangle_flow_rate(w, h), 5e-4)
s = np.linspace(0, 1, 401)
sol = stokes_roofed(np.sqrt(3) / 2 * 2.0 * (1 - s), 1.0, 161)
check("equilateral triangle (side 2): flow rate vs sqrt(3) side^4/320", sol["q"], np.sqrt(3) * 16 / 320, 1e-3)
check("equilateral triangle: wall stress / local gap at mid-bank", sol["tau_bed"][200] / (np.sqrt(3) / 2 * 2.0 * 0.5), 1.0, 1e-3)

print("-- threshold cross-section --")
o = threshold_section(0.1, 16, 161, 41); l = lubrication(0.1)
check("mu = 0.1: width/depth -> shallow limit pi/mu", o["W_over_amax"], l["W_over_amax"], 1e-2)
check("mu = 0.1: discharge coefficient -> 1/(9 mu)", o["chi"], l["chi"], 3e-2)
o = threshold_section(0.7, 16, 161, 41)
check("mu = 0.7: threshold condition residual at the nodes", o["max_residual"], 0.0, 1e-8, relative=False)
check("mu = 0.7: edge slope = mu", abs(np.gradient(o["a"], o["y"])[-3:]).mean(), 0.7, 3e-2)
of = threshold_section(0.7, 24, 241, 61)
check("mu = 0.7: W/a_avg, coarse vs fine discretisation", o["W_over_aavg"], of["W_over_aavg"], 1e-3)
check("mu = 0.7: chi, coarse vs fine discretisation", o["chi"], of["chi"], 2e-3)
# independent test of the converged shape: re-solve the flow on a finer grid and test the Coulomb condition
sf = np.linspace(0, 1, 481)
af = np.interp(sf * of["half_width"], of["y"], of["a"]); af[-1] = 0.0
sol = stokes_roofed(af, of["half_width"], 121)
t2 = (sol["a_s"] / of["half_width"]) ** 2
need = np.sqrt(np.clip(1 - t2 / 0.49, 0, None) / (1 + t2))
core = sf < 0.93
check("mu = 0.7: Coulomb condition on a finer grid (max |tau/tau_c - required|, y < 0.93 b)", np.abs(sol["tau_bed"][core] - need[core]).max(), 0.0, 5e-3, relative=False)

print("-- aquifer response --")
R = PipeResponse(pol_box(), 0.10, with_fields=True)
check("pipe length on the grid (hole centre to tip)", R.length, 0.10, 1e-9)
check("reciprocity: max |G - G^T| / max |G|", np.abs(R.G - R.G.T).max() / np.abs(R.G).max(), 0.0, 1e-8, relative=False)
check("pipe at the upstream head takes no water: max inflow / inflow of an open pipe", max(R.inflow(np.ones(R.n), 1.0).max(), 0.0) / R.inflow(np.zeros(R.n), 1.0).sum(), 0.0, 1e-8, relative=False)
tot, pipe = R.upstream_flow(np.zeros(R.n), 1.0), R.inflow(np.zeros(R.n), 1.0).sum()
check("water balance: flow through the filter exceeds the pipe inflow by the direct flow into the hole (5-40 %)", float(1.05 < tot / pipe < 1.40), 1.0, 0.0, relative=False)
Rt = PipeResponse(okamura_box(d=0.005), 0.09)
check("levee geometry (free surface, anisotropy): reciprocity", np.abs(Rt.G - Rt.G.T).max() / np.abs(Rt.G).max(), 0.0, 1e-8, relative=False)
big = okamura_box(d=0.010, Lx=0.76, Wy=0.24, D=0.12, x_exit=0.22, x_river=0.60, pipe_width=0.010)
check("geometric similarity: doubling every length multiplies N_min by 4 (scale law L^(-1/2))", fold(PipeResponse(big, 0.18))[0] / fold(Rt)[0], 4.0, 1e-6)

print("-- pipe + aquifer --")
fake = types.SimpleNamespace(n=1, G=np.array([[-0.8]]), g=np.array([0.05]), box=types.SimpleNamespace(d=0.01))
N_exact = 256 * 0.8 ** 3 * 0.01 ** 3 / (27 * 0.05 ** 4)
check("one-station system: fold vs closed form 256 |G|^3 d^3 / (27 g^4)", fold(fake)[0], N_exact, 1e-5)
Nmin, sol = fold(R)
check("box, l = 10 cm: fold (regression value, 5 mm cells)", Nmin, 0.8255, 5e-3)
check("no solution just below the fold", float(solve_pipe(R, 0.98 * Nmin, sol["h"]) is None), 1.0, 0.0, relative=False)
up = solve_pipe(R, 4.0 * Nmin, sol["h"])
check("pipe head never exceeds the upstream head (h_tip/H < 1)", float(up["h"][-1] < 1.0 and sol["h"][-1] < 1.0), 1.0, 0.0, relative=False)
Nm2, br = trace_branches(R, N_start_factor=50.0)
Ns = np.array([b["N"] for b in br]); kind = np.array([b["branch"] for b in br])
check("continuation through the fold: smallest N on the path = N_min", Ns.min(), Nmin, 2e-3)
check("continuation: two solutions at 4 N_min with different pipe head loss", float(kind.min() == 0 and kind.max() == 1 and Ns[kind == 1].max() > 4 * Nmin), 1.0, 0.0, relative=False)

print("-- pipe law --")
o = threshold_section(0.77, 16, 161, 41)
law = PipeLaw(o, 0.5)
Hf = fold_law(R, law, 3.0e-4)[0]
check("dimensional solver: fold head = (N_min Lambda / k)^(1/4)", Hf, head_from_N(Nmin, law.Lam, 3.0e-4), 1e-4)
check("tortuosity T = 1.2: fold head x T^(3/4)", fold_law(R, PipeLaw(o, 0.5, tortuosity=1.2), 3.0e-4)[0] / Hf, 1.2 ** 0.75, 1e-4)
lw = PipeLaw(o, 0.5, W0=0.007)
check("width fixed by the tip: slope continuous where the section becomes self-formed", lw.slope(np.array([lw.Q_x * 0.999]))[0][0] / lw.slope(np.array([lw.Q_x * 1.001]))[0][0], 1.0, 2e-3)
Q = np.array([0.02, 0.2, 0.7]) * lw.Q_x
check("width fixed by the tip: dS/dQ vs finite difference", (lw.slope(Q)[1] / ((lw.slope(Q * 1.0001)[0] - lw.slope(Q)[0]) / (Q * 1e-4))).mean(), 1.0, 1e-3)
check("width fixed by the tip raises the fold head", float(fold_law(R, lw, 3.0e-4)[0] > Hf), 1.0, 0.0, relative=False)
lc = PipeLaw(o, 0.5, W0=0.007, a_cap=0.0012)
check("depth cap: slope never below A / a_cap", lc.slope(np.array([1e-9, 1e-7, 1e-5, 1e-3]))[0].min(), lc.A / 0.0012, 1e-12)
check("depth cap raises (or keeps) the fold head", float(fold_law(R, lc, 3.0e-4)[0] >= fold_law(R, lw, 3.0e-4)[0] * (1 - 1e-6)), 1.0, 0.0, relative=False)
sw = solve_seepage_width(R, o, 0.5, 0.06, 3.0e-4, 2.5, W_min=0.004)
check("seepage-width pipe: W = Q_pipe / (k l i*) at the solution", sw[1], sw[0]["Q"][0] / (3.0e-4 * R.length * 2.5), 2e-4)

print("-- flow in a section of arbitrary shape; seepage through the bed; prescribed pipe heads --")
yy = np.linspace(-4.0, 4.0, 321)
check("general solver, rectangle 8 x 1: flow rate vs series solution", stokes_section(yy, np.ones_like(yy), 61)["q"], rectangle_flow_rate(8.0, 1.0), 5e-4)
of = threshold_section(0.7, 24, 241, 61)
yy = np.linspace(-of["half_width"], of["half_width"], 481)
aa = np.interp(np.abs(yy), of["y"], of["a"]); aa[0] = aa[-1] = 0.0
sg = stokes_section(yy, aa, 81)
check("general solver on the threshold section (mu = 0.7): discharge coefficient vs symmetric solver", sg["q"], of["chi"], 5e-4)
check("general solver on the threshold section: bed stress at the centre = tau_c", 2.0 * sg["dudn"][240], 1.0, 1e-3)
yy = np.linspace(0.0, 6.0, 361)
aa = 1.2 * np.sin(np.pi * yy / 6.0) ** 1.5 * (1.0 + 0.3 * yy / 6.0); aa[0] = aa[-1] = 0.0
check("general solver: a section and its mirror image carry the same flow", stokes_section(yy, aa, 61)["q"] / stokes_section(yy, aa[::-1], 61)["q"], 1.0, 1e-10)
os_ = threshold_section(0.1, 16, 161, 41, sigma=0.3)
check("seepage, mu = 0.1, sigma = 0.3: depth -> shallow limit (1 - sigma)", os_["a_max"], 0.7, 5e-3)
check("seepage, mu = 0.1, sigma = 0.3: width -> shallow limit pi/mu (unchanged)", 2.0 * os_["half_width"], np.pi / 0.1, 1e-2)
check("seepage, mu = 0.1, sigma = 0.3: discharge coefficient -> (1 - sigma)^3 / (9 mu)", os_["chi"], 0.7 ** 3 / 0.9, 3e-2)
os_ = threshold_section(0.77, 16, 161, 41, sigma=0.4)
check("seepage, mu = 0.77, sigma = 0.4: threshold condition residual at the nodes", os_["max_residual"], 0.0, 1e-8, relative=False)
check("seepage: slope of the bank at the roof satisfies cos(phi) - sigma = sin(phi)/mu", abs(np.gradient(os_["a"], os_["y"])[-3:]).mean(), edge_slope(0.77, 0.4), 3e-2)
check("seepage: bed stress at the centre = (1 - sigma) tau_c", os_["tau_bed"][0], 0.6, 1e-9)
hp = 0.2 + 0.3 * R.s_st / R.s_st[-1]
pp = prescribed_pipe(pol_box(), R.x_tip, lambda x: np.interp(x, R.x_st, hp), 0.010, 1.0)
check("prescribed pipe heads: inflow per station vs response matrices", np.abs(pp["q"] * 0.005 / R.inflow(hp, 1.0) - 1.0).max(), 0.0, 1e-6, relative=False)
check("prescribed pipe heads: water balance (upstream face = pipe + hole)", pp["Q_up"] / (pp["q"].sum() * 0.005 + pp["Q_hole"]), 1.0, 1e-8)
pc = prescribed_pipe(pol_box(), 0.48, lambda x: 0.2 + 0.6 * (x - 0.125) / 0.355, 0.020, 1.0)
check("prescribed pipe heads, pipe through the bed: water balance", pc["Q_up"] / (pc["q"].sum() * 0.005 + pc["Q_hole"]), 1.0, 1e-8)

print("-- pipe with seepage through its bed coupled to the sand; inertia; starting guess --")
from pipetheory.section import momentum_coefficient
from pipetheory.seepage import SeepageLaw, section_family, solve_pipe_seepage, fold_seepage
tab = section_family(0.7701)
check("section family, sigma = 0.9: depth -> exact limit 1 - sigma", tab[-1, 1], 0.1, 1e-2)
check("section family, sigma = 0.9: width -> pi/mu", tab[-1, 2], np.pi / 0.7701, 4e-2)
check("section family, sigma = 0.9: discharge coefficient -> (1 - sigma)^3/(9 mu)", tab[-1, 3], 1e-3 / (9 * 0.7701), 5e-2)
o77 = threshold_section(0.7701)
check("section family, sigma = 0: same as the section without seepage (chi; the table keeps 5 digits)", tab[0, 3], o77["chi"], 1e-5)
k_ = 3.0e-4
ls = SeepageLaw(0.7701, 0.5, k_, 0.99, 1.0)
Qs, qs = np.array([2e-7, 1e-6, 4e-6]), np.array([2e-5, 1e-5, 3e-6])
S_, sg_, W_, a_ = ls.local(Qs, qs)
check("local law with seepage: sigma = 2 q / (pi k i_f W) at the solution", np.abs(sg_ * np.pi * k_ * 0.99 * W_ / (2 * qs) - 1).max(), 0.0, 1e-6, relative=False)
check("local law with seepage: Q = chi(sigma) g S a_c^4 / nu at the solution", np.abs(ls.chi(sg_) * 9.81 * ls.A0 ** 4 / (1e-6 * S_ ** 3) / Qs - 1).max(), 0.0, 1e-9, relative=False)
check("local law with seepage: section_at(S, q) returns the same sigma and discharge", np.abs(np.r_[ls.section_at(S_, qs)[0] / sg_, ls.section_at(S_, qs)[3] / Qs] - 1).max(), 0.0, 1e-6, relative=False)
ls.strength = 0.0
lp = PipeLaw(o77, 0.5)
check("local law, seepage force switched off: slope of the pipe law", np.abs(ls.local(Qs, qs)[0] / lp.slope(Qs)[0] - 1).max(), 0.0, 3e-6, relative=False)
Hf0, sf0 = fold_law(R, lp, k_)
ss = solve_pipe_seepage(R, ls, 2.0 * Hf0, solve_pipe_law(R, lp, 2.0 * Hf0, k_, sf0["h"] * 2.0)["h"])
Hfs, sfs = fold_seepage(R, ls, 2.0 * Hf0, ss["h"])
check("coupled problem, seepage force switched off: head at the fold of the pipe law", Hfs, Hf0, 5e-4)
ls.strength = 1.0
for f_ in (0.25, 0.5, 0.75, 1.0):
    ls.strength = f_
    ss = solve_pipe_seepage(R, ls, 2.0 * Hf0, ss["h"])
Hf1, sf1 = fold_seepage(R, ls, 2.0 * Hf0, ss["h"])
check("coupled problem with seepage: the head at the fold is lower (same flat-bed threshold stress)", float(Hf1 < Hf0), 1.0, 0.0, relative=False)
check("coupled problem with seepage: water balance, discharge at the exit = sum of the inflows", sf1["Q"][0] / (sf1["q"].sum() * 0.005), 1.0, 1e-10)
check("coupled problem with seepage: sigma at mid-pipe = 2 q / (pi k i_f W)", sf1["sigma"][9] * np.pi * k_ * 0.99 * sf1["W"][9] / (2 * sf1["q"][9]), 1.0, 1e-6)
check("coupled problem with seepage: heads = integral of the slope", np.abs(sf1["h"] - np.cumsum(R.ds * sf1["S"])).max() / sf1["h"][-1], 0.0, 1e-8, relative=False)
sc_ = np.linspace(0.0, 1.0, 241)
ac_ = np.cos(0.5 * np.pi * sc_); ac_[-1] = 0.0
check("momentum coefficient of the shallow cosine section -> 36/25", momentum_coefficient(dict(a=ac_, half_width=40.0)), 1.44, 1e-3)
li = SeepageLaw(0.7701, 0.5, k_, strength=0.0, inertia=1.4, shape=o77["area"] / (2 * o77["half_width"] * o77["a_max"]))
Si = li.local(Qs, qs)
area_ = o77["area"] * (li.A0 / lp.slope(Qs)[0]) ** 2
dM = (1.4 * (Qs * 1.0001) ** 2 / (area_ * 1.0001 ** (2.0 / 3.0)) - 1.4 * Qs ** 2 / area_) / (Qs * 1e-4) * qs      # d(beta Q^2/area)/ds, area ~ Q^(2/3)
check("inertial term = d(beta Q^2 / area)/ds / (g area) for a section that grows as Q^(2/3)", np.abs((Si[0] - lp.slope(Qs)[0]) / (dM / (9.81 * area_)) - 1).max(), 0.0, 1e-3, relative=False)
og = threshold_section(0.77, 16, 161, 41, init=(1.5, 1.0))
check("free boundary: same solution from a starting guess half as wide again", og["chi"], threshold_section(0.77, 16, 161, 41)["chi"], 1e-9)

print("-- weak inertia: wide pipe (exact first order), similarity solution, self-formed coefficient --")
from fractions import Fraction as Fr
from pipetheory.inertia import wide_pipe_coefficients, wide_threshold_pipe, similarity_one_wall, self_formed_coefficient
cw = wide_pipe_coefficients()
check("wide pipe: coefficients 81/35, -54/35 (gradient), -19/35, -6/35 (bed), 16/35, -6/35 (roof), exact",
      float(cw == dict(S_V=Fr(81, 35), S_h=Fr(-54, 35), bed_V=Fr(-19, 35), bed_h=Fr(-6, 35), roof_V=Fr(16, 35), roof_h=Fr(-6, 35), beta0=Fr(6, 5))), 1.0, 0.0, relative=False)
check("wide pipe: gradient = wall stresses + growth of the momentum flux, identically",
      float(cw["S_V"] - cw["bed_V"] - cw["roof_V"] - 2 * cw["beta0"] == 0 and cw["S_h"] - cw["bed_h"] - cw["roof_h"] + cw["beta0"] == 0), 1.0, 0.0, relative=False)
wt = wide_threshold_pipe()
check("wide threshold pipe: slope 2/7, depth -11/210, exact / momentum-integral closure 40/21",
      float((wt["slope"], wt["depth"], wt["ratio"]) == (Fr(2, 7), Fr(-11, 210), Fr(40, 21))), 1.0, 0.0, relative=False)
Ks, bs, rs = similarity_one_wall(0.01)
check("similarity solution (Navier-Stokes), R = 0.01: gradient correction / R -> 27/140", (Ks - 1) / 0.01, 27 / 140, 1e-3)
check("similarity solution, R = 0.01: bed stress correction / R -> -19/210", (bs - 1) / 0.01, -19 / 210, 2e-3)
check("similarity solution, R = 0.01: roof stress correction / R -> 8/105", (rs - 1) / 0.01, 8 / 105, 1e-3)
Ks, bs, rs = similarity_one_wall(2.0)
check("similarity solution, R = 2: first order gives the gradient to 1.5 %", (1 + 27 / 140 * 2.0) / Ks, 1.0, 1.5e-2)
check("self-formed section, mu = 0.77: c = 4 beta chi / (3 m^2)", self_formed_coefficient(o77), 0.0289, 1e-2)

print("-- weak inertia: self-formed pipe in the shallow limit --")
from pipetheory.inertia import shallow_self_formed, edge_exponent
ss_, su_ = shallow_self_formed("strip"), shallow_self_formed("uniform")
check("shallow limit, seepage of a strip: momentum identity, integral of cos^2 (T + V) = 16/15", ss_["check"], 16 / 15, 1e-12)
check("shallow limit, uniform inflow: the same identity", su_["check"], 16 / 15, 1e-12)
check("shallow limit: kinetic-energy identity, integral of cos^4 (2T + V) = 8/7 (strip)", ss_["I_c"], 8 / 7, 1e-12)
check("shallow limit: the same identity (uniform inflow)", su_["I_c"], 8 / 7, 1e-12)
check("shallow limit, strip: s_hat (regression value) within 0.5 % of the momentum-integral closure 8/25", ss_["s_hat"], 0.319053, 1e-5)
check("shallow limit, strip: exact / closure", ss_["ratio"], 1.0, 5e-3)
check("shallow limit, uniform inflow: s_hat (regression value)", su_["s_hat"], 0.555954, 1e-5)
check("shallow limit: quadrature converged (s_hat with 200 and 800 nodes)", shallow_self_formed("strip", 200, 48)["s_hat"], shallow_self_formed("strip", 800, 160)["s_hat"], 1e-8)
check("shallow limit: power-law inflow with p = 1/2 reproduces the strip", shallow_self_formed(0.5, 1600, 200)["s_hat"], ss_["s_hat"], 1e-5)
check("shallow limit: power-law inflow with p = 0 reproduces the uniform inflow", shallow_self_formed(0.0, 400, 96)["s_hat"], su_["s_hat"], 1e-8)
check("edge exponent of the seepage: 1/2 for a strip, 0.368 for mu = 0.77", edge_exponent(0.0) + edge_exponent(0.7701), 0.5 + 0.368, 1e-3)
check("shallow limit with the edge exponent of mu = 0.77: exact / closure (regression value)", shallow_self_formed(edge_exponent(0.7701), 1600, 200)["ratio"], 1.215, 2e-3)

print(f"\n{sum(results)} of {len(results)} checks passed in {time.time() - t0:.0f} s")
sys.exit(0 if all(results) else 1)
