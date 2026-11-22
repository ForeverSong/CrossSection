"""Large-scale test: how wide a strip of the bed does the pipe drain, and does it matter?

With the depth held at 4 d50 the pipe has to be wide to carry its discharge: a laminar slot of that depth at the
threshold gradient S* needs a width  W = 12 nu Q / (g S* a^3),  about 0.7 m for 20 mL/s.  The earlier large-scale
runs drained the bed through a strip 5 cm wide.  Here the strip is 0.05 to 1.2 m wide (1.2 m is the whole bed) (uniform along the
pipe) and the same quantities are evaluated:
  - loading of the tip K at the measured critical head (taken as 1.28 m: 1.44 m less the entry loss that the model
    does not have, see large_scale_data.py), and the head needed for the small-scale value K_c;
  - gradient over the first 0.8 m ahead of the tip (measured 0.29-0.40 while the tip advances);
  - discharge, and the slot width that this discharge needs, to see which strip width is consistent with itself.

usage:  python run_large_scale_width.py [--out ../tmp] [--widths 0.05,0.1,0.2,0.4,0.8,1.2] [--lengths 0.8,1.65,3.0] [--kc 0.086]
"""
import argparse
import os
import time
import numpy as np
from pipetheory.aquifer import Box, PipeResponse
from pipetheory.coupled import solve_pipe_law
from pipetheory.pipelaw import PipeLaw
from pipetheory.section import threshold_section
from pipe_width_data import read_csv
from run_tip_criterion import measures

HERE = os.path.dirname(os.path.abspath(__file__))
K_BED, NU, D50, CAP, H_CRIT = 9.0e-5, 1.52e-6, 0.185e-3, 4.0, 1.28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tmp"))
    ap.add_argument("--widths", default="0.05,0.1,0.2,0.4,0.8,1.2")
    ap.add_argument("--lengths", default="0.8,1.65,3.0")
    ap.add_argument("--kc", type=float, default=0.086)
    ap.add_argument("--cap", type=float, default=CAP, help="depth ceiling in grain diameters (4: read from small-scale depths; 5.28: depth at which the self-formed width equals the tip width)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--taper", action="store_true", help="strip as wide as the slot needs at each station (iterated), instead of uniform strips")
    a = ap.parse_args()
    sd = next(r for r in read_csv("pol2022_sands.csv") if r["sand"] == "FPH")
    sec = threshold_section(np.tan(np.radians(float(sd["repose_deg"]))), 16, 161, 41)
    tau = float(sd["tau_eq_Pa"]) / (sec["a_max"] * sec["aavg_over_amax"])
    law = PipeLaw(sec, tau, W0=30 * D50, a_cap=a.cap * D50, nu=NU)
    need = lambda q: 12 * NU * q / (9.81 * law.S_floor * (a.cap * D50) ** 3)
    out = open(os.path.join(a.out, ("paper1_large_scale_taper" if a.taper else "paper1_large_scale_width") + a.tag + ".txt"), "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()

    if a.taper:
        taper(a, law, need, say); out.close(); return
    say(f"FPH sand, bed 9.5 x 1.2 x 1.0 m, grid 2.5 cm; threshold pipe with depth <= {a.cap:g} d50 (gradient {law.S_floor:.3f}); head {H_CRIT} m")
    say("length  strip   Q exit  Q mid   width needed [m]   tip head   gradient over   K at this head   head for K_c    time")
    say(" [m]     [m]    [mL/s]  [mL/s]   exit     mid        [m]      0.8 m ahead      [m^1/2]         [m]          [s]")
    for l in [float(x) for x in a.lengths.split(",")]:
        for fp in [float(x) for x in a.widths.split(",")]:
            t0 = time.time()
            R = PipeResponse(Box(Lx=9.5, Wy=1.2, D=1.0, x_exit=1.3, x_river=8.5, d=0.025, pipe_width=fp, hole_radius=0.0065), l, with_fields=True, rtol=1e-8)

            def state(H, h0=None):
                sol = solve_pipe_law(R, law, H, K_BED, h0)
                return None if sol is None else (sol, measures(R, sol, H, K_BED, D50))
            s1 = state(H_CRIT)
            if s1 is None:
                say(f"{l:5.2f}   {fp:5.2f}   no solution at {H_CRIT} m"); continue
            sol, m = s1
            x, roof = R.roof_head(sol["h"], H_CRIT)
            g08 = (np.interp(R.x_tip + 0.8, x, roof) - sol["h"][-1]) / 0.8
            Ha, Ka, Hb, Hreq = H_CRIT, m["K"], 0.8 * H_CRIT, float("nan")
            for _ in range(5):                                   # secant on the head; K is nearly linear in it
                sb = state(Hb, sol["h"] * Hb / H_CRIT)
                if sb is None:                                   # below the fold of the bed criterion
                    Hb = 0.5 * (Ha + Hb); continue
                Kb = sb[1]["K"]; Hreq = Hb
                if abs(Kb - a.kc) < 2e-4 or abs(Kb - Ka) < 1e-9:
                    break
                Ha, Ka, Hb = Hb, Kb, Hb + (a.kc - Kb) * (Hb - Ha) / (Kb - Ka)
            qm = sol["Q"][R.n // 2]
            say(f"{l:5.2f}   {fp:5.2f}   {sol['Q'][0] * 1e6:5.1f}   {qm * 1e6:5.1f}    {need(sol['Q'][0]):5.2f}   {need(qm):5.2f}      {sol['h'][-1]:5.3f}       {g08:5.2f}           {m['K']:5.3f}"
                f"          {Hreq:5.2f}       {time.time() - t0:5.0f}")
    out.close()


def head_required(R, law, kc, hi=1.8):
    """Least head at which a threshold pipe exists and its tip is loaded to kc: bisection from above, a head with no
    solution (below the fold of the bed criterion) counting as too low.  Returns (head, solution, K, 'tip' or 'bed')."""
    sol = solve_pipe_law(R, law, hi, K_BED, law.S_floor * R.s_st)
    if sol is None:
        sol = solve_pipe_law(R, law, hi, K_BED)
    if sol is None:
        return None
    best = (hi, sol, measures(R, sol, hi, K_BED, D50)["K"], "tip")
    lo = 0.1
    for _ in range(12):
        H = 0.5 * (lo + best[0])
        sb = solve_pipe_law(R, law, H, K_BED, best[1]["h"] * H / best[0])
        if sb is None:
            lo, best = H, best[:3] + ("bed",)
            continue
        K = measures(R, sb, H, K_BED, D50)["K"]
        if K >= kc:
            best = (H, sb, K, "tip")
        else:
            lo, best = H, best[:3] + ("tip",)
    return best


def taper(a, law, need, say):
    """Strip width at each station = width the depth-limited slot needs for the local pipe discharge at the head that
    loads the tip to K_c (not below two cells, not above the bed).  Fixed point on (widths, head), under-relaxed."""
    say(f"Strip as wide as the pipe needs at each station; least head at which the tip is loaded to K_c = {a.kc} m^1/2")
    say("length   head [m] by iteration                         held by   width [m] 0.1, 0.25, 0.5, 1 m behind the tip, mid-pipe, exit   Q exit [mL/s]   gradient 0.8 m ahead")
    box = Box(Lx=9.5, Wy=1.2, D=1.0, x_exit=1.3, x_river=8.5, d=0.025, pipe_width=0.05, hole_radius=0.0065)
    for l in [float(x) for x in a.lengths.split(",")]:
        w, hist, r = None, [], None
        for it in range(8):
            R = PipeResponse(box, l, with_fields=True, rtol=1e-8, width=w)
            r = head_required(R, law, a.kc)
            if r is None:
                break
            H, sol = r[0], r[1]
            hist.append(H)
            w_new = np.clip(need(sol["Q"]), 0.05, box.Wy)
            w = w_new if w is None else 0.5 * (w + w_new)
            if it > 1 and abs(hist[-1] - hist[-2]) < 0.004:
                break
        if r is None:
            say(f"{l:5.2f}    no solution"); continue
        back = R.x_tip - R.x_st
        x, roof = R.roof_head(sol["h"], H)
        g08 = (np.interp(R.x_tip + 0.8, x, roof) - sol["h"][-1]) / 0.8
        wd = 2 * R.jws * box.d
        pick = [wd[np.argmin(abs(back - b))] for b in (0.1, 0.25, 0.5, 1.0) if b < l] + [wd[R.n // 2], wd[0]]
        say(f"{l:5.2f}    " + " ".join(f"{h:.2f}" for h in hist).ljust(44) + f"  {r[3]:4s}     " + " ".join(f"{v:.2f}" for v in pick).ljust(48) + f"  {sol['Q'][0] * 1e6:5.1f}          {g08:.2f}")


if __name__ == "__main__":
    main()
