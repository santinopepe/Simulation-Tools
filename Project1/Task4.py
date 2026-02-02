import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode

# Elastic pendulum RHS + analytic Jacobian (helps Newton/BDF at large k)
def elastic_pendulum(k: float):
    def rhs(t, y):
        y1, y2, y3, y4 = y
        r = np.hypot(y1, y2)
        if r == 0.0:
            lam = 0.0
        else:
            lam = k * (r - 1.0) / r
        return np.array([y3, y4, -y1 * lam, -y2 * lam - 1.0], float)

    def jac(t, y):
        y1, y2, y3, y4 = y
        r = np.hypot(y1, y2)
        if r == 0.0:
            return np.zeros((4, 4))
        lam = k * (r - 1.0) / r
        lam_y1 = k * y1 / (r**3)
        lam_y2 = k * y2 / (r**3)

        J = np.zeros((4, 4))
        J[0, 2] = 1.0
        J[1, 3] = 1.0
        J[2, 0] = -lam - y1 * lam_y1
        J[2, 1] = -y1 * lam_y2
        J[3, 0] = -y2 * lam_y1
        J[3, 1] = -lam - y2 * lam_y2
        return J
    return rhs, jac

def run_cvode(k, t_end, y0, opts):
    rhs, jac = elastic_pendulum(k)
    prob = Explicit_Problem(rhs, y0, name=f"Elastic pendulum k={k}")
    prob.jac = jac

    # Branch: allow pure Explicit Euler (Assimulo) for comparison
    if opts.get("solver", "CVode") == "ExplicitEuler":
        from assimulo.solvers import ExplicitEuler
        sim = ExplicitEuler(prob)
        sim.h = opts.get("h", 1e-3)  # fixed step
        t, y = sim.simulate(t_end, opts.get("nout", int(t_end / sim.h)))
        stats = dict(sim.statistics)
        return t, y, stats

    sim = CVode(prob)
    sim.reset()
    sim.discr = opts.get("discr", "BDF")        # 'BDF' or 'Adams'
    iter_opt = opts.get("iter", "Newton")
    if iter_opt.lower().startswith("func"):
        iter_opt = "FixedPoint"
    sim.iter  = iter_opt
    sim.maxord = opts.get("maxord", 5)          # 1–5 for BDF, 1–12 for Adams
    sim.rtol = opts.get("rtol", 1e-6)
    sim.atol = opts.get("atol", 1e-8)
    sim.maxh = opts.get("maxh", 0.0)            # 0 => no limit
    sim.inith = opts.get("inith", 0.0)          # 0 => CVODE chooses

    t, y = sim.simulate(t_end, opts.get("nout", 2000))
    stats = dict(sim.statistics)  # copy for later inspection
    return t, y, stats

def sweep():
    # Low vs high oscillation scenarios
    cases = [
        dict(k=1.0,   y0=np.array([1.0, 1.0, 0.0, 0.0]), t_end=30.0, label="k = 1"),
        dict(k=1e1,   y0=np.array([1.0, 1.0, 0.0, 0.0]), t_end=30.0, label="k = 10"),
        dict(k=1e2,   y0=np.array([1.0, 1.0, 0.0, 0.0]), t_end=30.0, label="k = 100"),
        dict(k=1e3,   y0=np.array([1.0, 1.0, 0.0, 0.0]), t_end=30.0, label="k = 1000"),
    ]

    # Solver setups to compare
    setups = [
        # CVODE with BDF order forced to 2 (matches “BDF2 with Newton”)
        dict(name="CVODE BDF2 Newton", solver="CVode", discr="BDF", iter="Newton",
             maxord=2, rtol=1e-2, atol=1e-5),
        # CVODE with BDF order capped at 4 (approx “BDF4 with Newton”)
        dict(name="CVODE BDF4 Newton", solver="CVode", discr="BDF", iter="Newton",
             maxord=4, rtol=1e-2, atol=1e-5),
        # Explicit Euler baseline (Assimulo explicit solver, fixed step)
        dict(name="Explicit Euler", solver="ExplicitEuler", h=0.00001, nout=30000),
    ]

    for case in cases:
        print(f"\n=== k={case['k']} ({case['label']}) ===")

        # One time-series plot and one trajectory plot per case
        fig_ts, ax_ts = plt.subplots(figsize=(6, 4))
        fig_tr, ax_tr = plt.subplots(figsize=(4.5, 4.5))
        ax_ts.set_title(f"Positions vs time – k={case['k']}")
        ax_ts.set_xlabel("t"); ax_ts.set_ylabel("position")
        ax_tr.set_title(f"Trajectory y1–y2 – k={case['k']}")
        ax_tr.set_xlabel("y1"); ax_tr.set_ylabel("y2")
        ax_tr.set_aspect("equal", adjustable="box")

        for s in setups:
            t, y, st = run_cvode(
                k=case["k"],
                t_end=case["t_end"],
                y0=case["y0"],
                opts={**s, "nout": 1200 if case["k"] < 1e5 else 800}
            )
            y1, y2 = y[:, 0], y[:, 1]
            print(f"{s['name']}: steps={st.get('nsteps')}, f-evals={st.get('nfevals')}, "
                  f"order_used={st.get('qlast')}, err_test_fails={st.get('netf')}")

            ax_ts.plot(t, y1, label=f"y1 – {s['name']}")
            ax_ts.plot(t, y2, linestyle="--", label=f"y2 – {s['name']}")
            ax_tr.plot(y1, y2, label=s['name'])

        ax_ts.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
        ax_tr.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
        fig_tr.tight_layout()
        fig_ts.tight_layout()
    plt.show()

if __name__ == "__main__":
    sweep()
