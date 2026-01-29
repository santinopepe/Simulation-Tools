import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode, ExplicitEuler


def ElasticPendulumRHS(k: float):
    """
    Returns rhs(t, y) (Right hand side) for the project. 
    We use lambda ecuations that are on stated. 
    """
    def rhs(t, y):
        y1, y2, y3, y4 = y
        r = np.hypot(y1, y2) # √(x² + y²)

        if r == 0.0: # Security check so there is no 0 division. 
            lam = 0.0
        else:
            lam = k * ((r - 1.0) / r)  # lambda(y1,y2) 

        dy1 = y3
        dy2 = y4
        dy3 = -y1 * lam
        dy4 = -y2 * lam - 1.0
        return np.array([dy1, dy2, dy3, dy4], dtype=float)
    return rhs


def Solver(k=1.0, t_end=20.0, rtol=1e-8, atol=1e-10, nout=2000, y0=None):
    rhs = ElasticPendulumRHS(k)

    if y0 is None:
        # Initial conditions
        y0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    prob = Explicit_Problem(rhs, y0, name=f"Elastic pendulum (k={k})")

    sim = CVode(prob)
    sim.reset()
    sim.discr = 'BDF'      # CVode: linear multistep method BDF
    sim.iter  = 'Newton'   # CVode: Newton nonlinear solver
    sim.rtol = rtol
    sim.atol = atol

    t, y = sim.simulate(t_end, nout)
    return t, y, sim


def Plot(t, y, title=""):
    y1, y2, y3, y4 = y.T

    plt.figure()
    plt.plot(t, y1, label="y1")
    plt.plot(t, y2, label="y2")
    plt.xlabel("t")
    plt.ylabel("position")
    plt.title("Positions vs time")
    plt.legend()

    plt.figure()
    plt.plot(y1, y2)
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title("Trajectory (y1,y2)")
    plt.axis("equal")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


# ---------- Diagnostics helpers ----------

def equilibrium_state(k: float):
    """
    Static equilibrium (x=0, vx=vy=0) with y < 0:
      k(r-1) = 1  -> r = 1 + 1/k, y = -r
    """
    r_eq = 1.0 + 1.0 / k
    return np.array([0.0, -r_eq, 0.0, 0.0], dtype=float)


def energy(y: np.ndarray, k: float):
    """
    Total energy (up to a constant):
      E = 1/2(v^2) + y2 + 1/2 k (r-1)^2
    """
    y1, y2, y3, y4 = y.T
    r = np.hypot(y1, y2)
    K = 0.5 * (y3**2 + y4**2)
    Vg = y2
    Vs = 0.5 * k * (r - 1.0)**2
    return K + Vg + Vs


def rel_drift(E: np.ndarray):
    E0 = E[0]
    denom = max(1.0, abs(E0))
    return (E.max() - E.min()) / denom


def max_abs(a: np.ndarray):
    return float(np.max(np.abs(a)))


def try_set_method(sim, discr=None, iter=None):
    """
    Some Assimulo/CVode builds accept sim.discr / sim.iter; others may differ.
    This helper won't crash your tests if a field isn't supported.
    """
    if discr is not None:
        try:
            sim.discr = discr
        except Exception:
            pass
    if iter is not None:
        try:
            sim.iter = iter
        except Exception:
            pass


# ---------- Tests ----------

def test_equilibrium(k=1.0, t_end=50.0, nout=1000, rtol=1e-8, atol=1e-10, make_plots=False):
    y0 = equilibrium_state(k)
    t, y, sim = Solver(k=k, t_end=t_end, rtol=rtol, atol=atol, nout=nout, y0=y0)

    dev = np.max(np.linalg.norm(y - y0, axis=1))
    if make_plots:
        plt.figure()
        plt.plot(t, np.linalg.norm(y - y0, axis=1))
        plt.xlabel("t")
        plt.ylabel("||y(t)-y_eq||")
        plt.title(f"Equilibrium deviation (k={k})")
        plt.show()

    return {"equilibrium_max_dev": float(dev), "y0": y0}


def test_energy_drift(k=1.0, t_end=100.0, nout=2000, rtol=1e-8, atol=1e-10, make_plots=False, y0=None):
    if y0 is None:
        # A non-trivial IC (avoid being exactly at equilibrium)
        y0 = np.array([0.5, -1.0, 0.0, 0.0], dtype=float)

    t, y, sim = Solver(k=k, t_end=t_end, rtol=rtol, atol=atol, nout=nout, y0=y0)
    E = energy(y, k)
    drift = rel_drift(E)

    if make_plots:
        plt.figure()
        plt.plot(t, E)
        plt.xlabel("t")
        plt.ylabel("E(t)")
        plt.title(f"Energy vs time (k={k})")
        plt.show()

    return {"energy_rel_drift": float(drift), "E0": float(E[0]), "Emin": float(E.min()), "Emax": float(E.max())}


def test_symmetry(k=5.0, t_end=50.0, nout=1500, rtol=1e-8, atol=1e-10):
    # Mirror in x: (x, y, vx, vy) -> (-x, y, -vx, vy)
    y0A = np.array([ 0.5, -1.0, 0.2, 0.0], dtype=float)
    y0B = np.array([-0.5, -1.0,-0.2, 0.0], dtype=float)

    tA, yA, _ = Solver(k=k, t_end=t_end, rtol=rtol, atol=atol, nout=nout, y0=y0A)
    tB, yB, _ = Solver(k=k, t_end=t_end, rtol=rtol, atol=atol, nout=nout, y0=y0B)

    # compare at same output points (we used same nout and t_end, so t arrays should align)
    err_y1 = max_abs(yA[:, 0] + yB[:, 0])
    err_y2 = max_abs(yA[:, 1] - yB[:, 1])
    err_y3 = max_abs(yA[:, 2] + yB[:, 2])
    err_y4 = max_abs(yA[:, 3] - yB[:, 3])

    return {
        "sym_err_y1": err_y1,
        "sym_err_y2": err_y2,
        "sym_err_y3": err_y3,
        "sym_err_y4": err_y4
    }


def test_tolerance_convergence(k=1.0, t_end=50.0, nout=2000):
    # Reference
    t_ref, y_ref, _ = Solver(k=k, t_end=t_end, rtol=1e-10, atol=1e-12, nout=nout,
                             y0=np.array([0.5, -1.0, 0.0, 0.0], dtype=float))

    results = []
    for rtol in [1e-6, 1e-8]:
        atol = rtol * 1e-2
        t, y, _ = Solver(k=k, t_end=t_end, rtol=rtol, atol=atol, nout=nout,
                         y0=np.array([0.5, -1.0, 0.0, 0.0], dtype=float))
        final_err = float(np.linalg.norm(y[-1] - y_ref[-1], ord=np.inf))
        traj_err = float(np.max(np.linalg.norm(y - y_ref, axis=1)))
        results.append({"rtol": rtol, "atol": atol, "final_inf_err": final_err, "max_traj_err": traj_err})

    return {"reference_tol": (1e-10, 1e-12), "comparisons": results}


def test_k_sweep(k_values=(1, 10, 100, 1000), t_end=30.0, nout=1200, rtol=1e-8, atol=1e-10, make_plots=False):
    y0 = np.array([0.5, -1.0, 0.0, 0.0], dtype=float)
    out = []
    for k in k_values:
        t, y, sim = Solver(k=k, t_end=t_end, rtol=rtol, atol=atol, nout=nout, y0=y0)
        r = np.hypot(y[:,0], y[:,1])
        out.append({
            "k": float(k),
            "r_min": float(r.min()),
            "r_max": float(r.max()),
            "energy_rel_drift": float(rel_drift(energy(y, k)))
        })

        if make_plots:
            plt.figure()
            plt.plot(y[:,0], y[:,1])
            plt.xlabel("y1")
            plt.ylabel("y2")
            plt.title(f"Trajectory (k={k})")
            plt.axis("equal")
            plt.show()

    return out



def run_tests(make_plots=False):
    print("\n==== Elastic Pendulum: Task 1 Test Suite ====\n")

    # 1) Equilibrium test
    eq = test_equilibrium(k=1.0, t_end=50.0, nout=1000, rtol=1e-8, atol=1e-10, make_plots=make_plots)
    print("[1] Equilibrium test (k=1):")
    print("    max deviation ||y - y_eq|| =", eq["equilibrium_max_dev"])
    print("    y_eq =", eq["y0"])

    # 2) Energy drift test
    en = test_energy_drift(k=1.0, t_end=100.0, nout=2000, rtol=1e-8, atol=1e-10, make_plots=make_plots)
    print("\n[2] Energy drift test (k=1):")
    print("    relative drift =", en["energy_rel_drift"])
    print("    E0/Emin/Emax =", en["E0"], en["Emin"], en["Emax"])

    # 3) Symmetry test
    sym = test_symmetry(k=5.0, t_end=50.0, nout=1500, rtol=1e-8, atol=1e-10)
    print("\n[3] Symmetry (mirror) test (k=5):")
    print("    max |y1A + y1B| =", sym["sym_err_y1"])
    print("    max |y2A - y2B| =", sym["sym_err_y2"])
    print("    max |y3A + y3B| =", sym["sym_err_y3"])
    print("    max |y4A - y4B| =", sym["sym_err_y4"])

    # 4) Tolerance convergence
    conv = test_tolerance_convergence(k=1.0, t_end=50.0, nout=2000)
    print("\n[4] Tolerance convergence (k=1, ref rtol=1e-10 atol=1e-12):")
    for row in conv["comparisons"]:
        print(f"    rtol={row['rtol']:.0e}, atol={row['atol']:.0e} -> "
              f"final_inf_err={row['final_inf_err']:.3e}, max_traj_err={row['max_traj_err']:.3e}")

    # 5) k sweep
    ks = test_k_sweep(k_values=(1, 10, 100, 1000), t_end=30.0, nout=1200, rtol=1e-8, atol=1e-10, make_plots=make_plots)
    print("\n[5] k-sweep summary (same IC):")
    for row in ks:
        print(f"    k={int(row['k']):4d}: r in [{row['r_min']:.4f}, {row['r_max']:.4f}], "
              f"energy drift={row['energy_rel_drift']:.3e}")

    print("\n==== Done ====\n")


if __name__ == "__main__":
    # run_tests(make_plots=True);
    t, y, sim = Solver(k=1, t_end=100.0, rtol=1e-8, atol=1e-10, nout=660)
    Plot(t, y, title="")

