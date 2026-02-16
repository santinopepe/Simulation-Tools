import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from assimulo.problem import Explicit_Problem

from importlib.machinery import SourceFileLoader

_loader = SourceFileLoader(
    "second_order",
    os.path.join(os.path.dirname(__file__), "2nd_Order.py"),
)
second_order = _loader.load_module()
NewmarkBeta = second_order.NewmarkBeta


def elastic_pendulum_acc(k: float):
    """
    Returns a(t, q, v) for the elastic pendulum.
    q = [x, y], v = [vx, vy]
    """

    def acc(t, q, v):
        x, y = q
        r = np.hypot(x, y)
        lam = 0.0 if r == 0.0 else k * (r - 1.0) / r
        ax = -x * lam
        ay = -y * lam - 1.0
        return np.array([ax, ay], dtype=float)

    return acc


def solve_newmark(k=1.0, t_end=20.0, h=0.01, beta=0.0, gamma=0.5, y0=None, nout=2000):
    """
    Integrate the elastic pendulum with explicit Newmark-β.
    State vector ordering: [x, y, vx, vy].
    """
    if y0 is None:
        y0 = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)

    acc = elastic_pendulum_acc(k)

    def rhs_first_order(t, y):
        q = y[:2]
        v = y[2:]
        a = acc(t, q, v)
        return np.hstack((v, a))

    prob = Explicit_Problem(rhs_first_order, y0, name=f"Elastic pendulum Newmark (k={k})")
    prob.acc = acc  # let Second_Order use acceleration directly



    sim = NewmarkBeta(prob, beta=beta, gamma=gamma)
    sim.h = h

    t, y = sim.simulate(t_end, nout)
    return t, y, sim


def compare_with_bdf(k=1.0, t_end=30.0, h=0.01, nout=3000):
    """
    Compare Newmark-β with existing BDF2/BDF4 solvers (located in Project1).
    Saves trajectory and position-time plots.
    """
    project1_path = os.path.join(os.path.dirname(__file__), "..", "Project1")
    sys.path.insert(0, project1_path)
    from BDF2_Assimulo import BDF_2
    from BDF4_Assimulo import BDF_4

    def elastic_rhs(t, y):
        x, y_pos, vx, vy = y
        r = np.hypot(x, y_pos)
        lam = 0.0 if r == 0.0 else k * (r - 1.0) / r
        return np.array([vx, vy, -x * lam, -y_pos * lam - 1.0], dtype=float)

    y0 = np.array([5.0, 5, 0.0, 0.0], dtype=float)

    # Newmark
    t_nm, y_nm, _ = solve_newmark(k=k, t_end=t_end, h=h, y0=y0, nout=nout)

    # BDF2
    prob2 = Explicit_Problem(elastic_rhs, y0, name=f"Elastic pendulum BDF2 (k={k})")
    sim2 = BDF_2(prob2)
    sim2.h = h
    t_bdf2, y_bdf2 = sim2.simulate(t_end, nout)

    # BDF4
    prob4 = Explicit_Problem(elastic_rhs, y0, name=f"Elastic pendulum BDF4 (k={k})")
    sim4 = BDF_4(prob4)
    sim4.h = h
    t_bdf4, y_bdf4 = sim4.simulate(t_end, nout)

    # --- plots ---
    def plot_trajectory():
        plt.figure(figsize=(6, 5))
        plt.plot(y_nm[:, 0], y_nm[:, 1], label="Newmark-β", lw=2)
        plt.plot(y_bdf2[:, 0], y_bdf2[:, 1], "--", label="BDF2", lw=1.5)
        plt.plot(y_bdf4[:, 0], y_bdf4[:, 1], ":", label="BDF4", lw=1.5)
        plt.xlabel("x"); plt.ylabel("y"); plt.title(f"Trajectory (k={k})")
        plt.axis("equal"); plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig("traj_newmark_vs_bdf.png", dpi=180)
        print("Saved traj_newmark_vs_bdf.png")

    def plot_time_series():
        plt.figure(figsize=(8, 4.5))
        for lbl, tvec, yvec, style in [
            ("x – Newmark", t_nm, y_nm[:, 0], "b-"),
            ("y – Newmark", t_nm, y_nm[:, 1], "b--"),
            ("x – BDF2", t_bdf2, y_bdf2[:, 0], "r-"),
            ("y – BDF2", t_bdf2, y_bdf2[:, 1], "r--"),
            ("x – BDF4", t_bdf4, y_bdf4[:, 0], "g-"),
            ("y – BDF4", t_bdf4, y_bdf4[:, 1], "g--"),
        ]:
            plt.plot(tvec, yvec, style, label=lbl, lw=1.3)
        plt.xlabel("t"); plt.ylabel("position")
        plt.title(f"Positions vs time (k={k})")
        plt.grid(True, alpha=0.3); plt.legend(fontsize=9, ncol=2)
        plt.tight_layout(); plt.savefig("pos_newmark_vs_bdf.png", dpi=180)
        print("Saved pos_newmark_vs_bdf.png")

    plot_trajectory()
    plot_time_series()

    return dict(
        newmark=dict(t=t_nm, y=y_nm),
        bdf2=dict(t=t_bdf2, y=y_bdf2),
        bdf4=dict(t=t_bdf4, y=y_bdf4),
    )


if __name__ == "__main__":
    compare_with_bdf(k=1, t_end=30.0, h=0.002, nout=4000)
