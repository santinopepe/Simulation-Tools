"""Run elastic pendulum with Newmark-β and HHT-α side by side.

Defaults match the existing NewmarkElastic setup but add HHT-α.
Plots are saved to PNGs; nothing is shown interactively.
"""

import argparse
import os

import matplotlib

# Avoid GUI pop‑ups
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from assimulo.problem import Explicit_Problem
from importlib.machinery import SourceFileLoader


def load_solvers():
    """Load NewmarkBeta and HHTAlpha from 2nd_Order.py without installing the module."""
    here = os.path.dirname(__file__)
    loader = SourceFileLoader("second_order", os.path.join(here, "2nd_Order.py"))
    mod = loader.load_module()
    return mod.NewmarkBeta, mod.HHTAlpha


def elastic_pendulum_acc(k: float):
    """Return acceleration function a(t, q, v) for elastic pendulum with unit mass, g=1, l0=1."""

    def acc(t, q, v):
        x, y = q
        r = np.hypot(x, y)
        lam = 0.0 if r == 0.0 else k * (r - 1.0) / r
        ax = -x * lam
        ay = -y * lam - 1.0
        return np.array([ax, ay], dtype=float)

    return acc


def build_problem(k: float, y0: np.ndarray):
    """Create an Explicit_Problem with both nonlinear acc and linearized M,C,K,f for MCK-based solvers."""
    acc = elastic_pendulum_acc(k)

    def rhs_first_order(t, y):
        q = y[:2]
        v = y[2:]
        a = acc(t, q, v)
        return np.hstack((v, a))

    prob = Explicit_Problem(rhs_first_order, y0, name=f"Elastic pendulum (k={k})")

    # Nonlinear acceleration (used in acc-mode solvers)
    prob.acc = acc

    # Linearized data for MCK-mode (required by HHT implementation here)
    prob.M = np.eye(2)
    prob.C = np.zeros((2, 2))
    prob.K = k * np.eye(2)
    prob.f = lambda t: np.array([0.0, -1.0])  # gravity

    return prob


def simulate_newmark(prob, h, t_end, nout, beta=0.0, gamma=0.5):
    NewmarkBeta, _ = load_solvers()
    sim = NewmarkBeta(prob, beta=beta, gamma=gamma)
    sim.h = h
    t, y = sim.simulate(t_end, nout)
    return t, y, sim


def simulate_hht(prob, h, t_end, nout, alpha=-0.05):
    _, HHTAlpha = load_solvers()
    sim = HHTAlpha(prob, alpha=alpha)
    sim.h = h
    t, y = sim.simulate(t_end, nout)
    return t, y, sim


def plot_results(t_nm, y_nm, t_hht, y_hht, k):
    plt.figure(figsize=(6, 5))
    plt.plot(y_nm[:, 0], y_nm[:, 1], label="Newmark-β", lw=2.0)
    plt.plot(y_hht[:, 0], y_hht[:, 1], "--", label="HHT-α", lw=1.6)
    plt.xlabel("x"); plt.ylabel("y"); plt.title(f"Trajectory (k={k})")
    plt.axis("equal"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("traj_newmark_vs_hht.png", dpi=180)
    print("Saved traj_newmark_vs_hht.png")

    plt.figure(figsize=(8, 4.5))
    plt.plot(t_nm, y_nm[:, 0], label="x – Newmark", lw=1.4)
    plt.plot(t_nm, y_nm[:, 1], "--", label="y – Newmark", lw=1.2)
    plt.plot(t_hht, y_hht[:, 0], label="x – HHT", lw=1.4)
    plt.plot(t_hht, y_hht[:, 1], "--", label="y – HHT", lw=1.2)
    plt.xlabel("t"); plt.ylabel("position")
    plt.title(f"Positions vs time (k={k})")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=9, ncol=2)
    plt.tight_layout(); plt.savefig("pos_newmark_vs_hht.png", dpi=180)
    print("Saved pos_newmark_vs_hht.png")


def parse_args():
    p = argparse.ArgumentParser(description="Run elastic pendulum with Newmark-β and HHT-α")
    p.add_argument("--k", type=float, default=1.0, help="spring constant k")
    p.add_argument("--t_end", type=float, default=30.0, help="final time")
    p.add_argument("--h", type=float, default=0.002, help="step size")
    p.add_argument("--nout", type=int, default=4000, help="output points")
    p.add_argument("--alpha", type=float, default=-0.05, help="HHT-α parameter (in [-1/3, 0])")
    return p.parse_args()


def main():
    args = parse_args()
    y0 = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)

    prob = build_problem(args.k, y0)

    t_nm, y_nm, _ = simulate_newmark(prob, h=args.h, t_end=args.t_end, nout=args.nout)
    t_hht, y_hht, _ = simulate_hht(prob, h=args.h, t_end=args.t_end, nout=args.nout, alpha=args.alpha)

    plot_results(t_nm, y_nm, t_hht, y_hht, args.k)


if __name__ == "__main__":
    main()
