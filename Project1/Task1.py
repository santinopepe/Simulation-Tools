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




if __name__ == "__main__":
    t, y, sim = Solver(k=1000000, t_end=30.0, rtol=1e-8, atol=1e-10, nout=660)
    Plot(t, y, title="")

