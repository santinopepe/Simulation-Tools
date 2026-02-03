import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Explicit_Problem
import sys
import os

# Import BDF2 and BDF4 solvers
sys.path.insert(0, os.path.dirname(__file__))
from BDF2_Assimulo import BDF_2
from BDF4_Assimulo import BDF_4


def ElasticPendulumRHS(k: float):
    """
    Returns rhs(t, y) (Right hand side) for the elastic pendulum.
    """
    def rhs(t, y):
        y1, y2, y3, y4 = y
        r = np.hypot(y1, y2)  # √(x² + y²)

        if r == 0.0:  # Security check so there is no 0 division
            lam = 0.0
        else:
            lam = k * ((r - 1.0) / r)  # lambda(y1,y2)

        dy1 = y3
        dy2 = y4
        dy3 = -y1 * lam
        dy4 = -y2 * lam - 1.0
        return np.array([dy1, dy2, dy3, dy4], dtype=float)
    return rhs


def Solver_BDF2(k=1.0, t_end=20.0, h=0.02, nout=2000, y0=None):
    rhs = ElasticPendulumRHS(k)

    if y0 is None:
        y0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    prob = Explicit_Problem(rhs, y0, name=f"Elastic pendulum BDF-2 (k={k})")
    
    sim = BDF_2(prob)
    sim.h = h
    
    t, y = sim.simulate(t_end, nout)
    return t, y, sim


def Solver_BDF4(k=1.0, t_end=20.0, h=0.02, nout=2000, y0=None):
    """Solver using BDF-4 method with fixed step size."""
    rhs = ElasticPendulumRHS(k)

    if y0 is None:
        y0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    prob = Explicit_Problem(rhs, y0, name=f"Elastic pendulum BDF-4 (k={k})")
    
    sim = BDF_4(prob)
    sim.h = h
    
    t, y = sim.simulate(t_end, nout)
    return t, y, sim



def plot_comparison(k=1.0, t_end=50.0, h=0.02, nout=2000):
    """Generate separate comparison plots for BDF-2 vs BDF-4: trajectory and position vs time."""
    #aca cambias la posicion inicial
    y0 = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
    
    # Run both solvers
    t2, y2, sim2 = Solver_BDF2(k=k, t_end=t_end, h=h, nout=nout, y0=y0)
    t4, y4, sim4 = Solver_BDF4(k=k, t_end=t_end, h=h, nout=nout, y0=y0)
    
    # Extract components
    y1_2, y2_2, y3_2, y4_2 = y2.T
    y1_4, y2_4, y3_4, y4_4 = y4.T
    
    # Plot 1: Trajectory (separate figure)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(y1_2, y2_2, 'b-', label='BDF-2', linewidth=2, alpha=0.8)
    ax1.plot(y1_4, y2_4, 'r--', label='BDF-4', linewidth=2, alpha=0.8)
    ax1.set_xlabel('y1 (position)', fontsize=12)
    ax1.set_ylabel('y2 (position)', fontsize=12)
    ax1.set_title(f'Trajectory (k={k})', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    plt.tight_layout()
    plt.savefig('trajectory.png', dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: trajectory.png")
    plt.show()
    
    # Plot 2: Position vs time (separate figure) - both y1 and y2
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(t2, y1_2, 'b-', label='BDF-2: y1 (x)', linewidth=2, alpha=0.8)
    ax2.plot(t2, y2_2, 'b--', label='BDF-2: y2 (y)', linewidth=2, alpha=0.8)
    ax2.plot(t4, y1_4, 'r-', label='BDF-4: y1 (x)', linewidth=2, alpha=0.8)
    ax2.plot(t4, y2_4, 'r--', label='BDF-4: y2 (y)', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Position', fontsize=12)
    ax2.set_title(f'Positions vs Time (k={k})', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('position_vs_time.png', dpi=150, bbox_inches='tight')
    print(f"  Plot saved: position_vs_time.png")
    plt.show()



if __name__ == "__main__":
    
    # aca cambias las condiciones iniciales
    plot_comparison(k=1000.0, t_end=30.0, h=0.002, nout=2000)
