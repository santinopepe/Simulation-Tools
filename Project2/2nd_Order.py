import numpy as np

from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import ID_PY_OK, ID_PY_COMPLETE
from assimulo.exception import AssimuloException
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve


class Second_Order(Explicit_ODE):
    """
    Base class for fixed-step second-order integrators.
    We store the Assimulo state as y = [u, u_dot].
    For Newmark/HHT we also keep the current acceleration a_n internally.
    """

    def __init__(self, problem):
        super().__init__(problem)

        if len(self.y0) % 2 != 0:
            raise AssimuloException("Second_Order expects [u, u_dot] stacked in the state vector.")
        self.ndof = len(self.y0) // 2

        # fixed step
        self.options["h"] = 0.00001
        self.supports["one_step_mode"] = True

        # Try to read linear elastodynamics data (M, C, K, f)
        self._M, self._C, self._K = self._extract_mck(problem)
        self._force = self._extract_force(problem)
        self._mode = "mck" if (self._M is not None and self._K is not None and self._force is not None) else "acc"

        # Internal storage for acceleration a_n
        self._a = None

        # Cache effective matrices per step size h (useful since last step may have different h)
        self._A_cache = {}


    def _set_h(self, h):
        self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]

    h = property(_get_h, _set_h)

   
    def _split(self, y):
        return y[: self.ndof], y[self.ndof :]

    def _combine(self, u, udot):
        return np.hstack((u, udot))

 
    def _extract_mck(self, problem):
        def first_present(names):
            for name in names:
                val = getattr(problem, name, None)
                if val is not None:
                    return val
            return None

        M = first_present(("M", "Mass_mat", "Mass"))
        K = first_present(("K", "Stiffness_mat", "Stiffness"))
        C = first_present(("C", "Damping_mat", "Damping"))

        if M is not None:
            M = np.asarray(M) if not issparse(M) else M
        if K is not None:
            K = np.asarray(K) if not issparse(K) else K
        if C is not None:
            C = np.asarray(C) if not issparse(C) else C

        return M, C, K

    def _extract_force(self, problem):
        F = getattr(problem, "F", None) or getattr(problem, "f", None) or getattr(problem, "force", None)
        if F is None:
            return None
        if callable(F):
            return lambda t: np.asarray(F(t))
        else:
            f0 = np.asarray(F)
            return lambda t: f0

    def _solve(self, A, b):
        if issparse(A):
            if spsolve is None:
                raise AssimuloException("Sparse matrix provided but SciPy sparse solver not available.")
            return np.asarray(spsolve(A, b))
        return np.linalg.solve(A, b)

    def _apply(self, A, x):
        # Matrix-vector product, sparse or dense
        return A @ x

    def acceleration_from_mck(self, t, u, udot):
        """
        a = M^{-1} ( f(t) - C u_dot - K u )
        """
        if self._force is None:
            raise AssimuloException("No external force provided.")
        rhs = self._force(t) - self._apply(self._K, u)
        if self._C is not None:
            rhs = rhs - self._apply(self._C, udot)
        return self._solve(self._M, rhs)

    def acceleration_from_problem(self, t, u, udot):
        """
        General fallback: use problem.acc(t,u,udot) if present,
        else problem.rhs(t, y) assuming returns [v, a].
        """
        if hasattr(self.problem, "acc"):
            return np.asarray(self.problem.acc(t, u, udot))
        z = self._combine(u, udot)
        rhs_val = self.problem.rhs(t, z)
        return np.asarray(rhs_val[self.ndof :])

    def acceleration(self, t, u, udot):
        return (
            self.acceleration_from_mck(t, u, udot)
            if self._mode == "mck"
            else self.acceleration_from_problem(t, u, udot)
        )


    def step(self, t, y, tf, opts):
        if opts["initialize"]:
            self._initialize_acceleration(t, y)
            self.solver_iterator = self._iter(t, y, tf)
        return next(self.solver_iterator)

    def integrate(self, t, y, tf, opts):
        if opts["initialize"]:
            self._initialize_acceleration(t, y)
        results = list(self._iter(t, y, tf))
        flags, tlist, ylist = zip(*results)
        return flags[-1], tlist, ylist

    def _initialize_acceleration(self, t, y):
        u0, ud0 = self._split(y)
        self._a = self.acceleration(t, u0, ud0)

    def _iter(self, t, y, tf):
        h = min(self.h, abs(tf - t))
        while t + h < tf:
            t, y = self._step(t, y, h)
            self.statistics["nsteps"] += 1
            yield ID_PY_OK, t, y
            h = min(self.h, abs(tf - t))
        else:
            if h == 0:
                h = tf - t
            t, y = self._step(t, y, h)
            self.statistics["nsteps"] += 1
            yield ID_PY_COMPLETE, t, y

    # subclasses implement _step(t,y,h)
    def _step(self, t, y, h):
        raise NotImplementedError


class NewmarkBeta(Second_Order):
    """
    Newmark-β method:
      - implicit scheme=
      - reduced to one d×d linear solve per step via =
      - then update u_dot and u_ddot
    Explicit special case: only when C=0 and β=0 (and typically γ=1/2).
    """

    def __init__(self, problem, beta=0.25, gamma=0.5):
        super().__init__(problem)
        self.beta = beta
        self.gamma = gamma

    def _effective_matrix(self, h):
        key = ("newmark", float(h), float(self.beta), float(self.gamma), id(self._M), id(self._K), id(self._C))
        if key in self._A_cache:
            return self._A_cache[key]

        beta = self.beta
        gamma = self.gamma

        if beta == 0.0:
            # no effective matrix in explicit branch
            return None

        A = (1.0 / (beta * h * h)) * self._M + self._K
        if self._C is not None:
            A = A + (gamma / (beta * h)) * self._C

        self._A_cache[key] = A
        return A

    def _step(self, t, y, h):
        beta = self.beta
        gamma = self.gamma

        u_n, ud_n = self._split(y)
        a_n = self._a

        if self._mode == "acc":
            u_tilde = u_n + h * ud_n + (0.5 - beta) * (h * h) * a_n
            ud_tilde = ud_n + (1.0 - gamma) * h * a_n

            a_np1 = self.acceleration(t + h, u_tilde, ud_tilde)
            u_np1 = u_tilde + beta * (h * h) * a_np1
            ud_np1 = ud_tilde + gamma * h * a_np1

            self._a = a_np1
            return t + h, self._combine(u_np1, ud_np1)

        # Explicit case: only valid if beta=0 and no damping (C=0)
        if beta == 0.0:
            if self._C is not None and (np.any(self._C != 0) if not issparse(self._C) else self._C.nnz != 0):
                raise AssimuloException("Explicit Newmark requires C=0 when beta=0.")
            u_np1 = u_n + ud_n * h + 0.5 * a_n * (h * h)
            a_np1 = self._solve(self._M, self._force(t + h) - self._apply(self._K, u_np1))
            ud_np1 = ud_n + (1.0 - gamma) * a_n * h + gamma * a_np1 * h

            self._a = a_np1
            return t + h, self._combine(u_np1, ud_np1)

        # Implicit reduced system: solve for u_{n+1}
        A = self._effective_matrix(h)

        # RHS:
        b = self._force(t + h)
        b = b + self._apply(
            self._M,
            (u_n / (beta * h * h)) + (ud_n / (beta * h)) + ((1.0 / (2.0 * beta) - 1.0) * a_n),
        )

        if self._C is not None:
            termC = (gamma * u_n / (beta * h)) - ((1.0 - gamma / beta) * ud_n) - ((1.0 - gamma / (2.0 * beta)) * h * a_n)
            b = b + self._apply(self._C, termC)

        u_np1 = self._solve(A, b)

        ud_np1 = (gamma / beta) * ((u_np1 - u_n) / h) + (1.0 - gamma / beta) * ud_n + (1.0 - gamma / (2.0 * beta)) * h * a_n

 
        a_np1 = ((u_np1 - u_n) / (beta * h * h)) - (ud_n / (beta * h)) - ((1.0 / (2.0 * beta) - 1.0) * a_n)

        self._a = a_np1
        return t + h, self._combine(u_np1, ud_np1)

    # properties
    @property
    def beta(self):
        return float(self.options["beta"])

    @beta.setter
    def beta(self, beta):
        self.options["beta"] = float(beta)

    @property
    def gamma(self):
        return float(self.options["gamma"])

    @gamma.setter
    def gamma(self, gamma):
        self.options["gamma"] = float(gamma)

    def print_statistics(self, verbose):
        self.log_message("Final Run Statistics: %s \n" % self.problem.name, verbose)
        self.log_message(" Step-length          : %s " % (self.options["h"]), verbose)
        self.log_message("\nSolver options:\n", verbose)
        self.log_message(" Solver            : NewmarkBeta (PDF form)", verbose)
        self.log_message(" beta              : %s" % self.beta, verbose)
        self.log_message(" gamma             : %s\n" % self.gamma, verbose)


class HHTAlpha(Second_Order):
    """
    HHT-α method exactly as in the PDF:
      - shifted equilibrium (11)
      - reduced to one d×d linear solve per step via (12)
      - then update u_dot via (6') and u_ddot via (5')
    Parameter tying:
      γ = 1/2 - α
      β = ((1-α)/2)^2
      α ∈ [-1/3, 0]
    """

    def __init__(self, problem, alpha=-0.05):
        super().__init__(problem)
        self.alpha = alpha  # sets beta/gamma

    def _effective_matrix(self, h):
        key = ("hht", float(h), float(self.alpha), id(self._M), id(self._K), id(self._C))
        if key in self._A_cache:
            return self._A_cache[key]

        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        A = (1.0 / (beta * h * h)) * self._M + (1.0 + alpha) * self._K
        if self._C is not None:
            A = A + (1.0 + alpha) * (gamma / (beta * h)) * self._C

        self._A_cache[key] = A
        return A

    def _step(self, t, y, h):
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        u_n, ud_n = self._split(y)
        a_n = self._a

        A = self._effective_matrix(h)

        # RHS:
        b = (1.0 + alpha) * self._force(t + h) - alpha * self._force(t)

        b = b + self._apply(
            self._M,
            (u_n / (beta * h * h)) + (ud_n / (beta * h)) + ((1.0 / (2.0 * beta) - 1.0) * a_n),
        )

        if self._C is not None:
            termC = (gamma * u_n / (beta * h)) - ((1.0 - gamma / beta) * ud_n) - ((1.0 - gamma / (2.0 * beta)) * h * a_n)
            b = b + (1.0 + alpha) * self._apply(self._C, termC)

            b = b + alpha * self._apply(self._C, ud_n)

        b = b + alpha * self._apply(self._K, u_n)

        u_np1 = self._solve(A, b)

        # Then the same Newmark kinematics:
        ud_np1 = (gamma / beta) * ((u_np1 - u_n) / h) + (1.0 - gamma / beta) * ud_n + (1.0 - gamma / (2.0 * beta)) * h * a_n
        a_np1 = ((u_np1 - u_n) / (beta * h * h)) - (ud_n / (beta * h)) - ((1.0 / (2.0 * beta) - 1.0) * a_n)

        self._a = a_np1
        return t + h, self._combine(u_np1, ud_np1)

    # properties
    @property
    def alpha(self):
        return float(self.options["alpha"])

    @alpha.setter
    def alpha(self, alpha):
        alpha = float(alpha)
        if alpha < -1.0 / 3.0 or alpha > 0.0:
            raise AssimuloException("alpha must be in [-1/3, 0] (PDF condition).")
        self.options["alpha"] = alpha
        self.options["gamma"] = 0.5 - alpha
        self.options["beta"] = ((1.0 - alpha) / 2.0) ** 2

    @property
    def beta(self):
        return float(self.options["beta"])

    @property
    def gamma(self):
        return float(self.options["gamma"])

    def print_statistics(self, verbose):
        self.log_message("Final Run Statistics: %s \n" % self.problem.name, verbose)
        self.log_message(" Step-length          : %s " % (self.options["h"]), verbose)
        self.log_message("\nSolver options:\n", verbose)
        self.log_message(" Solver            : HHTAlpha (PDF form)", verbose)
        self.log_message(" alpha             : %s" % self.alpha, verbose)
        self.log_message(" beta              : %s" % self.beta, verbose)
        self.log_message(" gamma             : %s\n" % self.gamma, verbose)
