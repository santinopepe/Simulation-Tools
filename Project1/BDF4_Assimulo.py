from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
from assimulo.problem import Explicit_Problem
import numpy as np
import scipy.linalg as SL


class BDF_4(Explicit_ODE):
    """
    BDF-4 (paso fijo) + Newton en cada paso (para sistemas no lineales).

    - BDF es implícito: u_{n+1} aparece dentro de f(t_{n+1}, u_{n+1}) :contentReference[oaicite:6]{index=6}
    - Newton: resolver F'(u^j) Δu = -F(u^j), u^{j+1} = u^j + Δu :contentReference[oaicite:7]{index=7}
    - Predictor: usar u_n como punto inicial (pedido por el Project01) :contentReference[oaicite:8]{index=8}
    - Jacobiano numérico (si no hay jac analítico): diferencias finitas centradas :contentReference[oaicite:9]{index=9}
    """

    tol = 1e-8
    maxit = 50
    maxsteps = 20000
    jac_eps = 1e-8

    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)

        # Opciones del solver
        self.options["h"] = 0.01

        # Estadísticas
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
        # Custom counters (Statistics only allows predefined keys)
        self.nnewton = 0
        self.njacs = 0

    def _set_h(self, h):
        self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]

    h = property(_get_h, _set_h)

    # ---------------------------
    # Helpers: Jacobiano
    # ---------------------------
    def _rhs(self, t, y):
        """Wrapper para contar evaluaciones de f."""
        self.statistics["nfcns"] += 1
        return self.problem.rhs(t, y)

    def _jacobian(self, t, y):
        """
        Jacobiano Jf(t,y) = ∂f/∂y.
        Si el problema provee jac, úsalo. Si no, finite difference central. :contentReference[oaicite:10]{index=10}
        """
        # Intentar jac analítico si existe
        jac = getattr(self.problem, "jac", None)
        if callable(jac):
            self.njacs += 1
            return jac(t, y)

        # Jacobiano numérico por diferencias finitas centradas:
        # ∂fi/∂xj(x) ≈ (fi(x + h e_j) - fi(x - h e_j)) / (2h) :contentReference[oaicite:11]{index=11}
        self.njacs += 1
        y = np.asarray(y, dtype=float)
        n = y.size
        J = np.zeros((n, n), dtype=float)
        eps = self.jac_eps

        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            fp = self._rhs(t, y + eps * ej)
            fm = self._rhs(t, y - eps * ej)
            J[:, j] = (fp - fm) / (2.0 * eps)       #calculo de la derivada por diferencias finitas

        return J

    # ---------------------------
    # Integración
    # ---------------------------
    def integrate(self, t, y, tf, opts):
        """
        Integra desde (t,y) hasta tf con paso fijo h.
        BDF-4 necesita 3 pasos previos, así que los primeros 3 pasos usan Euler explícito.
        """
        h = self.options["h"]
        h = min(h, abs(tf - t))

        # Lists for storing the result
        tres = []
        yres = []

        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1

            if i == 0:  # first step
                t_np1, y_np1 = self.step_EE(t, y, h)
            elif i == 1:  # second step
                t_np1, y_np1 = self.step_EE(t, y, h)
            elif i == 2:  # third step
                t_np1, y_np1 = self.step_EE(t, y, h)
            else:  # from fourth step onwards, use BDF4
                t_np1, y_np1 = self.step_BDF4([t, t_nm1, t_nm2, t_nm3],
                                              [y, y_nm1, y_nm2, y_nm3],
                                              h)
            
            # Shift history (like BDF2 does: t,t_nm1=t_np1,t)
            if i >= 2:
                t_nm3, y_nm3 = t_nm2, y_nm2
            if i >= 1:
                t_nm2, y_nm2 = t_nm1, y_nm1
            t_nm1, y_nm1 = t, y
            t, y = t_np1, y_np1

            tres.append(t)
            yres.append(y.copy())

            h = min(self.h, np.abs(tf - t))
        else:
            raise Exception('Final time not reached within maximum number of steps')

        return ID_PY_OK, tres, yres

    def step_EE(self, t, y, h):
        """
        Un paso de Euler explícito (starter).
        """
        return t + h, y + h * self._rhs(t, y)

    # ---------------------------
    # Paso BDF4 + Newton
    # ---------------------------
  
    def step_BDF4(self, T, Y, h):
        f = self._rhs

        t_n, t_nm1, t_nm2, t_nm3 = T        #timepos previos
        y_n, y_nm1, y_nm2, y_nm3 = Y        #estados previos

        t_np1 = t_n + h

        a0 = 25.0 / 12.0
        a1 = -4.0
        a2 = 3.0
        a3 = -4.0 / 3.0
        a4 = 1.0 / 4.0

        old_part = a1 * y_n + a2 * y_nm1 + a3 * y_nm2 + a4 * y_nm3

        # predictor (zero order): u^(0) = y_n
        u = y_n.copy()

        for it in range(self.maxit):
            self.nnewton += 1

            Fu = a0 * u + old_part - h * f(t_np1, u)        #error 

            # Convergencia por residual
            if SL.norm(Fu, ord=np.inf) < self.tol:
                return t_np1, u

            # Jacobiano: F'(u) = a0 I - h Jf
            Jf = self._jacobian(t_np1, u)
            A = a0 * np.eye(u.size) - h * Jf

            du = SL.solve(A, -Fu)
            u_new = u + du

            # Convergencia por incremento
            if SL.norm(du, ord=np.inf) < self.tol * (1.0 + SL.norm(u_new, ord=np.inf)):
                return t_np1, u_new

            u = u_new

        raise Exception(f'Newton could not converge within {self.maxit} iterations')


    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name), verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : ' + str(self.statistics["nsteps"]), verbose)
        self.log_message(' Number of Function Evaluations : ' + str(self.statistics["nfcns"]), verbose)
        self.log_message(' Number of Newton iterations    : ' + str(self.nnewton), verbose)
        self.log_message(' Number of Jacobian builds      : ' + str(self.njacs), verbose)

        self.log_message('\nSolver options:\n', verbose)
        self.log_message(' Solver            : BDF4', verbose)
        self.log_message(' Solver type       : Fixed step + Newton\n', verbose)

if __name__ == "__main__":
    # Péndulo elástico (mismo modelo que en Task1)
    def rhs(t, y):
        y1, y2, y3, y4 = y
        r = np.hypot(y1, y2)
        lam = 0.0 if r == 0 else  (r - 1.0) / r # Por ahi falta multiplicar por k
        return np.array([y3, y4, -y1 * lam, -y2 * lam - 1.0], float)

    y0 = np.array([1.0, 0.0, 0.0, 0.0], float)
    prob = Explicit_Problem(rhs, y0, name="Elastic pendulum")

    sim = BDF_4(prob)
    sim.h = 0.02        # paso preferido
    sim.maxsteps = 15000
    t, y = sim.simulate(10.0)  # tiempo final

    sim.print_statistics()
    sim.plot()
