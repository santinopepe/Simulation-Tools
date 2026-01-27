from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
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

    tol = 1e-10
    maxit = 20
    maxsteps = 5000
    jac_eps = 1e-8

    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)

        # Opciones del solver
        self.options["h"] = 0.01

        # Estadísticas
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
        self.statistics["nnewton"] = 0
        self.statistics["njacs"] = 0

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
            self.statistics["njacs"] += 1
            return jac(t, y)

        # Jacobiano numérico por diferencias finitas centradas:
        # ∂fi/∂xj(x) ≈ (fi(x + h e_j) - fi(x - h e_j)) / (2h) :contentReference[oaicite:11]{index=11}
        self.statistics["njacs"] += 1
        y = np.asarray(y, dtype=float)
        n = y.size
        J = np.zeros((n, n), dtype=float)
        eps = self.jac_eps

        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            fp = self._rhs(t, y + eps * ej)
            fm = self._rhs(t, y - eps * ej)
            J[:, j] = (fp - fm) / (2.0 * eps)

        return J

    # ---------------------------
    # Integración
    # ---------------------------
    def integrate(self, t, y, tf, opts):
        """
        Integra desde (t,y) hasta tf con paso fijo h (recortando el último).
        BDF-4 necesita 4 valores previos (u_n..u_{n-3}), así que arrancamos con
        3 pasos de un método de un paso (Euler explícito) para obtenerlos. :contentReference[oaicite:12]{index=12}
        """
        h = min(self.h, abs(tf - t))

        tres = []
        yres = []

        # Guardamos historia:
        # y_n, y_{n-1}, y_{n-2}, y_{n-3}
        y_n = np.asarray(y, dtype=float)
        t_n = t

        # Para inicializar u1,u2,u3 usamos 3 pasos de Euler explícito (starter).
        # El apunte dice que un k-step necesita starting values y que se obtienen con métodos de 1 paso. :contentReference[oaicite:13]{index=13}
        hist_y = [y_n.copy()]  # u0
        hist_t = [t_n]

        for _ in range(3):
            if t_n >= tf:
                break
            self.statistics["nsteps"] += 1
            t_np1, y_np1 = self.step_EE(t_n, y_n, h)
            t_n, y_n = t_np1, y_np1
            hist_t.append(t_n)
            hist_y.append(y_n.copy())
            tres.append(t_n)
            yres.append(y_n.copy())
            h = min(self.h, abs(tf - t_n))

        if t_n >= tf:
            return ID_PY_OK, tres, yres

        # Ahora tenemos u0,u1,u2,u3 (si tf lo permitió)
        # Ordenamos la historia para BDF4:
        # y_n = u3, y_{n-1}=u2, y_{n-2}=u1, y_{n-3}=u0
        y_n   = hist_y[-1]
        y_nm1 = hist_y[-2]
        y_nm2 = hist_y[-3]
        y_nm3 = hist_y[-4]
        t_n   = hist_t[-1]

        for i in range(self.maxsteps):
            if t_n >= tf:
                break
            self.statistics["nsteps"] += 1

            t_np1, y_np1 = self.step_BDF4([t_n, t_n - h, t_n - 2*h, t_n - 3*h],
                                          [y_n, y_nm1, y_nm2, y_nm3],
                                          h)

            # Shift de historia
            y_nm3, y_nm2, y_nm1, y_n = y_nm2, y_nm1, y_n, y_np1
            t_n = t_np1

            tres.append(t_n)
            yres.append(y_n.copy())

            h = min(self.h, abs(tf - t_n))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')

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
        """
        BDF-4 con Newton:

        Forma general implícita BDF: sum α u_{n+1-i} = h f(t_{n+1}, u_{n+1}) :contentReference[oaicite:14]{index=14}
        Newton para resolver el sistema no lineal del paso :contentReference[oaicite:15]{index=15}
        Predictor: u^(0) = u_n (pedido por el Project01) :contentReference[oaicite:16]{index=16}
        """
        f = self._rhs

        t_n, t_nm1, t_nm2, t_nm3 = T
        y_n, y_nm1, y_nm2, y_nm3 = Y

        t_np1 = t_n + h

        # Coeficientes estándar de BDF4 (paso constante):
        # (25/12) u_{n+1} - 4 u_n + 3 u_{n-1} - (4/3) u_{n-2} + (1/4) u_{n-3} = h f(t_{n+1}, u_{n+1})
        # Es un caso particular de la fórmula general BDF del apunte. :contentReference[oaicite:17]{index=17}
        a0 = 25.0 / 12.0
        a1 = -4.0
        a2 = 3.0
        a3 = -4.0 / 3.0
        a4 = 1.0 / 4.0

        # Reordenamos para la función no lineal F(u)=0:
        # F(u) = a0*u + a1*y_n + a2*y_nm1 + a3*y_nm2 + a4*y_nm3 - h f(t_np1, u)
        # (Esto coincide con la forma del apunte: Fn+1(u)=αk u - hβk f(...) - función de valores viejos) :contentReference[oaicite:18]{index=18}
        old_part = a1 * y_n + a2 * y_nm1 + a3 * y_nm2 + a4 * y_nm3

        # Predictor: usar y_n como u^(0) :contentReference[oaicite:19]{index=19}
        u = y_n.copy()

        for it in range(self.maxit):
            self.statistics["nnewton"] += 1

            Fu = a0 * u + old_part - h * f(t_np1, u)

            # criterio simple (norma infinito)
            if SL.norm(Fu, ord=np.inf) < self.tol:
                return t_np1, u

            # Jacobiano de F:
            # F'(u) = a0*I - h * ∂f/∂u(t_np1, u)  (exactamente como en el apunte) :contentReference[oaicite:20]{index=20}
            Jf = self._jacobian(t_np1, u)
            A = a0 * np.eye(u.size) - h * Jf

            # Resolver el sistema lineal A Δu = -F(u) (paso Newton) :contentReference[oaicite:21]{index=21}
            du = SL.solve(A, -Fu)

            u_new = u + du

            if SL.norm(du, ord=np.inf) < self.tol * (1.0 + SL.norm(u_new, ord=np.inf)):
                return t_np1, u_new

            u = u_new

        raise Explicit_ODE_Exception(f'Newton could not converge within {self.maxit} iterations')

    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name), verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : ' + str(self.statistics["nsteps"]), verbose)
        self.log_message(' Number of Function Evaluations : ' + str(self.statistics["nfcns"]), verbose)
        self.log_message(' Number of Newton iterations    : ' + str(self.statistics["nnewton"]), verbose)
        self.log_message(' Number of Jacobian builds      : ' + str(self.statistics["njacs"]), verbose)

        self.log_message('\nSolver options:\n', verbose)
        self.log_message(' Solver            : BDF4', verbose)
        self.log_message(' Solver type       : Fixed step + Newton\n', verbose)
