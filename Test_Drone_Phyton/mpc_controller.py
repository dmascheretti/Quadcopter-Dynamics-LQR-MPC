import casadi as ca
import numpy as np
import control as ct
from parametri_drone import A_lin, B_lin, F_eq

class MPC_Lineare:
    def __init__(self, Ts=0.01, N=60):
        self.Ts = Ts # Orizzonte temporale più fine
        self.N  = N  # Orizzonte di predizione più lungo

        # Discretizzazione ZOH — identica a MATLAB c2d
        sys_c  = ct.StateSpace(A_lin, B_lin, np.eye(12), np.zeros((12,4)))
        sys_d  = sys_c.sample(Ts, method='zoh')
        self.Ad = sys_d.A
        self.Bd = sys_d.B

        # Pesi dal codice di riferimento MATLAB
        Q_pos = np.diag([100, 100, 100])
        Q_ang = np.diag([20, 20, 20])
        Q_vel = np.diag([15, 15, 15])
        # Costruiamo la matrice Q 12x12 in modo corretto e chiaro.
        # Lo stato è [pos, ang, vel, ang_vel]. Penalizziamo i primi 3.
        self.Q = np.block([
            [Q_pos,           np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), Q_ang,           np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), Q_vel,           np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]
        ])
        self.R = np.diag([0.1, 0.1, 0.1, 0.1]) # Penalità sulla VARIAZIONE del comando

        self.F = F_eq * np.ones(4)         # forza di hovering

        self.X_sol = None
        self.U_sol = None

        self._build()

    def _build(self):
        opti = ca.Opti()

        # Variabili
        X = opti.variable(12, self.N + 1) # Stato assoluto
        U = opti.variable(4,  self.N)     # Comando assoluto

        # Parametri
        x0_p     = opti.parameter(12)
        xref_p   = opti.parameter(12)
        u_prev_p = opti.parameter(4) # Comando precedente per calcolo delta_u

        # Condizione iniziale
        opti.subject_to(X[:, 0] == x0_p)

        cost = 0
        for k in range(self.N):
            # Dinamica lineare (modello basato su deviazioni)
            opti.subject_to(
                X[:, k+1] == self.Ad @ X[:, k] + self.Bd @ (U[:, k] - self.F)
            )

            # Funzione di costo (come da codice MATLAB)
            err = X[:, k] - xref_p
            cost += err.T @ self.Q @ err

            if k == 0:
                delta_u = U[:, 0] - u_prev_p
            else:
                delta_u = U[:, k] - U[:, k-1]
            cost += delta_u.T @ self.R @ delta_u

        # Vincoli fisici
        opti.subject_to(opti.bounded(0, U, 13)) # Saturazione motori
        opti.subject_to(opti.bounded(-np.deg2rad(22), X[3:5, 1:], np.deg2rad(22)))

        opti.minimize(cost)

        # Solver (usiamo IPOPT che è già qui, SQPMethod è un'alternativa)
        opts = {
            'print_time': False,
            'ipopt': {
                'print_level': 0,
                'warm_start_init_point': 'yes'
            }
        }
        opti.solver('ipopt', opts)

        self.opti     = opti
        self.X_var    = X
        self.U_var    = U
        self.x0_p     = x0_p
        self.xref_p   = xref_p
        self.u_prev_p = u_prev_p

    def calcola(self, x_curr, target_pos, u_prev):
        # Stato di riferimento — target solo su posizione
        xref      = np.zeros(12)
        xref[0:3] = target_pos

        self.opti.set_value(self.x0_p,     x_curr)
        self.opti.set_value(self.xref_p,   xref)
        self.opti.set_value(self.u_prev_p, u_prev)

        # Warm start
        if self.X_sol is not None:
            Xg = np.hstack([self.X_sol[:, 1:], self.X_sol[:, -1:]])
            Ug = np.hstack([self.U_sol[:, 1:], self.U_sol[:, -1:]])
            self.opti.set_initial(self.X_var, Xg)
            self.opti.set_initial(self.U_var, Ug)

        # Risolvi il problema di ottimizzazione
        try:
            sol = self.opti.solve()
            u_opt      = sol.value(self.U_var[:, 0])
            self.X_sol = sol.value(self.X_var)
            self.U_sol = sol.value(self.U_var)
        except Exception as e:
            print(f"ATTENZIONE: Solver MPC fallito. Causa: {e}")
            print("Ritorno al comando di hover di emergenza.")
            # Se il solver fallisce, usa un'azione sicura (hover)
            u_opt = self.F.copy()
            # Resetta le soluzioni per non usare un warm start errato al prossimo step
            self.X_sol = None
            self.U_sol = None

        return u_opt