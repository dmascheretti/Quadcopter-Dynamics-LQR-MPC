"""
Controllore LPV-MPC (Linear Parameter-Varying Model Predictive Control).

Questo modulo implementa il controllore MPC per il drone Skydio X2.
La logica e i parametri sono stati allineati con i file di riferimento
del progetto Simulink, in particolare `S_init_sim_LPV.m`.

Caratteristiche principali:
- **LPV (Linear Parameter-Varying)**: A differenza di un MPC LTI (lineare e
  invariante nel tempo), questo controllore aggiorna le matrici di stato (Ad)
  e di ingresso (Bd) ad ogni passo, linearizzando il modello attorno all'assetto
  corrente del drone. Questo viene fatto tramite `lpv_scheduler.py`.
- **Pesi e Costi**: I pesi delle matrici Q e R e la funzione di costo sono
  stati scelti per replicare il comportamento del modello Simulink.
- **Vincoli**: Sono stati implementati i vincoli fisici su spinta dei motori
  e angoli massimi di rollio e beccheggio, come definiti in `S_init_sim_LPV.m`.
- **Solver**: Utilizza CasADi con il solver IPOPT. IPOPT è stato scelto per la
  sua stabilità con problemi parametrici come questo, a differenza di `sqpmethod`
  che, sebbene usato in Simulink per il real-time, è risultato meno robusto
  in questo ambiente Python.
"""

import casadi as ca
import numpy as np
import control as ct
from parametri_drone import A_lin, B_lin, F_eq
from lpv_scheduler import aggiorna_matrici, discretizza_zoh, calcola_B_c

# Parametri del controllore MPC (da S_init_sim_LPV.m)
_Ts_DEFAULT = 0.01
_N_DEFAULT  = 60

# Pesi (da S_init_sim_LPV.m)
_Q_POS = np.diag([100.0, 100.0, 100.0])
_Q_ANG = np.diag([ 20.0,  20.0,  20.0])
_Q_VEL = np.diag([ 15.0,  15.0,  15.0])
_R_MOT = np.diag([  0.1,   0.1,   0.1,  0.1])

# Vincoli (da S_init_sim_LPV.m)
_U_MIN        = 0.0
_U_MAX        = 13.0  # Allineato a ctrlrange="0 13" in x2.xml
_LIMITE_ANG   = np.deg2rad(22.0)   # ±22°


class MPC_Lineare:
    """
    Controllore MPC Lineare a Parametri Variabili per quadricottero.

    Ad ogni chiamata di calcola(), lo schedulatore LPV aggiorna le
    matrici Ad e Bd in base all'assetto corrente del drone, replicando
    il comportamento del blocco LPV_Scheduler in Simulink.
    """

    def __init__(self, Ts=_Ts_DEFAULT, N=_N_DEFAULT):
        self.Ts = Ts
        self.N  = N

        # Pesi per la funzione di costo
        self.Q_pos = _Q_POS
        self.Q_ang = _Q_ANG
        self.Q_vel = _Q_VEL
        self.R_mot = _R_MOT

        # Forza di equilibrio per motore per l'hovering (da `parametri_drone_sky.m`)
        # F_eq = m*g/4 = 1.325*9.81/4 ≈ 3.25 N (senza canting,
        # coerente con S_init_sim_LPV.m: F_eq_val = m*g/4)
        self.F_eq = F_eq * np.ones(4)

        # Matrici LTI iniziali (calcolate una volta e usate come riferimento)
        # Verranno poi aggiornate ad ogni step dallo schedulatore LPV.
        sys_c     = ct.StateSpace(A_lin, B_lin,
                                  np.eye(12), np.zeros((12, 4)))
        sys_d     = sys_c.sample(Ts, method='zoh')
        self.Ad0  = sys_d.A   # matrici LTI di riferimento
        self.Bd0  = sys_d.B

        # B_c continua per la discretizzazione LPV a ogni step
        self.B_c  = calcola_B_c()

        # Soluzioni precedenti per warm start
        self.X_sol = None
        self.U_sol = None

        # Costruisce il problema CasADi con Ad/Bd parametrici
        self._build()

        print(f"[MPC_LPV] Inizializzato: Ts={Ts}s, N={N}, "
              f"U_max={_U_MAX}N, angoli=±{np.rad2deg(_LIMITE_ANG):.0f}°")

    def _build(self):
        """
        Costruisce il problema di ottimizzazione CasADi.
        Corrisponde alla sezione di setup in S_init_sim_LPV.m.

        Differenza chiave rispetto alla versione LTI:
        Ad e Bd sono parametri opti (aggiornati a ogni step),
        non costanti numeriche — identico a:
            Ad_param = opti.parameter(12, 12);
            Bd_param = opti.parameter(12, 4);
        in S_init_sim_LPV.m.
        """
        opti = ca.Opti()

        # Variabili di decisione: stati predetti e comandi futuri
        X = opti.variable(12, self.N + 1)  # stati predetti
        U = opti.variable(4,  self.N)      # forze motore

        # Parametri, aggiornati ad ogni chiamata del solver
        x0_p     = opti.parameter(12)      # stato corrente
        target_p = opti.parameter(3)       # posizione target
        u_prev_p = opti.parameter(4)       # comando precedente
        Ad_p     = opti.parameter(12, 12)  # matrice A discreta (LPV)
        Bd_p     = opti.parameter(12,  4)  # matrice B discreta (LPV)

        # Vincolo sullo stato iniziale
        opti.subject_to(X[:, 0] == x0_p)

        # Vincoli di dinamica del sistema, che legano stati e comandi
        # lungo l'orizzonte di predizione. Usano le matrici LPV parametriche.
        # Logica identica a S_init_sim_LPV.m:
        #   X(:,k+1) == Ad_param*X(:,k) + Bd_param*(U(:,k) - F_eq)
        for k in range(self.N):
            opti.subject_to(
                X[:, k+1] == Ad_p @ X[:, k]
                            + Bd_p @ (U[:, k] - self.F_eq)
            )

        # Vincoli fisici: saturazione motori (da x2.xml)
        opti.subject_to(opti.bounded(_U_MIN, U, _U_MAX))

        # Limiti angoli roll/pitch: ±22° (da S_init_sim_LPV.m)
        # Su X[:, 1:] — escludo il primo step (stato corrente misurato)
        opti.subject_to(opti.bounded(
            -_LIMITE_ANG, X[3:5, 1:], _LIMITE_ANG
        ))

        # Funzione di costo da minimizzare.
        # La struttura è allineata a S_init_sim_LPV.m:
        #   err_pos = X(1:3,k) - target_pos   (errore posizione)
        #   angoli  = X(4:6,k)                (penalizzati a zero)
        #   vel_lin = X(7:9,k)                (penalizzate a zero)
        #   delta_u = variazione comando
        cost = 0
        # Conversione a MX una sola volta fuori dal ciclo
        Q_pos_mx = ca.MX(self.Q_pos)
        Q_ang_mx = ca.MX(self.Q_ang)
        Q_vel_mx = ca.MX(self.Q_vel)
        R_mot_mx = ca.MX(self.R_mot)

        for k in range(self.N):
            err_pos = X[0:3, k] - target_p
            angoli  = X[3:6, k]
            vel_lin = X[6:9, k]

            if k == 0:
                delta_u = U[:, 0] - u_prev_p
            else:
                delta_u = U[:, k] - U[:, k-1]

            cost += (err_pos.T @ Q_pos_mx @ err_pos
                   + angoli.T  @ Q_ang_mx @ angoli
                   + vel_lin.T @ Q_vel_mx @ vel_lin
                   + delta_u.T @ R_mot_mx @ delta_u)

        opti.minimize(cost)

        # Configurazione del solver. Usiamo IPOPT, che in Python è più stabile
        # per problemi con matrici parametriche e warm start).
        # sqpmethod è preferito in Simulink (S_init_sim_LPV.m) per la
        # velocità real-time, ma in Python puro IPOPT è più robusto.
        opts = {
            'print_time': False,
            'ipopt': {
                'print_level'              : 0,
                'warm_start_init_point'    : 'yes',
                'warm_start_bound_push'    : 1e-6,
                'warm_start_mult_bound_push': 1e-6,
                'mu_strategy'              : 'adaptive',
                'tol'                      : 1e-4,
                'max_iter'                 : 200,
                'sb'                       : 'yes',
            }
        }
        opti.solver('ipopt', opts)
        self._solver_name = 'ipopt'

        # Salvataggio dei riferimenti alle variabili e parametri di CasADi
        self.opti     = opti
        self.X_var    = X
        self.U_var    = U
        self.x0_p     = x0_p
        self.target_p = target_p
        self.u_prev_p = u_prev_p
        self.Ad_p     = Ad_p
        self.Bd_p     = Bd_p

        # Inizializzazione con matrici LTI (primo step)
        opti.set_value(self.Ad_p, self.Ad0)
        opti.set_value(self.Bd_p, self.Bd0)

    def calcola(self, x_curr, target_pos, u_prev):
        """
        Risolve il problema MPC per lo step corrente.

        Flusso (corrisponde al loop in Simulink):
          1. Legge phi, theta, psi dallo stato corrente
          2. Chiama lo schedulatore LPV per aggiornare Ad, Bd
          3. Aggiorna i parametri del problema CasADi
          4. Applica warm start dalle soluzioni precedenti
          5. Risolve e restituisce il primo controllo ottimale

        Args:
            x_curr (np.ndarray): Vettore di stato corrente (12,).
            target_pos (np.ndarray): Posizione target [x, y, z] (3,).
            u_prev (np.ndarray): Comando motori al passo precedente (4,).

        Returns:
            np.ndarray: Comando motori ottimale per lo step corrente (4,).
        """
        # 1. Schedulazione LPV
        # Leggo l'assetto corrente (indici 3,4,5 del vettore di stato)
        phi, theta, psi = x_curr[3], x_curr[4], x_curr[5]

        # Aggiorno Ad e Bd in base all'assetto attuale
        # (equivalente al blocco LPV_Scheduler in Simulink)
        Ad_lpv, Bd_lpv = aggiorna_matrici(phi, theta, psi, self.Ts)

        # 2. Aggiornamento dei parametri del problema di ottimizzazione
        self.opti.set_value(self.Ad_p,     Ad_lpv)
        self.opti.set_value(self.Bd_p,     Bd_lpv)
        self.opti.set_value(self.x0_p,     x_curr)
        self.opti.set_value(self.target_p, target_pos)
        self.opti.set_value(self.u_prev_p, u_prev)

        # 3. Warm Start: inizializza il solver con la soluzione precedente
        # Shifta la soluzione precedente di un passo
        # (stesso meccanismo di u_guess in Q_test_close_LMPC.m)
        if self.X_sol is not None:
            Xg = np.hstack([self.X_sol[:, 1:], self.X_sol[:, -1:]])
            Ug = np.hstack([self.U_sol[:, 1:], self.U_sol[:, -1:]])
            self.opti.set_initial(self.X_var, Xg)
            self.opti.set_initial(self.U_var, Ug)
        else:
            # Prima iterazione: inizializza U con hovering
            self.opti.set_initial(
                self.U_var,
                np.tile(self.F_eq.reshape(-1, 1), (1, self.N))
            )

        # 4. Risoluzione del problema di ottimizzazione
        try:
            sol = self.opti.solve()
            u_opt      = sol.value(self.U_var[:, 0])
            self.X_sol = sol.value(self.X_var)
            self.U_sol = sol.value(self.U_var)

        except Exception as e:
            # Fallback sicuro: hovering
            # Resetta il warm start per non propagare una soluzione errata
            err_str = str(e)
            # Sopprimi il traceback lungo di CasADi, mostra solo la causa
            causa = err_str.split("\n")[-2] if "\n" in err_str else err_str
            print(f"[MPC_LPV] Solver fallito "
                  f"(phi={np.rad2deg(phi):.1f}°, "
                  f"theta={np.rad2deg(theta):.1f}°): {causa}")
            print("[MPC_LPV] Applico hover di emergenza.")
            u_opt      = self.F_eq.copy()
            self.X_sol = None
            self.U_sol = None
            # Re-inizializza il warm start con hovering per il prossimo step
            self.opti.set_initial(
                self.U_var,
                np.tile(self.F_eq.reshape(-1, 1), (1, self.N))
            )

        return u_opt