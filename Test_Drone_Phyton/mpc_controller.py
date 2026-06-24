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
- **Obstacle Avoidance**: il costo include un termine di repulsione
  gaussiana centrato sul punto di ostacolo più vicino rilevato dal
  LiDAR (lidar_sim.py), passato tramite l'argomento opzionale
  `obs_pos` di calcola(). Non è un vincolo (subject_to): è una
  penalità additiva nella funzione di costo, scelta deliberatamente
  per evitare il rischio di infeasibility tipico dei vincoli rigidi
  e per ottenere un gradiente liscio che fa deviare la traiettoria
  con anticipo, in modo fluido, anziché con una correzione brusca
  all'ultimo istante.
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

# ── Obstacle Avoidance: repulsione nel costo (non vincolo) ──────────
# Il termine repulsivo è una penalità quadratica-gaussiana additiva
# nella funzione di costo, non un vincolo (subject_to). Questa scelta
# è deliberata: un vincolo (hard o soft-con-slack) può rendere il
# problema infeasible o introdurre un "gomito" nel gradiente vicino
# al bordo del vincolo. Una penalità di costo è invece sempre
# risolvibile per costruzione e offre un gradiente liscio che fa
# deviare la traiettoria con anticipo, in modo fluido.
#
#   costo_ostacolo(X_k) = W_OBS * exp(-||X_k[0:3] - obs_pos||² / (2*SIGMA_OBS²))
#
# Quando nessun ostacolo è rilevato dal LiDAR, obs_pos viene spostato
# a un punto lontanissimo (vedi main_volo.py): l'esponenziale decade
# a un valore numericamente nullo e il termine diventa trascurabile,
# senza bisogno di attivare/disattivare il termine nella struttura
# del problema CasADi.
#
# NOTA SUL TUNING (problema riscontrato, due fasi):
#
# Fase 1 — con SIGMA_OBS=0.6 il drone si schiantava: l'orizzonte di
# predizione di 0.6s (Ts=0.01s, N=60) è troppo corto perché il drone,
# vincolato a ±22°, possa fisicamente completare una deviazione
# laterale se comincia a reagire solo quando è già vicinissimo.
#
# Fase 2 — alzando sigma a 1.2 e il peso a 2500 per "vedere prima",
# il drone iniziava a fuggire anche da ostacoli lontani (2-3m) e
# irrilevanti, perché a quelle distanze il costo repulsivo superava
# il costo di restare vicino al target: il gaussiano era troppo
# "piatto" e si estendeva ben oltre la zona di reale pericolo.
#
# Soluzione: una curva più localizzata (sigma più piccolo) ma con
# picco più alto (peso maggiore). Il campo deve essere trascurabile
# a 3m (il drone ignora ostacoli lontani), già significativo verso
# 1-1.5m (dove deve iniziare a virare, dato che l'orizzonte gli
# concede solo quella finestra per farlo), e molto forte sotto 0.5m
# (barriera quasi invalicabile in prossimità reale).
#
#   costo_ostacolo(X_k) = W_OBS * exp(-||X_k[0:3] - obs_pos||² / (2*SIGMA_OBS²))
#
_SIGMA_OBS    = 0.8     # "raggio" del campo repulsivo [m]
_W_OBS        = 2000.0  # peso della repulsione nel costo totale


class MPC_Lineare:
    """
    Controllore MPC Lineare a Parametri Variabili per quadricottero.

    Ad ogni chiamata di calcola(), lo schedulatore LPV aggiorna le
    matrici Ad e Bd in base all'assetto corrente del drone, replicando
    il comportamento del blocco LPV_Scheduler in Simulink.
    """

    def __init__(self, Ts=_Ts_DEFAULT, N=_N_DEFAULT, debug_obstacle=False):
        self.Ts = Ts
        self.N  = N
        self.debug_obstacle = debug_obstacle  # stampa diagnostica costi

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
        self._target_prev = None  # per rilevare cambi bruschi di target
        self._obs_pos_filtrato = None  # punto ostacolo con smoothing
                                        # esponenziale (riduce i salti
                                        # discreti tra raggi LiDAR
                                        # adiacenti, causa di
                                        # oscillazioni nel comando)

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
        obs_pos_p  = opti.parameter(3)     # posizione ostacolo più vicino
                                            # (rilevato dal LiDAR); usato
                                            # solo nel termine repulsivo
                                            # del costo, nessun vincolo
        peso_obs_p = opti.parameter()      # peso scalare direzionale: riduce
                                            # la repulsione se l'ostacolo non
                                            # è nella direzione del target
                                            # (calcolato in calcola())

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

            # ── Repulsione ostacolo (solo primi N//3 step) ──
            # Limitare ai primi ~0.2s di predizione riduce la magnitudine
            # totale del costo repulsivo di 3x rispetto all'orizzonte
            # completo, ribilanciandolo con il tracking cost.
            # Il peso peso_obs_p è scalato direzionalmente in calcola().
            if k < self.N // 3:
                dist_sq_obs = ca.sumsqr(X[0:3, k] - obs_pos_p)
                cost += peso_obs_p * ca.exp(-dist_sq_obs / (2 * _SIGMA_OBS**2))

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
                'max_iter'                 : 350,  # alzato da 200: la
                # repulsione gaussiana rende il problema non convesso,
                # serve più margine quando il warm start non è vicino
                # alla soluzione (es. dopo un cambio brusco di target)
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
        self.obs_pos_p  = obs_pos_p
        self.peso_obs_p = peso_obs_p

        # Inizializzazione con matrici LTI (primo step)
        opti.set_value(self.Ad_p, self.Ad0)
        opti.set_value(self.Bd_p, self.Bd0)

        # Inizializzazione ostacolo: punto lontanissimo, repulsione
        # numericamente nulla finché main_volo.py non aggiorna il
        # parametro con un rilevamento reale del LiDAR
        opti.set_value(self.obs_pos_p,  np.array([1e3, 1e3, 1e3]))
        opti.set_value(self.peso_obs_p, _W_OBS)

    def calcola(self, x_curr, target_pos, u_prev, obs_pos=None):
        """
        Risolve il problema MPC per lo step corrente.

        Flusso (corrisponde al loop in Simulink):
          1. Legge phi, theta, psi dallo stato corrente
          2. Chiama lo schedulatore LPV per aggiornare Ad, Bd
          3. Aggiorna i parametri del problema CasADi (incluso
             l'eventuale ostacolo rilevato dal LiDAR)
          4. Applica warm start dalle soluzioni precedenti
          5. Risolve e restituisce il primo controllo ottimale

        Args:
            x_curr (np.ndarray): Vettore di stato corrente (12,).
            target_pos (np.ndarray): Posizione target [x, y, z] (3,).
            u_prev (np.ndarray): Comando motori al passo precedente (4,).
            obs_pos (np.ndarray|None): Posizione [x,y,z] del punto di
                ostacolo più vicino rilevato dal LiDAR. Se None (default,
                nessun ostacolo rilevato), il termine repulsivo viene
                resa numericamente nulla spostando il parametro interno
                a un punto lontanissimo.

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

        # Ostacolo: se non rilevato, punto lontanissimo -> repulsione
        # numericamente nulla (exp(-grande) ≈ 0), nessun effetto sul costo
        #
        # SMOOTHING: il punto rilevato dal LiDAR può "saltare" in modo
        # discreto da uno step all'altro, perché la scansione usa un
        # numero finito di raggi (es. con un cilindro stretto a media
        # distanza, può capitare che un solo raggio lo intercetti, e
        # quel raggio specifico cambia leggermente mentre il drone si
        # muove). Questo causava oscillazioni nel comando MPC, perché
        # il termine repulsivo "saltava" insieme al punto. Un filtro
        # esponenziale (media mobile pesata sul valore precedente)
        # rende il punto usato nel costo continuo nel tempo, anche se
        # il rilevamento grezzo non lo è.
        ALPHA_SMOOTH = 0.3  # 0=nessun aggiornamento, 1=nessun filtro
        if obs_pos is not None:
            obs_pos_raw = np.asarray(obs_pos, dtype=np.float64)
            if self._obs_pos_filtrato is None:
                self._obs_pos_filtrato = obs_pos_raw.copy()
            else:
                # Se il nuovo punto è molto lontano dal precedente,
                # è più probabile che sia un ostacolo diverso (non
                # rumore di scansione): in tal caso non filtrare,
                # adotta subito il nuovo punto.
                salto = np.linalg.norm(obs_pos_raw - self._obs_pos_filtrato)
                if salto > 1.0:
                    self._obs_pos_filtrato = obs_pos_raw.copy()
                else:
                    self._obs_pos_filtrato = (
                        ALPHA_SMOOTH * obs_pos_raw
                        + (1 - ALPHA_SMOOTH) * self._obs_pos_filtrato
                    )
            self.opti.set_value(self.obs_pos_p, self._obs_pos_filtrato)

            # Peso direzionale: se l'ostacolo è dietro o molto laterale
            # rispetto alla direzione del target, la repulsione è ridotta
            # (il drone non deve deviare per ostacoli che non ostruiscono
            # il percorso). cos_angolo ∈ [0.05, 1.0] → peso ∈ [5%, 100%].
            dir_target = target_pos - x_curr[0:3]
            dir_obs    = self._obs_pos_filtrato - x_curr[0:3]
            n_t = np.linalg.norm(dir_target)
            n_o = np.linalg.norm(dir_obs)
            if n_t > 0.1 and n_o > 0.1:
                cos_angolo = np.dot(dir_target, dir_obs) / (n_t * n_o)
                peso_dir   = float(np.clip(cos_angolo, 0.05, 1.0))
            else:
                peso_dir = 1.0
            self.opti.set_value(self.peso_obs_p, _W_OBS * peso_dir)
        else:
            self._obs_pos_filtrato = None
            self.opti.set_value(self.obs_pos_p,  np.array([1e3, 1e3, 1e3]))
            self.opti.set_value(self.peso_obs_p, _W_OBS)

        # ── DEBUG diagnostico (attivabile con debug_obstacle=True) ──
        # Confronta il peso "attrattivo" verso il target con il peso
        # "repulsivo" dell'ostacolo per lo stato corrente.
        if self.debug_obstacle and obs_pos is not None:
            dist_target = np.linalg.norm(x_curr[0:3] - target_pos)
            dist_obs    = np.linalg.norm(x_curr[0:3] - np.asarray(obs_pos))
            costo_target_singolo = 100.0 * dist_target**2
            costo_obs_singolo    = _W_OBS * np.exp(
                -dist_obs**2 / (2 * _SIGMA_OBS**2)
            )
            print(f"[DEBUG OBS] dist_target={dist_target:.2f}m "
                  f"dist_obs={dist_obs:.2f}m | "
                  f"costo_target≈{costo_target_singolo:.0f} "
                  f"costo_obs≈{costo_obs_singolo:.1f}")

        # 3. Warm Start: inizializza il solver con la soluzione precedente
        # Shifta la soluzione precedente di un passo
        # (stesso meccanismo di u_guess in Q_test_close_LMPC.m)
        #
        # ECCEZIONE: se il target è cambiato bruscamente rispetto allo
        # step precedente (es. l'utente ha appena richiesto un nuovo
        # punto molto distante), il vecchio warm start punta verso una
        # traiettoria non più rilevante. Forzare IPOPT a partire da lì
        # è spesso peggio che ripartire da una stima neutra: la
        # combinazione di warm start "sbagliato" + funzione di costo
        # non convessa (il termine repulsivo è una gaussiana, non un
        # termine quadratico) può far convergere il solver verso un
        # punto stazionario instabile o non convergere affatto entro
        # max_iter, da cui il fallback su hovering osservato.
        target_cambiato_bruscamente = False
        if self._target_prev is not None:
            salto_target = np.linalg.norm(target_pos - self._target_prev)
            # Soglia: un salto superiore a quanto il drone può
            # percorrere nell'intero orizzonte di predizione (stima
            # conservativa con velocità max ~2 m/s su N*Ts secondi)
            soglia_salto = 2.0 * (self.N * self.Ts) + 0.5
            target_cambiato_bruscamente = salto_target > soglia_salto

        self._target_prev = target_pos.copy()

        if self.X_sol is not None and not target_cambiato_bruscamente:
            Xg = np.hstack([self.X_sol[:, 1:], self.X_sol[:, -1:]])
            Ug = np.hstack([self.U_sol[:, 1:], self.U_sol[:, -1:]])
            self.opti.set_initial(self.X_var, Xg)
            self.opti.set_initial(self.U_var, Ug)
        else:
            # Target cambiato bruscamente o prima iterazione: warm
            # start neutro. Stato: rampa lineare dallo stato corrente
            # verso il nuovo target (più informativo del solo
            # hovering fisso, ma senza l'azzardo del vecchio warm
            # start ormai irrilevante).
            if target_cambiato_bruscamente:
                Xg = np.zeros((12, self.N + 1))
                for k in range(self.N + 1):
                    alpha = k / self.N
                    Xg[0:3, k] = (1 - alpha) * x_curr[0:3] + alpha * target_pos
                self.opti.set_initial(self.X_var, Xg)
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
            info_salto = (" [TARGET CAMBIATO BRUSCAMENTE]"
                          if target_cambiato_bruscamente else "")
            print(f"[MPC_LPV] Solver fallito "
                  f"(phi={np.rad2deg(phi):.1f}°, "
                  f"theta={np.rad2deg(theta):.1f}°){info_salto}: {causa}")
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