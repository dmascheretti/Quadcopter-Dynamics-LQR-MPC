# Quadcopter Dynamics, LQR & MPC

Modellazione matematica, simulazione non lineare e controllo avanzato (LQR discreto,
NMPC, LMPC e MPC a linearizzazione successiva) di un quadricottero sottoattuato, con
obstacle avoidance basato su LiDAR simulato.

Codice sviluppato per la tesi di laurea triennale in Ingegneria Informatica
**"Controllo Predittivo Lineare e Obstacle Avoidance per Quadricottero"**,
Università degli Studi di Bergamo, a.a. 2025/2026.

- **Relatore:** Prof. Antonio Ferramosca
- **Correlatore:** Marcelo Alves dos Santos

---

## Panoramica

Il lavoro percorre l'intera catena dalla modellazione al controllo:

1. **Modellazione** — derivazione simbolica delle equazioni del moto con
   formalismo di Eulero-Lagrange, verificata per confronto con Newton-Eulero e
   tramite la proprietà di antisimmetria di `M_dot - 2C`.
2. **Linearizzazione** — Jacobiani analitici attorno all'hovering, verifica di
   controllabilità, discretizzazione ZOH.
3. **Sintesi dei controllori** — LQR discreto come baseline, poi NMPC (modello non
   lineare, RK4), LMPC (modello linearizzato, QP convesso) e infine la formulazione
   a **linearizzazione successiva**, che ricalcola le matrici del modello attorno
   all'assetto corrente ad ogni passo di campionamento.
4. **Validazione** — simulazione su modello ad alta fedeltà dello Skydio X2 in
   MuJoCo, con scenari di hovering, tracking multi-waypoint, disturbo di vento e
   obstacle avoidance.

---

## Struttura del repository

```
├── matlab/            pipeline di modellazione, linearizzazione e controllo
│   └── MAT/           workspace intermedi (.mat) generati dagli script
└── python/            implementazione MPC + MuJoCo
    └── modelli/       modello XML dello Skydio X2 e scene di simulazione
```

### `matlab/` — pipeline sequenziale

Gli script vanno eseguiti **in ordine alfabetico**: ciascuno salva il workspace in
`MAT/` e lo step successivo lo ricarica.

| Script | Funzione |
|---|---|
| `A_coordinate.m` | Inizializza l'ambiente simbolico: coordinate generalizzate, derivate, parametri inerziali |
| `B_dinamica.m` | Imposta la struttura per l'approccio di Eulero-Lagrange |
| `C_matrici.m` | Calcola simbolicamente `R`, `W_eta`, `M(q)`, `C(q,q̇)` (simboli di Christoffel), `G(q)`, `B(q)` |
| `D_simulazione.m` | Sostituisce i parametri numerici e genera funzioni callable con `matlabFunction` |
| `E_equilibrio.m` | Jacobiani nel punto di hovering, verifica di controllabilità (rango 12) |
| `F1_test_NL.m` | Simulazione non lineare in anello aperto (`ode45`) |
| `F2_test_L.m` | Simulazione del sistema LTI linearizzato (`lsim`) |
| `F3_confronto.m` | Confronto tra i due modelli, dominio di validità della linearizzazione |
| `G_gamma.m` | Matrice di allocazione con rotor canting, spinta di hovering |
| `H_LQR.m` | Discretizzazione ZOH e sintesi del guadagno LQR |
| `I_test_tuning.m` | Confronto di tre profili di tuning (lento, bilanciato, aggressivo) |
| `J_LQR_NL.m` | LQR in anello chiuso sul modello non lineare, con saturazione attuatori |
| `K_tracking.m` | Tracking verso target distante: evidenzia la divergenza dell'LQR oltre 15° |
| `L_tracking_mission.m` | Tracking multi-waypoint con commutazione su tolleranza euclidea |
| `M_MPC_setup.m` | Setup CasADi per NMPC: discretizzazione RK4, vincoli, solutore IPOPT |
| `N_test_open.m` | NMPC in anello aperto: traiettoria ottima su singolo orizzonte |
| `O_test_close.m` | NMPC in anello chiuso con receding horizon e warm-start |
| `P_setup_LMPC.m` | Setup LMPC: modello linearizzato discretizzato, QP convesso |
| `Q_test_close_LMPC.m` | LMPC in anello chiuso su impianto non lineare (`ode45`) |
| `S_init_sim_LPV.m` | Configurazione con matrici aggiornabili come parametri del solutore |

### `python/` — simulazione MuJoCo

| Modulo | Funzione |
|---|---|
| `main_volo.py` | Ciclo principale: fisica MuJoCo, controllore, LiDAR, target via UDP, logging |
| `mpc_controller.py` | Controllore a linearizzazione successiva (CasADi + IPOPT), penalità repulsiva |
| `linearization_scheduler.py` | Ricalcolo di `A_c(φ,θ,ψ)` e discretizzazione ZOH (`scipy.linalg.expm`) |
| `lidar_sim.py` | LiDAR simulato: 32 raggi su 360°, range 5 m, raycasting via `mj_ray` |
| `joystick_client.py` | Client UDP per il comando interattivo del target da gamepad |
| `joystick_controller_tastiera.py` | Variante da tastiera con HUD |
| `parametri_drone.py` | Matrici `A_lin`, `B_lin` e spinta di hovering dello Skydio X2 |
| `leggi_fisica.py` | Estrazione di massa e inerzie dal file XML di MuJoCo |
| `plot_risultati.py` | Grafici da log `.pkl`: traiettoria 3D, angoli, forze, errore di posizione |

---

## Requisiti

**MATLAB** R2021b o successivo, con:
- Symbolic Math Toolbox
- Control System Toolbox
- [CasADi](https://web.casadi.org/) per MATLAB (per gli script `M`–`S`)

**Python** 3.10 o successivo:

```bash
pip install casadi mujoco numpy scipy matplotlib
```

---

## Esecuzione

### Pipeline MATLAB

Da `matlab/`, esegui gli script in ordine alfabetico. La cartella `MAT/` viene
popolata automaticamente:

```matlab
A_coordinate
B_dinamica
C_matrici
% ... e così via
```

### Simulazione MuJoCo

```bash
cd python
python main_volo.py
```

Il target si comanda via UDP. In una seconda shell:

```bash
python joystick_controller_tastiera.py   # da tastiera
python joystick_client.py                # da gamepad
```

I log vengono salvati in formato `.pkl` e si analizzano con:

```bash
python plot_risultati.py
```

---

## Parametri principali

| Parametro | Valore |
|---|---|
| Passo di campionamento | 10 ms |
| Orizzonte di predizione (LMPC / lin. successiva) | 60 passi |
| Orizzonte di predizione (NMPC) | 40 passi |
| Massa Skydio X2 | 1.325 kg |
| Spinta di hovering per motore | 3.25 N |
| Saturazione motori | 13 N |
| Limite angolare | ±22° |

---

## Note

- Il modello generico usato nella fase MATLAB ha configurazione a croce con rotori
  inclinati di 2°; il modello Skydio X2 ha configurazione a X senza canting.
- Il modello di predizione del controllore Python usa i parametri fisici estratti
  dal file XML dello Skydio X2.
- `R_init_sim.m` è una versione preliminare confluita in `P_setup_LMPC.m` e non è
  più richiamata dalla pipeline.

---

## Licenza

Codice rilasciato a scopo didattico e di consultazione.
Il modello dello Skydio X2 proviene da
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Apache License 2.0).
