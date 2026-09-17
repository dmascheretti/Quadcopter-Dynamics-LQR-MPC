"""
Loop principale per la simulazione del volo del drone.

Questo script orchestra l'intera simulazione, integrando:
- La fisica del drone Skydio X2 gestita da MuJoCo.
- Il controllore LPV-MPC implementato in `mpc_controller.py`.
- La ricezione di comandi di target via UDP da un client esterno.
- Una funzione opzionale per simulare un disturbo di vento costante.
- Un sensore LiDAR simulato (lidar_sim.py) per la percezione
  dell'ambiente, con scansione completa a 360° e visualizzazione
  dei raggi e dei punti di impatto nel viewer. In questa fase il
  LiDAR è puramente percettivo: scansiona e visualizza, ma non è
  ancora collegato al controllore MPC.
- Il logging dei dati di volo per analisi successive.

Per controllare il drone, è necessario avviare in parallelo uno dei
client UDP, come `joystick_client.py` o `joystick_controller_tastiera.py`.
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import socket
import json
import os
import pickle

from mpc_controller import MPC_Lineare
from parametri_drone import F_eq
from lidar_sim import LidarSim, visualizza_lidar

# Configurazione della simulazione

# Disturbo del vento
# Mettere a False per simulazione pulita (senza disturbi)
ABILITA_VENTO = False
WIND_FORCE    = np.array([2.0, -1.0, 0.0])  # [N] lungo X, Y, Z

# Scena da caricare: False = x2.xml (3 ostacoli, quella usata per
# tutti i test/figure del Cap. 5 validati); True = x2_copertina.xml
# (10 ostacoli variati, creata solo per lo screenshot del Cap. 1 —
# non è mai stata testata per i risultati di obstacle avoidance).
USA_SCENA_COPERTINA = True

# Logging dei dati
ABILITA_LOG   = True
LOG_FILE      = "risultati_simulazione.pkl"

# Parametri UDP per la ricezione del target
UDP_IP   = "127.0.0.1"
UDP_PORT = 5005

# Configurazione LiDAR
ABILITA_LIDAR     = True
LIDAR_N_RAGGI     = 32      # numero di raggi nella scansione completa
                             # (alzato da 16: con 16 raggi la spaziatura
                             # angolare di 22.5° era più larga della
                             # larghezza apparente di un ostacolo
                             # cilindrico tipico a 2m di distanza, ~9°.
                             # Il drone "perdeva" l'ostacolo tra un
                             # raggio e l'altro durante il movimento,
                             # causando salti discreti nel punto
                             # rilevato e oscillazioni nel comando MPC)
LIDAR_SETTORE_DEG = 360.0   # scansione completa a 360°
                             # (necessaria: il drone trasla in ogni
                             # direzione senza dover ruotare in yaw,
                             # quindi un settore frontale lascerebbe
                             # punti ciechi su fianchi e retro)
LIDAR_RANGE_MAX   = 5.0     # distanza massima rilevabile [m]
LIDAR_OGNI_N_STEP = 1       # esegui la scansione ogni step fisico
                             # (era 2: con l'ostacolo vicino al target,
                             # un rilevamento più frequente riduce il
                             # ritardo tra movimento del drone e
                             # aggiornamento del punto di ostacolo
                             # passato all'MPC, contribuendo a
                             # comandi più stabili)
LIDAR_DIST_ATTIVAZIONE = 2.5 # [m] soglia di distanza: ostacoli oltre
                              # questa distanza NON vengono passati
                              # all'MPC come penalità repulsiva. Vengono
                              # comunque visualizzati dal viewer. Evita
                              # che il drone "abbia paura" di ostacoli
                              # lontani che non interferiscono ancora
                              # con la traiettoria.

# Bypass waypoint: quando un ostacolo blocca il percorso diretto
# drone→target, si sostituisce temporaneamente il target con un
# punto laterale libero. L'MPC non deve più trovare una traiettoria
# "attraverso" l'ostacolo, problema che causa divergenze del solver.
BYPASS_MARGINE    = 1.4   # [m] distanza laterale del waypoint di bypass
                           # dall'asse drone→target proiettato sull'ostacolo
                           # (alzato da 1.2: più margine riduce il rischio
                           # che il bypass sia ancora nella zona repulsiva)
BYPASS_ANGOLO_MAX = 0.40  # [rad] ≈ 23°: se il raggio LiDAR più vicino
                           # alla direzione del target sta entro questo
                           # angolo E colpisce qualcosa prima del target,
                           # il percorso è considerato bloccato
BYPASS_ALPHA      = 0.35  # coefficiente smoothing esponenziale del bypass
BYPASS_DWELL_MIN  = 60    # [step] permanenza minima dopo l'attivazione
                           # (~0.6s a Ts=0.01s): nei primi BYPASS_DWELL_MIN
                           # passi la condizione di disattivazione non viene
                           # valutata, per limitare il chattering on/off
                           # quando il drone è sull'edge della zona di
                           # attivazione. Non elimina il chattering, ne
                           # limita la frequenza: se dai log risulta
                           # un'oscillazione lenta (non rapida) va rivisto.
BYPASS_TARGET_CAMBIO_SOGLIA = 0.5  # [m] se durante il dwell il target
                           # reale si sposta oltre questa soglia (es.
                           # nuovo comando joystick), il dwell si azzera
                           # e si ricalcola il bypass subito
                           # 0=fisso sul valore precedente, 1=nessun filtro

# Inizializzazione del controllore MPC
print("Inizializzazione MPC LPV...")
mpc = MPC_Lineare(Ts=0.01, N=60)
print("MPC pronto.")
# Inizializzazione di MuJoCo
xml_path = ("skydio_x2/x2_copertina.xml" if USA_SCENA_COPERTINA
            else "skydio_x2/x2.xml")
if not os.path.exists(xml_path):
    print(f"ERRORE: {xml_path} non trovato.")
    exit()

model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

# ID corpo drone per applicare forze esterne e per escluderlo
# dalla scansione LiDAR (altrimenti i raggi colpirebbero il
# drone stesso all'origine)
body_id_x2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x2")

# ID del body della sfera target: serve per escludere la sua
# geometria dal raycasting del LiDAR. La sfera resta visibile a
# schermo (gruppo geom di default), ma senza questa esclusione il
# drone la rileverebbe come un ostacolo quando le si avvicina,
# generando un conflitto diretto nel costo dell'MPC tra il termine
# attrattivo (target) e quello repulsivo (ostacolo).
body_id_target = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "target_waypoint"
)

# Inizializzazione del LiDAR
# (inizializzata sempre, anche a LiDAR disabilitato: calcola_target_effettivo()
# e il loop principale la referenziano incondizionatamente)
ultima_scansione = None
if ABILITA_LIDAR:
    lidar = LidarSim(
        model,
        n_raggi=LIDAR_N_RAGGI,
        settore_deg=LIDAR_SETTORE_DEG,
        range_max=LIDAR_RANGE_MAX,
        bodyexclude=body_id_x2,
        escludi_body_ids=(body_id_target,),
    )
    print(f"LiDAR attivo: {LIDAR_N_RAGGI} raggi, "
          f"settore {LIDAR_SETTORE_DEG:.0f}°, "
          f"range {LIDAR_RANGE_MAX:.1f}m")
else:
    print("LiDAR disabilitato.")

# Inizializza dalla keyframe hover
mujoco.mj_resetDataKeyframe(model, data, 0)

# Legge la posizione iniziale reale dal modello
# (evita un transitorio brusco al primo step MPC)
target = data.qpos[0:3].copy()

if ABILITA_VENTO:
    print(f"Vento attivo: {WIND_FORCE} N")
else:
    print("Vento disabilitato.")

# Inizializzazione del socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# Preparazione della struttura per il logging
if ABILITA_LOG:
    log = {
        't'      : [],   # tempo simulazione [s]
        'pos'    : [],   # posizione [x,y,z] m
        'euler'  : [],   # angoli [phi,theta,psi] rad
        'vel'    : [],   # velocità lineari [vx,vy,vz] m/s
        'ang_vel': [],   # velocità angolari [p,q,r] rad/s
        'u'      : [],   # forze motore [F1,F2,F3,F4] N
        'target' : [],   # target corrente [x,y,z] m
        'vento'  : WIND_FORCE.tolist() if ABILITA_VENTO else None,
        'lidar_dist_min': [],  # distanza minima rilevata dal LiDAR [m]
        # --- metriche computazionali / bypass ---
        't_solve'    : [],   # tempo di soluzione del QP [s]
        'target_eff' : [],   # target effettivo (bypass o reale)
        'obs_pos'    : [],   # punto ostacolo passato all'MPC (nan se assente)
        'obs_attivo' : [],   # True se la repulsione è attiva
    }

# Variabili per il loop di simulazione
u_prev     = F_eq * np.ones(4)
loop_count = 0
step_fisica = 1  # MPC gira ogni step fisico (Ts=0.01s)

# Stato del bypass waypoint (persistente tra le chiamate)
_bypass_pos_filtrato       = None   # posizione bypass con smoothing esponenziale
_bypass_attivo             = False  # True se il bypass era attivo allo step precedente
_bypass_dwell_rimanente    = 0      # passi residui di permanenza minima obbligatoria
_bypass_target_attivazione = None   # target reale al momento dell'attivazione del bypass


def _valuta_bypass_raw(drone_pos, target_reale, scan):
    """
    Valuta se, in base alla geometria istantanea, serve un bypass e, se
    sì, ne calcola la posizione grezza (prima dello smoothing). Restituisce
    None se il bypass non è necessario o non calcolabile in questo istante.
    Non gestisce isteresi/dwell: quella logica vive in
    calcola_target_effettivo(), questa funzione è pura geometria.
    """
    if scan is None:
        return None

    dir_t  = target_reale - drone_pos
    dist_t = np.linalg.norm(dir_t)
    if dist_t < 0.3:
        return None

    dir_tn     = dir_t / dist_t
    ang_target = np.arctan2(dir_tn[1], dir_tn[0])

    angoli = scan['angoli_abs']
    delta  = (angoli - ang_target + np.pi) % (2 * np.pi) - np.pi
    idx    = int(np.argmin(np.abs(delta)))

    bypass_necessario = (
        np.abs(delta[idx]) <= BYPASS_ANGOLO_MAX
        and scan['hit'][idx]
        and scan['distanze'][idx] < dist_t - 0.2
    )
    if not bypass_necessario:
        return None

    obs_pt = scan['punti'][idx]
    proj   = drone_pos + np.dot(obs_pt - drone_pos, dir_tn) * dir_tn

    perp_r = np.array([ dir_tn[1], -dir_tn[0], 0.0])
    perp_l = np.array([-dir_tn[1],  dir_tn[0], 0.0])

    ang_r = np.arctan2(perp_r[1], perp_r[0])
    ang_l = np.arctan2(perp_l[1], perp_l[0])
    dr    = (angoli - ang_r + np.pi) % (2 * np.pi) - np.pi
    dl    = (angoli - ang_l + np.pi) % (2 * np.pi) - np.pi
    d_r   = scan['distanze'][int(np.argmin(np.abs(dr)))]
    d_l   = scan['distanze'][int(np.argmin(np.abs(dl)))]
    perp  = perp_r if d_r >= d_l else perp_l

    bypass_raw    = proj + perp * BYPASS_MARGINE
    bypass_raw[2] = drone_pos[2]
    return bypass_raw


def calcola_target_effettivo(drone_pos, target_reale, scan):
    """
    Restituisce il target effettivo da passare all'MPC.

    Se un ostacolo blocca il percorso diretto drone→target, calcola un
    waypoint laterale (bypass) sul lato più aperto. Applica:
    - smoothing esponenziale (BYPASS_ALPHA) per evitare salti bruschi
      del target che invalidano il warm start di IPOPT
    - isteresi sull'angolo di attivazione: il bypass si disattiva solo
      quando il percorso è chiaramente libero (soglia x1.5), non appena
      esce dalla zona borderline, evitando oscillazioni rapide on/off
    - permanenza minima (dwell, BYPASS_DWELL_MIN passi): appena il
      bypass si attiva, la condizione di disattivazione non viene
      nemmeno valutata per un tempo minimo, per limitare ulteriormente
      il chattering rapido; l'unica eccezione è un cambio sostanziale
      del target reale (es. nuovo comando joystick), che azzera
      immediatamente il dwell e forza una rivalutazione
    """
    global _bypass_pos_filtrato, _bypass_attivo
    global _bypass_dwell_rimanente, _bypass_target_attivazione

    # --- Bypass già attivo e ancora in dwell: non valutare la
    # disattivazione, salvo cambio sostanziale del target reale. ---
    if _bypass_attivo and _bypass_dwell_rimanente > 0:
        cambio_target = (
            _bypass_target_attivazione is not None
            and np.linalg.norm(target_reale - _bypass_target_attivazione)
                > BYPASS_TARGET_CAMBIO_SOGLIA
        )
        if not cambio_target:
            _bypass_dwell_rimanente -= 1
            # Aggiorna la stima se possibile (segue un ostacolo che si
            # sposta leggermente nella scansione); altrimenti mantiene
            # l'ultimo waypoint filtrato invece di sterzare di scatto.
            bypass_raw = _valuta_bypass_raw(drone_pos, target_reale, scan)
            if bypass_raw is not None and _bypass_pos_filtrato is not None:
                _bypass_pos_filtrato = (
                    BYPASS_ALPHA * bypass_raw
                    + (1.0 - BYPASS_ALPHA) * _bypass_pos_filtrato
                )
            if _bypass_pos_filtrato is not None:
                return _bypass_pos_filtrato.copy()
        else:
            # Target cambiato: azzera il dwell, si ricalcola tutto sotto.
            _bypass_dwell_rimanente = 0
            _bypass_attivo = False
            _bypass_pos_filtrato = None

    # --- Valutazione normale (bypass non attivo, o dwell esaurito) ---
    if scan is None:
        _bypass_pos_filtrato = None
        _bypass_attivo = False
        return target_reale.copy()

    dist_t = np.linalg.norm(target_reale - drone_pos)
    if dist_t < 0.3:
        _bypass_pos_filtrato = None
        _bypass_attivo = False
        return target_reale.copy()

    # Isteresi: soglia più ampia per disattivare il bypass rispetto ad
    # attivarlo (si applica solo dopo il dwell, quando si può ancora
    # essere in bypass ma con permanenza minima già esaurita).
    if _bypass_attivo:
        dir_t      = target_reale - drone_pos
        dir_tn     = dir_t / dist_t
        ang_target = np.arctan2(dir_tn[1], dir_tn[0])
        angoli     = scan['angoli_abs']
        delta      = (angoli - ang_target + np.pi) % (2 * np.pi) - np.pi
        idx        = int(np.argmin(np.abs(delta)))
        bypass_necessario = (
            np.abs(delta[idx]) <= BYPASS_ANGOLO_MAX * 1.5
            and scan['hit'][idx]
            and scan['distanze'][idx] < dist_t - 0.2
        )
        if not bypass_necessario:
            _bypass_pos_filtrato = None
            _bypass_attivo = False
            return target_reale.copy()
        bypass_raw = _valuta_bypass_raw(drone_pos, target_reale, scan)
        if bypass_raw is None:
            _bypass_pos_filtrato = None
            _bypass_attivo = False
            return target_reale.copy()
    else:
        bypass_raw = _valuta_bypass_raw(drone_pos, target_reale, scan)
        if bypass_raw is None:
            return target_reale.copy()

    # Smoothing esponenziale: evita salti bruschi che invaliderebbero il
    # warm start di IPOPT e causerebbero fallimenti del solver.
    appena_attivato = not _bypass_attivo
    if _bypass_pos_filtrato is None:
        _bypass_pos_filtrato = bypass_raw.copy()
    else:
        _bypass_pos_filtrato = (
            BYPASS_ALPHA * bypass_raw
            + (1.0 - BYPASS_ALPHA) * _bypass_pos_filtrato
        )

    if appena_attivato:
        _bypass_dwell_rimanente    = BYPASS_DWELL_MIN
        _bypass_target_attivazione = target_reale.copy()

    _bypass_attivo = True
    return _bypass_pos_filtrato.copy()


def visualizza_bypass(viewer, bypass_pos):
    """Disegna una sfera ciano in corrispondenza del waypoint di bypass."""
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.12, 0.0, 0.0]),
        pos=np.array(bypass_pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.array([0.0, 0.9, 0.9, 0.8])
    )
    viewer.user_scn.ngeom += 1


def visualizza_traiettoria(viewer, punti):
    """Disegna sfere arancioni lungo la traiettoria predetta dall'MPC."""
    if punti is None:
        return
    rgba = np.array([1.0, 0.65, 0.0, 0.8])
    size = np.array([0.02, 0.0, 0.0])
    for p in punti:
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=size,
            pos=np.array(p, dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=rgba
        )
        viewer.user_scn.ngeom += 1

# Avvio del loop principale di simulazione
print(f"Simulazione avviata.")
print(f"In ascolto per target su UDP {UDP_IP}:{UDP_PORT}")
print("Avviare joystick_controller_tastiera.py o joystick_client.py")
print("Chiudere il viewer per terminare e salvare i dati.")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        t_start = time.time()

        # Applica la forza del vento, se abilitata
        if ABILITA_VENTO:
            data.xfrc_applied[body_id_x2, 0:3] = WIND_FORCE

        # Ricezione del target via UDP
        # svuota il buffer: prende l'ultimo pacchetto disponibile
        # (evita che si accumuli ritardo se il client è più veloce)
        ultimo_pacchetto = None
        try:
            while True:
                dati_udp, _ = sock.recvfrom(1024)
                ultimo_pacchetto = dati_udp
        except BlockingIOError:
            pass

        if ultimo_pacchetto is not None:
            try:
                td = json.loads(ultimo_pacchetto.decode())
                target[0] = float(td['x'])
                target[1] = float(td['y'])
                target[2] = float(td['z'])
            except (json.JSONDecodeError, KeyError):
                pass  # pacchetto malformato, ignora

        # Visualizza sfera target nella scena
        data.mocap_pos[0] = target

        # Calcolo del comando MPC
        if loop_count % step_fisica == 0:

            # Lettura stato da MuJoCo
            pos     = data.qpos[0:3].copy()
            quat_mj = data.qpos[3:7].copy()   # MuJoCo: [w, x, y, z]

            # Conversione quaternione → angoli di Eulero ZYX
            # scipy usa [x,y,z,w], inversione finale → [phi, theta, psi]
            rot   = R.from_quat([quat_mj[1], quat_mj[2],
                                  quat_mj[3], quat_mj[0]])
            euler = rot.as_euler('ZYX')[::-1]   # [phi, theta, psi]

            vel     = data.qvel[0:3].copy()
            ang_vel = data.qvel[3:6].copy()

            x_curr = np.concatenate([pos, euler, vel, ang_vel])

            # Scansione LiDAR — eseguita PRIMA della chiamata all'MPC,
            # così il controllore usa il rilevamento più recente
            # possibile (non quello del passo precedente) per il
            # termine repulsivo nel costo.
            if ABILITA_LIDAR and loop_count % LIDAR_OGNI_N_STEP == 0:
                yaw_corrente = euler[2]
                ultima_scansione = lidar.scansiona(
                    data, pos, yaw=yaw_corrente
                )

            # Punto di ostacolo da passare all'MPC: il più vicino
            # rilevato nell'ultima scansione disponibile. None se il
            # LiDAR è disabilitato o non ha ancora scansionato nulla
            # (primo step) — in tal caso mpc.calcola() usa il default
            # "lontanissimo" che rende la repulsione trascurabile.
            obs_pos_lidar = None
            if ABILITA_LIDAR and ultima_scansione is not None:
                idx = ultima_scansione['idx_min']
                if (ultima_scansione['hit'][idx]
                        and ultima_scansione['dist_min'] < LIDAR_DIST_ATTIVAZIONE):
                    obs_pos_lidar = ultima_scansione['punti'][idx]

            # Bypass waypoint: se un ostacolo blocca il percorso
            # diretto verso il target, sostituisce temporaneamente
            # il target con un punto laterale libero, evitando il
            # conflitto attrattivo/repulsivo che causa divergenze
            # del solver quando il target è dietro/vicino a un ostacolo.
            target_eff = calcola_target_effettivo(
                pos, target, ultima_scansione
            )

            # Calcolo comando ottimale (con eventuale repulsione ostacolo)
            _t0      = time.perf_counter()
            u_opt    = mpc.calcola(x_curr, target_eff, u_prev,
                                    obs_pos=obs_pos_lidar)
            _t_solve = time.perf_counter() - _t0
            u_prev   = u_opt

            # Log dati
            if ABILITA_LOG:
                log['t'].append(loop_count * model.opt.timestep)
                log['pos'].append(pos.copy())
                log['euler'].append(euler.copy())
                log['vel'].append(vel.copy())
                log['ang_vel'].append(ang_vel.copy())
                log['u'].append(u_opt.copy())
                log['target'].append(target.copy())
                log['lidar_dist_min'].append(
                    ultima_scansione['dist_min']
                    if (ABILITA_LIDAR and ultima_scansione is not None)
                    else np.nan
                )
                log['t_solve'].append(_t_solve)
                log['target_eff'].append(target_eff.copy())
                log['obs_pos'].append(
                    obs_pos_lidar.copy() if obs_pos_lidar is not None
                    else np.full(3, np.nan)
                )
                log['obs_attivo'].append(obs_pos_lidar is not None)

        # Azzeramento scena utente e ridisegno di tutti gli elementi
        # grafici aggiuntivi (LiDAR + traiettoria MPC) in questo frame
        viewer.user_scn.ngeom = 0

        if ABILITA_LIDAR and ultima_scansione is not None:
            visualizza_lidar(viewer, ultima_scansione)

        # Visualizza sfera ciano se il bypass è attivo (target_eff != target)
        try:
            if np.linalg.norm(target_eff - target) > 0.05:
                visualizza_bypass(viewer, target_eff)
        except NameError:
            pass  # target_eff non ancora calcolato al primo frame

        if mpc.X_sol is not None:
            traj = mpc.X_sol[0:3, :].T
            visualizza_traiettoria(viewer, traj)

        # Avanzamento della simulazione fisica
        data.ctrl[:] = u_prev
        mujoco.mj_step(model, data)
        viewer.sync()

        loop_count += 1

        # Sincronizzazione tempo reale
        dt = model.opt.timestep - (time.time() - t_start)
        if dt > 0:
            time.sleep(dt)

# Salvataggio dei dati di log a fine simulazione
if ABILITA_LOG and len(log['t']) > 0:
    # Converti in array numpy per comodità
    for k in ['pos', 'euler', 'vel', 'ang_vel', 'u', 'target',
              'target_eff', 'obs_pos']:
        log[k] = np.array(log[k])
    log['t'] = np.array(log['t'])
    log['lidar_dist_min'] = np.array(log['lidar_dist_min'])
    log['t_solve']    = np.array(log['t_solve'])
    log['obs_attivo'] = np.array(log['obs_attivo'])

    with open(LOG_FILE, 'wb') as f:
        pickle.dump(log, f)
    print(f"\nDati salvati in '{LOG_FILE}' ({len(log['t'])} step, "
          f"{log['t'][-1]:.1f}s di simulazione)")
    print("Usare plot_risultati.py per visualizzare i grafici.")

    ts = log['t_solve'] * 1e3
    print(f"\nTempo di soluzione [ms]: media {ts.mean():.2f}  "
          f"mediana {np.median(ts):.2f}  p95 {np.percentile(ts, 95):.2f}  "
          f"max {ts.max():.2f}")
    print(f"Passi oltre Ts=10ms: {100 * (ts > 10).mean():.1f}%")
    if log['obs_attivo'].any():
        print(f"  con ostacolo:   media {ts[log['obs_attivo']].mean():.2f} ms")
    if (~log['obs_attivo']).any():
        print(f"  senza ostacolo: media {ts[~log['obs_attivo']].mean():.2f} ms")
else:
    print("\nSimulazione terminata (log disabilitato o nessun dato).")

sock.close()