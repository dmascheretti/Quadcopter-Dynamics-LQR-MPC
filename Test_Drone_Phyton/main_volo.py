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
ABILITA_VENTO = True
WIND_FORCE    = np.array([2.0, -1.0, 0.0])  # [N] lungo X, Y, Z

# Logging dei dati
ABILITA_LOG   = True
LOG_FILE      = "risultati_simulazione.pkl"

# Parametri UDP per la ricezione del target
UDP_IP   = "127.0.0.1"
UDP_PORT = 5005

# Configurazione LiDAR
ABILITA_LIDAR     = True
LIDAR_N_RAGGI     = 16      # numero di raggi nella scansione completa
LIDAR_SETTORE_DEG = 360.0   # scansione completa a 360°
                             # (necessaria: il drone trasla in ogni
                             # direzione senza dover ruotare in yaw,
                             # quindi un settore frontale lascerebbe
                             # punti ciechi su fianchi e retro)
LIDAR_RANGE_MAX   = 5.0     # distanza massima rilevabile [m]
LIDAR_OGNI_N_STEP = 2       # esegui la scansione ogni N step fisici
                             # (riduce il costo computazionale; il
                             # disegno usa l'ultima scansione disponibile)

# Inizializzazione del controllore MPC
print("Inizializzazione MPC LPV...")
mpc = MPC_Lineare(Ts=0.01, N=60)
print("MPC pronto.")
# Inizializzazione di MuJoCo
xml_path = "skydio_x2/x2.xml"
if not os.path.exists(xml_path):
    print(f"ERRORE: {xml_path} non trovato.")
    exit()

model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

# ID corpo drone per applicare forze esterne e per escluderlo
# dalla scansione LiDAR (altrimenti i raggi colpirebbero il
# drone stesso all'origine)
body_id_x2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x2")

# Inizializzazione del LiDAR
if ABILITA_LIDAR:
    lidar = LidarSim(
        model,
        n_raggi=LIDAR_N_RAGGI,
        settore_deg=LIDAR_SETTORE_DEG,
        range_max=LIDAR_RANGE_MAX,
        bodyexclude=body_id_x2,
    )
    ultima_scansione = None
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
    }

# Variabili per il loop di simulazione
u_prev     = F_eq * np.ones(4)
loop_count = 0
step_fisica = 1  # MPC gira ogni step fisico (Ts=0.01s)

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

            # Calcolo comando ottimale
            u_opt  = mpc.calcola(x_curr, target, u_prev)
            u_prev = u_opt

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

        # Scansione LiDAR (a frequenza ridotta rispetto al fisico)
        if ABILITA_LIDAR and loop_count % LIDAR_OGNI_N_STEP == 0:
            yaw_corrente = euler[2] if loop_count > 0 else 0.0
            ultima_scansione = lidar.scansiona(
                data, pos, yaw=yaw_corrente
            )

        # Azzeramento scena utente e ridisegno di tutti gli elementi
        # grafici aggiuntivi (LiDAR + traiettoria MPC) in questo frame
        viewer.user_scn.ngeom = 0

        if ABILITA_LIDAR and ultima_scansione is not None:
            visualizza_lidar(viewer, ultima_scansione)

        if mpc.X_sol is not None:
            traj = mpc.X_sol[0:3, :].T
            # Decommentare per visualizzare anche la traiettoria predetta
            # visualizza_traiettoria(viewer, traj)

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
    for k in ['pos', 'euler', 'vel', 'ang_vel', 'u', 'target']:
        log[k] = np.array(log[k])
    log['t'] = np.array(log['t'])
    log['lidar_dist_min'] = np.array(log['lidar_dist_min'])

    with open(LOG_FILE, 'wb') as f:
        pickle.dump(log, f)
    print(f"\nDati salvati in '{LOG_FILE}' ({len(log['t'])} step, "
          f"{log['t'][-1]:.1f}s di simulazione)")
    print("Usare plot_risultati.py per visualizzare i grafici.")
else:
    print("\nSimulazione terminata (log disabilitato o nessun dato).")

sock.close()