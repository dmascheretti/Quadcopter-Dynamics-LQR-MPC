import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

from mpc_controller import MPC_Lineare
from parametri_drone import F_eq

# ── INIZIALIZZAZIONE ──────────────────────────────────────────
print("Inizializzazione MPC Lineare...")
# Usiamo i parametri del codice di riferimento MATLAB
mpc = MPC_Lineare(Ts=0.01, N=60)
print("MPC pronto.")

# ── MUJOCO ───────────────────────────────────────────────────
xml_path = "skydio_x2/x2.xml"
if not os.path.exists(xml_path):
    print(f"ERRORE: {xml_path} non trovato.")
    exit()

model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

# Inizializza dalla keyframe hover
mujoco.mj_resetDataKeyframe(model, data, 0)

u_prev     = F_eq * np.ones(4)
loop_count = 0
# Ts del controllore è 0.01s, timestep della fisica è 0.01s.
# Quindi il controllore deve girare ad ogni step della fisica.
step_fisica = 1

# ── TARGET ───────────────────────────────────────────────────

# Definiamo una sequenza di waypoint da seguire (un quadrato in aria)
waypoints = [
    np.array([1.5, 1.0, 1.0]),   # Waypoint 1
    np.array([1.5, -1.0, 1.5]),  # Waypoint 2
    np.array([-1.5, -1.0, 1.0]), # Waypoint 3
    np.array([-1.5, 1.0, 1.5]),  # Waypoint 4
    np.array([0.0, 0.0, 1.0])    # Ritorno al centro
]
current_waypoint_index = 0
target = waypoints[current_waypoint_index]
distanza_soglia = 0.15 # Raggio di 15cm per considerare un waypoint raggiunto

def visualizza_traiettoria(viewer, punti):
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

# ── LOOP PRINCIPALE ──────────────────────────────────────────
print(f"Simulazione avviata. Inseguimento di {len(waypoints)} waypoints.")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        t_start = time.time()

        # --- LOGICA DI INSEGUIMENTO WAYPOINT ---
        # Calcola la distanza dal target attuale
        dist_dal_target = np.linalg.norm(data.qpos[0:3] - target)

        # Se il drone è abbastanza vicino, passa al waypoint successivo
        if dist_dal_target < distanza_soglia:
            if current_waypoint_index < len(waypoints) - 1:
                current_waypoint_index += 1
                target = waypoints[current_waypoint_index]
                print(f"Waypoint raggiunto! Prossimo target: {target}")

        # Aggiorna sfera target nella scena
        data.mocap_pos[0] = target

        if loop_count % step_fisica == 0:

            # ── Lettura stato da MuJoCo ──────────────────────
            pos     = data.qpos[0:3].copy()
            quat_mj = data.qpos[3:7].copy()       # [w, x, y, z]

            # Converti quaternione in angoli ZYX (come Raffo)
            rot   = R.from_quat([quat_mj[1], quat_mj[2],
                                  quat_mj[3], quat_mj[0]])  # scipy: [x,y,z,w]
            euler = rot.as_euler('ZYX')[::-1]                # → [phi, theta, psi]

            vel     = data.qvel[0:3].copy()
            ang_vel = data.qvel[3:6].copy()

            x_curr = np.concatenate([pos, euler, vel, ang_vel])

            # ── MPC ──────────────────────────────────────────
            u_opt  = mpc.calcola(x_curr, target, u_prev)
            u_prev = u_opt

            # ── Traiettoria predetta ─────────────────────────
            if mpc.X_sol is not None:
                traj = mpc.X_sol[0:3, :].T
                # La visualizzazione della traiettoria è stata disabilitata per massimizzare le performance.
                # visualizza_traiettoria(viewer, traj)

        # ── Fisica ───────────────────────────────────────────
        data.ctrl[:] = u_prev
        mujoco.mj_step(model, data)
        viewer.sync()

        loop_count += 1

        dt = model.opt.timestep - (time.time() - t_start)
        if dt > 0:
            time.sleep(dt)