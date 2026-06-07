import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

xml_path = "skydio_x2/x2.xml" 

if not os.path.exists(xml_path):
    print(f"ERRORE: Non trovo il file {xml_path}")
else:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        while viewer.is_running():
            step_start = time.time()
            
            # --- 1. LETTURA DELLO STATO (Sensori) ---
            # Posizione lineare (X, Y, Z)
            pos = data.qpos[0:3]
            
            # Orientamento (Quaternione: w, x, y, z in MuJoCo)
            quat_mujoco = data.qpos[3:7] 
            
            # Conversione da Quaternione a Eulero (Roll, Pitch, Yaw)
            # Scipy usa l'ordine x, y, z, w, quindi riordiniamo i valori di MuJoCo
            r = R.from_quat([quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]])
            euler = r.as_euler('xyz') # Restituisce [Roll, Pitch, Yaw] in radianti
            
            # Velocità lineari (Vx, Vy, Vz) e angolari (Wx, Wy, Wz)
            vel = data.qvel[0:3]
            ang_vel = data.qvel[3:6]
            
            # Creiamo il vettore di stato X esattamente come lo vuole il tuo MPC (12x1)
            # [X, Y, Z, Roll, Pitch, Yaw, Vx, Vy, Vz, Wx, Wy, Wz]
            x_curr = np.concatenate((pos, euler, vel, ang_vel))
            
            # --- 2. IL CERVELLO (CasADi MPC) ---
            # Qui inserirai la chiamata alla tua funzione MPC tradotta in Python.
            # Esempio fittizio:
            # U_ottimo = mio_mpc_calcola(x_curr, target)
            
            # Per ora, mettiamo un comando fisso per non farlo cadere
            # ATTENZIONE: 100 è probabilmente un valore enorme. Inizia con valori più bassi.
            U_ottimo = [6.0, 6.0, 6.0, 6.0] 
            
            # --- 3. AZIONE ---
            data.ctrl[:] = U_ottimo
            
            # Avanza la fisica
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Sincronizzazione del tempo per simulare in "tempo reale"
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)