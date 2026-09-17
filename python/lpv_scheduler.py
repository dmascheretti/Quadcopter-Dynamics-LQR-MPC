"""
Schedulatore LPV (Linear Parameter-Varying) per il controllore MPC.

Questo modulo implementa la logica di uno schedulatore LPV, simile a quella
del blocco LPV_Scheduler in Simulink.

Il suo compito è ricalcolare la matrice di stato `A_c` ad ogni istante,
linearizzando il modello del drone attorno all'assetto corrente (phi, theta, psi).
Questo approccio è più accurato del modello LTI (Linear Time-Invariant), che
linearizza solo attorno al punto di hover (angoli nulli).

La matrice di ingresso `B_c`, invece, non dipende dall'assetto e viene
calcolata una sola volta.

Questo codice si basa sulle equazioni e sulla logica derivate dai seguenti
file MATLAB del progetto di riferimento:
- `parametri_drone_sky.m` (modello Newton-Euler e linearizzazione)
- `S_init_sim_LPV.m` (impostazione del problema di ottimizzazione)
"""

import numpy as np
from scipy.linalg import expm

# Parametri fisici del drone Skydio X2 (da `leggi_fisica.py`)
_PARAMS_DEFAULT = {
    'm'   : 1.3250,
    'Ixx' : 0.060711,
    'Iyy' : 0.036468,
    'Izz' : 0.025412,
    'g'   : 9.81,
    # Matrice di allocazione Gamma (config a X, senza canting)
    # Motori: 1=Dietro-Dx, 2=Dietro-Sx, 3=Davanti-Sx, 4=Davanti-Dx (numerazione Simulink)
    'lx'  : 0.14,    # braccio beccheggio [m]
    'ly'  : 0.18,    # braccio rollio [m]
    'c'   : 0.0201,  # coefficiente coppia imbardata
}


def calcola_B_c(params=None):
    """
    Calcola la matrice di ingresso B continua (12x4).

    Questa matrice è invariante rispetto all'assetto, poiché le derivate
    parziali delle forze rispetto agli input dei motori, valutate all'equilibrio,
    non dipendono dagli angoli di Eulero.
    """
    if params is None:
        params = _PARAMS_DEFAULT

    m   = params['m']
    Ixx = params['Ixx']
    Iyy = params['Iyy']
    Izz = params['Izz']
    lx  = params['lx']
    ly  = params['ly']
    c   = params['c']

    # Matrice Gamma allocazione a X (da parametri_drone_sky.m)
    Gamma = np.array([
        [ 1,    1,    1,    1  ],
        [-ly,   ly,   ly,  -ly ],
        [ lx,   lx,  -lx,  -lx],
        [-c,    c,   -c,    c  ]
    ])

    # La matrice B_c ha termini non nulli solo per le righe delle accelerazioni.
    # L'effetto degli input (forze motori) si manifesta sulle accelerazioni
    # lineari (ddz) e angolari (ddphi, ddtheta, ddpsi).
    B_c = np.zeros((12, 4))

    # Riga 8 (ddz): accelerazione verticale.
    # Il segno è positivo perché usiamo una convenzione z-up (standard in MuJoCo).
    # Aumentare la spinta aumenta vz (il drone sale). CORRETTO.
    #
    # NOTA: in S_init_sim_LPV.m compare "B_lin(9,:) = -B_lin(9,:)"
    # perché il blocco Drone_Physics in Simulink usa una convenzione
    # z-down (o NED) internamente, quindi richiede l'inversione.
    # In Python+MuJoCo la convenzione è z-up: il segno NON va invertito.
    B_c[8,  :] = +Gamma[0, :] / m
    B_c[9,  :] =  Gamma[1, :] / Ixx
    B_c[10, :] =  Gamma[2, :] / Iyy
    B_c[11, :] =  Gamma[3, :] / Izz

    return B_c


def calcola_A_c(phi, theta, psi, params=None):
    """
    Calcola la matrice di stato A continua (12x12), linearizzata attorno
    all'assetto corrente (phi, theta, psi).

    A differenza del modello LTI (linearizzato solo a phi=theta=psi=0),
    in questo modello LPV i termini che legano gli angoli alle accelerazioni
    lineari dipendono dall'assetto istantaneo.

    DERIVAZIONE (da `parametri_drone_sky.m`):
    -----------
    Dal modello Newton-Euler di parametri_drone_sky.m:

        ddx = (T/m)*(cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))
        ddy = (T/m)*(cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))
        ddz = (T/m)* cos(phi)*cos(theta) - g

    Linearizzando rispetto a (phi, theta, psi) attorno allo stato corrente
    (non all'origine), e assumendo la spinta totale T pari a quella di equilibrio (m*g):

        dddx/dphi   = g*(-sin(phi)*sin(theta)*cos(psi) + cos(phi)*sin(psi))
        dddx/dtheta = g*( cos(phi)*cos(theta)*cos(psi))
        dddx/dpsi   = g*(-cos(phi)*sin(theta)*sin(psi) + sin(phi)*cos(psi))

        dddy/dphi   = g*(-sin(phi)*sin(theta)*sin(psi) - cos(phi)*cos(psi))
        dddy/dtheta = g*( cos(phi)*cos(theta)*sin(psi))
        dddy/dpsi   = g*( cos(phi)*sin(theta)*cos(psi) - sin(phi)*sin(psi))

        dddz/dphi   = g*(-sin(phi)*cos(theta))
        dddz/dtheta = g*(-cos(phi)*sin(theta))
        dddz/dpsi   = 0  (indipendente da psi)

    Per phi=theta=psi=0, si riottengono i termini del modello LTI:
        dddx/dtheta = g  →  A[6,4] = g   (coerente con `parametri_drone.py`)
        dddy/dphi   = -g →  A[7,3] = -g
        tutti gli altri = 0               ✓
    """
    if params is None:
        params = _PARAMS_DEFAULT

    g = params['g']

    cp = np.cos(phi);  sp = np.sin(phi)
    ct = np.cos(theta); st = np.sin(theta)
    cs = np.cos(psi);  ss = np.sin(psi)

    A_c = np.zeros((12, 12))

    # Blocco cinematica: q_dot dipende da q_dot (identità 6x6)
    A_c[0:6, 6:12] = np.eye(6)

    # Riga 6 (ddx): derivate parziali rispetto a phi, theta, psi
    A_c[6, 3] = g * (-sp * st * cs + cp * ss)   # dddx/dphi
    A_c[6, 4] = g * ( cp * ct * cs)              # dddx/dtheta
    A_c[6, 5] = g * (-cp * st * ss + sp * cs)   # dddx/dpsi  (segno corretto)

    # Riga 7 (ddy): derivate parziali rispetto a phi, theta, psi
    A_c[7, 3] = g * (-sp * st * ss - cp * cs)   # dddy/dphi
    A_c[7, 4] = g * ( cp * ct * ss)              # dddy/dtheta
    A_c[7, 5] = g * ( cp * st * cs - sp * ss)   # dddy/dpsi

    # Riga 8 (ddz): derivate parziali rispetto a phi, theta
    A_c[8, 3] = g * (-sp * ct)                   # dddz/dphi
    A_c[8, 4] = g * (-cp * st)                   # dddz/dtheta
    # A_c[8, 5] = 0  già zero

    # Le accelerazioni angolari (righe 9-11) non dipendono
    # dall'assetto nel modello Newton-Euler linearizzato

    return A_c


def discretizza_zoh(A_c, B_c, Ts):
    """
    Discretizzazione ZOH tramite esponenziale di matrice.
    Equivalente a c2d(sys, Ts, 'zoh') di MATLAB.

    Utilizza la matrice aumentata:
        M = [A_c  B_c]   →  expm(M*Ts) = [Ad  Bd]
            [ 0    0 ]                    [ 0   I]
    """
    n = A_c.shape[0]
    m = B_c.shape[1]

    M = np.zeros((n + m, n + m))
    M[0:n, 0:n] = A_c
    M[0:n, n:]  = B_c
    eM = expm(M * Ts)

    Ad = eM[0:n, 0:n]
    Bd = eM[0:n, n:]
    return Ad, Bd


def aggiorna_matrici(phi, theta, psi, Ts, params=None):
    """
    Calcola le matrici di stato discrete Ad e Bd per l'assetto corrente.

    Questa è la funzione principale dello schedulatore, chiamata ad ogni
    passo dal controllore MPC.

    Args:
        phi, theta, psi (float): Angoli di Eulero correnti [rad].
        Ts (float): Passo di campionamento [s].
        params (dict, optional): Parametri fisici del drone.

    Returns:
        (np.ndarray, np.ndarray): Tuple contenente le matrici Ad (12x12) e Bd (12x4).
    """
    if params is None:
        params = _PARAMS_DEFAULT

    A_c = calcola_A_c(phi, theta, psi, params)
    B_c = calcola_B_c(params)
    Ad, Bd = discretizza_zoh(A_c, B_c, Ts)
    return Ad, Bd