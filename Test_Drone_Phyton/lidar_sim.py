"""
lidar_sim.py
============
Sensore LiDAR simulato per il drone, basato sul raycasting nativo
di MuJoCo (mujoco.mj_ray).

Questo modulo è puramente percettivo: scansiona l'ambiente e
restituisce dati grezzi (distanze, punti di impatto, angoli).
Non ha alcuna dipendenza dal controllore MPC — l'integrazione con
il controllo verrà fatta in main_volo.py in uno step successivo,
una volta verificato che la percezione funzioni e si visualizzi
correttamente.

IMPORTANTE — esclusione del marker del target dal raycasting
---------------------------------------------------------------
La sfera che visualizza il target (target_waypoint nell'XML) è una
geometria nella scena MuJoCo. I flag contype="0" conaffinity="0" la
rendono fisicamente non collidibile, ma NON la escludono dal
raycasting di mj_ray — quei flag riguardano solo la fisica dei
contatti, non la visibilità ai raggi. Senza un'esclusione esplicita,
quando il drone si avvicina al proprio target il LiDAR lo rileva
come un ostacolo, generando un conflitto diretto nel costo dell'MPC
tra il termine attrattivo (target) e quello repulsivo (ostacolo),
quasi sovrapposti e in competizione numerica — causa di un crash
del solver osservato in fase di test.

NOTA su un primo tentativo di fix, poi corretto: inizialmente la
sfera era stata assegnata a un gruppo geom dedicato (group=4),
escluso dal raycasting tramite la maschera geomgroup di mj_ray.
Questo però rendeva la sfera anche INVISIBILE nel viewer 3D, perché
il parametro group in MuJoCo controlla contemporaneamente sia il
raycasting sia la visibilità grafica — non sono canali separati.
La sfera è stata quindi riportata al gruppo di default (0, sempre
visibile), e l'esclusione dal LiDAR è ora realizzata identificando
il body_id del target (passato come geomid_target_body al
costruttore di LidarSim) e scartando il risultato di mj_ray quando
il geomid colpito appartiene a quel body specifico.

Funzionamento di mj_ray
------------------------
mujoco.mj_ray(model, data, pnt, vec, geomgroup, flg_static,
              bodyexclude, geomid)

Riceve un punto di origine (pnt) e una direzione unitaria (vec) nel
frame del mondo, e restituisce la distanza al primo oggetto colpito
lungo quel raggio. Se il raggio non colpisce nulla entro il range
massimo, la funzione restituisce -1.

Geometria della scansione
--------------------------
Di default vengono generati 16 raggi su una scansione completa a
360° nel piano orizzontale (sul piano X-Y, a quota fissa pari
all'altezza del drone). Una scansione a 360° è necessaria perché il
drone, durante il volo, si muove tipicamente per traslazione pura
(comandata da target via UDP) senza che l'angolo di yaw segua
necessariamente la direzione di moto: un settore frontale limitato
lascerebbe "punti ciechi" su fianchi e retro del drone, mancando
ostacoli laterali durante traslazioni non rettilinee rispetto al
proprio asse di imbardata. Il parametro settore_deg permette comunque
di restringere il campo se in futuro servisse un settore direzionale
(es. radar frontale).
"""

import numpy as np
import mujoco


class LidarSim:
    """
    LiDAR simulato a raggi multipli sul piano orizzontale.

    Parametri
    ---------
    model        : mujoco.MjModel
    n_raggi      : int   — numero di raggi nella scansione
    settore_deg  : float — ampiezza angolare totale del settore [deg].
                            Default 360° (scansione completa): con il
                            drone in volo libero (traslazioni in ogni
                            direzione senza necessariamente ruotare in
                            yaw), un settore frontale limitato lascia
                            "punti ciechi" su lati e retro. La scansione
                            a 360° garantisce che nessun ostacolo venga
                            mancato indipendentemente dall'orientamento
                            del drone.
    range_max    : float — distanza massima rilevabile [m]
    bodyexclude  : int   — ID del body da escludere dalla scansione
                            (tipicamente il corpo del drone stesso,
                            per evitare auto-intersezioni)
    escludi_gruppi : list[int] — gruppi geom (0-5) da escludere dal
                            raycasting. Di default esclude il gruppo 4,
                            riservato a elementi puramente visivi come
                            la sfera del target. Senza questa esclusione,
                            il LiDAR rileverebbe il marker del target
                            stesso come un ostacolo quando il drone gli
                            si avvicina, generando un conflitto diretto
                            tra il termine attrattivo (target) e quello
                            repulsivo (ostacolo) nella funzione di costo
                            dell'MPC — la causa di un crash del solver
                            osservato in fase di test, dato che i due
                            termini diventano quasi sovrapposti e
                            numericamente in competizione diretta.
    escludi_body_ids : list[int] — ID dei body le cui geometrie devono
                            essere ignorate dal LiDAR (oltre al body
                            del drone stesso, passato separatamente
                            in bodyexclude). Tipicamente qui va l'ID
                            del body "target_waypoint": la sua sfera
                            visiva resta visibile a schermo (gruppo
                            geom di default) ma viene scartata dal
                            raycasting per evitare che il drone la
                            rilevi come un ostacolo quando le si
                            avvicina, generando un conflitto diretto
                            tra il termine attrattivo (target) e
                            quello repulsivo (ostacolo) nel costo
                            dell'MPC — causa di un crash del solver
                            osservato in fase di test.
    """

    def __init__(self, model, n_raggi=16, settore_deg=360.0,
                 range_max=5.0, bodyexclude=-1, escludi_body_ids=()):
        self.model       = model
        self.n_raggi     = n_raggi
        self.settore_rad = np.deg2rad(settore_deg)
        self.range_max   = range_max
        self.bodyexclude = bodyexclude

        # ID dei geom (non dei body!) appartenenti ai body da
        # escludere. mj_ray restituisce un geomid, quindi qui
        # pre-calcoliamo la corrispondenza geom -> body una volta,
        # per evitare di rifare la ricerca ad ogni raggio scansionato.
        self._geomid_da_escludere = set()
        for body_id in escludi_body_ids:
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] == body_id:
                    self._geomid_da_escludere.add(geom_id)

        # Angoli dei raggi nel settore di scansione.
        # Per una scansione completa a 360°, endpoint=False evita di
        # generare due raggi sovrapposti a 0° e 360° (stesso raggio).
        # Per un settore parziale (<360°), endpoint=True copre
        # correttamente i due estremi del settore.
        settore_completo = abs(settore_deg - 360.0) < 1e-6
        self.angoli_relativi = np.linspace(
            -self.settore_rad / 2, self.settore_rad / 2,
            n_raggi, endpoint=(not settore_completo)
        )

        # Buffer riutilizzabile richiesto dalla firma di mj_ray
        self._geomid_buf = np.zeros(1, dtype=np.int32)

    # ------------------------------------------------------------
    def _spara_raggio(self, data, origine, direzione, max_tentativi=4):
        """
        Spara un singolo raggio, ignorando eventuali colpi su geomid
        appartenenti ai body esclusi (es. il marker del target).

        mj_ray non supporta nativamente l'esclusione di più body in
        una sola chiamata (bodyexclude accetta un solo ID). Per
        escludere anche il target, ripetiamo il raggio "spostando"
        virtualmente l'origine appena oltre il punto colpito quando
        quel punto appartiene a un geom escluso — così il raggio
        continua a propagarsi e può rilevare un eventuale ostacolo
        reale posizionato oltre il marker del target.

        Ritorna (distanza_totale, geomid_colpito) — distanza
        cumulata dall'origine ORIGINALE del raggio, non dall'ultimo
        punto di ripartenza.
        """
        origine_corrente = origine.copy()
        distanza_accumulata = 0.0

        for _ in range(max_tentativi):
            dist = mujoco.mj_ray(
                self.model, data,
                origine_corrente, direzione,
                None,                  # nessun filtro di gruppo: la
                                        # sfera target resta nel
                                        # gruppo di default e quindi
                                        # visibile/raggiungibile, la
                                        # escludiamo qui sotto via geomid
                1,                     # flg_static: includi geometrie statiche
                self.bodyexclude,
                self._geomid_buf
            )

            if dist < 0:
                # Nessun colpo lungo il resto del raggio
                return -1.0, -1

            geomid = int(self._geomid_buf[0])

            if geomid not in self._geomid_da_escludere:
                # Colpo "vero": un ostacolo reale, non il target
                return distanza_accumulata + dist, geomid

            # Colpo su geometria esclusa (es. sfera target): avanza
            # l'origine appena oltre quel punto e ripeti il raggio
            margine = 1e-3
            origine_corrente = (origine_corrente
                                 + (dist + margine) * direzione)
            distanza_accumulata += dist + margine

            if distanza_accumulata > self.range_max:
                return -1.0, -1

        # Troppi tentativi (caso raro, es. geometrie escluse
        # sovrapposte): consideriamo il raggio libero
        return -1.0, -1

    # ------------------------------------------------------------
    def scansiona(self, data, drone_pos, yaw=0.0, quota_scan=None):
        """
        Esegue una scansione completa e ritorna i risultati.

        Parametri
        ---------
        data        : mujoco.MjData
        drone_pos   : array(3,) — posizione corrente del drone [x,y,z]
        yaw         : float     — angolo di imbardata corrente [rad].
                                   Con scansione a 360° il yaw ruota
                                   semplicemente la griglia di raggi
                                   (il primo raggio punta sempre nella
                                   direzione del muso del drone), ma
                                   la coppertura risultante è identica
                                   per qualunque valore di yaw essendo
                                   l'intero giro completo. È comunque
                                   utile passarlo per coerenza con il
                                   frame del drone e per un eventuale
                                   futuro restringimento del settore.
        quota_scan  : float|None — quota a cui scansionare; se None
                                    usa la quota corrente del drone

        Ritorna
        -------
        risultato : dict con:
          'distanze'  : array (n_raggi,) — distanza rilevata
                        (range_max se nessun colpo)
          'punti'     : array (n_raggi, 3) — coordinate assolute
                        del punto di impatto (o del punto limite
                        a range_max se nessun colpo)
          'angoli_abs': array (n_raggi,) — angolo assoluto di ogni
                        raggio nel frame del mondo [rad]
          'hit'       : array (n_raggi,) bool — True se il raggio
                        ha colpito un ostacolo
          'dist_min'  : float — distanza minima rilevata in tutta
                        la scansione (utile per un check rapido)
          'idx_min'   : int — indice del raggio con distanza minima
        """
        origine = np.array(drone_pos, dtype=np.float64).copy()
        if quota_scan is not None:
            origine[2] = quota_scan

        n = self.n_raggi
        distanze   = np.full(n, self.range_max)
        punti      = np.zeros((n, 3))
        angoli_abs = np.zeros(n)
        hit        = np.zeros(n, dtype=bool)

        for i, ang_rel in enumerate(self.angoli_relativi):
            ang_abs = yaw + ang_rel
            angoli_abs[i] = ang_abs

            direzione = np.array([
                np.cos(ang_abs),
                np.sin(ang_abs),
                0.0
            ])

            dist, geom_colpito = self._spara_raggio(data, origine, direzione)

            if dist >= 0 and dist <= self.range_max:
                distanze[i] = dist
                hit[i]      = True
                punti[i]    = origine + dist * direzione
            else:
                # Nessun colpo: punto al limite del range, per la
                # visualizzazione (raggio "verde" a piena lunghezza)
                distanze[i] = self.range_max
                hit[i]      = False
                punti[i]    = origine + self.range_max * direzione

        idx_min = int(np.argmin(distanze))

        return {
            'distanze'  : distanze,
            'punti'     : punti,
            'angoli_abs': angoli_abs,
            'hit'       : hit,
            'dist_min'  : float(distanze[idx_min]),
            'idx_min'   : idx_min,
            'origine'   : origine,
        }


# ──────────────────────────────────────────────────────────────────
# VISUALIZZAZIONE
# ──────────────────────────────────────────────────────────────────

def visualizza_lidar(viewer, scan_result, colore_libero=(0.2, 0.9, 0.2, 0.5),
                      colore_colpito=(0.9, 0.2, 0.2, 0.8),
                      raggio_punto=0.04):
    """
    Disegna i raggi LiDAR e i punti di impatto nel viewer MuJoCo.

    Ogni raggio è rappresentato come una linea sottile (capsule)
    dall'origine al punto rilevato: verde se il raggio non ha
    colpito nulla (a range massimo), rosso se ha colpito un ostacolo.
    Nei punti di impatto viene inoltre disegnata una piccola sfera
    rossa più visibile, che costituisce la "nuvola di punti" del
    LiDAR.

    Va chiamata ad ogni step PRIMA di viewer.sync(), dopo aver
    azzerato il contatore di geometrie della scena utente
    (viewer.user_scn.ngeom = 0) se si vogliono ridisegnare anche
    altri elementi (es. traiettoria MPC) nello stesso frame.
    """
    origine    = scan_result['origine']
    punti      = scan_result['punti']
    hit        = scan_result['hit']

    for i in range(len(punti)):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 1:
            break  # spazio insufficiente nella scena per altre geometrie

        colore = colore_colpito if hit[i] else colore_libero

        # ── Linea del raggio (capsule sottile da origine a punto) ──
        _disegna_linea(viewer, origine, punti[i], colore, raggio=0.005)

        # ── Punto di impatto evidenziato (solo se ha colpito) ──
        if hit[i]:
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([raggio_punto, 0.0, 0.0]),
                pos=np.array(punti[i], dtype=np.float64),
                mat=np.eye(3).flatten(),
                rgba=np.array(colore_colpito)
            )
            viewer.user_scn.ngeom += 1


def _disegna_linea(viewer, p1, p2, rgba, raggio=0.005):
    """
    Disegna un segmento da p1 a p2 come capsule sottile.
    Funzione di utilità interna per visualizza_lidar().
    """
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return

    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)

    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.array([raggio, 0.0, 0.0]),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float64)
    )
    # mjv_connector orienta e dimensiona automaticamente la capsule
    # per congiungere i due punti dati — è la funzione standard di
    # MuJoCo per disegnare segmenti/linee nella scena utente.
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        raggio,
        p1,
        p2
    )
    viewer.user_scn.ngeom += 1