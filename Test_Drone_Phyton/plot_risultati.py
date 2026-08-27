"""
plot_risultati.py
=================
Genera i grafici dai dati salvati da main_volo.py.

Uso:
    python plot_risultati.py                  # carica risultati_simulazione.pkl
    python plot_risultati.py mio_file.pkl     # carica file specifico

Grafici prodotti:
  1. Traiettoria 3D con target
  2. Posizione X, Y, Z vs tempo con target
  3. Angoli di Eulero vs tempo
  4. Forze motori vs tempo
  5. Errore di posizione (norma) vs tempo
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import time

# ═══════════════════════════════════════════════════════════════
# CARICAMENTO DATI
# ═══════════════════════════════════════════════════════════════

log_file = sys.argv[1] if len(sys.argv) > 1 else "risultati_simulazione.pkl"
log_file_abs = os.path.abspath(log_file)

# Stampo cwd e path assoluto: elimina ogni ambiguita' su DOVE lo
# script sta effettivamente leggendo/scrivendo (causa piu' comune
# di "i grafici non si aggiornano": lo script viene lanciato da una
# cartella diversa da quella in cui si stanno guardando i PNG).
print(f"Cartella di lavoro corrente: {os.getcwd()}")
print(f"Carico il file: {log_file_abs}")

if not os.path.exists(log_file):
    print(f"ERRORE: File '{log_file_abs}' non trovato.")
    print("Eseguire prima main_volo.py con ABILITA_LOG = True")
    sys.exit(1)

mtime = os.path.getmtime(log_file)
size_bytes = os.path.getsize(log_file)
print(f"File .pkl modificato il: {time.ctime(mtime)}")
print(f"Dimensione file: {size_bytes} byte")

if size_bytes == 0:
    print("ERRORE: il file .pkl e' vuoto (0 byte).")
    print("Probabile causa: main_volo.py e' stato interrotto PRIMA di")
    print("completare la scrittura (processo ancora attivo, kill forzato,")
    print("oppure OneDrive/antivirus stava sincronizzando il file in quel")
    print("momento). Rilanciare main_volo.py e chiudere il viewer con la X,")
    print("attendendo il messaggio 'Dati salvati in...' in console prima")
    print("di lanciare di nuovo questo script.")
    sys.exit(1)

try:
    with open(log_file, 'rb') as f:
        log = pickle.load(f)
except (pickle.UnpicklingError, EOFError) as e:
    print(f"ERRORE nell'apertura del file .pkl: {e}")
    print("Il file esiste e non e' vuoto, ma il contenuto e' corrotto o")
    print("incompleto: quasi certamente main_volo.py era ancora nel bel")
    print("mezzo della scrittura (con open(...) as f: pickle.dump(log, f))")
    print("quando questo script ha provato a leggerlo, oppure il processo")
    print("main_volo.py e' stato terminato a forza (kill/IDE stop) prima")
    print("che il salvataggio finisse. Rilanciare la simulazione e")
    print("attendere che il terminale stampi 'Dati salvati in...' prima")
    print("di eseguire di nuovo plot_risultati.py.")
    sys.exit(1)
except PermissionError as e:
    print(f"ERRORE di permessi nell'apertura del file .pkl: {e}")
    print("Il file e' probabilmente ancora aperto/bloccato da un altro")
    print("processo (es. main_volo.py non ancora terminato del tutto,")
    print("o un altro editor/programma che lo tiene aperto). Chiudere")
    print("tutti i processi collegati e riprovare.")
    sys.exit(1)

t      = log['t']
pos    = log['pos']       # (N, 3)
euler  = log['euler']     # (N, 3) — [phi, theta, psi]
vel    = log['vel']        # (N, 3)
u      = log['u']          # (N, 4)
target = log['target']    # (N, 3)
vento  = log['vento']

print(f"Dati caricati: {len(t)} step, {t[-1]:.1f}s di simulazione")
if vento is not None:
    print(f"Simulazione con vento: {vento} N")
else:
    print("Simulazione senza disturbi.")

# ═══════════════════════════════════════════════════════════════
# GRAFICI
# ═══════════════════════════════════════════════════════════════

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})


def salva(fig, nome):
    """Salva la figura stampando il percorso assoluto di destinazione,
    cosi' e' sempre chiaro dove il file viene effettivamente scritto
    indipendentemente dalla cartella da cui si lancia lo script."""
    path_assoluto = os.path.abspath(nome)
    fig.savefig(path_assoluto, dpi=150, bbox_inches='tight')
    print(f"  Salvato: {path_assoluto}")


# Range comune (in metri), usato sia dalla traiettoria 3D sia dai
# grafici di posizione nel tempo: senza un range esplicito, matplotlib
# auto-scala ogni asse per conto suo e se un asse resta quasi costante
# (es. drone fermo in hovering) zooma su un intervallo vicino allo zero
# macchina (es. 1e-18, puro rumore numerico floating-point), facendo
# sembrare che il segnale "oscilli" quando in realta' e' piatto.
margine = 0.3
y_min = min(pos.min(), target.min()) - margine
y_max = max(pos.max(), target.max()) + margine

# ── Figura 1: Traiettoria 3D ─────────────────────────────────
fig1 = plt.figure(figsize=(8, 6))
ax3d = fig1.add_subplot(111, projection='3d')

ax3d.plot(pos[:,0], pos[:,1], pos[:,2],
          'b-', linewidth=1.5, label='Traiettoria effettiva')
ax3d.plot(target[:,0], target[:,1], target[:,2],
          'r--', linewidth=1.0, alpha=0.7, label='Target')
ax3d.scatter(pos[0,0], pos[0,1], pos[0,2],
             c='green', s=60, zorder=5, label='Partenza')
ax3d.scatter(pos[-1,0], pos[-1,1], pos[-1,2],
             c='red', s=60, zorder=5, label='Fine')

# Rilevamento automatico dei waypoint "veri": non ogni singolo cambio
# di target (potrebbe essere solo di passaggio), ma solo i tratti in
# cui il target e' rimasto costante per almeno SOGLIA_STOP_S secondi,
# cioe' il drone si e' effettivamente fermato in quel punto.
SOGLIA_STOP_S = 3.0

cambio    = np.any(np.abs(np.diff(target, axis=0)) > 1e-6, axis=1)
confini   = np.where(cambio)[0] + 1
seg_inizio = np.concatenate(([0], confini))
seg_fine   = np.concatenate((confini, [len(target)]))

idx_waypoint = [i0 for i0, i1 in zip(seg_inizio, seg_fine)
                if t[i1 - 1] - t[i0] >= SOGLIA_STOP_S]

if idx_waypoint:
    waypoint_pos = target[idx_waypoint]
    ax3d.scatter(waypoint_pos[:,0], waypoint_pos[:,1], waypoint_pos[:,2],
                 c='orange', marker='*', s=200, zorder=6,
                 edgecolors='black', linewidths=0.5, label='Waypoint')

ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z [m]')
ax3d.set_title('Traiettoria 3D — quasi-LPV-MPC')
ax3d.legend()
# Range esplicito su tutti e tre gli assi (stesso range della Fig.2):
# e' questo che elimina davvero il problema, non solo la notazione.
# Senza set_xlim3d/set_ylim3d/set_zlim3d l'asse 3D si auto-scala per
# conto suo e puo' zoomare su rumore numerico anche col ticklabel
# in formato "plain".
ax3d.set_xlim3d(y_min, y_max)
ax3d.set_ylim3d(y_min, y_max)
ax3d.set_zlim3d(y_min, y_max)
for asse in ('x', 'y', 'z'):
    ax3d.ticklabel_format(style='plain', useOffset=False, axis=asse)
plt.tight_layout()
salva(fig1, 'traiettoria_3d.png')

# ── Figura 2: Posizioni X, Y, Z vs tempo ────────────────────
fig2, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
etichette = ['X [m]', 'Y [m]', 'Z [m]']
colori    = ['tab:blue', 'tab:orange', 'tab:green']

for i, (ax, etichetta, colore) in enumerate(zip(axes, etichette, colori)):
    ax.plot(t, pos[:, i],    color=colore,  label='Effettiva')
    ax.plot(t, target[:, i], color=colore,
            linestyle='--', alpha=0.6, label='Target')
    ax.set_ylabel(etichetta)
    ax.set_ylim(y_min, y_max)
    # Notazione decimale semplice, niente esponenti ne' offset
    # (l'offset e' proprio cio' che genera le etichette tipo "1e-18")
    ax.ticklabel_format(style='plain', useOffset=False, axis='y')
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Tempo [s]')
fig2.suptitle('Posizioni vs Tempo — quasi-LPV-MPC (scala comune in metri)')
plt.tight_layout()
salva(fig2, 'posizioni_tempo.png')

# ── Figura 3: Angoli di Eulero ───────────────────────────────
fig3, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
nomi_angoli = [r'$\phi$ (rollio) [°]',
               r'$\theta$ (beccheggio) [°]',
               r'$\psi$ (imbardata) [°]']
colori_ang  = ['tab:purple', 'tab:red', 'tab:brown']

euler_deg = np.rad2deg(euler)

# Range comune per i tre angoli (stessa logica della Fig.2): include
# sempre la banda +-22 nel calcolo, cosi' il limite resta visibile
# anche se un angolo e' rimasto quasi piatto vicino a zero
margine_ang = 5.0
y_min_ang = min(euler_deg.min(), -22) - margine_ang
y_max_ang = max(euler_deg.max(),  22) + margine_ang

for i, (ax, nome, colore) in enumerate(zip(axes, nomi_angoli, colori_ang)):
    ax.plot(t, euler_deg[:, i], color=colore)
    ax.set_ylabel(nome)
    ax.set_ylim(y_min_ang, y_max_ang)
    ax.ticklabel_format(style='plain', useOffset=False, axis='y')
    # Linee limite ±22° (vincolo MPC) solo per phi e theta
    if i < 2:
        ax.axhline( 22, color='gray', linestyle=':', alpha=0.7,
                    label='Limite ±22°')
        ax.axhline(-22, color='gray', linestyle=':', alpha=0.7)
        ax.legend(loc='upper right', fontsize=8)

# Annotazione dei valori stazionari di phi e theta: solo se il vento
# e' attivo, calcola la media dell'ultimo 20% della simulazione
# (assumendo che il transitorio si sia gia' esaurito) e la mostra con
# una freccia, evidenza fisica della compensazione del vento senza
# termine integrale esplicito.
if vento is not None:
    n_coda = max(1, int(0.2 * len(t)))
    t_freccia = t[-1] - 0.15 * (t[-1] - t[0])

    phi_ss = euler_deg[-n_coda:, 0].mean()
    axes[0].annotate(
        rf'$\phi_{{ss}} \approx {phi_ss:.1f}^\circ$',
        xy=(t_freccia, phi_ss),
        xytext=(t_freccia, phi_ss + 0.35 * (y_max_ang - y_min_ang)),
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=10, ha='center',
    )

    theta_ss = euler_deg[-n_coda:, 1].mean()
    axes[1].annotate(
        rf'$\theta_{{ss}} \approx {theta_ss:.1f}^\circ$',
        xy=(t_freccia, theta_ss),
        xytext=(t_freccia, theta_ss + 0.35 * (y_max_ang - y_min_ang)),
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=10, ha='center',
    )

axes[-1].set_xlabel('Tempo [s]')
fig3.suptitle('Angoli di Eulero vs Tempo — quasi-LPV-MPC (scala comune)')
plt.tight_layout()
salva(fig3, 'angoli_eulero.png')

# ── Figura 4: Forze motori ───────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(10, 4))
for i in range(4):
    ax4.plot(t, u[:, i], label=f'Motore {i+1}')
ax4.axhline(13.0, color='red', linestyle='--', alpha=0.7,
            label='Saturazione (13 N)')
ax4.axhline(0.0,  color='gray', linestyle=':', alpha=0.5)
ax4.set_xlabel('Tempo [s]')
ax4.set_ylabel('Forza [N]')
ax4.set_title('Forze Motori vs Tempo — quasi-LPV-MPC (scala fissa 0-14N)')
# Scala fissa 0-14N: sempre la stessa finestra fisica (0 N a poco sopra
# la saturazione 13 N), cosi' i grafici di run diverse sono confrontabili
ax4.set_ylim(-0.5, 14)
ax4.ticklabel_format(style='plain', useOffset=False, axis='y')
ax4.legend()
plt.tight_layout()
salva(fig4, 'forze_motori.png')

# ── Figura 5: Errore di posizione ────────────────────────────
errore = np.linalg.norm(pos - target, axis=1)
fig5, ax5 = plt.subplots(figsize=(10, 3))
ax5.plot(t, errore, 'tab:red')
ax5.set_xlabel('Tempo [s]')
ax5.set_ylabel('Errore posizione [m]')
ax5.set_title('Norma dell\'errore di posizione vs Tempo — quasi-LPV-MPC')
ax5.fill_between(t, errore, alpha=0.2, color='tab:red')
ax5.set_ylim(bottom=0)
# Stessa correzione della Fig.2: niente notazione scientifica/offset,
# altrimenti un errore quasi nullo verrebbe zoomato su rumore numerico
ax5.ticklabel_format(style='plain', useOffset=False, axis='y')
plt.tight_layout()
salva(fig5, 'errore_posizione.png')

print("\nFatto.")
plt.show()
