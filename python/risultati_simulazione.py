"""
Script per generare i grafici dei risultati della simulazione.

Questo script carica un file di log in formato `.pkl` (generato da `main_volo.py`)
e produce una serie di grafici per analizzare la performance del controllore.

Utilizzo da riga di comando:
- `python risultati_simulazione.py`: Carica il file di default `risultati_simulazione.pkl`.
- `python risultati_simulazione.py <nome_file.pkl>`: Carica un file specifico.

I grafici generati e salvati come file .png sono:
- Traiettoria 3D (drone vs target)
- Posizione (X, Y, Z) nel tempo
- Angoli di Eulero (roll, pitch, yaw) nel tempo
- Spinta dei motori nel tempo
- Norma dell'errore di posizione nel tempo
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Caricamento dei dati dal file di log
log_file = sys.argv[1] if len(sys.argv) > 1 else "risultati_simulazione.pkl"

if not os.path.exists(log_file):
    print(f"ERRORE: File '{log_file}' non trovato.")
    print("Eseguire prima main_volo.py con ABILITA_LOG = True")
    sys.exit(1)

with open(log_file, 'rb') as f:
    log = pickle.load(f)

t      = log['t']
pos    = log['pos']       # (N, 3)
euler  = log['euler']     # (N, 3) — [phi, theta, psi]
vel    = log['vel']        # (N, 3)
u      = log['u']          # (N, 4)
target = log['target']    # (N, 3)
vento  = log['vento']

# Campi opzionali (assenti nei log generati prima dell'introduzione del
# bypass/repulsione): fallback per non rompere l'analisi di log vecchi.
target_eff = log.get('target_eff', target)
obs_attivo = log.get('obs_attivo', None)
if obs_attivo is not None:
    obs_attivo = np.asarray(obs_attivo, dtype=bool)

print(f"Dati caricati: {len(t)} step, {t[-1]:.1f}s di simulazione")
if vento is not None:
    print(f"Simulazione con vento: {vento} N")
else:
    print("Simulazione senza disturbi.")
if obs_attivo is not None:
    n_attivi = int(np.sum(obs_attivo))
    print(f"Repulsione ostacolo attiva per {n_attivi}/{len(t)} step "
          f"({100*n_attivi/len(t):.1f}%)")
else:
    print("Campo 'obs_attivo' non presente nel log (log generato con "
          "una versione precedente di main_volo.py).")


def trova_intervalli_attivi(flag, tempo):
    """Restituisce la lista di intervalli (t_start, t_end) in cui `flag`
    è True in modo continuo, per evidenziare graficamente i tratti in cui
    la repulsione ostacolo era attiva."""
    flag = np.asarray(flag, dtype=bool)
    intervalli = []
    in_tratto = False
    t_start = None
    for i, v in enumerate(flag):
        if v and not in_tratto:
            t_start = tempo[i]
            in_tratto = True
        elif not v and in_tratto:
            intervalli.append((t_start, tempo[i]))
            in_tratto = False
    if in_tratto:
        intervalli.append((t_start, tempo[-1]))
    return intervalli

# Impostazioni globali per i grafici
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

# Grafico 1: Traiettoria 3D
fig1 = plt.figure(figsize=(8, 6))
ax3d = fig1.add_subplot(111, projection='3d')

ax3d.plot(pos[:,0], pos[:,1], pos[:,2],
          'b-', linewidth=1.5)  # senza label: coperta dall'arancione
                                  # quando la repulsione è quasi sempre
                                  # attiva; niente voce fantasma in legenda
ax3d.plot(target[:,0], target[:,1], target[:,2],
          'r--', linewidth=1.0, alpha=0.7, label='Target')

# Evidenzia sulla traiettoria volata i punti in cui la repulsione era
# effettivamente attiva: prova visiva diretta di dove l'ostacolo ha
# influenzato la traiettoria (a differenza di un semplice cambio target).
if obs_attivo is not None and np.any(obs_attivo):
    ax3d.scatter(pos[obs_attivo, 0], pos[obs_attivo, 1], pos[obs_attivo, 2],
                 c='orange', s=18, alpha=0.7, zorder=4,
                 label='Traiettoria con repulsione attiva')
    # Waypoint di bypass effettivamente inseguiti (solo dove differisce
    # dal target reale di più di 5 cm, come in visualizza_bypass()).
    divergenza_bp = np.linalg.norm(target - target_eff, axis=1)
    bypass_attivo = divergenza_bp > 0.05
    if np.any(bypass_attivo):
        ax3d.plot(target_eff[bypass_attivo, 0], target_eff[bypass_attivo, 1],
                   target_eff[bypass_attivo, 2], 'c.', markersize=3,
                   alpha=0.6, label='Waypoint di bypass')

ax3d.scatter(pos[0,0], pos[0,1], pos[0,2],
             c='green', s=60, zorder=5, label='Partenza')
ax3d.scatter(pos[-1,0], pos[-1,1], pos[-1,2],
             c='red', s=60, zorder=5, label='Fine')

ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z [m]')
ax3d.set_title('Traiettoria 3D — quasi-LPV-MPC')
ax3d.legend()
# Angolo di vista fissato: la vista di default di matplotlib genera
# ambiguità di profondità (punti vicini nello spazio possono sembrare
# distanti). elev=22, azim=-55 dà una buona lettura del piano X-Y
# mantenendo visibile la variazione in Z. Regolare se necessario.
ax3d.view_init(elev=22, azim=-55)
plt.tight_layout()
plt.savefig('traiettoria_3d.png', dpi=150, bbox_inches='tight')

# Grafico 2: Posizioni X, Y, Z nel tempo
fig2, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
etichette = ['X [m]', 'Y [m]', 'Z [m]']
colori    = ['tab:blue', 'tab:orange', 'tab:green']

for i, (ax, etichetta, colore) in enumerate(zip(axes, etichette, colori)):
    ax.plot(t, pos[:, i],    color=colore,  label='Effettiva')
    ax.plot(t, target[:, i], color=colore,
            linestyle='--', alpha=0.6, label='Target')
    ax.set_ylabel(etichetta)
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Tempo [s]')
fig2.suptitle('Posizioni vs Tempo — quasi-LPV-MPC')
plt.tight_layout()
plt.savefig('posizioni_tempo.png', dpi=150, bbox_inches='tight')

# Grafico 3: Angoli di Eulero nel tempo
fig3, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
nomi_angoli = [r'$\phi$ (rollio) [°]',
               r'$\theta$ (beccheggio) [°]',
               r'$\psi$ (imbardata) [°]']
colori_ang  = ['tab:purple', 'tab:red', 'tab:brown']

for i, (ax, nome, colore) in enumerate(zip(axes, nomi_angoli, colori_ang)):
    ax.plot(t, np.rad2deg(euler[:, i]), color=colore)
    ax.set_ylabel(nome)
    # Linee limite ±22° (vincolo MPC) solo per phi e theta
    if i < 2:
        ax.axhline( 22, color='gray', linestyle=':', alpha=0.7,
                    label='Limite ±22°')
        ax.axhline(-22, color='gray', linestyle=':', alpha=0.7)
        ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Tempo [s]')
fig3.suptitle('Angoli di Eulero vs Tempo — quasi-LPV-MPC')
plt.tight_layout()
plt.savefig('angoli_eulero.png', dpi=150, bbox_inches='tight')

# Grafico 4: Spinta dei motori nel tempo
fig4, ax4 = plt.subplots(figsize=(10, 4))
for i in range(4):
    ax4.plot(t, u[:, i], label=f'Motore {i+1}')
ax4.axhline(13.0, color='red', linestyle='--', alpha=0.7,
            label='Saturazione (13 N)')
ax4.axhline(0.0,  color='gray', linestyle=':', alpha=0.5)
ax4.set_xlabel('Tempo [s]')
ax4.set_ylabel('Forza [N]')
ax4.set_title('Forze Motori vs Tempo — quasi-LPV-MPC')
ax4.legend()
plt.tight_layout()
plt.savefig('forze_motori.png', dpi=150, bbox_inches='tight')

# Grafico 5: Norma dell'errore di posizione nel tempo
# ATTENZIONE: 'errore' è la distanza dal target REALE (non da target_eff),
# quindi un salto può derivare sia da un nuovo comando target sia da un
# bypass. Le due cause vengono ora distinte esplicitamente nel grafico.
errore = np.linalg.norm(pos - target, axis=1)
fig5, ax5 = plt.subplots(figsize=(10, 3.5))
ax5.plot(t, errore, 'tab:red', label='Errore da target reale', zorder=3)
ax5.set_xlabel('Tempo [s]')
ax5.set_ylabel('Errore posizione [m]', color='tab:red')
ax5.tick_params(axis='y', labelcolor='tab:red')
ax5.set_title('Norma dell\'errore di posizione vs Tempo — quasi-LPV-MPC')
ax5.fill_between(t, errore, alpha=0.15, color='tab:red', zorder=1)

# Sfondo arancione: tratti in cui la repulsione ostacolo era attiva.
if obs_attivo is not None:
    intervalli = trova_intervalli_attivi(obs_attivo, t)
    for j, (t0, t1) in enumerate(intervalli):
        ax5.axvspan(t0, t1, color='tab:orange', alpha=0.15, zorder=0,
                    label='Repulsione ostacolo attiva' if j == 0 else None)

# Divergenza target reale <-> target effettivo: quantifica quanto il
# bypass ha effettivamente deviato la rotta dal target reale.
divergenza_bp = np.linalg.norm(target - target_eff, axis=1)
ax5b = ax5.twinx()
ax5b.plot(t, divergenza_bp, color='tab:blue', linestyle='--',
          linewidth=1.2, alpha=0.85, zorder=2,
          label='Divergenza bypass |target - target_eff|')
ax5b.set_ylabel('Divergenza bypass [m]', color='tab:blue')
ax5b.tick_params(axis='y', labelcolor='tab:blue')

linee5, etichette5   = ax5.get_legend_handles_labels()
linee5b, etichette5b = ax5b.get_legend_handles_labels()
ax5.legend(linee5 + linee5b, etichette5 + etichette5b,
           loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('errore_posizione.png', dpi=150, bbox_inches='tight')

print("\nGrafici salvati:")
for nome in ['traiettoria_3d.png', 'posizioni_tempo.png',
             'angoli_eulero.png', 'forze_motori.png',
             'errore_posizione.png']:
    print(f"  {nome}")

plt.show()