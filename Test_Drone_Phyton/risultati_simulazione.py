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

print(f"Dati caricati: {len(t)} step, {t[-1]:.1f}s di simulazione")
if vento is not None:
    print(f"Simulazione con vento: {vento} N")
else:
    print("Simulazione senza disturbi.")

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
          'b-', linewidth=1.5, label='Traiettoria effettiva')
ax3d.plot(target[:,0], target[:,1], target[:,2],
          'r--', linewidth=1.0, alpha=0.7, label='Target')
ax3d.scatter(pos[0,0], pos[0,1], pos[0,2],
             c='green', s=60, zorder=5, label='Partenza')
ax3d.scatter(pos[-1,0], pos[-1,1], pos[-1,2],
             c='red', s=60, zorder=5, label='Fine')

ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z [m]')
ax3d.set_title('Traiettoria 3D — LPV-MPC')
ax3d.legend()
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
fig2.suptitle('Posizioni vs Tempo — LPV-MPC')
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
fig3.suptitle('Angoli di Eulero vs Tempo — LPV-MPC')
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
ax4.set_title('Forze Motori vs Tempo — LPV-MPC')
ax4.legend()
plt.tight_layout()
plt.savefig('forze_motori.png', dpi=150, bbox_inches='tight')

# Grafico 5: Norma dell'errore di posizione nel tempo
errore = np.linalg.norm(pos - target, axis=1)
fig5, ax5 = plt.subplots(figsize=(10, 3))
ax5.plot(t, errore, 'tab:red')
ax5.set_xlabel('Tempo [s]')
ax5.set_ylabel('Errore posizione [m]')
ax5.set_title('Norma dell\'errore di posizione vs Tempo — LPV-MPC')
ax5.fill_between(t, errore, alpha=0.2, color='tab:red')
plt.tight_layout()
plt.savefig('errore_posizione.png', dpi=150, bbox_inches='tight')

print("\nGrafici salvati:")
for nome in ['traiettoria_3d.png', 'posizioni_tempo.png',
             'angoli_eulero.png', 'forze_motori.png',
             'errore_posizione.png']:
    print(f"  {nome}")

plt.show()