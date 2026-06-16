"""
Client UDP per controllare il drone da tastiera.

Questo script permette di pilotare il drone usando i tasti della tastiera.
A differenza di versioni precedenti, il movimento è continuo tenendo premuto
un tasto. Include limiti di sicurezza, un HUD informativo e una funzione di
reset rapido.

Tasti di controllo:
  W / S : avanti / indietro (asse X)
  A / D : sinistra / destra (asse Y)
  Q / E : su / giù (asse Z)
  R     : reset alla posizione di hover (0, 0, 1.0)
  ESC   : esci
"""

import pygame
import socket
import json
import time

# Configurazione
UDP_IP   = "127.0.0.1"
UDP_PORT = 5005

# Velocità di spostamento del target [m per secondo]
VEL_XY = 0.8   # m/s lungo X e Y
VEL_Z  = 0.5   # m/s lungo Z

# Limiti di sicurezza
X_MIN, X_MAX = -5.0,  5.0
Y_MIN, Y_MAX = -5.0,  5.0
Z_MIN, Z_MAX =  0.2,  3.0

# Frequenza di invio UDP e aggiornamento schermo
FREQ_HZ = 50
DT      = 1.0 / FREQ_HZ

# Inizializzazione
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

pygame.init()
screen = pygame.display.set_mode((420, 300))
pygame.display.set_caption("Drone Keyboard Controller")

try:
    font       = pygame.font.SysFont("Consolas", 16)
    font_small = pygame.font.SysFont("Consolas", 13)
except Exception:
    font       = pygame.font.Font(None, 18)
    font_small = pygame.font.Font(None, 15)

# Target iniziale: hover a 1m
target_x = 0.0
target_y = 0.0
target_z = 1.0

target_precedente = None  # per stampare solo quando cambia

print("=== Controllo Drone da Tastiera ===")
print(f"  W/S → X  |  A/D → Y  |  Q/E → Z  |  R → Reset hover  |  ESC → Esci")
print(f"  Invio a {UDP_IP}:{UDP_PORT} a {FREQ_HZ}Hz")


def disegna_hud(screen, font, font_small, tx, ty, tz, keys):
    """Visualizza lo stato nella finestra pygame."""
    screen.fill((20, 20, 30))

    titolo = font.render("Keyboard Controller — Drone Skydio X2",
                         True, (200, 200, 255))
    screen.blit(titolo, (10, 10))

    righe = [
        f"Target attuale:",
        f"  X = {tx:+.2f} m     [W avanti / S indietro]",
        f"  Y = {ty:+.2f} m     [D destra  / A sinistra]",
        f"  Z = {tz:+.2f} m     [Q su      / E giù     ]",
        f"",
        f"Tasti attivi: " + (
            " ".join(k for k, v in keys.items() if v) or "nessuno"
        ),
        f"",
        f"[R] Reset hover (0, 0, 1.0)",
        f"[ESC] Esci",
    ]

    for i, riga in enumerate(righe):
        colore = (180, 220, 180)
        surf = font_small.render(riga, True, colore)
        screen.blit(surf, (10, 50 + i * 22))

    pygame.display.flip()


# Loop principale
running = True

while running:
    t_loop = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            # Reset immediato alla posizione di hover
            if event.key == pygame.K_r:
                target_x = 0.0
                target_y = 0.0
                target_z = 1.0
                print("Reset → hover (0.0, 0.0, 1.0)")

    # `pygame.key.get_pressed()` restituisce lo stato attuale di tutti
    # i tasti — permette il movimento continuo tenendo premuto
    keys_pressed = pygame.key.get_pressed()

    # Dizionario per HUD
    stato_tasti = {
        'W': bool(keys_pressed[pygame.K_w]),
        'S': bool(keys_pressed[pygame.K_s]),
        'A': bool(keys_pressed[pygame.K_a]),
        'D': bool(keys_pressed[pygame.K_d]),
        'Q': bool(keys_pressed[pygame.K_q]),
        'E': bool(keys_pressed[pygame.K_e]),
    }

    # Calcolo spostamento in base al DT (indipendente dalla frequenza)
    dx = 0.0
    dy = 0.0
    dz = 0.0

    if keys_pressed[pygame.K_w]: dx += VEL_XY * DT
    if keys_pressed[pygame.K_s]: dx -= VEL_XY * DT
    if keys_pressed[pygame.K_d]: dy += VEL_XY * DT
    if keys_pressed[pygame.K_a]: dy -= VEL_XY * DT
    if keys_pressed[pygame.K_q]: dz += VEL_Z  * DT
    if keys_pressed[pygame.K_e]: dz -= VEL_Z  * DT

    target_x = max(X_MIN, min(X_MAX, target_x + dx))
    target_y = max(Y_MIN, min(Y_MAX, target_y + dy))
    target_z = max(Z_MIN, min(Z_MAX, target_z + dz))

    # Invio del target via UDP
    target_dict = {
        "x": round(target_x, 3),
        "y": round(target_y, 3),
        "z": round(target_z, 3),
    }
    sock.sendto(
        json.dumps(target_dict).encode(),
        (UDP_IP, UDP_PORT)
    )

    # Stampa in console solo se il target è cambiato
    target_attuale = (round(target_x, 2),
                      round(target_y, 2),
                      round(target_z, 2))
    if target_attuale != target_precedente:
        print(f"Target → x={target_x:+.2f}  y={target_y:+.2f}  "
              f"z={target_z:+.2f}")
        target_precedente = target_attuale

    # Aggiornamento dell'interfaccia grafica (HUD)
    disegna_hud(screen, font, font_small,
                target_x, target_y, target_z, stato_tasti)

    # Sincronizzazione del loop per mantenere la frequenza desiderata
    dt_rim = DT - (time.time() - t_loop)
    if dt_rim > 0:
        time.sleep(dt_rim)

print("Controller terminato.")
sock.close()
pygame.quit()