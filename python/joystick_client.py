"""
Client UDP per controllare il drone con un joystick/gamepad.

Nota su Pygame:
È fondamentale creare una finestra con `pygame.display.set_mode()`. Senza un
display attivo, Pygame su alcuni sistemi operativi non cattura gli eventi del
joystick, rendendolo di fatto inutilizzabile.

Modalità di controllo:
- INCREMENTALE (default): Le levette spostano il target di un piccolo passo
  ad ogni frame. Rilasciandole, il drone mantiene la posizione. Ideale per
  un controllo di precisione.
- ASSOLUTA: Le levette mappano la posizione del target in un'area predefinita.
  Rilasciandole, il target torna al centro. Attivabile con `MODALITA_ASSOLUTA = True`.

Debug e mappatura assi:
- Per trovare i numeri degli assi del tuo controller, imposta `DEBUG_ASSI = True`.
- La mappatura di default è pensata per controller stile Xbox/PlayStation:
    - Levetta sinistra (verticale, asse 1): avanti/indietro
    - Levetta sinistra (orizzontale, asse 0): sinistra/destra
    - Levetta destra (verticale, asse 3): quota (su/giù)
  Modifica le costanti AXIS_* se necessario.
"""

import pygame
import socket
import json
import time
import sys

# Configurazione
UDP_IP   = "127.0.0.1"
UDP_PORT = 5005

# Mettere DEBUG_ASSI = True per trovare i numeri giusti
# sul proprio controller
DEBUG_ASSI = False

AXIS_X = 1   # levetta sinistra verticale  → avanti/indietro
AXIS_Y = 0   # levetta sinistra orizzontale→ sinistra/destra
AXIS_Z = 3   # levetta destra verticale    → quota

# Valori sotto questa soglia vengono trattati come zero
# (elimina il drift delle levette a riposo)
DEADZONE = 0.08

MODALITA_ASSOLUTA = False  # True = range fisso, False = incrementale

# Modalità assoluta: range del target
X_RANGE = 3.0   # target tra -3.0 e +3.0 m
Y_RANGE = 3.0
Z_BASE  = 1.0   # quota base [m]
Z_RANGE = 1.5   # target tra Z_BASE-Z_RANGE e Z_BASE+Z_RANGE

# Modalità incrementale: velocità di spostamento del target
VEL_XY = 0.04   # m per frame (a 50Hz = 2 m/s massimo)
VEL_Z  = 0.02

# Limiti di sicurezza in modalità incrementale
X_MIN, X_MAX = -5.0, 5.0
Y_MIN, Y_MAX = -5.0, 5.0
Z_MIN, Z_MAX =  0.2, 3.0

# Frequenza di invio dei pacchetti UDP
FREQ_HZ  = 50              # pacchetti UDP al secondo
DT       = 1.0 / FREQ_HZ

def applica_deadzone(val, dz):
    """Azzera i valori sotto la soglia, riscala il resto a [0,1]."""
    if abs(val) < dz:
        return 0.0
    segno = 1.0 if val > 0 else -1.0
    return segno * (abs(val) - dz) / (1.0 - dz)


def leggi_asse(joy, asse, inverti=False):
    """Legge un asse con deadzone e inversione opzionale."""
    try:
        val = joy.get_axis(asse)
    except Exception:
        return 0.0
    val = applica_deadzone(val, DEADZONE)
    return -val if inverti else val


def disegna_hud(screen, font, target_x, target_y, target_z,
                ax, ay, az, modalita):
    """Disegna lo stato attuale sulla finestra pygame."""
    screen.fill((20, 20, 30))

    titolo = font.render("Joystick Controller — Drone Skydio X2",
                         True, (200, 200, 255))
    screen.blit(titolo, (10, 10))

    mod_str = "ASSOLUTA" if modalita else "INCREMENTALE"
    testo = [
        f"Modalità: {mod_str}",
        f"",
        f"Target attuale:",
        f"  X = {target_x:+.2f} m",
        f"  Y = {target_y:+.2f} m",
        f"  Z = {target_z:+.2f} m",
        f"",
        f"Assi joystick:",
        f"  AXIS_X ({AXIS_X}) = {ax:+.3f}",
        f"  AXIS_Y ({AXIS_Y}) = {ay:+.3f}",
        f"  AXIS_Z ({AXIS_Z}) = {az:+.3f}",
        f"",
        f"[ESC] Esci",
    ]

    for i, riga in enumerate(testo):
        surf = font.render(riga, True, (180, 220, 180))
        screen.blit(surf, (10, 50 + i * 22))

    pygame.display.flip()

# Inizializzazione
pygame.init()
pygame.joystick.init()

# FONDAMENTALE: senza display attivo pygame non processa
# gli eventi del joystick su molti sistemi operativi
screen = pygame.display.set_mode((420, 360))
pygame.display.set_caption("Drone Joystick Controller")

try:
    font = pygame.font.SysFont("Consolas", 16)
except Exception:
    font = pygame.font.Font(None, 18)

if pygame.joystick.get_count() == 0:
    print("ERRORE: Nessun joystick trovato.")
    print("Connetti un controller e riprova.")
    pygame.quit()
    sys.exit(1)

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Controller trovato: {joystick.get_name()}")
print(f"  Assi disponibili: {joystick.get_numaxes()}")
print(f"  Tasti disponibili: {joystick.get_numbuttons()}")

if DEBUG_ASSI:
    print("\nMODALITÀ DEBUG ASSI attiva.")
    print("Muovi le levette per vedere i valori degli assi.")
    print("Premi ESC per uscire.\n")

# Socket UDP per inviare il target
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"Invio target a {UDP_IP}:{UDP_PORT}")

# Stato iniziale del target
target_x = 0.0
target_y = 0.0
target_z = Z_BASE

# Loop principale
print("\nControllore avviato. Chiudi la finestra o premi ESC per uscire.")

try:
    while True:
        t_loop = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt

        # Lettura assi con deadzone
        # Asse X (avanti/indietro): levetta sinistra verticale
        # Invertiamo perché "su" = valore negativo sul joystick
        ax = leggi_asse(joystick, AXIS_X, inverti=True)
        # Asse Y (sinistra/destra): levetta sinistra orizzontale
        ay = leggi_asse(joystick, AXIS_Y, inverti=False)
        # Asse Z (quota): levetta destra verticale, su = sale
        az = leggi_asse(joystick, AXIS_Z, inverti=True)

        # Stampa i valori di tutti gli assi se in modalità debug
        if DEBUG_ASSI:
            valori = [joystick.get_axis(i)
                      for i in range(joystick.get_numaxes())]
            print("Assi: " + "  ".join(
                f"[{i}]={v:+.3f}" for i, v in enumerate(valori)
            ))

        # Calcolo della posizione target in base alla modalità
        if MODALITA_ASSOLUTA:
            # Le levette definiscono direttamente la posizione target
            target_x =  ax * X_RANGE
            target_y =  ay * Y_RANGE
            target_z =  Z_BASE + az * Z_RANGE
            target_z =  max(Z_MIN, min(Z_MAX, target_z))
        else:
            # Le levette incrementano/decrementano il target
            # (il drone mantiene la posizione quando si lascia la levetta)
            target_x += ax * VEL_XY
            target_y += ay * VEL_XY
            target_z += az * VEL_Z

            # Limiti di sicurezza
            target_x = max(X_MIN, min(X_MAX, target_x))
            target_y = max(Y_MIN, min(Y_MAX, target_y))
            target_z = max(Z_MIN, min(Z_MAX, target_z))

        # Invio del target via UDP
        payload = json.dumps({
            "x": round(target_x, 3),
            "y": round(target_y, 3),
            "z": round(target_z, 3),
        }).encode('utf-8')
        sock.sendto(payload, (UDP_IP, UDP_PORT))

        # Aggiornamento dell'interfaccia grafica (HUD)
        disegna_hud(screen, font,
                    target_x, target_y, target_z,
                    ax, ay, az, MODALITA_ASSOLUTA)

        # Sincronizzazione del loop per mantenere la frequenza desiderata
        dt_rim = DT - (time.time() - t_loop)
        if dt_rim > 0:
            time.sleep(dt_rim)

except KeyboardInterrupt:
    print("\nController terminato.")
finally:
    sock.close()
    pygame.quit()