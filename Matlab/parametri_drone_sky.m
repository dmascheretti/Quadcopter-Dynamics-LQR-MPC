% ==============================================================
% GENERATORE MATRICI LPV PER PYTHON - MODELLO SKYDIO X2 (MuJoCo)
% ==============================================================
clear; clc;

% 1. PARAMETRI FISICI (Estratti da MuJoCo)
m   = 1.3250;
Ixx = 0.060711;
Iyy = 0.036468;
Izz = 0.025412;
lx  = 0.14;   % Distanza asse X (beccheggio)
ly  = 0.18;   % Distanza asse Y (rollio)
c   = 0.0201; % Coefficiente coppia imbardata
g   = 9.81;

% 2. STATO E INGRESSI SIMBOLICI (Corretto: nomi esatti!)
syms x y z phi theta psi real
syms dx dy dz dphi dtheta dpsi real
syms F1 F2 F3 F4 real

q     = [x; y; z; phi; theta; psi];
q_dot = [dx; dy; dz; dphi; dtheta; dpsi];
X     = [q; q_dot]; 
U     = [F1; F2; F3; F4];

% 3. MATRICE GAMMA (Configurazione a X da file XML)
% Motori: 1(Dietro-Dx), 2(Dietro-Sx), 3(Davanti-Sx), 4(Davanti-Dx)
Gamma = [
    1,   1,   1,   1;   % Spinta Z
    -ly,  ly,  ly, -ly;   % Rollio (tau_phi)
    lx,  lx, -lx, -lx;   % Beccheggio (tau_theta)
    -c,   c,  -c,   c    % Imbardata (tau_psi)
    ];

Tau = Gamma * U; 
T_tot = Tau(1); tau_phi = Tau(2); tau_theta = Tau(3); tau_psi = Tau(4);

% 4. DINAMICA DEL SISTEMA (Newton-Euler)
% Accelerazioni lineari
ddx = (cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi)) * (T_tot / m);
ddy = (cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi)) * (T_tot / m);
ddz = (cos(phi)*cos(theta)) * (T_tot / m) - g;

% Accelerazioni angolari 
ddphi   = (tau_phi   + (Iyy - Izz)*dtheta*dpsi) / Ixx;
ddtheta = (tau_theta + (Izz - Ixx)*dphi*dpsi)   / Iyy;
ddpsi   = (tau_psi   + (Ixx - Iyy)*dphi*dtheta) / Izz;

f = [q_dot; ddx; ddy; ddz; ddphi; ddtheta; ddpsi];

% 5. PUNTO DI EQUILIBRIO (Hovering)
X_eq = zeros(12,1);
F_eq_val = (m * g) / 4;
U_eq = [F_eq_val; F_eq_val; F_eq_val; F_eq_val];

% 6. CALCOLO JACOBIANI (Linearizzazione)
A_sym = jacobian(f, X);
B_sym = jacobian(f, U);

A_lin = double(subs(A_sym, [X; U], [X_eq; U_eq]));
B_lin = double(subs(B_sym, [X; U], [X_eq; U_eq]));

% 7. AUTOGENERAZIONE CODICE PYTHON
fprintf('\n======================================================\n');
fprintf(' COPIA IL TESTO QUI SOTTO E INCOLLALO IN VS CODE\n');
fprintf(' (Sostituisci tutto il contenuto di parametri_drone.py)\n');
fprintf('======================================================\n\n');

fprintf('import numpy as np\n\n');
fprintf('F_eq = %.6f\n\n', F_eq_val);

fprintf('A_lin = np.array([\n');
for i=1:12
    fprintf('    [');
    fprintf('%8.4f, ', A_lin(i,1:end-1));
    fprintf('%8.4f]', A_lin(i,end));
    if i<12, fprintf(',\n'); else, fprintf('\n'); end
end
fprintf('])\n\n');

fprintf('B_lin = np.array([\n');
for i=1:12
    fprintf('    [');
    fprintf('%8.4f, ', B_lin(i,1:end-1));
    fprintf('%8.4f]', B_lin(i,end));
    if i<12, fprintf(',\n'); else, fprintf('\n'); end
end
fprintf('])\n\n');
fprintf('======================================================\n');