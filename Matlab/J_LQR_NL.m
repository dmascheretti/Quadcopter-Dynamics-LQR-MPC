% J - LQR SU MODELLO NON LINEARE 

clear; clc; close all;

load('MAT/step7_workspace.mat');
load('MAT/step4_workspace.mat', 'M_fun', 'C_fun', 'G_n', 'B_fun');

parametri_drone;

Gamma_num = double(Gamma_num);

% Dati del controllore
f_s = 100;       
Ts = 1 / f_s;    

% Tuning bilanciato
Q = diag([[10 10 10], [50 50 50], [1 1 1], [1 1 1]]);
R = diag([0.1, 0.1, 0.1, 0.1]); 
[K_lqr, ~, ~] = lqrd(A_lin, B_lin, Q, R, Ts);

% Calcolo della spinta di Hovering (F_eq) per ogni motore
% In hovering, i 4 motori si dividono (circa) equamente il peso del drone MA con
% angolo diverso da 0 
alpha_cant = deg2rad(2); % Ipotesi angolo di due gradi

% !! DOVREI CAMBIARE ANCHE LA MATRICE GAMMA PRIMA RIGA MA COS 2 CIRCA 1 !!

F_eq = (m_val * g_val) / (4 * cos(alpha_cant)) * ones(4, 1);

t_sim_total = 5; % Durata simulazione [s]
N_steps = round(t_sim_total / Ts); % Numero passi in base al Ts

% Log delle variabili per i grafici
t_log = zeros(1, N_steps);
X_log = zeros(12, N_steps);
U_log = zeros(4, N_steps);

% Condizione Iniziale (Fuori equilibrio)
% [x y z phi theta psi u v w p q r]
x_start = zeros(12, 1);
x_start(3) = 1.0;  % Errore quota (1 metro)
x_start(5) = 0.2;  % Errore pitch
X_log(:, 1) = x_start;
t_log(1) = 0;

% Saturazione
F_min = 0;         % Minimo non fa nulla
F_max = 12;        % Spinta massima 12 N per esempio


% Qua come usare initial ma nel modello non lineare -> calcolo passo per
% passo, tengo nei log e poi stampo i log
for k = 1:(N_steps-1)
    
    delta_x = X_log(:, k); 
    delta_u = -K_lqr * delta_x;
    
    F_motori = F_eq + delta_u;
    
    % Saturazione 
    F_motori = max(min(F_motori, F_max), F_min);
    
    % Salvo i comandi nei log
    U_log(:, k) = F_motori;
   
    % drone_dynamics non usa forze ma T + momenti
    % Trasformo con matrice gamma per dare poi a drone_dynamics i valori
    % corretti
    U_virtual = Gamma_num * F_motori;
    
    % ode 45
    t_span = [t_log(k), t_log(k) + Ts];
    
    % Passiamo U virtuali non le forze
    [t_ode, X_ode] = ode45(@(t, x) drone_dynamics(t, x, U_virtual, M_fun, C_fun, G_n, B_fun), t_span, X_log(:, k));
    
    % Salvo stati per X e t ( calcolati con ode ) nei grafici
    X_log(:, k+1) = X_ode(end, :)';
    t_log(k+1) = t_span(2);
end


U_log(:, end) = U_log(:, end-1);

figure('Name', 'LQR su Modello NON-Lineare (ODE45)', 'Color', 'w', 'Position', [100, 100, 1000, 600]);

% Theta
subplot(2,2,1); hold on;
plot(t_log, X_log(5, :), 'r', 'LineWidth', 1.5);
plot([t_log(1) t_log(end)], [0 0], 'k--', 'LineWidth', 1);
grid on; title('\theta)');
xlabel('Tempo [s]'); ylabel('Angolo [rad]');

% Z
subplot(2,2,2); hold on;
plot(t_log, X_log(3, :), 'g', 'LineWidth', 1.5);
plot([t_log(1) t_log(end)], [0 0], 'k--', 'LineWidth', 1);
grid on; title('Z');
xlabel('Tempo [s]'); ylabel('Posizione [m]');

% X
subplot(2,2,3); hold on;
plot(t_log, X_log(1, :), 'b', 'LineWidth', 1.5);
plot([t_log(1) t_log(end)], [0 0], 'k--', 'LineWidth', 1);
grid on; title('X');
xlabel('Tempo [s]'); ylabel('Posizione [m]');

% Sforzo motori
subplot(2,2,4); hold on;
plot(t_log, U_log(1, :), 'LineWidth', 1.2);
plot(t_log, U_log(2, :), 'LineWidth', 1.2);
plot(t_log, U_log(3, :), 'LineWidth', 1.2);
plot(t_log, U_log(4, :), 'LineWidth', 1.2);
plot([t_log(1) t_log(end)], [F_eq(1) F_eq(1)], 'k--', 'LineWidth', 1); % Linea Hovering
grid on; title('Spinta dei 4 Motori (F1, F2, F3, F4)');
xlabel('Tempo [s]'); ylabel('Forza [N]');
legend('M1', 'M2', 'M3', 'M4', 'Hovering', 'Location', 'best');