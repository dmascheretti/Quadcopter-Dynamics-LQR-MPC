% K - LQR TRACKING NL

clear; clc; close all;

load('MAT/step7_workspace.mat');
load('MAT/step4_workspace.mat', 'M_fun', 'C_fun', 'G_n', 'B_fun');
parametri_drone;
Gamma_num = double(Gamma_num);

% Dati del controllore
f_s = 100;       
Ts = 1 / f_s;    

disp (Ts);
% Tuning bilanciato
Q = diag([[10 10 10], [50 50 50], [1 1 1], [1 1 1]]);
R = diag([0.1, 0.1, 0.1, 0.1]); 
[K_lqr, ~, ~] = lqrd(A_lin, B_lin, Q, R, Ts);

% Calcolo della spinta di Hovering (F_eq) con angolo 2 gradi
alpha_cant = deg2rad(2); 
F_eq = (m_val * g_val) / (4 * cos(alpha_cant)) * ones(4, 1);

t_sim_total = 30; % t di simulaziome
N_steps = round(t_sim_total / Ts); 

% Log delle variabili
t_log = zeros(1, N_steps);
X_log = zeros(12, N_steps);
U_log = zeros(4, N_steps);

% Target !! Se metto ad exe ( 20, 10, 10) gli angoli di Eulero assumono
% valori troppo elevati !! 
% !!!
% Non funziona piu la linearizazzione e tutto diverge sia angoli che
% posizioni
% !!!
x_target = zeros(12, 1);
x_target(1) = 3.0;  % Vai a X = 3 metri
x_target(2) = 2.0;  % Vai a Y = 2 metri
x_target(3) = 4.0;  % Vai a Z = 4 metri

% Ipotesi : partiamo da fermi 
X_log(:, 1) = zeros(12,1); 

% Posso anche far partire da stati diversi

t_log(1) = 0;

% Saturazione
F_min = 0;         
F_max = 12;        

for k = 1:(N_steps-1)
    
    % Calcolo distanza dal target ad ogni step 

    delta_x = X_log(:, k) - x_target; % Dove sono ora - dove devo arrivare
    
    delta_u = -K_lqr * delta_x;
    
    F_motori = F_eq + delta_u;
    
    % Saturazione 
    F_motori = max(min(F_motori, F_max), F_min);
    
    U_log(:, k) = F_motori;
   
    % Trasformo con matrice gamma
    U_virtual = Gamma_num * F_motori;
    
    % Modello non lineare 
    t_span = [t_log(k), t_log(k) + Ts];
    [t_ode, X_ode] = ode45(@(t, x) drone_dynamics(t, x, U_virtual, M_fun, C_fun, G_n, B_fun), t_span, X_log(:, k));
    
    X_log(:, k+1) = X_ode(end, :)';
    t_log(k+1) = t_span(2);
end

U_log(:, end) = U_log(:, end-1);

figure('Name', 'LQR Tracking', 'Color', 'w', 'Position', [100, 100, 1000, 600]);
tabgroup = uitabgroup(gcf);


% Stessa finestra
tab1 = uitab(tabgroup, 'Title', 'Posizioni Tracking');
tab2 = uitab(tabgroup, 'Title', 'Angoli di Eulero');

% X
subplot(2,2,1, 'Parent', tab1); hold on;
plot(t_log, X_log(1, :), 'b', 'LineWidth', 1.5);
yline(x_target(1), 'k--', 'Target X', 'LineWidth', 1);
grid on; title('X');
xlabel('Tempo [s]'); ylabel('Posizione [m]');

% Y
subplot(2,2,2, 'Parent', tab1); hold on;
plot(t_log, X_log(2, :), 'r', 'LineWidth', 1.5);
yline(x_target(2), 'k--', 'Target Y', 'LineWidth', 1);
grid on; title('Y');
xlabel('Tempo [s]'); ylabel('Posizione [m]');

% Z
subplot(2,2,3, 'Parent', tab1); hold on;
plot(t_log, X_log(3, :), 'g', 'LineWidth', 1.5);
yline(x_target(3), 'k--', 'Target Z', 'LineWidth', 1);
grid on; title('Z');
xlabel('Tempo [s]'); ylabel('Posizione [m]');

% Motori
subplot(2,2,4, 'Parent', tab1); hold on;
plot(t_log, U_log, 'LineWidth', 1.2);
grid on; title('Spinta Motori');
xlabel('Tempo [s]'); ylabel('Forza [N]');

% Phi
subplot(3,1,1, 'Parent', tab2); hold on;
plot(t_log, rad2deg(X_log(4, :)), 'm', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 1);
grid on; title('\phi');
ylabel('Angolo [deg]');

% Theta
subplot(3,1,2, 'Parent', tab2); hold on;
plot(t_log, rad2deg(X_log(5, :)), 'r', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 1);
grid on; title('\theta');
ylabel('Angolo [deg]');

% Psi
subplot(3,1,3, 'Parent', tab2); hold on;
plot(t_log, rad2deg(X_log(6, :)), 'b', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 1);
grid on; title('\psi');
ylabel('Angolo [deg]');
xlabel('Tempo [s]');
