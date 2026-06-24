% K - LQR TRACKING NL
clear; clc; close all;
load('MAT/step7_workspace.mat');
load('MAT/step4_workspace.mat', 'M_fun', 'C_fun', 'G_n', 'B_fun');
parametri_drone;
Gamma_num = double(Gamma_num);

f_s = 100; Ts = 1 / f_s;
Q = diag([[10 10 10], [50 50 50], [1 1 1], [1 1 1]]);
R = diag([0.1, 0.1, 0.1, 0.1]);
[K_lqr, ~, ~] = lqrd(A_lin, B_lin, Q, R, Ts);

alpha_cant = deg2rad(2);
F_eq = (m_val * g_val) / (4 * cos(alpha_cant)) * ones(4, 1);

t_sim_total = 30;
N_steps = round(t_sim_total / Ts);
t_log = zeros(1, N_steps);
X_log = zeros(12, N_steps);
U_log = zeros(4, N_steps);

x_target = zeros(12, 1);
x_target(1) = 7.5;
x_target(2) = 2.0;
x_target(3) = 0.0;

X_log(:, 1) = zeros(12,1);
t_log(1) = 0;
F_min = 0; F_max = 12;

for k = 1:(N_steps-1)
    delta_x = X_log(:, k) - x_target;
    delta_u  = -K_lqr * delta_x;
    F_motori = F_eq + delta_u;
    F_motori = max(min(F_motori, F_max), F_min);
    U_log(:, k) = F_motori;
    U_virtual = Gamma_num * F_motori;
    t_span = [t_log(k), t_log(k) + Ts];
    [~, X_ode] = ode45(@(t,x) drone_dynamics(t,x,U_virtual,M_fun,C_fun,G_n,B_fun), ...
                        t_span, X_log(:,k));
    X_log(:,k+1) = X_ode(end,:)';
    t_log(k+1) = t_span(2);
end
U_log(:,end) = U_log(:,end-1);

% ── Trova il momento in cui theta supera la soglia critica ──────────
SOGLIA_DEG = 15;
theta_deg   = rad2deg(X_log(5,:));
idx_soglia  = find(abs(theta_deg) > SOGLIA_DEG, 1);
t_soglia    = t_log(idx_soglia);

% ── Figura unica per presentazione ─────────────────────────────────
fig = figure('Color', [0.12 0.14 0.17], 'Position', [100 100 1200 620]);

BLUE   = [0.00 0.45 0.70];
RED    = [0.85 0.10 0.10];
GREEN  = [0.00 0.60 0.30];
PURPLE = [0.55 0.15 0.68];
GRAY   = [0.35 0.35 0.35];
GRIDF  = [0.32 0.35 0.40];

% ── Riga 1: posizioni ───────────────────────────────────────────────
subplot(2,3,1);
plot(t_log, X_log(1,:), 'Color', BLUE, 'LineWidth', 2.5); hold on;
yline(x_target(1), '--', 'Color', GRAY, 'LineWidth', 1.5, 'Label', 'Target');
if ~isempty(idx_soglia)
    xline(t_soglia, ':', 'Color', RED, 'LineWidth', 2);
end
grid on; set(gca, 'GridColor', GRIDF, 'GridAlpha', 1, ...
    'Box','on', 'FontSize', 11, 'Color', [0.18 0.20 0.24]);
xlabel('Tempo [s]'); ylabel('x [m]'); title('\bfPosizione X', 'FontSize',12);

subplot(2,3,2);
plot(t_log, X_log(2,:), 'Color', RED, 'LineWidth', 2.5); hold on;
yline(x_target(2), '--', 'Color', GRAY, 'LineWidth', 1.5, 'Label', 'Target');
if ~isempty(idx_soglia)
    xline(t_soglia, ':', 'Color', RED, 'LineWidth', 2);
end
grid on; set(gca, 'GridColor', GRIDF, 'GridAlpha', 1, ...
    'Box','on', 'FontSize', 11, 'Color', [0.18 0.20 0.24]);
xlabel('Tempo [s]'); ylabel('y [m]'); title('\bfPosizione Y', 'FontSize',12);

subplot(2,3,3);
plot(t_log, X_log(3,:), 'Color', GREEN, 'LineWidth', 2.5); hold on;
yline(x_target(3), '--', 'Color', GRAY, 'LineWidth', 1.5, 'Label', 'Target');
if ~isempty(idx_soglia)
    xline(t_soglia, ':', 'Color', RED, 'LineWidth', 2);
end
grid on; set(gca, 'GridColor', GRIDF, 'GridAlpha', 1, ...
    'Box','on', 'FontSize', 11, 'Color', [0.18 0.20 0.24]);
xlabel('Tempo [s]'); ylabel('z [m]'); title('\bfPosizione Z', 'FontSize',12);

% ── Riga 2: angoli ──────────────────────────────────────────────────
subplot(2,3,4);
plot(t_log, rad2deg(X_log(4,:)), 'Color', PURPLE, 'LineWidth', 2.5);
grid on; set(gca, 'GridColor', GRIDF, 'GridAlpha', 1, ...
    'Box','on', 'FontSize', 11, 'Color', [0.18 0.20 0.24]);
xlabel('Tempo [s]'); ylabel('\phi [°]'); title('\bfRollio \phi', 'FontSize',12);

subplot(2,3,5);
plot(t_log, theta_deg, 'Color', RED, 'LineWidth', 2.5); hold on;
grid on; set(gca, 'GridColor', GRIDF, 'GridAlpha', 1, ...
    'Box','on', 'FontSize', 11, 'Color', [0.18 0.20 0.24]);
xlabel('Tempo [s]'); ylabel('\theta [°]'); title('\bfBeccheggio \theta', 'FontSize',12);

subplot(2,3,6);
plot(t_log, rad2deg(X_log(6,:)), 'Color', BLUE, 'LineWidth', 2.5);
grid on; set(gca, 'GridColor', GRIDF, 'GridAlpha', 1, ...
    'Box','on', 'FontSize', 11, 'Color', [0.18 0.20 0.24]);
xlabel('Tempo [s]'); ylabel('\psi [°]'); title('\bfImbardata \psi', 'FontSize',12);

sgtitle('LQR — Tracking verso [7.5, 2.0, 0.0] m', ...
    'FontSize', 13, 'FontWeight', 'bold', 'Color', 'white');
exportgraphics(fig, 'lqr_divergenza.png', 'Resolution', 200, 'BackgroundColor', [0.12 0.14 0.17]);
disp('Salvato: lqr_divergenza.png');