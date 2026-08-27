% Q - SIMULAZIONE CLOSED-LOOP MPC LINEARE (LTI)
clear; clc; close all;
import casadi.*
% Carico dati dinamica non lineare (per simulare la realta' fisica)
load('MAT/step4_workspace.mat', 'M_fun', 'C_fun', 'G_n', 'B_fun');
% Carico matrici linearizzate (usate internamente dal controllore MPC)
load('MAT/step7_workspace.mat');
parametri_drone;
Gamma_num = double(Gamma_num);
Ts = 0.01;         % Tempo di campionamento
N = 60;            % Orizzonte di predizione
limite_angoli = pi/6;
% Pesi per la funzione di costo (Definita Positiva)
% Aggiungo matrici anche per velocita e angoli in mood che valga la
% linearizzazione e il controllo non diverga come avveniva solo con matrice
% pesi delle posizioni test 1
Q_pos = diag([30, 30, 30]);
Q_vel = diag([15, 15, 15]);
Q_ang = diag([50, 50, 50]);
R_mot = diag([10, 10, 10, 10]);
alpha_cant = deg2rad(2);
F_eq = (m_val * g_val) / (4 * cos(alpha_cant)) * ones(4, 1);
% Richiamo file setup per matematica MPC (Script P)
P_setup_LMPC;
T_sim = 10 ;
N_steps = round(T_sim / Ts);
% Salvataggio dati per grafici
X_log = zeros(12, N_steps);
U_log = zeros(4, N_steps);
T_log = (0:N_steps-1) * Ts;
x_corrente = zeros(12, 1); % Condizioni iniziali
bersaglio = [7.5; 2; 0];     % Target
% Ipotesi iniziale per il solutore
u_guess = repmat(F_eq, 1, N);
u_applicato_precedente = F_eq;
for k = 1:N_steps
    opti.set_value(x0_param, x_corrente);
    opti.set_value(target_pos, bersaglio);
    opti.set_value(u_prev_param, u_applicato_precedente);
    % Warm starting
    opti.set_initial(U, u_guess);
    try
        % Risoluzione problema ottimizzazione (LTI, estremamente veloce)
        sol = opti.solve();
        u_opt = sol.value(U);
        % Prendo prima u di controllo
        u_applicato = u_opt(:, 1);
        % Sposto al passo successivo
        u_guess = [u_opt(:, 2:end), u_opt(:, end)];
    catch
        disp('Errore soluzione non trovata');
        u_applicato = F_eq; % Hovering se rileva problemi
    end
    % Salvo dati per grafici
    X_log(:, k) = x_corrente;
    U_log(:, k) = u_applicato;
    % Trasformo in numeri per ode 45
    U_virtual = double(full(Gamma_num * u_applicato));
    x_corrente_num = double(full(x_corrente));
    G_n_num = double(full(G_n));
    t_span_num = double([T_log(k), T_log(k) + Ts]);
    [~, X_ode] = ode45(@(t, x) drone_dynamics(t, x, U_virtual, M_fun, C_fun, G_n_num, B_fun), t_span_num, x_corrente_num);
    x_corrente = X_ode(end, :)';
    u_applicato_precedente = u_applicato;
    % Stampa lo stato ogni 10 passi
    if mod(k, 10) == 0
        fprintf('Tempo: %05.2f s | Quota Z: %05.2f m | X: %05.2f m | Y: %05.2f m\n', ...
            k*Ts, x_corrente(3), x_corrente(1), x_corrente(2));
    end
end
% Traiettoria
figure('Name', 'MPC LTI - Traiettoria di Volo', 'Color', 'w', 'Position', [100, 100, 600, 500]);
plot3(X_log(1,:), X_log(2,:), X_log(3,:), 'r-', 'LineWidth', 2); hold on; % Linea rossa per LTI
plot3(0, 0, 0, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
plot3(bersaglio(1), bersaglio(2), bersaglio(3), 'bp', 'MarkerSize', 12, 'MarkerFaceColor', 'b');
grid on; xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title('Traiettoria (MPC LTI su Drone Non Lineare)');
legend('Traiettoria Effettiva', 'Partenza', 'Target', 'Location', 'best');
axis equal;
view(3);
% Sforzo motori
figure('Name', 'MPC LTI - Saturazione Motori', 'Color', 'w', 'Position', [750, 100, 600, 400]);
plot(T_log, U_log', 'LineWidth', 1.5); hold on;
yline(12, 'r--', 'Saturazione (12 N)', 'LineWidth', 2);
ylabel('Forza Singolo Motore [N]'); xlabel('Tempo [s]');
title('Motori (MPC LTI)');
legend('Motore 1', 'Motore 2', 'Motore 3', 'Motore 4', 'Location', 'east');
grid on;