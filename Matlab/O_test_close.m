% O - SIMULAZIONE CLOSED-LOOP NMPC
clear; clc; close all;
import casadi.*

load('MAT/step4_workspace.mat', 'M_fun', 'C_fun', 'G_n', 'B_fun');
load('MAT/step7_workspace.mat');
parametri_drone;
Gamma_num = double(Gamma_num);

Ts            = 0.01;
N             = 40;
limite_angoli = pi/6;   % ±30°

Q_pos = diag([100, 100, 100]);
R_mot = diag([1,   1,   1,   1]);

alpha_cant = deg2rad(2);
F_eq = (m_val * g_val) / (4 * cos(alpha_cant)) * ones(4, 1);

% Setup problema MPC (costruisce f, F, opti, X, U, parametri)
M_MPC_setup;

T_sim   = 10;
N_steps = round(T_sim / Ts);

X_log = zeros(12, N_steps);
U_log = zeros(4,  N_steps);
T_log = (0:N_steps-1) * Ts;

x_corrente             = zeros(12, 1);
bersaglio              = [3; 4; 3];

% Guess iniziale: hovering uniforme
u_guess                = repmat(F_eq, 1, N);
u_applicato_precedente = F_eq;
u_sol_prev             = repmat(F_eq, 1, N);

% Inizializza anche le variabili di stato del guess a zero
% cosi il punto iniziale e' feasible per costruzione
opti.set_initial(X, zeros(12, N+1));
opti.set_initial(U, repmat(F_eq, 1, N));

for k = 1:N_steps

    % Controllo divergenza
    if any(abs(x_corrente(4:5)) > pi/2)
        fprintf('[!] Divergenza angolare a t=%.2f s\n', k*Ts);
        N_steps = k-1;
        break;
    end

    % Aggiorna parametri
    opti.set_value(x0_param,     x_corrente);
    opti.set_value(target_pos,   bersaglio);
    opti.set_value(u_prev_param, u_applicato_precedente);

    % Guess warm-start
    opti.set_initial(U, u_guess);

    try
        sol   = opti.solve();
        u_opt = sol.value(U);
        u_applicato = u_opt(:, 1);
        % Trasla il guess per il passo successivo
        u_guess     = [u_opt(:, 2:end), u_opt(:, end)];
        u_sol_prev  = u_opt;

    catch ME
        % Fallback: secondo step della soluzione precedente
        % (ha momenti non nulli, non causa divergenza)
        u_applicato = u_sol_prev(:, 2);
        u_guess     = [u_sol_prev(:, 2:end), u_sol_prev(:, end)];
        if k <= 5
            % Stampa errore solo nei primi passi per diagnostica
            fprintf('[warn] t=%.2fs IPOPT: %s\n', k*Ts, ME.message);
        elseif k == 6
            fprintf('[warn] (messaggi successivi soppressi)\n');
        end
    end

    X_log(:, k) = x_corrente;
    U_log(:, k) = u_applicato;

    % Avanza la dinamica con RK4
    x_corrente = full(F(x_corrente, u_applicato));
    u_applicato_precedente = u_applicato;

    if mod(k, 50) == 0
        fprintf('t=%05.2f s | Z=%+5.2f m | X=%+5.2f m | Y=%+5.2f m\n', ...
                k*Ts, x_corrente(3), x_corrente(1), x_corrente(2));
    end
end

X_log = X_log(:, 1:N_steps);
U_log = U_log(:, 1:N_steps);
T_log = T_log(1:N_steps);

% ---- FIGURE ----
figure('Name','NMPC - Traiettoria','Color','w','Position',[100 100 600 500]);
plot3(X_log(1,:), X_log(2,:), X_log(3,:), 'b-', 'LineWidth', 2); hold on;
plot3(0, 0, 0, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
plot3(bersaglio(1), bersaglio(2), bersaglio(3), ...
      'rp', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
grid on; xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title('Traiettoria NMPC');
legend('Traiettoria effettiva','Partenza','Target [3,4,3]','Location','best');
view(45, 25);

figure('Name','NMPC - Forze Motori','Color','w','Position',[750 100 600 400]);
plot(T_log, U_log', 'LineWidth', 1.5); hold on;
yline(12, 'r--', 'Saturazione 12 N', 'LineWidth', 2);
yline(F_eq(1), 'k--', sprintf('F_{eq} = %.2f N', F_eq(1)), 'LineWidth', 1);
ylabel('Forza [N]'); xlabel('Tempo [s]');
title('Forze motori NMPC');
legend('M1','M2','M3','M4','Location','east');
grid on;