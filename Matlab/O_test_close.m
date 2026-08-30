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

% Pesi allineati a Q_test_close_LMPC.m. Una prima configurazione con il
% solo Q_pos (motivata dalla maggiore fedelta' del modello RK4 interno)
% produceva overshoot crescente: senza penalita' sulla velocita' il
% drone arriva veloce sul target e non ha incentivo a frenare in tempo.
Q_pos = diag([30, 30, 30]);
Q_vel = diag([15, 15, 15]);
Q_ang = diag([50, 50, 50]);
R_mot = diag([10, 10, 10, 10]);

alpha_cant = deg2rad(2);
F_eq = (m_val * g_val) / (4 * cos(alpha_cant)) * ones(4, 1);

% Setup problema MPC (costruisce f, F, opti, X, U, parametri)
fprintf('[diag] Inizio costruzione problema MPC (M_MPC_setup)...\n');
tic_setup = tic;
M_MPC_setup;
fprintf('[diag] Setup completato in %.1f s.\n', toc(tic_setup));

% T_sim = 7 s: con una mediana di ~5.3 s/step servono ~60 min, ma la
% finestra copre la convergenza completa sul target [3,4,3] (raggiunto
% intorno ai 5-6 s), rendendo la traiettoria confrontabile con quella
% dell'LMPC. Con T_sim = 4 s il drone si fermava a due terzi del percorso.
T_sim   = 7;
N_steps = round(T_sim / Ts);

X_log   = zeros(12, N_steps);
U_log   = zeros(4,  N_steps);
T_log   = (0:N_steps-1) * Ts;
t_solve = zeros(N_steps, 1);   % tempo di soluzione dell'NLP per passo [s]
ok_step = false(N_steps, 1);   % true se il solutore e' arrivato a convergenza

x_corrente             = zeros(12, 1);
bersaglio              = [3; 4; 3];

% Guess iniziale: hovering uniforme
u_guess                = repmat(F_eq, 1, N);
u_applicato_precedente = F_eq;
u_sol_prev             = repmat(F_eq, 1, N);

% Inizializza anche le variabili di stato del guess a zero,
% cosi' il punto iniziale e' feasible per costruzione
opti.set_initial(X, zeros(12, N+1));
opti.set_initial(U, repmat(F_eq, 1, N));

t_wall = tic;

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

    if k <= 10
        fprintf('[diag] Inizio solve step k=%d...\n', k);
    end

    t0 = tic;
    try
        sol         = opti.solve();
        t_solve(k)  = toc(t0);
        ok_step(k)  = true;

        if k <= 10
            fprintf('[diag]   step k=%d risolto in %.2f s\n', k, t_solve(k));
        end

        u_opt       = sol.value(U);
        u_applicato = u_opt(:, 1);
        % Trasla il guess per il passo successivo
        u_guess     = [u_opt(:, 2:end), u_opt(:, end)];
        u_sol_prev  = u_opt;

    catch ME
        % Il tempo va registrato anche sui passi falliti: lasciarlo a
        % zero falserebbe le statistiche verso il basso.
        t_solve(k)  = toc(t0);
        ok_step(k)  = false;

        % Fallback: secondo step della soluzione precedente
        % (ha momenti non nulli, non causa divergenza)
        u_applicato = u_sol_prev(:, 2);
        u_guess     = [u_sol_prev(:, 2:end), u_sol_prev(:, end)];

        if k <= 5
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
        fprintf('t=%05.2f s | Z=%+5.2f m | X=%+5.2f m | Y=%+5.2f m | wall %.1f min\n', ...
                k*Ts, x_corrente(3), x_corrente(1), x_corrente(2), toc(t_wall)/60);
    end
end

X_log   = X_log(:, 1:N_steps);
U_log   = U_log(:, 1:N_steps);
T_log   = T_log(1:N_steps);
t_solve = t_solve(1:N_steps);
ok_step = ok_step(1:N_steps);

%% ---- Metriche tempo di soluzione ----
% Il primo passo e' escluso: non dispone di warm-start e non e'
% rappresentativo del regime.
ts = t_solve(2:end) * 1e3;

fprintf('\n=== TEMPO DI SOLUZIONE NMPC (N=%d, Ts=%.0f ms) ===\n', N, Ts*1e3);
fprintf('n campioni : %d  (falliti: %d)\n', numel(ts), sum(~ok_step(2:end)));
fprintf('media      : %8.2f ms\n', mean(ts));
fprintf('mediana    : %8.2f ms\n', median(ts));
fprintf('p95        : %8.2f ms\n', prctile(ts, 95));
fprintf('max        : %8.2f ms\n', max(ts));
fprintf('oltre Ts   : %8.1f %%\n', 100*mean(ts > Ts*1e3));
fprintf('durata tot : %8.1f min\n', toc(t_wall)/60);

% Errore finale di posizione
err_fin = norm(X_log(1:3, end) - bersaglio);
fprintf('errore finale di posizione: %.3f m\n', err_fin);

%% ---- FIGURE ----
% Tutte le figure sono forzate su tema chiaro per coerenza con il resto
% del documento e per la stampa.
schiarisci = @() set(gca, 'Color','w', 'XColor','k', 'YColor','k', 'ZColor','k');

% --- Traiettoria 3D ---
f1 = figure('Name','NMPC - Traiettoria','Color','w','Position',[100 100 600 500]);
plot3(X_log(1,:), X_log(2,:), X_log(3,:), 'b-', 'LineWidth', 2); hold on;
plot3(0, 0, 0, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
plot3(bersaglio(1), bersaglio(2), bersaglio(3), ...
      'rp', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
grid on; xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title('Traiettoria NMPC');
legend('Traiettoria effettiva','Partenza','Target [3,4,3]','Location','best');
view(45, 25);
schiarisci();
set(findall(f1,'Type','text'), 'Color','k');

% --- Forze motori ---
f2 = figure('Name','NMPC - Forze Motori','Color','w','Position',[750 100 600 400]);
plot(T_log, U_log', 'LineWidth', 1.5); hold on;
yline(12, 'r--', 'Saturazione 12 N', 'LineWidth', 2);
yline(F_eq(1), 'k--', sprintf('F_{eq} = %.2f N', F_eq(1)), 'LineWidth', 1);
ylim([0 13]);
ylabel('Forza [N]'); xlabel('Tempo [s]');
title('Forze motori NMPC');
legend('M1','M2','M3','M4','Location','east');
grid on;
schiarisci();
set(findall(f2,'Type','text'), 'Color','k');

% --- Angoli di Eulero ---
f3 = figure('Name','NMPC - Angoli','Color','w','Position',[750 550 600 400]);
plot(T_log, rad2deg(X_log(4:6,:)'), 'LineWidth', 1.5); hold on;
yline( 30, 'r--', 'Limite +30°', 'LineWidth', 1.5);
yline(-30, 'r--', 'Limite -30°', 'LineWidth', 1.5);
ylabel('Angolo [°]'); xlabel('Tempo [s]');
title('Angoli di Eulero NMPC');
legend('\phi','\theta','\psi','Location','best');
grid on;
schiarisci();
set(findall(f3,'Type','text'), 'Color','k');

%% ---- ESPORTAZIONE PDF VETTORIALE ----
if ~exist('pdf', 'dir'); mkdir('pdf'); end
exportgraphics(f1, 'pdf/NMPC_traiettoria.pdf', 'ContentType', 'vector');
exportgraphics(f2, 'pdf/NMPC_motori.pdf',      'ContentType', 'vector');
exportgraphics(f3, 'pdf/NMPC_angoli.pdf',      'ContentType', 'vector');
fprintf('Figure esportate in pdf/\n');

%% ---- SALVATAGGIO DATI ----
save('MAT/nmpc_risultati.mat', 'X_log', 'U_log', 'T_log', ...
     't_solve', 'ok_step', 'bersaglio', 'N', 'Ts', 'T_sim');
fprintf('Dati salvati in MAT/nmpc_risultati.mat\n');