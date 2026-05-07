% N - MPC OPEN LOOP

clear; clc; close all;
import casadi.*

load('MAT/step4_workspace.mat', 'M_fun', 'C_fun', 'G_n', 'B_fun');
load('MAT/step7_workspace.mat');

Ts = 0.05;
N = 20;
parametri_drone;
Gamma_num = double(Gamma_num);

% Definite positive -> Lyapunov
Q_pos = diag([100, 100, 100]); % Mi interessa solo la posizione
R_mot = diag([0.1, 0.1, 0.1, 0.1]); 

alpha_cant = deg2rad(2); 
F_eq = (m_val * g_val) / (4 * cos(alpha_cant)) * ones(4, 1);

M_MPC_setup; % Qui creo solo ambiente non calocla nessun valore ancora

goal = [5; 2; 3];

x_start = zeros(12,1);
opti.set_value(x0_param, x_start); % Riempiamo parameter nello script precedente 
opti.set_value(target_pos, goal); % Stessa cosa

opti.set_initial(U, repmat(F_eq, 1, N)); % Velocizza processo, dico di aprtire con F eq

try
    sol = opti.solve(); % Testo tutte le traiettore e scelgo quella a costo minore
    disp('Soluzione trovata');

    U_ottima = sol.value(U); % Salvo qui la traiettoria

    % Indico forze 
    fprintf('M1: %.2f N | M2: %.2f N | M3: %.2f N | M4: %.2f N\n', ...
            U_ottima(1,1), U_ottima(2,1), U_ottima(3,1), U_ottima(4,1));

    X_pred = sol.value(X); 
    figure('Name', 'Predizione MPC', 'Color', 'w');
    plot3(X_pred(1,:), X_pred(2,:), X_pred(3,:), 'b--o', 'LineWidth', 1.5); hold on;
    plot3(0, 0, 0, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    plot3(goal(1), goal(2), goal(3), 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
    grid on; xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    title('Traiettoria Open-Loop'); legend('Waypoints', 'Partenza', 'Target');

catch e
    disp('Errore:');
    disp(e.message);
end

% In open loop non è detto che arrivi esattamente al target, si risolve
% chiudendo il loop e usando un ciclo for

% Se aumento la N invece arriva al tragurado correttamente ma ci metterà
% piu tempo