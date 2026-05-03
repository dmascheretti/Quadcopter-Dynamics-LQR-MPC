% O - SIMULAZIONE CLOSED-LOOP NMPC
clear; clc; close all;
import casadi.*

load('MAT/step4_workspace.mat', 'M_fun', 'C_fun', 'G_n', 'B_fun');
load('MAT/step7_workspace.mat');
parametri_drone;
Gamma_num = double(Gamma_num);

Ts = 0.05;         % Tempo di campionamento
N = 20; 
limite_angoli = pi/6;

% Pesi per la funzione di costo (Definita Positiva)
Q_pos = diag([100, 100, 100]); 
R_mot = diag([0.1, 0.1, 0.1, 0.1]); 

alpha_cant = deg2rad(2); 
F_eq = (m_val * g_val) / (4 * cos(alpha_cant)) * ones(4, 1);


M_MPC_setup; % Richiamo file setup per matematica MPC

T_sim = 20; % [s]
N_steps = round(T_sim / Ts);

% Salvataggio dati per grafici
X_log = zeros(12, N_steps);
U_log = zeros(4, N_steps);
T_log = (0:N_steps-1) * Ts;

x_corrente = zeros(12, 1); % Condizioni iniziali
bersaglio = [1; 7; 20];     % Taget

% Quello che facevamo con setInitial nello script precedente
u_guess = repmat(F_eq, 1, N);


for k = 1:N_steps
    
    % Aggiornamento valori
    opti.set_value(x0_param, x_corrente);
    opti.set_value(target_pos, bersaglio);
    
    % Ipotisi inziale per velocizzare processo
    opti.set_initial(U, u_guess);
    
    try
        % Risoluzione problema ottimizzazione
        sol = opti.solve();
        u_opt = sol.value(U);
        
        % Prendo prima u di controllo
        u_applicato = u_opt(:, 1);
        
        % Sposto al passo successivo
        u_guess = [u_opt(:, 2:end), u_opt(:, end)];
    
    catch  
        u_applicato = F_eq; % Hovering se rileva problemi
    end
    
    % Salvo dati per grafici
    X_log(:, k) = x_corrente;
    U_log(:, k) = u_applicato;
    
    % Applico controllo + simulo porossimi passi con RK4
    x_corrente_casadi = F(x_corrente, u_applicato);
    x_corrente = full(x_corrente_casadi);
    
    % Stampa lo stato ogni 10 passi 
    if mod(k, 10) == 0
        fprintf('Tempo: %05.2f s | Quota Z: %05.2f m | X: %05.2f m | Y: %05.2f m\n', ...
                k*Ts, x_corrente(3), x_corrente(1), x_corrente(2));
    end
end

% Traiettoria
figure('Name', 'MPC - Traiettoria di Volo', 'Color', 'w', 'Position', [100, 100, 600, 500]);
plot3(X_log(1,:), X_log(2,:), X_log(3,:), 'b-', 'LineWidth', 2); hold on;
plot3(0, 0, 0, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
plot3(bersaglio(1), bersaglio(2), bersaglio(3), 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
grid on; xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title('Traiettoria');
legend('Traiettoria Effettiva', 'Partenza', 'Target', 'Location', 'best');
view(3);

% Sforzo motori
figure('Name', 'MPC - Saturazione Motori', 'Color', 'w', 'Position', [750, 100, 600, 400]);
plot(T_log, U_log', 'LineWidth', 1.5); hold on;
yline(12, 'r--', 'Saturazione (12 N)', 'LineWidth', 2);
ylabel('Forza Singolo Motore [N]'); xlabel('Tempo [s]');
title('Motori');
legend('Motore 1', 'Motore 2', 'Motore 3', 'Motore 4', 'Location', 'east');
grid on;


% posso anche inserire oltre a vincoli interni anche vincoli esterni es
% ostacoli ecc ecc --> aggiungo in mpc setup tutte le zone che non voglio
% che vengano raggiunte

% ostacolo_pos = [2.5; 1.0; 1.5]; % Lo mettiamo a metà strada verso il bersaglio
% raggio_ostacolo = 0.8;          % Raggio di sicurezza (80 centimetri)

%    distanza_quadrato = (X(1, k+1) - ostacolo_pos(1))^2 + ...
%                        (X(2, k+1) - ostacolo_pos(2))^2 + ...
%                        (X(3, k+1) - ostacolo_pos(3))^2;
%    opti.subject_to(distanza_quadrato >= raggio_ostacolo^2);



% Se auemnta a 1 7 20 fa un effetto strano tipo fionda come mai?
