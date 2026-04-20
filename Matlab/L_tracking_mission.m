% K - LQR TRACKING NL (MULTI-WAYPOINT)
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

% Non considero molto energia ( può fare manovre aggressive )
R = diag([0.1, 0.1, 0.1, 0.1]); 

% R = diag([10, 10, 10, 10]); 
% R = diag([100, 100, 100, 100]); 

% Se aumento R diminuiscono gli angoli ( e aumenta il tempo ) 
% Con lqrd uso le matrivi continue e le rendo discrete inserendo il tempo
% di campionmento Ts negli argomenti
[K_lqr, ~, ~] = lqrd(A_lin, B_lin, Q, R, Ts);

% Calcolo della spinta di Hovering (F_eq) con angolo 2 gradi
alpha_cant = deg2rad(2); 
F_eq = (m_val * g_val) / (4 * cos(alpha_cant)) * ones(4, 1);

t_sim_total = 20; % tempo di simulazione di 15 secondi
N_steps = round(t_sim_total / Ts); % 15/0.01 = 15000

% Log delle variabili ( vettori di 0 lunghi N_steps che poi verranno
% riempiti durante la simulazione ) - 15000 righe
t_log = zeros(1, N_steps);
X_log = zeros(12, N_steps);
U_log = zeros(4, N_steps);
Target_log = zeros(3, N_steps);

% Lista waypoint da raggiungre ( sulle righe ) - trasposta
waypoints = [ 0.5,  0.5,  1.0;  
             -0.5,  1.0,  1.5;  
              1.0,  2.0,  0.0;
              0.0,  0.0,  1.0 ]';

% Posso usare questa tecnica anche per fare una traiettoria che in maniera
% diretta farebbe divergere il drone !

% Posso aggiungere quante righe voglio

wp_corrente = 1; % Attualmente puntiamo alla prima riga 
num_wp = size(waypoints, 2); % Il numero di waypoint è la dimensione della matrice


tolleranza = 0.2; % Tolleranza per cambiare setpoint, se mi avvicino di 20 cm 
% passo al setpoint successivo ( cosi non sta fermo a 0 )

% Inizializzo il primo Target 
x_target = zeros(12, 1);
x_target(1:3) = waypoints(:, wp_corrente); % Metto nelle prime tre posizioni
% i valori del primo waypoint ( o0vvero il target x y z sono quelle del
% primo waypoint )

% Ipotesi : partiamo da fermi ( per il grafico tutto a 0 a t = 0 )
X_log(:, 1) = zeros(12,1); 
t_log(1) = 0;

% Saturazione
F_min = 0;         
F_max = 12;        

for k = 1:(N_steps-1)
    
    dist = norm(X_log(1:3, k) - x_target(1:3)); % Distanza dal target

    if dist < tolleranza && wp_corrente < num_wp % Se è piu vicino della tolleranza e 
        % non ha finito i waypoint allora stampa OK + tempo
        disp(['WP ', num2str(wp_corrente), ' raggiunto al tempo t=', num2str(t_log(k)), 's']);
        wp_corrente = wp_corrente + 1;
        x_target(1:3) = waypoints(:, wp_corrente); % Aggiorno il target con il nuovo punto 
    end
    
    Target_log(:, k) = x_target(1:3); % Salvo per il grafico
    
    % Calcolo distanza dal target ad ogni step 
    delta_x = X_log(:, k) - x_target; 
    
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
Target_log(:, end) = x_target(1:3); % Completo il log dei target


figure('Name', 'LQR Tracking Multi-Waypoint', 'Color', 'w', 'Position', [100, 100, 1000, 600]);
tabgroup = uitabgroup(gcf);

% Stessa finestra
tab1 = uitab(tabgroup, 'Title', 'Posizioni Tracking');
tab2 = uitab(tabgroup, 'Title', 'Angoli di Eulero');

% X
subplot(2,2,1, 'Parent', tab1); hold on;
plot(t_log, X_log(1, :), 'b', 'LineWidth', 1.5);
plot(t_log, Target_log(1, :), 'k--', 'LineWidth', 1); 
grid on; title('X');
xlabel('Tempo [s]'); ylabel('Posizione [m]');

% Y
subplot(2,2,2, 'Parent', tab1); hold on;
plot(t_log, X_log(2, :), 'r', 'LineWidth', 1.5);
plot(t_log, Target_log(2, :), 'k--', 'LineWidth', 1);
grid on; title('Y');
xlabel('Tempo [s]'); ylabel('Posizione [m]');

% Z
subplot(2,2,3, 'Parent', tab1); hold on;
plot(t_log, X_log(3, :), 'g', 'LineWidth', 1.5);
plot(t_log, Target_log(3, :), 'k--', 'LineWidth', 1);
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

disegna_drone_3d(t_log, X_log', waypoints);

% 1 - Se tengo waypoints con valore 3...5 ecc ecc gli angoli impazziscono 
% sol. ridurre distanza oppure aumentare R ( la linearizzazione non
% funionava più )

% 2 - ha senso integrare un osservatore per il calcolo del vettore di stato
% stimato o ipotizziamo che il sistema sia completamente osservabile e i
% sensori perfetti?

% 3 - Ma se con angoli troppo elevati il drone diverge non si puo dividere
% il pezzetto di traiettoria in segmenti sempre piu piccoli in cui la
% linerizzazione vale ancora? E' questo che collega la teoria dell' LQR con
% quella del MPC?