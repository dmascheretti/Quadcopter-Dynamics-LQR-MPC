% H - LQR SU MODELLO LINARE 
clear; clc; close all;

load('MAT/step7_workspace.mat'); 

f_s = 100;         % Frequenza di campionamento [Hz]
Ts = 1 / f_s;      % Tempo di campionamento 10 ms


% Vettore di stato: x = [x y z phi theta psi dx dy dz dphi dtheta dpsi]

% Ad ogni elemento assegno un peso in base alla priorità 

pesi_posizione = [10, 10, 10];
pesi_angoli    = [50, 50, 50];  % Priorita alta
pesi_vel_lin   = [1, 1, 1];
pesi_vel_ang   = [1, 1, 1];     % Priotita bassa 

Q = diag([pesi_posizione, pesi_angoli, pesi_vel_lin, pesi_vel_ang]);

% Matrice R test aggressiva

rho = 0.1;
R = diag([rho, rho, rho, rho]);


% lqrd calcola il guadagno ottimale K per il sistema discreto
[K_lqr, ~, ~] = lqrd(A_lin, B_lin, Q, R, Ts);

disp(K_lqr(:, 1:6)); 


% Matrice C e D identita e zeri 

sys_continuo = ss(A_lin, B_lin, eye(12), zeros(12,4));

sys_discreto = c2d(sys_continuo, Ts, 'zoh');

% Sostiutisco u con -K * x_k e raccolgo u_k
A_cl = sys_discreto.A - sys_discreto.B * K_lqr;
sys_cl = ss(A_cl, zeros(12,4), eye(12), zeros(12,4), Ts);

% Condizioni iniziale 
x0 = zeros(12,1);
x0(5) = 0.2;  % Pitch
x0(3) = 1.0;  % z = 1 m

% Simulazione
t_sim = 0:Ts:5;
[Y, t_out, X] = initial(sys_cl, x0, t_sim);

figure('Name', 'LQR Lineare Discreto', 'Color', 'w');

subplot(2,1,1);
plot(t_out, X(:, 4), 'b', 'LineWidth', 1.5); hold on;
plot(t_out, X(:, 5), 'r', 'LineWidth', 1.5);
plot(t_out, X(:, 6), 'g', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 1);
grid on; title('Stabilizzazione Angoli di Assetto');
xlabel('Tempo [s]'); ylabel('Angolo [rad]');
legend('\phi', '\theta', '\psi');

subplot(2,1,2);
plot(t_out, X(:, 1), 'b', 'LineWidth', 1.5); hold on;
plot(t_out, X(:, 2), 'r', 'LineWidth', 1.5);
plot(t_out, X(:, 3), 'g', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 1);
grid on; title('Stabilizzazione Posizione');
xlabel('Tempo [s]'); ylabel('Posizione [m]');
legend('X', 'Y', 'Z');

save('MAT/step8_workspace.mat', 'K_lqr', 'Ts');