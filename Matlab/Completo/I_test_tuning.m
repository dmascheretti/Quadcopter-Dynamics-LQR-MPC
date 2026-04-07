% H2 - CONFRONTO CON TUNING

load('MAT/step7_workspace.mat'); 

Ts = 0.01;

% Matrici discerete come nello script precedente
sys_continuo = ss(A_lin, B_lin, eye(12), zeros(12,4));
sys_discreto = c2d(sys_continuo, Ts, 'zoh');
Ad = sys_discreto.A;
Bd = sys_discreto.B;

% CASO 1
Q1 = diag([[10 10 10], [50 50 50], [1 1 1], [1 1 1]]);
R1 = diag([0.1, 0.1, 0.1, 0.1]); 
[K1, ~, ~] = lqrd(A_lin, B_lin, Q1, R1, Ts);

% CASO 2
Q2 = diag([[2 2 2], [10 10 10], [1 1 1], [1 1 1]]);
R2 = diag([5.0, 5.0, 5.0, 5.0]); 
[K2, ~, ~] = lqrd(A_lin, B_lin, Q2, R2, Ts);

% CASO 3
Q3 = diag([[50 50 50], [200 200 200], [5 5 5], [5 5 5]]);
R3 = diag([0.001, 0.001, 0.001, 0.001]); 
[K3, ~, ~] = lqrd(A_lin, B_lin, Q3, R3, Ts);


t_sim = 0:Ts:5;
N = length(t_sim);

X1 = zeros(12, N); X2 = zeros(12, N); X3 = zeros(12, N);

% Condizione iniziale comune: Pitch 0.2 rad, Z = 1 m

x0 = zeros(12,1); x0(5) = 0.2; x0(3) = 1.0;
X1(:,1) = x0; X2(:,1) = x0; X3(:,1) = x0;

% Ciclo di simulazione (Tutti e 3 i droni in parallelo)
for k = 1:(N-1)
    % Drone 1 (Bilanciato)
    u1 = -K1 * X1(:, k);
    X1(:, k+1) = Ad * X1(:, k) + Bd * u1;
    
    % Drone 2 (Lento)
    u2 = -K2 * X2(:, k);
    X2(:, k+1) = Ad * X2(:, k) + Bd * u2;
    
    % Drone 3 (Aggressivo)
    u3 = -K3 * X3(:, k);
    X3(:, k+1) = Ad * X3(:, k) + Bd * u3;
end

figure('Name', 'Confronto Tuning LQR (H2)', 'Color', 'w', 'Position', [100, 100, 1000, 400]);

% Colori e stili per i tre droni
c1 = 'b'; s1 = '-';  % Blu   (Originale)
c2 = 'r'; s2 = '-'; % Rosso  (Lento)
c3 = 'g'; s3 = '-'; % Verde  (Aggressivo)

% Grafico 1 - Pitch
subplot(1,3,1); hold on;
plot(t_sim, X1(5, :), 'Color', c1, 'LineStyle', s1, 'LineWidth', 1.5); 
plot(t_sim, X2(5, :), 'Color', c2, 'LineStyle', s2, 'LineWidth', 1.5);
plot(t_sim, X3(5, :), 'Color', c3, 'LineStyle', s3, 'LineWidth', 1.5);
plot([t_sim(1) t_sim(end)], [0 0], 'k--', 'LineWidth', 1); 
grid on; title('\theta');
xlabel('Tempo [s]'); ylabel('Angolo [rad]');
legend('1. Bilanciato', '2. Lento (R alto)', '3. Aggressivo (R basso)');

% Grafico 2 - Z
subplot(1,3,2); hold on;
plot(t_sim, X1(3, :), 'Color', c1, 'LineStyle', s1, 'LineWidth', 1.5); 
plot(t_sim, X2(3, :), 'Color', c2, 'LineStyle', s2, 'LineWidth', 1.5);
plot(t_sim, X3(3, :), 'Color', c3, 'LineStyle', s3, 'LineWidth', 1.5);
plot([t_sim(1) t_sim(end)], [0 0], 'k--', 'LineWidth', 1);
grid on; title('Posizione z');
xlabel('Tempo [s]'); ylabel('Posizione [m]');


% Grafico 3 - X
subplot(1,3,3); hold on;
plot(t_sim, X1(1, :), 'Color', c1, 'LineStyle', s1, 'LineWidth', 1.5); 
plot(t_sim, X2(1, :), 'Color', c2, 'LineStyle', s2, 'LineWidth', 1.5);
plot(t_sim, X3(1, :), 'Color', c3, 'LineStyle', s3, 'LineWidth', 1.5);
plot([t_sim(1) t_sim(end)], [0 0], 'k--', 'LineWidth', 1);
grid on; title('x');
xlabel('Tempo [s]'); ylabel('Posizione [m]');