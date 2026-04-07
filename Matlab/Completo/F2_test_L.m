% F2 - SIMULAZIONE CON SISTEMA LINEARE 

load('MAT/step5_workspace.mat');
parametri_drone;

% Definizione matrici C e D nel sistema linearizzato
C_lin = eye(12);        % Matrice identita, ipotesi : leggo tutti i sensori
D_lin = zeros(12, 4);   % Matrice di zeri
sys_lin = ss(A_lin, B_lin, C_lin, D_lin); % Sistema linearizzato

% Vettori temporali
t_span = 0:0.01:5;   % 5 secondi per il Test 1 + passo 0.01
t_span_breve = 0:0.01:3; % 3 secondi per Test 2 e 3 + passo 0.01

%% TEST 1

% Hovering con perturbazione iniziale

x0 = zeros(12,1);       % Equilibrio
x0(3)  = 1.0;           % quota iniziale z=1m
x0(4)  = 0.1;           % piccolo angolo di roll iniziale

% Utilizzo le variazioni e non il valore reale, in questo caso delta u = 0
u_hover_lin = zeros(length(t_span), 4); 

% lsim simula il sistema lineare
[Y1_lin, t1_lin, X1_lin] = lsim(sys_lin, u_hover_lin, t_span, x0);

figure('Name','Hover con roll iniziale');
subplot(2,3,1); plot(t1_lin, X1_lin(:,1), 'r', 'LineWidth', 1.5); xlabel('t [s]'); ylabel('x [m]'); title('Posizione x');
subplot(2,3,2); plot(t1_lin, X1_lin(:,2), 'r', 'LineWidth', 1.5); xlabel('t [s]'); ylabel('y [m]'); title('Posizione y');
subplot(2,3,3); plot(t1_lin, X1_lin(:,3), 'r', 'LineWidth', 1.5); xlabel('t [s]'); ylabel('z [m]'); title('Quota z');
subplot(2,3,4); plot(t1_lin, rad2deg(X1_lin(:,4)), 'r', 'LineWidth', 1.5); ylabel('\phi [°]'); title('Roll \phi'); xlabel('t [s]');
subplot(2,3,5); plot(t1_lin, rad2deg(X1_lin(:,5)), 'r', 'LineWidth', 1.5); ylabel('\theta [°]'); title('Pitch \theta'); xlabel('t [s]');
subplot(2,3,6); plot(t1_lin, rad2deg(X1_lin(:,6)), 'r', 'LineWidth', 1.5); ylabel('\psi [°]'); title('Yaw \psi'); xlabel('t [s]');
sgtitle('Hover (\Delta u=0), roll_0=0.1 rad');


%% TEST 2

% Scalata in quota (+20% di spinta)
x0_level = zeros(12,1); 


delta_T = 0.2 * m_val * g_val; % Rispetto a m * g il delta è 0.2 * m * g (prima era 1.2)

u_salita_lin = zeros(length(t_span_breve), 4);
u_salita_lin(:, 1) = delta_T; % Applico il delta_T per tutto il tempo

[Y2_lin, t2_lin, X2_lin] = lsim(sys_lin, u_salita_lin, t_span_breve, x0_level);

figure('Name','Salita in quota');
plot(t2_lin, X2_lin(:,3),'b','LineWidth',2);
xlabel('t [s]'); ylabel('z [m]'); title(['\Delta T = ', num2str(delta_T), ' N']);
grid on;


%% TEST 3 

% Coppia di pitch (Il drone si sposta in avanti)

u_pitch_lin = zeros(length(t_span_breve), 4);
u_pitch_lin(:, 3) = 0.01; % delta_tau_theta = 0.01 Nm

[Y3_lin, t3_lin, X3_lin] = lsim(sys_lin, u_pitch_lin, t_span_breve, zeros(12,1));

figure('Name','Coppia di pitch');
subplot(1,2,1);
plot(t3_lin, rad2deg(X3_lin(:,5)),'r','LineWidth',2);
xlabel('t [s]'); ylabel('\theta [°]'); title('Pitch (\theta)');
subplot(1,2,2);
plot(t3_lin, X3_lin(:,1),'k','LineWidth',2);
xlabel('t [s]'); ylabel('x [m]'); title('Movimento su x');
sgtitle('\Delta\tau_\theta = 0.01 Nm');

save('MAT/step6b_workspace.mat');