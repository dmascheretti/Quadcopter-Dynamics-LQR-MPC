% F1 - SIMULAZIONE CON SISTEMA NON LINEARE 

load('MAT/step4_workspace.mat');

parametri_drone;

% Condizioni iniziali
x0 = zeros(12,1); % valori del vettore X a 0
x0(3)  = 1.0;   % quota iniziale z=1m
x0(4)  = 0.1;   % angolo di roll iniziale

% Posso fare stessi test con roll pitch e yaw per vedere come cambia la 
% posizione del drone nello spazio

% Tempo da 0 a 5 s
t_span = [0, 5];  

%% TEST 1

% T=mg, tau=0
u_hover = [m_val*g_val; 0; 0; 0]; % equilibrio ingressi e momenti = 0

% Risolve equazione differenziale con funzioni calcolate nello step
% dell'equilibrio - Funzione, asse temporale, condizioni inziali
ode_fun = @(t,x) drone_dynamics(t, x, u_hover, M_fun, C_fun, G_n, B_fun);
[t1, X1] = ode45(ode_fun, t_span, x0);

% Rappresentazione grafica

figure('Name','Hover con roll iniziale');
subplot(2,3,1); plot(t1, X1(:,1)); xlabel('t [s]'); ylabel('x [m]'); title('Posizione x');
subplot(2,3,2); plot(t1, X1(:,2)); xlabel('t [s]'); ylabel('y [m]'); title('Posizione y');
subplot(2,3,3); plot(t1, X1(:,3)); xlabel('t [s]'); ylabel('z [m]'); title('Quota z');
subplot(2,3,4); plot(t1, rad2deg(X1(:,4))); ylabel('\phi [°]'); title('Roll \phi'); xlabel('t [s]');
subplot(2,3,5); plot(t1, rad2deg(X1(:,5))); ylabel('\theta [°]'); title('Pitch \theta'); xlabel('t [s]');
subplot(2,3,6); plot(t1, rad2deg(X1(:,6))); ylabel('\psi [°]'); title('Yaw \psi'); xlabel('t [s]');
sgtitle('Hover (T=mg, tau=0), roll_0=0.1 rad');

%% TEST 2

% Test 2: Scalata in quota
u_salita = [m_val*g_val*1.2; 0; 0; 0];   % T maggiore del peso --> andrà in alto
x0_level = zeros(12,1); % Inizializzo equilibrio
x0_level(3) = 0; % Parto da 0 su z


ode_fun2 = @(t,x) drone_dynamics(t, x, u_salita, M_fun, C_fun, G_n, B_fun);
[t2, X2] = ode45(ode_fun2, [0,3], x0_level); % 3 è il tempo in cui poi si ferma

figure('Name','Salita in quota');
plot(t2, X2(:,3),'b','LineWidth',2);
xlabel('t [s]'); ylabel('z [m]'); title('Scalata: T = 1.2mg');
grid on;

%% TEST 3

% Test 3: Coppia di pitch, il drone si muoverà in avanti e aumenterà theta
u_pitch = [m_val*g_val; 0; 0.01; 0];    % piccola coppia di pitch, momento torcente theta
ode_fun3 = @(t,x) drone_dynamics(t, x, u_pitch, M_fun, C_fun, G_n, B_fun);
[t3, X3] = ode45(ode_fun3, [0,3], zeros(12,1));

figure('Name','Coppia di pitch');
subplot(1,2,1);
plot(t3, rad2deg(X3(:,5)),'r','LineWidth',2);
xlabel('t [s]'); ylabel('\theta [°]'); title('Pitch (\theta)');
subplot(1,2,2);
plot(t3, X3(:,1),'k','LineWidth',2);
xlabel('t [s]'); ylabel('x [m]'); title('Movimento su x');
sgtitle('tau_\theta = 0.01 Nm');


% disegna_drone_3d(t1, X1);
% disegna_drone_3d(t2, X2);
% disegna_drone_3d(t3, X3);

save('MAT/step6_workspace.mat');