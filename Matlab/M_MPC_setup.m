% M - MPC CASADI SETUP

import casadi.*

load('MAT/step7_workspace.mat');
load('MAT/step4_workspace.mat', 'M_fun', 'C_fun', 'G_n', 'B_fun');

parametri_drone;

Gamma_num = double(Gamma_num); % Matrice Gamma da step 7 

Ts = 0.05;         % Tempo campionamento
N = 20;            % Orizzonte predittivo 

Q_pos = diag([100, 100, 100]); % Agisce solo su x y z, gli angoli hanno i vincoli

R_mot = diag([0.1, 0.1, 0.1, 0.1]); 

alpha_cant = deg2rad(2); 
F_eq = (m_val * g_val) / (4 * cos(alpha_cant)) * ones(4, 1);

% 12 stati e 4 ingressi
x = MX.sym('x', 12); 
u = MX.sym('u', 4);  

limite_angoli = pi/6;

% Descrizione stati del sistema
% x = [x,y,z,ϕ,θ,ψ,x˙,y˙​,z˙,ϕ',θ',ψ']
phi_v    = x(4); theta_v  = x(5); psi_v    = x(6);
dphi_v   = x(10); dtheta_v = x(11); dpsi_v   = x(12);
qd_val   = x(7:12);

% Ricavo forze virtuali con Gamma + matrici M C B
U_virtual = Gamma_num * u;
M_n = M_fun(phi_v, theta_v, psi_v);
C_n = C_fun(phi_v, theta_v, psi_v, dphi_v, dtheta_v, dpsi_v);
B_n = B_fun(phi_v, theta_v, psi_v);

% Calcolo derivata seconda e creo vettore X ( ode )
q_ddot = M_n \ (-C_n*qd_val - G_n + B_n * U_virtual);
ode = [qd_val; q_ddot];

% Funzione x derivato = f ( x , u )
f = Function('f', {x, u}, {ode});

% Discretizzazione con Runge-Kutta 4
k1 = f(x, u);
k2 = f(x + Ts/2 * k1, u);
k3 = f(x + Ts/2 * k2, u);
k4 = f(x + Ts * k3, u);

% Formula per trovare la prossima posizione in base allo momento corrente
% Calcola avanzamento del drone
x_next = x + Ts/6 * (k1 + 2*k2 + 2*k3 + k4);
F = Function('F', {x, u}, {x_next});

opti = casadi.Opti();

% Traiettoria e comandi futuri
X = opti.variable(12, N+1); 
U = opti.variable(4, N);    

x0_param   = opti.parameter(12, 1); % Stato attuale
target_pos = opti.parameter(3, 1);  % Target 

% Inserisco vincoli

opti.subject_to(X(:,1) == x0_param); % Condizione iniziale
for k = 1:N
    opti.subject_to(X(:, k+1) == F(X(:, k), U(:, k))); % Rispetta leggi dinamica 
end
opti.subject_to(0 <= U <= 12); % Saturazione motori 0 - 12 N
opti.subject_to(-limite_angoli <= X(4:5, :) <= limite_angoli); % Limiti angoli (Pitch e Roll)

% Funzioni di costo, valuta le traiettorie calcolate, obiettivo costo
% minore possibile
cost = 0;
for k = 1:N
    err_pos = X(1:3, k) - target_pos;
    delta_u = U(:, k) - F_eq; 
    cost = cost + err_pos' * Q_pos * err_pos + delta_u' * R_mot * delta_u;
end
opti.minimize(cost); % Tiene solo quella con costo minore

% Impostazioni OPTS
p_opts = struct('expand', false);
s_opts = struct('max_iter', 100, 'print_level', 0);
opti.solver('ipopt', p_opts, s_opts);