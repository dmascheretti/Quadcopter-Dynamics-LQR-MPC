clear; clc;

load('MAT/step4_workspace.mat'); % M_fun, C_fun, B_fun
load('MAT/step7_workspace.mat'); % A_lin, B_lin
parametri_drone;

Ts = 0.01;  
N = 60; 
limite_angoli = pi/6;
alpha_cant = deg2rad(2); 
F_eq_val = (m_val * g_val) / (4 * cos(alpha_cant));
F_eq = F_eq_val * ones(4,1);

Q_pos = diag([30, 30, 30]);      
Q_vel = diag([15, 15, 15]);      
Q_ang = diag([50, 50, 50]);      
R_mot = diag([10, 10, 10, 10]);


import casadi.*


sys_c = ss(A_lin, B_lin, eye(12), zeros(12,4));
sys_d = c2d(sys_c, Ts, 'zoh');
Ad = sys_d.A;
Bd = sys_d.B;

x = MX.sym('x', 12);
u = MX.sym('u', 4); % Forze motori assolute

% Modello discreto lineare
x_next = Ad * x + Bd * (u - F_eq);
F_lin = Function('F_lin', {x, u}, {x_next});

opti = casadi.Opti();
X = opti.variable(12, N+1); 
U = opti.variable(4, N);    

% Parametri 
x0_param   = opti.parameter(12, 1); 
target_pos = opti.parameter(3, 1);  
u_prev_param = opti.parameter(4, 1);

% Vincoli
opti.subject_to(X(:,1) == x0_param);

for k = 1:N
    opti.subject_to(X(:, k+1) == F_lin(X(:, k), U(:, k)));
end

opti.subject_to(0 <= U <= 12); 
opti.subject_to(-limite_angoli <= X(4:5, 2:end) <= limite_angoli);

% Costo
cost = 0;
for k = 1:N
    if k == 1
        delta_u = U(:, 1) - u_prev_param; 
    else
        delta_u = U(:, k) - U(:, k-1);
    end
    
    err_pos = X(1:3, k) - target_pos; 
    angoli  = X(4:6, k);              
    vel_lin = X(7:9, k);              
    
    cost = cost + err_pos' * Q_pos * err_pos ...
                + vel_lin' * Q_vel * vel_lin ...
                + angoli'  * Q_ang * angoli ...
                + delta_u' * R_mot * delta_u;
end
opti.minimize(cost);

p_opts = struct('expand', true, 'print_time', false); 
s_opts = struct('max_iter', 100, ...
                'print_level', 0, ...  
                'sb', 'yes', ...         
                'print_user_options', 'no'); 
opti.solver('ipopt', p_opts, s_opts);