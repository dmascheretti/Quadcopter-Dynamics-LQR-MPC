
load('MAT/step4_workspace.mat');  % M_fun, C_fun, B_fun, G_n
load('MAT/step7_workspace.mat');  % A_lin, B_lin, Gamma_num, F_eq


B_lin(9, :) = -B_lin(9, :);

Ts  = 0.01;
N   = 60;
limite_angoli = deg2rad(22);
F_eq_val = (m_val * g_val) / (4 * cos(alpha_cant));
F_eq = F_eq_val * ones(4,1);

Q_pos = diag([100,  100,  100 ]);
Q_vel = diag([15,  15,  15 ]);
Q_ang = diag([20, 20, 20]);
R_mot = diag([0.1,  0.1,  0.1, 0.1]);

import casadi.*
opti = casadi.Opti();

X            = opti.variable(12, N+1);
U            = opti.variable(4,  N);
x0_param     = opti.parameter(12, 1);
target_pos   = opti.parameter(3,  1);
u_prev_param = opti.parameter(4,  1);
Ad_param     = opti.parameter(12, 12);
Bd_param     = opti.parameter(12, 4);

opti.subject_to(X(:,1) == x0_param);

for k = 1:N
    opti.subject_to(X(:,k+1) == Ad_param*X(:,k) + Bd_param*(U(:,k) - F_eq));
end

opti.subject_to(0 <= U <= 40);
opti.subject_to(-limite_angoli <= X(4:5, 2:end) <= limite_angoli);

cost = 0;
for k = 1:N
    if k == 1
        delta_u = U(:,1) - u_prev_param;
    else
        delta_u = U(:,k) - U(:,k-1);
    end
    err_pos = X(1:3,k) - target_pos;
    angoli  = X(4:6,k);
    vel_lin = X(7:9,k);
    cost = cost + err_pos'*Q_pos*err_pos ...
                + vel_lin'*Q_vel*vel_lin ...
                + angoli' *Q_ang*angoli  ...
                + delta_u'*R_mot*delta_u;
end
opti.minimize(cost);

opts = struct();
opts.qpsol = 'qrqp';             
opts.expand = true;              

opts.print_time = false;      
opts.print_iteration = false; 
opts.print_header = false;
opts.print_status = true;      

opts.qpsol_options.print_iter = false; 
opts.qpsol_options.print_header = false; 
opts.qpsol_options.print_info = false;   
opti.solver('sqpmethod', opts);

sys_c = ss(A_lin, B_lin, eye(12), zeros(12,4));
sys_d = c2d(sys_c, Ts, 'zoh');
Ad    = sys_d.A;
Bd    = sys_d.B;

opti.set_value(Ad_param,     Ad);
opti.set_value(Bd_param,     Bd);
opti.set_value(x0_param,     zeros(12,1));
opti.set_value(target_pos,   [3;4;3]);
opti.set_value(u_prev_param, F_eq);


% da fare:
% cambiare lpv calcolando seti matrici ad ogni passo come foglio note

% cambiare il blocco drone phys con uav toolbox o qualcoss di siile dando
% solo ingressi

% altrimenti non va 
xy_data_membrane = [-0.1 -0.1; 0.1 -0.1; 0.1 0.1; -0.1 0.1];