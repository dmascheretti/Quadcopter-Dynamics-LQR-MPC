% M - MPC CASADI SETUP
import casadi.*

% 12 stati e 4 ingressi
x = MX.sym('x', 12);
u = MX.sym('u', 4);

% x = [x,y,z,phi,theta,psi, dx,dy,dz, dphi,dtheta,dpsi]
phi_v    = x(4);  theta_v  = x(5);  psi_v   = x(6);
dphi_v   = x(10); dtheta_v = x(11); dpsi_v  = x(12);
qd_val   = x(7:12);

U_virtual = Gamma_num * u;
M_n = M_fun(phi_v, theta_v, psi_v);
C_n = C_fun(phi_v, theta_v, psi_v, dphi_v, dtheta_v, dpsi_v);
B_n = B_fun(phi_v, theta_v, psi_v);
q_ddot = M_n \ (-C_n*qd_val - G_n + B_n * U_virtual);
ode = [qd_val; q_ddot];

f = Function('f', {x, u}, {ode});

% Discretizzazione RK4
k1 = f(x, u);
k2 = f(x + Ts/2 * k1, u);
k3 = f(x + Ts/2 * k2, u);
k4 = f(x + Ts * k3, u);
x_next = x + Ts/6 * (k1 + 2*k2 + 2*k3 + k4);
F = Function('F', {x, u}, {x_next});

% ---- Problema di ottimizzazione ----
opti = casadi.Opti();

X = opti.variable(12, N+1);
U = opti.variable(4,  N);

x0_param     = opti.parameter(12, 1);
target_pos   = opti.parameter(3,  1);
u_prev_param = opti.parameter(4,  1);

opti.subject_to(X(:,1) == x0_param);
for k = 1:N
    opti.subject_to(X(:, k+1) == F(X(:, k), U(:, k)));
end

opti.subject_to(0 <= U <= 12);
% Vincoli angolari solo sugli stati futuri (k=2..N+1)
% X(:,1) e' fissato da x0_param, applicargli il vincolo
% rende il problema infeasible se il guess non lo rispetta esattamente
opti.subject_to(-limite_angoli <= X(4, 2:end) <= limite_angoli);
opti.subject_to(-limite_angoli <= X(5, 2:end) <= limite_angoli);

% Funzione di costo su stati futuri X(:,k+1)
cost = 0;
for k = 1:N
    if k == 1
        delta_u = U(:, 1) - u_prev_param;
    else
        delta_u = U(:, k) - U(:, k-1);
    end
    err_pos = X(1:3, k+1) - target_pos;
    vel_lin = X(7:9, k+1);
    angoli  = X(4:6, k+1);
    cost = cost + err_pos' * Q_pos * err_pos ...
                + vel_lin' * Q_vel * vel_lin ...
                + angoli'  * Q_ang * angoli ...
                + delta_u' * R_mot * delta_u;
end
opti.minimize(cost);

% expand=false obbligatorio: le funzioni M_fun/C_fun/B_fun usano
% LinsolQr che non supporta eval_sx richiesto da expand=true.
% (Tentativo con 'symbolicqr' + expand=true scartato: inlineando f/F
% 160 volte nel grafo SX piatto, la risoluzione lineare simbolica si
% moltiplica invece di essere condivisa come Function, e il costo
% dell'Hessiano esatto di IPOPT esplode invece di calare.)
p_opts = struct('expand', false, 'print_time', false);
% warm_start_init_point/bound_push: warm-start anche delle variabili
% duali (non solo del primale via opti.set_initial), come in Python
% (mpc_controller.py) — riduce le iterazioni quando il guess del passo
% precedente e' vicino alla soluzione.
s_opts = struct('max_iter',                   300,        ...
                'print_level',                0,          ...
                'sb',                         'yes',      ...
                'tol',                        1e-4,       ...
                'acceptable_tol',             1e-3,       ...
                'acceptable_iter',            5,           ...
                'warm_start_init_point',      'yes',      ...
                'warm_start_bound_push',      1e-6,       ...
                'warm_start_mult_bound_push', 1e-6,       ...
                'mu_strategy',                'adaptive', ...
                'hessian_approximation',      'limited-memory');
opti.solver('ipopt', p_opts, s_opts);