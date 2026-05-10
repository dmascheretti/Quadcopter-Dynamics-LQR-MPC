% P - MPC LINEARE (LTI) SETUP
import casadi.*

% Carico matrici linearizzate (A_lin e B_lin sono già numeriche)
load('MAT/step7_workspace.mat'); 

% Discretizzazione delle matrici (ZOH)
% delta_x_dot = A*delta_x + B*delta_u
sys_c = ss(A_lin, B_lin, eye(12), zeros(12,4));
sys_d = c2d(sys_c, Ts, 'zoh');
Ad = sys_d.A;
Bd = sys_d.B;

% Definizione variabili CasADi
x = MX.sym('x', 12);
u = MX.sym('u', 4); % Forze motori assolute

% Modello discreto lineare: x_next = Ad*x + Bd*(u - F_eq)
% Usiamo (u - F_eq) perché B_lin è calcolata come derivata rispetto alla forza
x_next = Ad * x + Bd * (u - F_eq); % corretto solo se x_eq = 0
F_lin = Function('F_lin', {x, u}, {x_next});

opti = casadi.Opti();

% Variabili di decisione
X = opti.variable(12, N+1); 
U = opti.variable(4, N);    

% Parametri (valori che cambiano ad ogni passo del loop)
x0_param   = opti.parameter(12, 1); 
target_pos = opti.parameter(3, 1);  
u_prev_param = opti.parameter(4, 1);

% Vincolo stato iniziale
opti.subject_to(X(:,1) == x0_param);

% Vincoli dinamica lineare
for k = 1:N
    opti.subject_to(X(:, k+1) == F_lin(X(:, k), U(:, k)));
end

% Vincoli fisici (Saturazione motori e angoli)
opti.subject_to(0 <= U <= 12); 

% Erorre se tengo cosi 
% opti.subject_to(-limite_angoli <= X(4:5, :) <= limite_angoli);

opti.subject_to(-limite_angoli <= X(4:5, 2:end) <= limite_angoli);

% Funzione di costo (identica all'NMPC)

% Se non metto vincolo terminale non ha senso calcolare equazione di
% Riccati per ottenere la P
cost = 0;
for k = 1:N
    if k == 1
        delta_u = U(:, 1) - u_prev_param; 
    else
        delta_u = U(:, k) - U(:, k-1);
    end
    
    % Errore di posizione rispetto al target
   err_pos = X(1:3, k) - target_pos; % Errore di posizione
    angoli  = X(4:6, k);              % Angoli Eulero
    vel_lin = X(7:9, k);              % Velocità lineari
    
    % Costo quadratico totale
    cost = cost + err_pos' * Q_pos * err_pos ...
                + vel_lin' * Q_vel * vel_lin ...
                + angoli'  * Q_ang * angoli ...
                + delta_u' * R_mot * delta_u;
end

p_opts = struct('expand', true, 'print_time', false); 
s_opts = struct('max_iter', 100, ...
                'print_level', 0, ...  
                'sb', 'yes', ...         
                'print_user_options', 'no'); 

opti.solver('ipopt', p_opts, s_opts);