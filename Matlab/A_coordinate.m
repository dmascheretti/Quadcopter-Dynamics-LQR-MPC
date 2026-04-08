% A - VARIABILI E STATI

% Coordinate generalizzate: q = [xi; eta]
% xi = [x; y; z]          → posizioni
% eta = [phi; theta; psi] → angoli di Eulero

% Vettore di stato completo EL: x = [q; q_dot] ∈ R^12

% Ingressi: Gamma = [T; tau_phi; tau_theta; tau_psi] ∈ R^4 : spinta verticale motori 
%                                                         + momenti sugli angoli di Eulero

% Parametri fisici drone sono nel file a parte 

syms rx ry rz real;
r_sym = [rx; ry; rz];

display (r_sym);

% CM coincide con Body Frame (r=0)
% rx_val = 0; ry_val = 0; rz_val = 0;

syms m g Ixx Iyy Izz l real

% Tensore di inerzia J
J = diag([Ixx, Iyy, Izz]);

display(J);

% Vettore di stato simbolico a valori reali 
syms x y z phi theta psi real               % q
syms dx dy dz dphi dtheta dpsi real         % q_dot, derivata prima del vettore q
syms ddx ddy ddz ddphi ddtheta ddpsi real   % q_ddot, derivata seconda del vettore q

% Coordinate generalizzate Eulero-Lagrange
xi  = [x; y; z];
eta = [phi; theta; psi]; % Angoli Eulero
q   = [xi; eta];         

display (q);
disp (length(q));

% Velocità generalizzate Eulero-Lagrange
dxi  = [dx; dy; dz];
deta = [dphi; dtheta; dpsi];
q_dot = [dxi; deta];       

display (q_dot);
disp(length(q_dot));


% Accelerazioni generalizzate Eulero-Lagrange
q_ddot = [ddx; ddy; ddz; ddphi; ddtheta; ddpsi];  

% Vettore Gamma degli ingressi (Thrust + momenti angolari)
syms T tau_phi tau_theta tau_psi real

Gamma = [T; tau_phi; tau_theta; tau_psi];          


% Vettore di stato completo X
x_state = [q; q_dot];    

display (x_state);

% Salvataggio in modo che uso stesse variabili negli step successivi
save('MAT/step1_workspace.mat');