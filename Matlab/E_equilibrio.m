% E - EQUILIBRIO 

% Equilibrio: q=0, x=0, y=0, z=0 + angoli Eulero = 0 (vettore x_eq=0)
% Ingresso: T=mg, momenti angolari = 0 -> hovering


load('MAT/step3_workspace.mat');

% test massa : fprintf('m = %f\n', double(m));

parametri_drone;

% Calcolo simbolico del vettore f(x,u)
% syms T_s tau_phi_s tau_theta_s tau_psi_s real % Ingressi simbolici


u_sym = [T; tau_phi; tau_theta; tau_psi]; % Uso quelli dello step 1

% Qui calcolo il mio sistema da q ddot equazioni rotazionali +
% traslazionali (\ indica inversa)

% f_sym è il mio sistema f (x,u)

q_ddot_sym = simplify(M_q \ (-C_q*q_dot - G_q + B_q*u_sym));
f_sym = [q_dot; q_ddot_sym];

display (f_sym);

x_sym      = [q; q_dot];

% Punto di equilibrio, vettore x=0, velocita uguale a 0 --> q = 0
x_eq = zeros(12,1);   % hover: z=0, origine assi nel centro del drone

% Motori spingono verso l'alto con m*g, gli altri momenti sono pari a zero
% (Uso i valori numerici nel file parametri_drone)

u_eq = [m_val*g_val; 0; 0; 0];

% Forza singolo rotore
f_singola = u_eq(1) / 4;
fprintf('Forza singolo rotore: %f N\n',f_singola);

% Calcolo delle derivate parziali per linearizzazione con jacobiani
A_sym = jacobian(f_sym, x_sym);  % Derivate parziali rispetto a X
B_sym = jacobian(f_sym, u_sym);  % Derivate parziali rispetto a u

display (A_sym);

display (B_sym);


% Sostituzione al punto di equilibrio x = vettore di 0, u = [mg, 0, 0, 0]
A_lin_simb = subs(A_sym, [x_sym; u_sym], [x_eq; u_eq]);
B_lin_simb = subs(B_sym, [x_sym; u_sym], [x_eq; u_eq]);

display (A_lin_simb);
display (B_lin_simb);


% In forma numerica subs per sostituire con valori in parametri_drone
A_lin = double(subs(A_lin_simb, [m, g, Ixx, Iyy, Izz], [m_val, g_val, Ixx_val, Iyy_val, Izz_val]));
B_lin = double(subs(B_lin_simb, [m, g, Ixx, Iyy, Izz], [m_val, g_val, Ixx_val, Iyy_val, Izz_val]));

fprintf('  Matrice A: %dx%d\n', size(A_lin));
disp (A_lin);


fprintf('  Matrice B: %dx%d\n', size(B_lin));
disp (B_lin);

if rank(ctrb(A_lin, B_lin)) == 12, disp('Sistema Controllabile'); end


save('MAT/step5_workspace.mat');