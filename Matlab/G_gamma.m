% 6 - MATRICE GAMMA

load('MAT/step3_workspace.mat'); 
parametri_drone; 

syms F1 F2 F3 F4 real
u_forze = [F1; F2; F3; F4]; 

syms l c real % c coefficiente per il calcolo del tau yaw

% Matrice che traforma T + momenti in forze
Gamma = [  1,  1,  1,  1;   % T = F1 + F2 + F3 + F4
           0,  l,  0, -l;   % tau_phi = l*F2 - l*F4 
          -l,  0,  l,  0;   % tau_theta = -l*F1 + l*F3
           c, -c,  c, -c ]; % tau_psi = c*F1 - c*F2 + c*F3 - c*F4

display (Gamma);

% Sostituisco e ottengo gamma numerica
Gamma_num = subs(Gamma, [l, c], [l_val, c_val]);

% Vettore ingressi 
U = Gamma_num * u_forze;


% Ottengo vettore ingressi in base alle forze
display (U);

% Nuovo sistema con ingressi con forze e non più T + momenti
q_ddot_sym_f = simplify(M_q \ (-C_q*q_dot - G_q + B_q*U));

% q_dot rimane la stessa, q_ddot cambia per vettore U
f_sym_f = [q_dot; q_ddot_sym_f];
x_sym = [q; q_dot];

% Equlibrio, tutto 0 ( x y z pari a 0 per comodità )
x_eq = zeros(12,1); 

% Angolo di due gradi ipotetico 
alpha_cant = deg2rad(2); 
Forza_eq = (m_val * g_val) / (4 * cos(alpha_cant));
u_forze_eq = [Forza_eq; Forza_eq; Forza_eq; Forza_eq];

% Jacobiani
A_sym_f = jacobian(f_sym_f, x_sym);
B_sym_f = jacobian(f_sym_f, u_forze);

% Sostituzione dello stato e dell'ingresso di equilibrio
A_lin_simb = subs(A_sym_f, [x_sym; u_forze], [x_eq; u_forze_eq]);
B_lin_simb = subs(B_sym_f, [x_sym; u_forze], [x_eq; u_forze_eq]);

A_lin = double(subs(A_lin_simb, [m, g, Ixx, Iyy, Izz], ...
                                [m_val, g_val, Ixx_val, Iyy_val, Izz_val]));
                            
% Riga 9 -> divisione delle forze
% Riga 10 -> solo colonna 2 e 4 ovvero forze 2 e 4 per generare rollio 
% Stessa cosa per righe 11 e 12 con beccheggio e imbardata
B_lin = double(subs(B_lin_simb, [m, g, Ixx, Iyy, Izz], ...
                                [m_val, g_val, Ixx_val, Iyy_val, Izz_val]));

display (A_lin);

display (B_lin);

% Verifica Controllabilità
matrice_controllabilita = ctrb(A_lin, B_lin);
rango = rank(matrice_controllabilita);

if rango == 12
    disp('Controllabile');
else
    disp('Non controllabile');
end

save('MAT/step7_workspace.mat');
