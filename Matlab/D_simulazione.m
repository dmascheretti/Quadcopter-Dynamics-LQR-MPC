% D- SIMULAZIONE CON VALORI NUMERICI E FUNZIONI

load('MAT/step3_workspace.mat'); % Carica le matrici simboliche 

parametri_drone; % Carico valori drone da file esterno


% Ottengo matrici con numeri, subs sostituisce simboli con numeri presenti
% in parametri_drone

% Matrice inerzia uso sia quella completa che quella semplificata
% Poi uso solo quella M_num

M_num = subs(M_q, [m, Ixx, Iyy, Izz], [m_val, Ixx_val, Iyy_val, Izz_val]);
M_num_completa = subs (M_q_completa, [m, Ixx, Iyy, Izz, rx, ry, rz], [m_val, Ixx_val, Iyy_val, Izz_val, rx_val, ry_val, rz_val]);
C_num = subs(C_q, [Ixx, Iyy, Izz], [Ixx_val, Iyy_val, Izz_val]);
B_num = B_q; % (La matrice B_q dipende solo dagli angoli, non ha massa/inerzie da sostituire)


display (M_num);
display (M_num_completa);
display(C_num);
display(B_num);


% Converto matrici in funzioni per velocizzare il calcolo negli step
% successivi
M_fun = matlabFunction(M_num, 'Vars', {phi, theta, psi});
C_fun = matlabFunction(C_num, 'Vars', {phi, theta, psi, dphi, dtheta, dpsi});
B_fun = matlabFunction(B_num, 'Vars', {phi, theta, psi});


% G_n è costante non calcolo funzione
G_n = [0; 0; m_val * g_val; 0; 0; 0]; 

display(G_n);

% Derivata di M simbolica e reale (regola catena con phi theta e psi)
dotM_sym = diff(M_q, phi)*dphi + diff(M_q, theta)*dtheta + diff(M_q, psi)*dpsi;
dotM_num = subs(dotM_sym, [m, Ixx, Iyy, Izz], [m_val, Ixx_val, Iyy_val, Izz_val]);


N = simplify(dotM_num - 2 * C_num);

% Verifica Antisimmetria !!! conj !!!
% fprintf('\nVerifica Antisimmetria (Diagonale deve essere nulla):\n');
% disp(simplify(N));

% Elementi sulla diagonale 
for i = 1:6
    e = N(i,i);

    % tolgo complessi coniugati
    elemento = subs(e, [conj(phi), conj(theta), conj(psi)], [phi, theta, psi]);

    % Semplifico e calcolo il valore reale dell'elemento
    elemento = simplify(subs(elemento, [real(phi), real(theta), real(psi)], [phi, theta, psi]));

   
    fprintf('Elemento (%d,%d): ', i, i);
    disp(elemento);
end


save('MAT/step4_workspace.mat');