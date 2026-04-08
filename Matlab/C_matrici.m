% C - DEFINZIONE MATRICI

% Carico varibaili dallo step 1
load('MAT/step1_workspace.mat');

% Test massa fprintf('m = %f\n', double(m));

% Matrice di Rotazione R_I ZYX  (da frame body a frame inerziale)
Rx = [1,          0,         0;
      0,    cos(phi),  sin(phi);
      0,   -sin(phi),  cos(phi)];

Ry = [cos(theta),  0, -sin(theta);
      0,           1,          0;
      sin(theta),  0,  cos(theta)];

Rz = [cos(psi),  sin(psi), 0;
     -sin(psi),  cos(psi), 0;
      0,         0,        1];

% R_B = rotazione da inerziale a body
R_B = Rx * Ry * Rz;

% R_I = rotazione da body a inerziale = R_B' (Trasposta)
R_I = simplify(R_B');

display(R_I);


% Matrice di Eulero W_eta per passare da derivate angoli a [p q r]
% omega = W_eta * eta_dot
W_eta = [1,  0,          -sin(theta);
         0,  cos(phi),    sin(phi)*cos(theta);
         0, -sin(phi),    cos(phi)*cos(theta)];


% Tensore di inerzia generalizzato J(eta) = W_eta' * J * W_eta
% Matrice in basso a dx della matrice M

J_eta = simplify(W_eta' * J * W_eta); 

display (J_eta); % conj sarebbe il coseno del complesso coniugato --> per numeri reali non importa 

% Matrice di inerzia M(q)
% Con r=0: S(r)=0, quindi i termini diagonali si annullano

% eye (3) matrice identità 3*3

M_q = [m*eye(3),      zeros(3,3);
       zeros(3,3),    J_eta];  

display (M_q);

% Matrice di Inerzia M(q) con r!=0 -> modello completo
S_r = [  0, -rz,  ry;
        rz,   0, -rx;
       -ry,  rx,   0];

M11 = m * eye(3);
M12 = -m * R_I * S_r * W_eta; 
M21 = M12';
M22 = J_eta; 

M_q_completa = [M11, M12;
                M21, M22];

display(M_q_completa);

% Matrice di Coriolis C(q,q_dot)

q_sym    = [x; y; z; phi; theta; psi];
q_dot_sym = [dx; dy; dz; dphi; dtheta; dpsi];

n = 6;
C_q = sym(zeros(n, n));

for k = 1:n
    for j = 1:n
        c_kj = sym(0);
        for i = 1:n
            m_kj = M_q(k,j);
            m_ki = M_q(k,i);
            m_ij = M_q(i,j);
            c_ijk = sym(0.5) * (diff(m_kj, q_sym(i)) + ...
                                diff(m_ki, q_sym(j)) - ...
                                diff(m_ij, q_sym(k)));
            c_kj = c_kj + c_ijk * q_dot_sym(i);
        end
        C_q(k,j) = c_kj;
    end
end
C_q = simplify(C_q);

display (C_q);


% Vettore di gravità G(q) 
% Energia potenziale: U = m*g*z  (z = quota inerziale, positiva verso l'alto)
% G(q) = dU/dq --> solo su asse z=m*g


U_pot = m * g * z;
G_q = sym(zeros(n,1));
for i = 1:n
    G_q(i) = diff(U_pot, q_sym(i));
end
G_q = simplify(G_q);

display(G_q);

% Matrice di ingresso B(q) con r = 0
e3 = [0; 0; 1];
f_col = R_I * e3;      % T agisce solo su asse z
B_q = [f_col,          zeros(3,3);  
       zeros(3,1),     eye(3)];  
B_q = simplify(B_q);

display (B_q);

% Ricavo stsse equazioni che avrei ricavato con Newton
q_ddot = M_q \ (-C_q*q_dot - G_q + B_q*Gamma);
display(q_ddot);

display (q_ddot(3));

save('MAT/step3_workspace.mat');