% B - MODELLAZIONE DINAMICA 

% Carico valori dallo step precedente 
load('MAT/step1_workspace.mat');

% L'equazione base Eulero-Lagrange: 
%
%  M(q)*q_ddot + C(q,q_dot)*q_dot + G(q) = B(q)*Gamma
%
% Formula inversa per ricavare derivata seconda di q e ricavare le
% equazioni:
% q_ddot = M(q)^{-1} * [-C*q_dot - G + B*Gamma]
%
% Le matrici M, C, G, B vengono calcolate nello file successivo.

fprintf('  M(q)*q_ddot + C*q_dot + G = B*Gamma\n');


save('MAT/step2_workspace.mat');