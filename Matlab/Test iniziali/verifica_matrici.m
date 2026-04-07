% =========================================================================
% SCRIPT: verifica_matrici.m
% DESCRIZIONE: Test matematici per validare le matrici del drone
% =========================================================================
clc; clear;

% Inventiamo degli angoli casuali per il test (in radianti)
phi = 0.2; 
theta = -0.15; 
psi = 0.8;

disp('======================================================');
disp('   INIZIO VERIFICA MATRICI');
disp('======================================================');

%% TEST 1: Ortogonalità della Matrice di Rotazione (R)
% La teoria dice che l'inversa di una matrice di rotazione è uguale 
% alla sua trasposta. Quindi R * R' DEVE dare la matrice identità (I).
disp('--- TEST 1: Matrice di Rotazione (R) ---');
R = rotazione_ZYX(phi, theta, psi);
Test_1 = round(R * R', 5); % Arrotondo per evitare errori numerici di MATLAB

disp('Risultato R * R'' (Deve essere la matrice Identità):');
disp(Test_1);
if isequal(Test_1, eye(3))
    disp('✅ TEST 1 SUPERATO! La rotazione è corretta.');
else
    disp('❌ ERRORE NEL TEST 1');
end
disp(' ');

%% TEST 2: Inversa della Matrice Cinematica (W)
% La teoria dice che moltiplicando una matrice per la sua inversa
% si ottiene sempre la matrice Identità.
disp('--- TEST 2: Matrici Cinematiche (W e W_inv) ---');
W = cinematica_W(phi, theta);
W_inv = cinematica_W_inv(phi, theta);
Test_2 = round(W * W_inv, 5);

disp('Risultato W * W_inv (Deve essere la matrice Identità):');
disp(Test_2);
if isequal(Test_2, eye(3))
    disp('✅ TEST 2 SUPERATO! L''inversa è calcolata perfettamente.');
else
    disp('❌ ERRORE NEL TEST 2');
end
disp(' ');

%% TEST 3: Il Test Supremo di Passività (Antisimmetria)
% La teoria di Eulero-Lagrange dice che la matrice (J_dot - 2*C) 
% deve essere ANTISIMMETRICA. (Diagonale a zero, e termini opposti invertiti di segno)
% NOTA: Per farlo serve derivare J nel tempo, lo facciamo numericamente:

disp('--- TEST 3: Proprietà di Antisimmetria (J_dot - 2C) ---');
% Parametri drone
Ixx = 0.0363; Iyy = 0.0363; Izz = 0.0615;
phid = 0.5; thetad = 0.1; psid = 0.2; % Velocità angolari inventate

% 1. Calcolo C
C = coriolis_lagrange(phi, theta, psi, phid, thetad, psid, Ixx, Iyy, Izz);

% 2. Calcolo J_dot (derivata di J) usando una piccola approssimazione numerica (dt)
dt = 0.0001;
J_t1 = inerzia_lagrange(phi, theta, Ixx, Iyy, Izz);
J_t2 = inerzia_lagrange(phi + phid*dt, theta + thetad*dt, Ixx, Iyy, Izz);
J_dot = (J_t2 - J_t1) / dt;

% 3. Calcolo la matrice di Test (N)
N = J_dot - 2*C;

disp('Matrice N = (J_dot - 2C). Deve essere ANTISIMMETRICA:');
disp(N);
disp('Verifica: N + N'' (Deve essere tutto ZERO):');
disp(round(N + N', 4));

disp('======================================================');