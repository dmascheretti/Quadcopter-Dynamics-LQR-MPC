% =========================================================================
% FILE: init_params.m
% DESCRIZIONE: Inizializzazione dei parametri fisici del Quadricottero
% RIFERIMENTO: Tesi G. Raffo, Capitolo 2, Tabella 2.1
% =========================================================================

clear all;  % Pulisce il workspace da vecchie variabili
clc;        % Pulisce la command window

disp('Caricamento dei parametri del drone in corso...');

% --- Parametri Fisici di Base ---
m = 2.24;       % [kg] Massa del quadricottero
g = 9.81;       % [m/s^2] Accelerazione di gravità terrestre
l = 0.332;      % [m] Distanza tra il centro di massa e i rotori (braccio)

% --- Parametri Aerodinamici (Eliche) ---
b = 9.5e-6;     % [N*s^2] Coefficiente di spinta dei rotori (Thrust)
k_tau = 1.7e-7; % [N*m*s^2] Coefficiente di resistenza aerodinamica (Drag)

% --- Momenti di Inerzia ---
% Poiché il drone è simmetrico e assumiamo il centro di massa coincidente
% con l'origine del Body frame, la matrice di inerzia è puramente diagonale.
Ixx = 0.0363;   % [kg*m^2] Inerzia sull'asse X (Rollio)
Iyy = 0.0363;   % [kg*m^2] Inerzia sull'asse Y (Beccheggio)
Izz = 0.0615;   % [kg*m^2] Inerzia sull'asse Z (Imbardata)

% Creazione della Matrice di Inerzia J (3x3 diagonale)
J = diag([Ixx, Iyy, Izz]); 

disp('Parametri caricati con successo nel Workspace!');