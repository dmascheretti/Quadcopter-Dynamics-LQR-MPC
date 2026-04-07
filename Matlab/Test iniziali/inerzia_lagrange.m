function J_eta = inerzia_lagrange(phi, theta, Ixx, Iyy, Izz)
% =========================================================================
% FUNZIONE: inerzia_lagrange.m
% DESCRIZIONE: Calcola la matrice di inerzia generalizzata J(eta) 
%              per il modello di Eulero-Lagrange.
% INPUT:
%   phi, theta - Angoli di rollio e beccheggio
%   Ixx, Iyy, Izz - Momenti di inerzia costanti
% OUTPUT:
%   J_eta - Matrice di inerzia 3x3 dipendente dall'assetto
% =========================================================================

    % Richiama la matrice cinematica W che hai creato prima!
    W = cinematica_W(phi, theta);
    
    % Matrice di inerzia fissa (Body frame)
    I = diag([Ixx, Iyy, Izz]);
    
    % Calcolo della matrice di inerzia Lagrangiana (Trasposta di W * I * W)
    J_eta = W' * I * W;

end