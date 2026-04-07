function M_matrix = inerzia_completa(m, phi, theta, Ixx, Iyy, Izz)
% =========================================================================
% FUNZIONE: inerzia_completa.m
% DESCRIZIONE: Calcola la matrice di inerzia TOTALE M(q) 6x6 
%              (Traslazione + Rotazione) per il modello completo a 6 GdL.
% =========================================================================

    % 1. Parte Traslazionale (Blocco 3x3 in alto a sinistra)
    % Usiamo eye(3) che crea la matrice identità [1 0 0; 0 1 0; 0 0 1]
    M_trasl = m * eye(3); 
    
    % 2. Parte Rotazionale (Blocco 3x3 in basso a destra)
    % Richiamiamo la matrice W e creiamo J_eta
    W = cinematica_W(phi, theta);
    I = diag([Ixx, Iyy, Izz]);
    J_eta = W' * I * W;
    
    % 3. Unione nella matrice completa 6x6
    % La funzione blkdiag unisce le due matrici mettendole sulla diagonale
    % e riempiendo il resto di zeri.
    M_matrix = blkdiag(M_trasl, J_eta);

end