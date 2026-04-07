function R = rotazione_ZYX(phi, theta, psi)
% =========================================================================
% FUNZIONE: rotazione_ZYX.m
% DESCRIZIONE: Calcola la matrice di rotazione R dal Body frame 
%              all'Inerziale usando la convenzione degli angoli 
%              di Eulero Z-Y-X (Yaw-Pitch-Roll).
% INPUT:
%   phi   - Angolo di rollio (radianti)
%   theta - Angolo di beccheggio (radianti)
%   psi   - Angolo di imbardata (radianti)
% OUTPUT:
%   R     - Matrice di rotazione 3x3
% =========================================================================

    % Pre-calcolo seni e coseni per efficienza computazionale
    c_phi = cos(phi);
    s_phi = sin(phi);
    
    c_theta = cos(theta);
    s_theta = sin(theta);
    
    c_psi = cos(psi);
    s_psi = sin(psi);

    % Costruzione della matrice di rotazione (Tesi Raffo, Eq. 2.9 o simili)
    R = zeros(3,3); % Inizializza la matrice per velocità
    
    R(1,1) = c_psi * c_theta;
    R(1,2) = c_psi * s_theta * s_phi - s_psi * c_phi;
    R(1,3) = c_psi * s_theta * c_phi + s_psi * s_phi;
    
    R(2,1) = s_psi * c_theta;
    R(2,2) = s_psi * s_theta * s_phi + c_psi * c_phi;
    R(2,3) = s_psi * s_theta * c_phi - c_psi * s_phi;
    
    R(3,1) = -s_theta;
    R(3,2) = c_theta * s_phi;
    R(3,3) = c_theta * c_phi;

end