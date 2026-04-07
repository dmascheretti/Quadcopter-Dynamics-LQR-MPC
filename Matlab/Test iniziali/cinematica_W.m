function W = cinematica_W(phi, theta)
% =========================================================================
% FUNZIONE: cinematica_W.m
% DESCRIZIONE: Calcola la matrice di trasformazione W che lega le
%              derivate degli angoli di Eulero alle velocità angolari
%              nel Body frame (p, q, r).
% INPUT:
%   phi   - Angolo di rollio (radianti)
%   theta - Angolo di beccheggio (radianti)
% OUTPUT:
%   W     - Matrice cinematica 3x3
% =========================================================================

    c_phi = cos(phi);
    s_phi = sin(phi);
    c_theta = cos(theta);
    s_theta = sin(theta);

    % Costruzione della matrice W (Tesi Raffo, Eq. 2.11)
    W = [1,      0,            -s_theta;
         0,  c_phi, c_theta * s_phi;
         0, -s_phi, c_theta * c_phi];

end