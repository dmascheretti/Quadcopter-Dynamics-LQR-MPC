function W_inv = cinematica_W_inv(phi, theta)
% =========================================================================
% FUNZIONE: cinematica_W_inv.m
% DESCRIZIONE: Calcola l'inversa della matrice di trasformazione W.
%              Serve per passare da (p, q, r) alle derivate di Eulero.
%              ATTENZIONE: Singolarità per theta = +/- 90 gradi (Gimbal Lock)
% INPUT:
%   phi   - Angolo di rollio (radianti)
%   theta - Angolo di beccheggio (radianti)
% OUTPUT:
%   W_inv - Matrice cinematica inversa 3x3
% =========================================================================

    c_phi = cos(phi);
    s_phi = sin(phi);
    c_theta = cos(theta);
    t_theta = tan(theta);

    % Controllo di sicurezza: evito la divisione per zero esatta
    if abs(c_theta) < 1e-6
        warning('ATTENZIONE: Vicinanza al Gimbal Lock (theta vicino a 90 gradi)!');
    end

    % Costruzione della matrice W inversa
    W_inv = [1, s_phi * t_theta, c_phi * t_theta;
             0, c_phi,          -s_phi;
             0, s_phi / c_theta, c_phi / c_theta];

end