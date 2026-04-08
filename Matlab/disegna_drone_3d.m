function disegna_drone_3d(t, X)

% FUNZIONE: disegna_drone_3d
% INPUT: 
%   t - vettore del tempo uscito da ode45
%   X - matrice degli stati (Nx12) uscita da ode45


    figure('Name', 'Simulazione 3D Drone', 'Color', 'w');
    view(3); grid on; hold on;
    axis equal;
    
    % Limiti
    xlim([-10, 10]); ylim([-10, 10]); zlim([-10, 10]);
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');

    % Disegno un pavimento verde/grigio
    patch([-2 2 2 -2], [-2 -2 2 2], [0 0 0 0], [0.8 0.9 0.8], 'FaceAlpha', 0.5);

    % Lunghezza bracci del drone (L)
    L = 0.225; 

    % Geometria del drone
    braccio1 = [L, 0, 0]'; braccio2 = [0, L, 0]';
    braccio3 = [-L, 0, 0]'; braccio4 = [0, -L, 0]';
    motori_body = [braccio1, braccio2, braccio3, braccio4];

    % Inizializzo le linee per il disegno (il rosso indica il "muso" / asse X)
    plot_bracci1 = plot3(0,0,0, 'r-o', 'LineWidth', 3, 'MarkerSize', 6, 'MarkerFaceColor', 'r'); 
    plot_bracci2 = plot3(0,0,0, 'k-o', 'LineWidth', 3, 'MarkerSize', 6, 'MarkerFaceColor', 'k');

    % Disegno la scia
    scia = plot3(0,0,0, 'b.', 'MarkerSize', 2);

    % Ciclo di animazione (avanza di 5 step alla volta per non renderlo lentissimo)
    for k = 1:5:length(t)
        % 1. Estraggo lo stato attuale dalla riga 'k' della matrice X
        x_k = X(k, 1); y_k = X(k, 2); z_k = X(k, 3);
        phi_k = X(k, 4); theta_k = X(k, 5); psi_k = X(k, 6);
        
        % 2. Calcolo la rotazione R_I (da Body a Inerziale) per questo istante
        Rx = [1 0 0; 0 cos(phi_k) sin(phi_k); 0 -sin(phi_k) cos(phi_k)];
        Ry = [cos(theta_k) 0 -sin(theta_k); 0 1 0; sin(theta_k) 0 cos(theta_k)];
        Rz = [cos(psi_k) sin(psi_k) 0; -sin(psi_k) cos(psi_k) 0; 0 0 1];
        R_I = (Rx * Ry * Rz)'; 
        
        % 3. Ruoto e traslo i bracci del drone
        motori_inerziali = R_I * motori_body + [x_k; y_k; z_k];
        
        % 4. Aggiorno il disegno a schermo
        set(plot_bracci1, 'XData', [motori_inerziali(1,3), motori_inerziali(1,1)], ...
                          'YData', [motori_inerziali(2,3), motori_inerziali(2,1)], ...
                          'ZData', [motori_inerziali(3,3), motori_inerziali(3,1)]);
                      
        set(plot_bracci2, 'XData', [motori_inerziali(1,4), motori_inerziali(1,2)], ...
                          'YData', [motori_inerziali(2,4), motori_inerziali(2,2)], ...
                          'ZData', [motori_inerziali(3,4), motori_inerziali(3,2)]);
        
        set(scia, 'XData', X(1:k, 1), 'YData', X(1:k, 2), 'ZData', X(1:k, 3));
        
        drawnow; % Forza il refresh dello schermo
        pause(0.05); % Aspetta 20 millisecondi tra un frame e l'altro
    end
end