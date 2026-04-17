function disegna_drone_3d(t, X, waypoints)
    figure('Name', 'Simulazione 3D Drone', 'Color', 'w', 'Position', [100, 100, 800, 800]);
    view(3); grid on; hold on;
    axis equal;
    
    xlim([-5, 5]); ylim([-2, 6]); zlim([0, 6]);
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    
    patch([-10 10 10 -10], [-10 -10 10 10], [0 0 0 0], [0.8 0.9 0.8], 'FaceAlpha', 0.5);
    
    if nargin > 2 && ~isempty(waypoints)
        plot3(waypoints(1,:), waypoints(2,:), waypoints(3,:), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    end

    L = 0.225; 
    
    braccio1 = [L, 0, 0]'; braccio2 = [0, L, 0]';
    braccio3 = [-L, 0, 0]'; braccio4 = [0, -L, 0]';
    motori_body = [braccio1, braccio2, braccio3, braccio4];
    
    plot_bracci1 = plot3(0,0,0, 'r-o', 'LineWidth', 3, 'MarkerSize', 6, 'MarkerFaceColor', 'r'); 
    plot_bracci2 = plot3(0,0,0, 'k-o', 'LineWidth', 3, 'MarkerSize', 6, 'MarkerFaceColor', 'k');
    
    scia = animatedline('Color', 'b', 'LineStyle', '--', 'LineWidth', 1.5);
    
    pause; 

    for k = 1:5:length(t)
        x_k = X(k, 1); y_k = X(k, 2); z_k = X(k, 3);
        phi_k = X(k, 4); theta_k = X(k, 5); psi_k = X(k, 6);
        
        Rx = [1 0 0; 0 cos(phi_k) sin(phi_k); 0 -sin(phi_k) cos(phi_k)];
        Ry = [cos(theta_k) 0 -sin(theta_k); 0 1 0; sin(theta_k) 0 cos(theta_k)];
        Rz = [cos(psi_k) sin(psi_k) 0; -sin(psi_k) cos(psi_k) 0; 0 0 1];
        R_I = (Rx * Ry * Rz)'; 
        
        motori_inerziali = R_I * motori_body + [x_k; y_k; z_k];
        
        set(plot_bracci1, 'XData', [motori_inerziali(1,3), motori_inerziali(1,1)], ...
                          'YData', [motori_inerziali(2,3), motori_inerziali(2,1)], ...
                          'ZData', [motori_inerziali(3,3), motori_inerziali(3,1)]);
                      
        set(plot_bracci2, 'XData', [motori_inerziali(1,4), motori_inerziali(1,2)], ...
                          'YData', [motori_inerziali(2,4), motori_inerziali(2,2)], ...
                          'ZData', [motori_inerziali(3,4), motori_inerziali(3,2)]);
        
        addpoints(scia, x_k, y_k, z_k);
        
        drawnow; 
    end
end