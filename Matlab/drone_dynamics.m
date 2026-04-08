% Funzione che viene richiamata da ode45:
% - Prende stati correnti --> x y z non mi servono !
% - Calcola nuove matrici
% - Ricava derivata seconda
% - Assembla nuovo vettore X e lo deriva (x_dot)
function x_dot = drone_dynamics(t, x, u, M_fun, C_fun, G_n, B_fun)
    phi_v    = x(4);
    theta_v  = x(5);
    psi_v    = x(6);
    dphi_v   = x(10);
    dtheta_v = x(11);
    dpsi_v   = x(12);
    qd_val   = x(7:12);

    M_n = M_fun(phi_v, theta_v, psi_v);
    C_n = C_fun(phi_v, theta_v, psi_v, dphi_v, dtheta_v, dpsi_v);
    B_n = B_fun(phi_v, theta_v, psi_v);

    q_ddot = M_n \ (-C_n*qd_val - G_n + B_n*u);
    x_dot  = [qd_val; q_ddot];
end