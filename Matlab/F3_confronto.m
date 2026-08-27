% F3 - CONFRONTO MODELLI

load('MAT/step6_workspace.mat');
load('MAT/step6b_workspace.mat');

X1_nl_allineato = interp1(t1, X1, t1_lin);
errore = X1_nl_allineato - X1_lin;

figure('Name', 'Lineare vs Non Lineare');

subplot(1,2,1);
plot(t1_lin, X1_nl_allineato(:,3), 'b', 'LineWidth', 2); hold on;
plot(t1_lin, X1_lin(:,3), 'r--', 'LineWidth', 2);
xlabel('t [s]'); ylabel('z [m]'); title('Quota Z');
legend('Non-Lineare (Reale)', 'Lineare (Approssimato)', 'Location', 'best');
grid on;

subplot(1,2,2);
plot(t1_lin, errore(:,3), 'b', 'LineWidth', 2);
yline(0, 'k--', 'LineWidth', 1);
xlabel('t [s]'); ylabel('\Deltaz [m]'); title('Errore su Z');
grid on;

sgtitle('Modelli sovrapposti');