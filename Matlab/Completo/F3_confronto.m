% F3 - CONFRONTO MODELLI

% Carico i dati dal test non lineare (ode45) e lineare (lsim)

load('MAT/step6_workspace.mat');    % Non Lineare

load('MAT/step6b_workspace.mat');   % Lineare

% Stessi tempi per non lineare e lineare
X1_nl_allineato = interp1(t1, X1, t1_lin);

% Calcolo della Differenza 
errore = X1_nl_allineato - X1_lin;

% Confronto grafico 1 - Test 1 (roll iniziale)
figure('Name', 'Lineare vs Non Lineare');

% Variazione di Z, nel lineare retta --> rimane costante
subplot(1,2,1);
plot(t1_lin, X1_nl_allineato(:,3), 'b', 'LineWidth', 2); hold on;
plot(t1_lin, X1_lin(:,3), 'r--', 'LineWidth', 2);
xlabel('t [s]'); ylabel('z [m]'); title('Quota Z');
legend('Non-Lineare (Reale)', 'Lineare (Approssimato)', 'Location', 'best');
grid on;

% Variazione del roll
subplot(1,2,2);
plot(t1_lin, rad2deg(X1_nl_allineato(:,4)), 'b', 'LineWidth', 2); hold on;
plot(t1_lin, rad2deg(X1_lin(:,4)), 'r--', 'LineWidth', 2);
xlabel('t [s]'); ylabel('\phi [°]'); title('Rollio \phi');
legend('Non-Lineare', 'Lineare', 'Location', 'best');
grid on;

sgtitle('Modelli sovrapposti');


% Grafico 2 - Test 1
figure('Name', 'Errore di Linearizzazione');

% Errore su Y, mi aspetto una differenza minima
subplot(1,2,1);
plot(t1_lin, errore(:,2), 'b', 'LineWidth', 1.5);
xlabel('t [s]'); ylabel('Errore Y [m]'); title('Differenza su Asse Y');
grid on;

% Errore su Z , mi aspetto una differenza elevata 1 - Z (NL)
subplot(1,2,2);
plot(t1_lin, errore(:,3), 'b', 'LineWidth', 1.5);
xlabel('t [s]'); ylabel('Errore Z [m]'); title('Differenza su Asse Z');
grid on;

sgtitle('Errore Assoluto (Non-Lineare meno Lineare)');


% controllore modello lineare lqr
% poi con non lineare 

% prima per hovering poi per tracking 

% trasformare T e momenti in forze con matrice gamma e poi calcolo sistema
% con u forze.