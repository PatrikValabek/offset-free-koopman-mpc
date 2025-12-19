clear; %close all;
%%
experiment = 2;
load("CT_u_"+string(experiment))
load("CT_x_"+string(experiment))
load("CT_y_sp_"+string(experiment))

%%
figure;

for k = 1:3
    subplot(3,1,k);  % three rows, one column, plot k

    plot(y(1:1000,k), 'LineWidth', 1.5); hold on;
    plot(y_sp(1:1000,k), '--', 'LineWidth', 1.5);

    ylabel(['Variable ' num2str(k)]);
    grid on;
    legend('y', 'y_{sp}', 'Location', 'best');
end

xlabel('Time step');

figure;

for k = 1:3
    subplot(3,1,k);  % three rows, one column, plot k

    plot(u(1:1000,k), 'LineWidth', 1.5); hold on;

    ylabel(['Variable ' num2str(k)]);
    grid on;
    legend('u', 'Location', 'best');
end

xlabel('Time step');