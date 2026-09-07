clear; %close all;
%%
experiment = 16;
load("T2D2_u_"+string(experiment))
load("T2D2_x_"+string(experiment))
load("T2D2_y_sp_"+string(experiment))

%%
% Column indices: if logs have 3 outputs, omit the first; if already 2, use both.
ny = size(y, 2);
if ny >= 3
    iy = 2:3;
else
    iy = 1:2;
end

figure;

% PU2x2: two retained outputs
for k = 1:2
    subplot(2,1,k);

    plot(y(1:end, iy(k)), 'LineWidth', 1.5); hold on;
    plot(y_sp(1:end, iy(k)), '--', 'LineWidth', 1.5);

    ylabel(['y ' num2str(iy(k))]);
    grid on;
    legend('y', 'y_{sp}', 'Location', 'best');
end

xlabel('Time step');

nu = size(u, 2);
if nu >= 3
    iu = 2:3;
else
    iu = 1:2;
end

figure;

% PU2x2: two retained inputs
for k = 1:2
    subplot(2,1,k);

    plot(u(1:end, iu(k)), 'LineWidth', 1.5); hold on;

    ylabel(['u ' num2str(iu(k))]);
    grid on;
    legend('u', 'Location', 'best');
end

xlabel('Time step');