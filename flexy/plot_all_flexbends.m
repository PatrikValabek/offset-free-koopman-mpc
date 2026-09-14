function plot_all_flexbends(dataFolder)
%PLOT_ALL_FLEXBENDS Plot all FlexBend identification experiments and inputs.
%
%   plot_all_flexbends
%   plot_all_flexbends(dataFolder)
%
% The function reproduces the raw experiments listed in ident.m:
%   pch1 : 40 -> 80
%   pch2 : 50 -> 80
%   pch3 : 20 -> 40
%   pch4 : 30 -> 10
%   pch5 : 70 -> 30
%   pch6 : 40 -> 90
%   pch7 : 40 -> 80   (Ts = 0.01 s)
%   pch8 : 50 -> 80   (Ts = 0.01 s)
%   pch9 : 35 -> 55   (Ts = 0.01 s)
%
% Data format in each matrix:
%   column 1 = time t
%   column 2 = FlexBend output y
%   column 3 = input u
%
% All experiments are displayed in one 3x3 tiled figure. Each tile uses
% two y-axes so that y and u remain readable even when their scales differ.

    if nargin < 1 || isempty(dataFolder)
        dataFolder = fullfile(fileparts(mfilename('fullpath')), 'data');
    end

    % File, variable, and step labels taken from ident.m
    files = { ...
        'pch1.mat', ...
        'pch2.mat', ...
        'pch3.mat', ...
        'pch4.mat', ...
        'pch5.mat', ...
        'pch6.mat', ...
        'pch7.mat', ...
        'pch8.mat', ...
        'pch9.mat'};

    vars = { ...
        'pch1', ...
        'pch2', ...
        'pch3', ...
        'pch4', ...
        'pch5', ...
        'pch6', ...
        'pch7', ...
        'pch8', ...
        'pch9'};

    stepLabels = { ...
        '40 \rightarrow 80', ...
        '50 \rightarrow 80', ...
        '20 \rightarrow 40', ...
        '30 \rightarrow 10', ...
        '70 \rightarrow 30', ...
        '40 \rightarrow 90', ...
        '40 \rightarrow 80', ...
        '50 \rightarrow 80', ...
        '35 \rightarrow 55'};

    figure('Name','FlexBend identification experiments','Color','w');
    tl = tiledlayout(3,3,'TileSpacing','compact','Padding','compact');
    title(tl,'FlexBend identification data');

    for k = 1:numel(files)
        filePath = fullfile(dataFolder, files{k});

        if ~isfile(filePath)
            error('File not found: %s', filePath);
        end

        S = load(filePath, vars{k});

        if ~isfield(S, vars{k})
            error('Variable "%s" not found in %s.', vars{k}, filePath);
        end

        data = S.(vars{k});

        if size(data,2) < 3
            error('%s must have at least 3 columns [t, y, u].', vars{k});
        end

        t = data(:,1);
        y = data(:,2);
        u = data(:,3);

        ax = nexttile;

        yyaxis(ax,'left')
        plot(ax,t,y,'LineWidth',1.25);
        ylabel(ax,'FlexBend y');

        yyaxis(ax,'right')
        plot(ax,t,u,'--','LineWidth',1.15);
        ylabel(ax,'Input u');

        xlabel(ax,'Time [s]');
        title(ax,sprintf('pch%d: %s',k,stepLabels{k}));
        grid(ax,'on');
        box(ax,'on');
    end
end
