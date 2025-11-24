%% Open ELab in MANAGER mode
%  In this mode, you can LIST and INSTALL devices
%
%  Example: 
%           elab_manager = ELab();  % Creating elab instance by calling 
%                                   % ELab class without parameters, 
%                                   % automatically triggers the MANAGER mode.
%           elab_manager.list();    % Displays list of devices available in
%                                   % elab master database.
%           
%           elab_manager.install('pct23'); % Installs library files for
%                                           % the device 'pct23'.
%

elab_manager = ELab();
elab_manager.list();

%% Open ELab in CONTROL mode
%  In this mode, you have full control over selected device
%
%  Example: 
%           elab_manager = ELab(DEVICE_NAME, MODE, ADDRESS, LOGGING, LOGGING_PERIOD, INTERNAL_SAMPLING_PERIOD, POLLING_PERIOD); 
%
%           where DEVICE_NAME (String) is a designated name of the device (e.g. 'pct23'),
%                 MODE (String) is mode switch with possible values 'MANAGER', 'CONTROL', 'MONITOR',
%                 ADDRESS (String) is HTTP address of elab master SCADA system,
%                 LOGGING (0 or 1) is switch for online data logging into elab master database,
%                 LOGGING_PERIOD (N seconds) defines how often the measured data is logged into database,
%                 INTERNAL_SAMPLING_PERIOD (N seconds) defines how often the device streams new data to the elab SCADA master,
%                 POLLING_PERIOD (N seconds) defines how often the ELab class refreshes the data from SCADA master (this should be set to Ts)
%

Ts = 1;%
device_name = 'pct23';
mode = 'control';
address = 'http://192.168.1.108:3030';%
logging = 0;%
logging_period = Ts;
internal_sampling_period = Ts;
polling_period = Ts;

% create instance of udaq28 device
pct23 = ELab(device_name, mode, address, logging, logging_period, internal_sampling_period, polling_period);

%% Using the device (measure/control)

% get all measured data at once
tags = pct23.getAllTags();
%%
% get specific tag
% temperature_1 = pct23.getTag('T1');
%pct23.off()
pct23.setTag('FSV',1);
% get value of specific tag
% flowrate_value = pct23.getTagValue('F1');

% set value of specific tag
% pct23.setTag('Pump1',100)

% set values of multiple tags at the same time
% pct23.setTags({'Pump1', 50, 'DV', 1})

% reset all control signals to default values
% pct23.off()

% set close the device
% pct23.close()
N = 2500;

y = zeros(N,3);
y_sp = zeros(N,3);
u_sp = zeros(N,3);
u = zeros(N,3);
u_prev = [50,50,25];

pct23.setTag('Pump1',50);
pct23.setTag('Pump2',50);
pct23.setTag('Heater',20);

double(pct23.getTag('T4').value)
pause(1)
double(pct23.getTag('T2').value)
pause(1)
double(pct23.getTag('T1').value)

%% one_loop_simulation

for i = 1:N
    % Start the timer
    tic;

    % Code to execute within the loop
    disp(['Iteration: ', num2str(i), ' at ', datestr(now)]);

    y3 = double(pct23.getTag('T4').value);
    y2 = double(pct23.getTag('T2').value);
    y1 = double(pct23.getTag('T1').value);

    y(i,:) = [y1 y2 y3];
    y(i,:)
    pct23.setTag('FSV',1);

    [y_sp(i,:), u(i,:), u_sp(i,:)] = CT_controll(u_prev, y(i,:), i);
    y_sp(i,:)
    u_sp(i,:)
    u(i,:)
    u_prev = u(i,:);
    
    pct23.setTag('Pump1',u(i,1)); % feed
    pct23.setTag('Pump2',u(i,2)); % heating media
    pct23.setTag('Heater',u(i,3)); % spiral

    % Wait until 1 second has passed since the start of the iteration
    elapsedTime = toc;
    pause(max(0, 1 - elapsedTime));

end

%%
%terminate(pyenv);
pct23.off();
pct23.setTag('FSV',1);


