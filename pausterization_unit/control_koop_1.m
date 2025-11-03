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
N = 360;
d = 60;
dif = 52-47.17824345;
y = zeros(N,1);
u = zeros(N,1);
u_dist = 20;
pause(1)
u_spiral = zeros(N,1);
P_spiral = 12;

%% Warmup
for i = 1:d
    % Start the timer
    tic;

    % Code to execute within the loop
    disp(['Iteration: ', num2str(i), ' at ', datestr(now)]);
    
    pct23.setTag('Pump1',50); % feed
    y(i) = double(pct23.getTag('T4').value);
    pct23.setTag('FSV',1);
    u(i) = koopman_controll(y(i)-dif);
    u(i)
    pct23.setTag('Pump2',u(i)); %heating
    
    value =  P_spiral*(71 - pct23.getTag('T2').value)
    u_spiral(i) = min(max(value, 0), 100);
    pct23.setTag('Heater',u_spiral(i));
    % Wait until 1 second has passed since the start of the iteration
    elapsedTime = toc;
    pause(max(0, 1 - elapsedTime));

end
%% OK

for i = d:2*d
    % Start the timer
    tic;

    % Code to execute within the loop
    disp(['Iteration: ', num2str(i), ' at ', datestr(now)]);
    
    pct23.setTag('Pump1',50); % feed
    y(i) = double(pct23.getTag('T4').value);
    pct23.setTag('FSV',1);
    u(i) = koopman_controll(y(i)-dif);
    u(i)
    pct23.setTag('Pump2',u(i)); %heating
    
    value =  P_spiral*(71 - pct23.getTag('T2').value)
    u_spiral(i) = min(max(value, 0), 100);
    pct23.setTag('Heater',u_spiral(i));
    % Wait until 1 second has passed since the start of the iteration
    elapsedTime = toc;
    pause(max(0, 1 - elapsedTime));
end
%% fail 1
for i = 2*d:3*d
    % Start the timer
    tic;

    % Code to execute within the loop
    disp(['Iteration: ', num2str(i), ' at ', datestr(now)]);
    
    pct23.setTag('Pump1',20); % feed
    y(i) = double(pct23.getTag('T4').value);
    pct23.setTag('FSV',1);
    u(i) = koopman_controll(y(i)-dif);
    u(i)
    pct23.setTag('Pump2',u(i)); %heating
    
    value =  P_spiral*(71 - pct23.getTag('T2').value)
    u_spiral(i) = min(max(value, 0), 100);
    pct23.setTag('Heater',u_spiral(i));
    % Wait until 1 second has passed since the start of the iteration
    elapsedTime = toc;
    pause(max(0, 1 - elapsedTime));
end
%% OK
for i = 3*d:4*d
    % Start the timer
    tic;

    % Code to execute within the loop
    disp(['Iteration: ', num2str(i), ' at ', datestr(now)]);
    
    pct23.setTag('Pump1',50); % feed
    y(i) = double(pct23.getTag('T4').value);
    pct23.setTag('FSV',1);
    u(i) = koopman_controll(y(i)-dif);
    u(i)
    pct23.setTag('Pump2',u(i)); %heating
    
    value =  P_spiral*(71 - pct23.getTag('T2').value)
    u_spiral(i) = min(max(value, 0), 100);
    pct23.setTag('Heater',u_spiral(i));
    % Wait until 1 second has passed since the start of the iteration
    elapsedTime = toc;
    pause(max(0, 1 - elapsedTime));
end
%% fail 2
for i = 4*d:5*d
    % Start the timer
    tic;

    % Code to execute within the loop
    disp(['Iteration: ', num2str(i), ' at ', datestr(now)]);
    
    pct23.setTag('Pump1',100); % feed
    y(i) = double(pct23.getTag('T4').value);
    pct23.setTag('FSV',1);
    u(i) = koopman_controll(y(i)-dif);
    u(i)
    pct23.setTag('Pump2',u(i)); %heating
    
    value =  P_spiral*(71 - pct23.getTag('T2').value)
    u_spiral(i) = min(max(value, 0), 100);
    pct23.setTag('Heater',u_spiral(i));
    % Wait until 1 second has passed since the start of the iteration
    elapsedTime = toc;
    pause(max(0, 1 - elapsedTime));
end
%% OK
for i = 5*d:6*d
    % Start the timer
    tic;

    % Code to execute within the loop
    disp(['Iteration: ', num2str(i), ' at ', datestr(now)]);
    
    pct23.setTag('Pump1',50); % feed
    y(i) = double(pct23.getTag('T4').value);
    pct23.setTag('FSV',1);
    u(i) = koopman_controll(y(i)-dif);
    u(i)
    pct23.setTag('Pump2',u(i)); %heating
    
    value =  P_spiral*(71 - pct23.getTag('T2').value)
    u_spiral(i) = min(max(value, 0), 100);
    pct23.setTag('Heater',u_spiral(i));
    % Wait until 1 second has passed since the start of the iteration
    elapsedTime = toc;
    pause(max(0, 1 - elapsedTime));
end

terminate(pyenv);
pct23.off();
pct23.setTag('FSV',1);
