function elab_pct23(block)
%MSFUNTMPL_BASIC A template for a Leve-2 M-file S-function
%   The M-file S-function is written as a MATLAB function with the
%   same name as the S-function. Replace 'msfuntmpl_basic' with the 
%   name of your S-function.
%
%   It should be noted that the M-file S-function is very similar
%   to Level-2 C-Mex S-functions. You should be able to get more
%   information for each of the block methods by referring to the
%   documentation for C-Mex S-functions.
%
%   Copyright 2003-2009 The MathWorks, Inc.



%%
%% The setup method is used to setup the basic attributes of the
%% S-function such as ports, parameters, etc. Do not add any other
%% calls to the main body of the function.
%%
setup(block);

%endfunction

%% Function: setup ===================================================
%% Abstract:
%%   Set up the S-function block's basic characteristics such as:
%%   - Input ports
%%   - Output ports
%%   - Dialog parameters
%%   - Options
%%
%%   Required         : Yes
%%   C-Mex counterpart: mdlInitializeSizes
%%

function setup(block)
% Register number of ports
block.NumInputPorts  = 8;
block.NumOutputPorts = 16;

% Setup port properties to be inherited or dynamic
block.SetPreCompInpPortInfoToDynamic;
block.SetPreCompOutPortInfoToDynamic;

% % Override input port properties
% block.InputPort(1).Dimensions        = 1;
% block.InputPort(1).DatatypeID  = 0;  % double
% block.InputPort(1).Complexity  = 'Real';
% block.InputPort(1).DirectFeedthrough = true;
% 
% % Override output port properties
% block.OutputPort(1).Dimensions       = 1;
% block.OutputPort(1).DatatypeID  = 0; % double
% block.OutputPort(1).Complexity  = 'Real';

% Register parameters
block.NumDialogPrms     = 3;
block.DialogPrmsTunable = {'Nontunable','Nontunable','Nontunable'}; % Ts, address, logging
                          

% 1 - sampling time
% 2 - connection type
% 3 - com port


% Register sample times
%  [0 offset]            : Continuous sample time
%  [positive_num offset] : Discrete sample time
%
%  [-1, 0]               : Inherited sample time
%  [-2, 0]               : Variable sample time
block.SampleTimes = [block.DialogPrm(1).Data 0];

% Specify the block simStateCompliance. The allowed values are:
%    'UnknownSimState', < The default setting; warn and assume DefaultSimState
%    'DefaultSimState', < Same sim state as a built-in block
%    'HasNoSimState',   < No sim state
%    'CustomSimState',  < Has GetSimState and SetSimState methods
%    'DisallowSimState' < Error out when saving or restoring the model sim state
block.SimStateCompliance = 'DefaultSimState';

%% -----------------------------------------------------------------
%% The M-file S-function uses an internal registry for all
%% block methods. You should register all relevant methods
%% (optional and required) as illustrated below. You may choose
%% any suitable name for the methods and implement these methods
%% as local functions within the same file. See comments
%% provided for each function for more information.
%% -----------------------------------------------------------------

block.RegBlockMethod('SetInputPortSamplingMode',@SetInputPortSamplingMode);
block.RegBlockMethod('PostPropagationSetup',    @DoPostPropSetup);
block.RegBlockMethod('InitializeConditions', @InitializeConditions);
block.RegBlockMethod('Start', @Start);
block.RegBlockMethod('Outputs', @Outputs);
block.RegBlockMethod('Update', @Update);
block.RegBlockMethod('Derivatives', @Derivatives);
block.RegBlockMethod('Terminate', @Terminate);

%end setup

function SetInputPortSamplingMode(block, idx, fd)
  block.InputPort(idx).SamplingMode = fd;
  for i = 1:block.NumOutputPorts
    block.OutputPort(i).SamplingMode = fd;
  end

%%
%% PostPropagationSetup:
%%   Functionality    : Setup work areas and state variables. Can
%%                      also register run-time methods here
%%   Required         : No
%%   C-Mex counterpart: mdlSetWorkWidths
%%
function DoPostPropSetup(block)
    if block.SampleTimes(1) == 0
        throw(MSLException(block.BlockHandle,'Dicrete sampling time required'));
    end


%%
%% InitializeConditions:
%%   Functionality    : Called at the start of simulation and if it is 
%%                      present in an enabled subsystem configured to reset 
%%                      states, it will be called when the enabled subsystem
%%                      restarts execution to reset the states.
%%   Required         : No
%%   C-MEX counterpart: mdlInitializeConditions
%%
function InitializeConditions(block)
    for i = 1:block.NumOutputPorts
        block.OutputPort(i).Data = 0;
    end
%end InitializeConditions


%%
%% Start:
%%   Functionality    : Called once at start of model execution. If you
%%                      have states that should be initialized once, this 
%%                      is the place to do it.
%%   Required         : No
%%   C-MEX counterpart: mdlStart
%%
function Start(block)
global elab_instance;
%try
    elab_instance = ELab('pct23', ...                   % target name
                         'control', ...                 % mode
                         block.DialogPrm(2).Data, ...   % address
                         block.DialogPrm(3).Data, ...   % logging
                         block.DialogPrm(1).Data, ...   % logging period Ts_log
                         block.DialogPrm(1).Data, ...   % internal sampling period Ts_int
                         block.DialogPrm(1).Data ...    % polling period Ts_poll
                         );
    pause(block.DialogPrm(1).Data+1);
% catch e
%     throw(MSLException(block.BlockHandle, ...
%             'Simulink:ELabError', ...
%             'Could not create an ELab instance.'));
% end
%endfunction

%%
%% Outputs:
%%   Functionality    : Called to generate block outputs in
%%                      simulation step
%%   Required         : Yes
%%   C-MEX counterpart: mdlOutputs
%%
function Outputs(block)
global elab_instance;
elab_instance.setTags({ ...
                       'Pump1',  block.InputPort(1).Data, ...
                       'Pump2',  block.InputPort(2).Data, ...
                       'Heater', block.InputPort(3).Data, ...
                       'DV',     block.InputPort(4).Data, ...
                       'FSV',    block.InputPort(5).Data, ...
                       'CWV',    block.InputPort(6).Data, ...
                       'TAFV',   block.InputPort(7).Data, ...
                       'TBFV',   block.InputPort(8).Data ...
                       });
tags = elab_instance.getAllTags();
block.OutputPort(1).Data = tags.L1.value;
block.OutputPort(2).Data = tags.F1.value;
block.OutputPort(3).Data = tags.PWR.value;
block.OutputPort(4).Data = tags.T1.value;
block.OutputPort(5).Data = tags.T2.value;
block.OutputPort(6).Data = tags.T3.value;
block.OutputPort(7).Data = tags.T4.value;
block.OutputPort(8).Data = tags.C1.value;
block.OutputPort(9).Data = tags.DVP.value;
block.OutputPort(10).Data = tags.HWOT.value;
block.OutputPort(11).Data = tags.TBLL.value;
block.OutputPort(12).Data = tags.TBHL.value;
block.OutputPort(13).Data = tags.FPSD.value;
block.OutputPort(14).Data = tags.WPSD.value;
block.OutputPort(15).Data = tags.WHSD.value;
block.OutputPort(16).Data = tags.VCSD.value;

%end Outputs

%%
%% Update:
%%   Functionality    : Called to update discrete states
%%                      during simulation step
%%   Required         : No
%%   C-MEX counterpart: mdlUpdate
%%
function Update(block)

%end Update

%%
%% Derivatives:
%%   Functionality    : Called to update derivatives of
%%                      continuous states during simulation step
%%   Required         : No
%%   C-MEX counterpart: mdlDerivatives
%%
function Derivatives(block)

%end Derivatives

%%
%% Terminate:
%%   Functionality    : Called at the end of simulation for cleanup
%%   Required         : Yes
%%   C-MEX counterpart: mdlTerminate
%%
function Terminate(block)
    global elab_instance;
    elab_instance.close();
    elab_instance.delete();
%end Terminate

