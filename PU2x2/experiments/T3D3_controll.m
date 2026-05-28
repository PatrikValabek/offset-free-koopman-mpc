function [y_s, u_opt, u_s] = T3D3_controll(u_prev, y, step)
    % Use py.controllers instead of just controllers
    py_result = py.controllers_T3D3.next_optimal_input(u_prev, y, step-1);
    
    % Extract both values
    y_s = double(py_result{1});
    u_opt = double(py_result{2});
    u_s = double(py_result{3});
    % Ensure column vectors
    y_s = y_s(:);
    u_opt = u_opt(:);
    u_s = u_s(:);
end