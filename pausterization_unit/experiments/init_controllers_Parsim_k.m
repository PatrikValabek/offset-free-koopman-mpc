%% Debug script with fixed syntax

fprintf('=== Checking Paths ===\n');

project_root = '/Users/patrik/Documents/Research/2025-2026/offset-free-koopman-mpc';
src_dir = fullfile(project_root, 'src');
experiments_dir = fullfile(project_root, 'pausterization_unit', 'experiments');

fprintf('Project root: %s\n', project_root);
fprintf('Src dir exists: %d\n', isfolder(src_dir));
fprintf('Experiments dir exists: %d\n', isfolder(experiments_dir));

fprintf('\n=== Python Path ===\n');
path_list = py.list(py.sys.path);  % Convert to list first
for i = 1:min(15, length(path_list))
    path_str = char(path_list{i});
    fprintf('  [%d] %s\n', i, path_str);
    
    if contains(path_str, 'src')
        fprintf('    ^ Contains src\n');
    end
    if contains(path_str, 'experiments')
        fprintf('    ^ Contains experiments\n');
    end
end

fprintf('\n=== Testing Imports ===\n');

% Test helper import
try
    helper = py.importlib.import_module('helper');
    fprintf('✓ helper imported successfully\n');
catch ME
    fprintf('✗ helper import failed: %s\n', ME.message);
    fprintf('  Adding src to path and retrying...\n');
    insert(py.sys.path, int32(0), src_dir);
    helper = py.importlib.import_module('helper');
    fprintf('✓ helper imported after adding path\n');
end

% Test controllers import
try
    controllers = py.importlib.import_module('controllers_Parsim_K');
    fprintf('✓ controllers imported successfully\n');
catch ME
    fprintf('✗ controllers import failed: %s\n', ME.message);
end
controllers.load()
controllers.tests()