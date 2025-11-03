import numpy as np
import joblib

matrix_C = False
# Load matrices A, B, and C 
A = np.load('../data/A_C_' + str(matrix_C) + '.npy')
B = np.load('../data/B_C_' + str(matrix_C) + '.npy')
C = np.load('../data/C_C_' + str(matrix_C) + '.npy')

nz, nu = B.shape  # state and input dimensions
ny = C.shape[0]  # output dimensions

# disturbance 
F = np.eye(ny)
nd = F.shape[1]

scaler = joblib.load('../data/scaler.pkl')
scalerU = joblib.load('../data/scalerU.pkl')

# begining
y_start_ns = np.array([0.5, 0.499999])
y_start = scaler.transform(y_start_ns.reshape(1, -1))

u_previous_ns = np.array([0.07045999, 0.47585951])
u_previous = scalerU.transform(u_previous_ns.reshape(1, -1))[0]

# observer
P0 = np.eye(nz+nd) 
Q = np.eye(nz+nd) * 0.1 
R = np.eye(ny) * 0.5

# controller
N = 20
Qy = np.eye(ny) * 0.1
Qu = np.eye(nu) * 5
u_min = scalerU.transform(np.array([[0.0, 0.0]]))[0]
u_max = scalerU.transform(np.array([[0.5, 1.0]]))[0]
y_min = scaler.transform(np.array([[-5.0, -5.0]]))[0]
y_max = scaler.transform(np.array([[5.0, 5.0]]))[0]

# simulation
sim_time = 500

# reference trajectory

# Parameters
num_signals = ny
num_samples = sim_time
change_interval = 100


# Initialize reference array
reference = np.zeros((num_signals, num_samples))
ref = np.array([[0.5, 0.5],[1.5, 0.8],[1.0, 0.9],[2.0, 1.7],[1.0, 0.9]])
#ref = np.array([[0.5, 0.5],[0.6, 0.4],[0.6, 0.5],[0.4, 0.4],[0.5, 0.5]])

# Generate new value every `change_interval` steps
for i in range(0, num_samples, change_interval):
    new_value = ref[i // change_interval].reshape(-1,1)  
    reference[:, i:i+change_interval] = new_value

# Optional: Clip in case final interval exceeds array length
reference_ns = reference[:, :num_samples]
reference = scaler.transform(reference_ns.T).T

reference[:,100]

sim_setup = {
    'y_start': y_start,
    'u_previous': u_previous,
    'y_start_ns': y_start_ns,
    'u_previous_ns': u_previous_ns,
    'P0': P0,
    'Q': Q,
    'R': R,
    'N': N,
    'Qy': Qy,
    'Qu': Qu,
    'u_min': u_min,
    'u_max': u_max,
    'y_min': y_min,
    'y_max': y_max,
    'sim_time': sim_time,
    'reference': reference,
    'reference_ns': reference_ns,
}

joblib.dump(sim_setup, "sim_setup.pkl")



