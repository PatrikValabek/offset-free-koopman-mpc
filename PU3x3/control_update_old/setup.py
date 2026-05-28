import numpy as np
import joblib
import os


# ---------------------------- Outputs and inputs ------------------------------

ny = 4  # [T1, T2, T4, L]
nu = 3
scaler = joblib.load('../data/scaler.pkl')  # 4-feature scaler for MPC models
scalerU = joblib.load('../data/scalerU.pkl')

u_min_ns = np.array([[30.0, 20.0, 0.0]])
u_max_ns = np.array([[ 100.0, 100.0, 50.0]])

u_min = scalerU.transform(u_min_ns.reshape(1, -1))[0]
u_max = scalerU.transform(u_max_ns.reshape(1, -1))[0]



# ---------------------------- Build references --------------------------------
sim_time = 500
change_interval = 250
c = 10

# References: [T1, T2, T4, L] where L = 10^[(T1-40)/5]
ref_y_matrix_temps = np.array([
    [45, 61, 40.261475],
    [40, 55, 39.35773633],
    [35, 50, 40.261475],
    [50, 55, 51.16012],
    [51, 60, 52.488274],
    [40, 56, 39.35773633],
])
ref_L = 10.0 ** ((ref_y_matrix_temps[:, 0] - 40.0) / 5.0)
ref_y_matrix = np.column_stack([ref_y_matrix_temps, ref_L])

# Input setpoints (non-scaled), one row per segment; same schedule as ref_y_matrix
ref_u_matrix = np.array([
    [75.0, 50.0, 20.0],
    [50.0, 55.0, 22.0],
    [50.0, 48.0, 18.0],
    [50.0, 52.0, 24.0],
    [75.0, 50.0, 20.0],
])

reference_ns = np.zeros((ny, sim_time))
reference_u_ns = np.zeros((nu, sim_time))
for i in range(0, sim_time, change_interval):
    idx = i // change_interval
    y_val = ref_y_matrix[idx]
    reference_ns[:, i:i+change_interval] = y_val.reshape(-1, 1)
    u_val = ref_u_matrix[idx]
    reference_u_ns[:, i:i+change_interval] = u_val.reshape(-1, 1)


reference = scaler.transform(reference_ns.T).T
reference_u = scalerU.transform(reference_u_ns.T).T

# ---------------------------- Initial conditions ------------------------------
# Start from first steady state

T1_start = 40.0
y_start_ns = np.array([[T1_start, 58.0, 40.5, 10.0 ** ((T1_start - 40.0) / 5.0)]])
y_start = scaler.transform(y_start_ns.reshape(1, -1))

u_previous_ns = np.array([[60, 60.0, 20.0]])
u_previous = scalerU.transform(u_previous_ns.reshape(1, -1))[0]


# ---------------------------- Observer/Controller -----------------------------
nd = ny

P0 = 1
Q = 0.1
Qd = 1
R =  100


N = 60
Qy_te = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 5.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 10.0]
])
Qu_te = np.array([
    [0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
])


Qy = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 5.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 10.0]
])
Qu = np.array([
    [0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
])
Qdu = np.array([
    [1, 0.0, 0.0],
    [0.0, 1, 0.0],
    [0.0, 0.0, 1]
])


u_min = scalerU.transform(u_min_ns.reshape(1, -1))[0]
u_max = scalerU.transform(u_max_ns.reshape(1, -1))[0]

# Conservative y bounds based on computed references with margins
y_min_ns = np.array([[0.0, 0.0, 0.0, 0.0]])
y_max_ns = np.array([[100.0, 100.0, 100.0, 1e6]])
y_min = scaler.transform(y_min_ns.reshape(1, -1))[0]
y_max = scaler.transform(y_max_ns.reshape(1, -1))[0]


# ---------------------------- Dump setup --------------------------------------
sim_setup = {
    'y_start': y_start,
    'u_previous': u_previous,
    'y_start_ns': y_start_ns,
    'u_previous_ns': u_previous_ns,
    'P0': P0,
    'Q': Q,
    'Qd': Qd,
    'R': R,
    'N': N,
    'Qy': Qy,
    'Qu': Qu,
    'Qdu': Qdu,
    'Qy_te': Qy_te,
    'Qu_te': Qu_te,
    'u_min': u_min,
    'u_max': u_max,
    'y_min': y_min,
    'y_max': y_max,
    'sim_time': sim_time,
    'reference': reference,
    'reference_ns': reference_ns,
    'reference_u': reference_u,
    'reference_u_ns': reference_u_ns,
    'notes': 'Pausterization unit setup with L = 10^[(T1-40)/5]',
}

out_path = os.path.join(os.path.dirname(__file__), "sim_setup.pkl")
joblib.dump(sim_setup, out_path)
