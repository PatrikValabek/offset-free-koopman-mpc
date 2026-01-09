import numpy as np
import joblib
import os


# ---------------------------- Outputs and inputs ------------------------------

ny = 3 
nu = 3
scaler = joblib.load('../data/scaler.pkl')
scalerU = joblib.load('../data/scalerU.pkl')

u_min_ns = np.array([[30.0, 20.0, 0.0]])
u_max_ns = np.array([[100.0, 100.0, 50.0]])

u_min = scalerU.transform(u_min_ns.reshape(1, -1))[0]
u_max = scalerU.transform(u_max_ns.reshape(1, -1))[0]



# ---------------------------- Build references --------------------------------
sim_time = 500
change_interval = 500

ref_y_matrix = np.array([[54.66027, 73.49036, 54.36351]])

reference_ns = np.zeros((ny, sim_time))
for i in range(0, sim_time, change_interval):
    idx = i // change_interval
    y_val = ref_y_matrix[idx]
    reference_ns[:, i:i+change_interval] = y_val.reshape(-1, 1)


reference = scaler.transform(reference_ns.T).T


# ---------------------------- Initial conditions ------------------------------
# Start from first steady state

y_start_ns = np.array([[60.0, 70.0, 62.0]])
y_start = scaler.transform(y_start_ns.reshape(1, -1))

u_previous_ns = np.array([[70.0, 90.0, 30.0]])
u_previous = scalerU.transform(u_previous_ns.reshape(1, -1))[0]


# ---------------------------- Observer/Controller -----------------------------
# Disturbance dimension: track outputs as measurable, no explicit disturbance states here
nd = ny

P0 = 1
Q = 0.5
# Q = np.block([
#     [np.eye(nx) * 0.1,  np.zeros((nx, nd))],   # Trust state model
#     [np.zeros((nd, nx)), np.eye(nd) * 1.0]      # Disturbance adapts fast
# ])
R =  0.1

N = 60
Qy = np.eye(ny) * 5.0
Qu = np.eye(nu) * 5.0

u_min = scalerU.transform(u_min_ns.reshape(1, -1))[0]
u_max = scalerU.transform(u_max_ns.reshape(1, -1))[0]

# Conservative y bounds based on computed references with margins

y_min_ns = np.array([[0.0, 0.0, 0.0]])
y_max_ns = np.array([[100.0, 100.0, 100.0]])
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
    'notes': 'Pausterization unit setup',
}

out_path = os.path.join(os.path.dirname(__file__), "sim_setup.pkl")
joblib.dump(sim_setup, out_path)