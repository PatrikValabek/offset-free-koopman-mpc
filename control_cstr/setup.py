import numpy as np
import joblib
import os

# This setup targets the CSTR series with recycle model `src/models/cstr_series.py`.
# We generate achievable steady-state references by simulating the ODEs to steady
# for several constant input sets, then assemble a piecewise-constant reference
# trajectory in the non-scaled domain. We keep identity scaling (ns == scaled).


# ---------------------------- CSTR model params -------------------------------
# These match defaults in `CSTRSeriesRecycle` for consistency
C_A_O = 97.35   # mol/m^3
T_O = 298.0     # K
C_B_O = 0.0     # mol/m^3
C_U_O = 0.0     # mol/m^3
V1 = 1e-3       # m^3
V2 = 2e-3       # m^3
U1A1 = 0.461    # kJ/(s·K)
U2A2 = 0.732    # kJ/(s·K)
rho = 1.05e3    # kg/m^3
cp = 3.766      # kJ/(kg·K)
R_const = 8.3145e-3  # kJ/(mol·K)

# Kinetics
k1_0 = 1.0e5;  E1 = 45.0
k2_0 = 9.8e9;  E2 = 70.0
k3_0 = 5.0e4;  E3 = 55.0
deltaH1 = 60.0; deltaH2 = 40.0; deltaH3 = 60.0


def cstr_ode(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Continuous-time ODEs for the two CSTRs with recycle.

    x: [C_A1, T1, C_A2, T2, C_B1, C_B2, C_U1, C_U2]
    u: [F, L, Tc1, Tc2]
    """
    C_A1, T1, C_A2, T2, C_B1, C_B2, C_U1, C_U2 = x
    F, L, Tc1, Tc2 = u

    # Guard against non-physical states
    C_A1 = max(C_A1, 0.0); C_A2 = max(C_A2, 0.0)
    C_B1 = max(C_B1, 0.0); C_B2 = max(C_B2, 0.0)
    C_U1 = max(C_U1, 0.0); C_U2 = max(C_U2, 0.0)
    T1 = max(T1, 1.0); T2 = max(T2, 1.0)

    # Arrhenius terms
    k1_T1 = k1_0 * np.exp(-E1 / (R_const * T1))
    k1_T2 = k1_0 * np.exp(-E1 / (R_const * T2))
    k2_T1 = k2_0 * np.exp(-E2 / (R_const * T1))
    k2_T2 = k2_0 * np.exp(-E2 / (R_const * T2))
    k3_T1 = k3_0 * np.exp(-E3 / (R_const * T1))

    # Reaction rates
    r1_1 = k1_T1 * (C_A1 ** 2)
    r2_1 = k2_T1 * C_A1
    r3_1 = k3_T1 * C_A1 * C_B1
    r1_2 = k1_T2 * (C_A2 ** 2)
    r2_2 = k2_T2 * C_A2

    # Reactor 1 balances
    dCA1dt = (C_A_O / V1) * F + (L / V1) * C_A2 - ((F + L) / V1) * C_A1 - 2.0 * r1_1 - r2_1 - r3_1
    dCB1dt = (C_B_O / V1) * F + (L / V1) * C_B2 - ((F + L) / V1) * C_B1 + r1_1 + r3_1
    dCU1dt = (C_U_O / V1) * F + (L / V1) * C_U2 - ((F + L) / V1) * C_U1 + r2_1
    dT1dt = (T_O / V1) * F + (L / V1) * T2 \
            - (U1A1 / (V1 * rho * cp)) * (T1 - Tc1) \
            - ((F + L) / V1) * T1 \
            + ((-deltaH1) * r1_1 + (-deltaH2) * r2_1 + (-deltaH3) * r3_1) / (rho * cp)

    # Reactor 2 balances
    dCA2dt = ((F + L) / V2) * (C_A1 - C_A2) - 2.0 * r1_2 - r2_2
    dCB2dt = ((F + L) / V2) * (C_B1 - C_B2) + r1_2
    dCU2dt = ((F + L) / V2) * (C_U1 - C_U2) + r2_2
    dT2dt = ((F + L) / V2) * (T1 - T2) \
            - (U2A2 / (V2 * rho * cp)) * (T2 - Tc2) \
            + ((-deltaH1) * r1_2 + (-deltaH2) * r2_2) / (rho * cp)

    return np.array([dCA1dt, dT1dt, dCA2dt, dT2dt, dCB1dt, dCB2dt, dCU1dt, dCU2dt])


def simulate_to_steady(u: np.ndarray, x0: np.ndarray = None, dt: float = 0.1, max_time: float = 5000.0,
                       tol: float = 1e-8) -> np.ndarray:
    """Forward simulate with constant inputs until steady-state (||dx|| small)."""
    if x0 is None:
        # start from feed conditions/no products
        x = np.array([C_A_O, T_O, C_A_O, T_O, 0.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        x = x0.astype(float).copy()

    n_steps = int(max_time / dt)
    for _ in range(n_steps):
        dx = cstr_ode(x, u)
        if np.linalg.norm(dx, ord=2) < tol:
            break
        # Semi-implicit Euler to improve stability a bit
        x = x + dt * dx
        # keep non-negative concentrations
        x[0] = max(x[0], 0.0)
        x[2] = max(x[2], 0.0)
        x[4] = max(x[4], 0.0)
        x[5] = max(x[5], 0.0)
        x[6] = max(x[6], 0.0)
        x[7] = max(x[7], 0.0)
    return x


# ---------------------------- Outputs and inputs ------------------------------
# We control 4 inputs: [F, L, Tc1, Tc2]
# States (nx=8): [C_A1, T1, C_A2, T2, C_B1, C_B2, C_U1, C_U2]
nx = 22
ny = 8  
nu = 4

# Reasonable bounds
F_min, F_max = 0.0e-5, 2.0e-4   # m^3/s
L_min, L_max = 0.0e-5, 2.0e-4   # m^3/s
Tc_min, Tc_max = 280.0, 330.0   # K

u_min_ns = np.array([F_min, L_min, Tc_min, Tc_min])
u_max_ns = np.array([F_max, L_max, Tc_max, Tc_max])


def outputs_from_state(x: np.ndarray) -> np.ndarray:
    # All states are outputs
    return x.copy()


# ---------------------------- Build references --------------------------------
sim_time = 500
change_interval = 100

# Choose a sequence of constant inputs within bounds; each will yield a steady state
u_targets = [
    np.array([1.5e-4, 7.5e-5, 290.0, 290.0]),
    np.array([1.8e-4, 1.0e-4, 295.0, 292.0]),
    np.array([2.2e-4, 1.2e-4, 300.0, 295.0]),
    np.array([2.6e-4, 1.5e-4, 305.0, 300.0]),
    np.array([2.0e-4, 1.0e-4, 298.0, 298.0]),
]

# Clip to bounds just in case
u_targets = [np.minimum(np.maximum(u, u_min_ns), u_max_ns) for u in u_targets]

# Compute steady states and corresponding outputs (achievable references)
steady_states = []
references_y_list = []
references_x_list = []
last_x = None
for u in u_targets:
    x_ss = simulate_to_steady(u, x0=last_x)
    y_ss = outputs_from_state(x_ss)
    steady_states.append((x_ss, u))
    references_y_list.append(y_ss)
    references_x_list.append(x_ss)
    last_x = x_ss

ref_y_matrix = np.vstack(references_y_list)  # shape (len(u_targets), ny==8)
ref_x_matrix = np.vstack(references_x_list)  # shape (len(u_targets), 8)

# Build piecewise-constant references for outputs and full state
reference_ns = np.zeros((ny, sim_time))
for i in range(0, sim_time, change_interval):
    idx = i // change_interval
    y_val = ref_y_matrix[idx]
    reference_ns[:, i:i+change_interval] = y_val.reshape(-1, 1)

# Identity scaling for this scenario (ns == scaled)
class _IdentityScaler:
    def transform(self, X):
        X = np.asarray(X)
        return X

scaler = joblib.load('../data/scaler.pkl')
scalerU = joblib.load('../data/scalerU.pkl')

reference = scaler.transform(reference_ns.T).T


# ---------------------------- Initial conditions ------------------------------
# Start from first steady state
x0_ss, u0_ss = steady_states[0]
y0_ss = outputs_from_state(x0_ss)

y_start_ns = y0_ss.copy()
y_start = scaler.transform(y_start_ns.reshape(1, -1))

u_previous_ns = u0_ss.copy()
u_previous = scalerU.transform(u_previous_ns.reshape(1, -1))[0]


# ---------------------------- Observer/Controller -----------------------------
# Disturbance dimension: track outputs as measurable, no explicit disturbance states here
nd = ny

P0 = np.eye(nx + nd)
Q = np.eye(nx + nd) * 0.1
# Q = np.block([
#     [np.eye(nx) * 0.1,  np.zeros((nx, nd))],   # Trust state model
#     [np.zeros((nd, nx)), np.eye(nd) * 1.0]      # Disturbance adapts fast
# ])
R = np.eye(ny) * 0.5

N = 20
Qy = np.eye(ny) * 5.0
Qu = np.eye(nu) * 0.1

u_min = scalerU.transform(u_min_ns.reshape(1, -1))[0]
u_max = scalerU.transform(u_max_ns.reshape(1, -1))[0]

# Conservative y bounds based on computed references with margins
y_vals = np.vstack([reference_ns.T])
y_mean = y_vals.mean(axis=0)
y_range = np.maximum(1e-6, np.ptp(y_vals, axis=0))
margin = 10
y_min_ns = y_mean - 2.0 * margin
y_max_ns = y_mean + 2.0 * margin
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
    'notes': 'CSTRSeriesRecycle setup with achievable steady-state references (y=[C_B2, T2])',
}

out_path = os.path.join(os.path.dirname(__file__), "sim_setup.pkl")
joblib.dump(sim_setup, out_path)