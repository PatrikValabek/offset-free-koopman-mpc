"""Simulation setup for N4SID offset-free MPC of the CSTR–separator.

Industrial scenario (grade campaign + unmeasured feed upsets)
-----------------------------------------------------------
CVs (same as identification): T1, T2, T3, xB3.
MVs: Q1, Q2, Q3, F10, F20, Fr.

The campaign is a bottoms-quality (xB3) grade change around the Li & Swartz
nominal point, with temperatures held as operating constraints. Two plant
disturbances that the controller does not measure are then applied:

  1. Cold fresh feed: modest T10 drop (preheat / utility drift).
  2. Slightly leaner fresh feed: modest xA10 drop (tank switch / impurity).

These magnitudes are chosen so both grade setpoints remain reachable inside
the identification input box (Q ≤ 25 kJ/s). A 10 K T10 cut at premium F10
needs ~ρ Cp F10 ΔT ≈ 40 kJ/s of extra heat, which saturates the heaters
and makes the NMPC benchmark infeasible.

References are plant steady states (achievable at the nominal parameters).
The disturbances stay on the plant only; the observer / targets / MPC are
unchanged and must reject them through the output-disturbance estimate.

Run this file from this directory so ``sim_setup.pkl`` is written here
(the helper QP classes load it from the process cwd).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
REPO_ROOT = HERE.parent.parent
SRC = REPO_ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.append(SRC.as_posix())

import models  # noqa: E402

# ---------------------------- Plant / scalers ---------------------------------
plant = models.CSTRSeparator()
y_names = list(plant.y_names)  # T1, T2, T3, xB3
u_names = list(plant.u_names)  # Q1, Q2, Q3, F10, F20, Fr
ny, nu = len(y_names), len(u_names)

scaler = joblib.load((DATA_DIR / "scaler_cstr_separator.pkl").as_posix())
scalerU = joblib.load((DATA_DIR / "scalerU_cstr_separator.pkl").as_posix())

u_min_ns = plant.mv_constraints[:, 0].copy()
u_max_ns = plant.mv_constraints[:, 1].copy()
u_min = scalerU.transform(u_min_ns.reshape(1, -1))[0]
u_max = scalerU.transform(u_max_ns.reshape(1, -1))[0]

# Operating window around the identification data (not hard equipment limits).
y_min_ns = np.array([310.0, 310.0, 310.0, 0.35])
y_max_ns = np.array([380.0, 380.0, 380.0, 0.85])
y_min = scaler.transform(y_min_ns.reshape(1, -1))[0]
y_max = scaler.transform(y_max_ns.reshape(1, -1))[0]


def simulate_to_ss(u: np.ndarray, x0: np.ndarray, n_steps: int = 400, Ts: float = 1.0):
    x = np.asarray(x0, dtype=float).copy()
    u_row = np.asarray(u, dtype=float).reshape(1, -1)
    for _ in range(n_steps):
        x = plant.step(x, u_row, Ts)
    return x, plant.measure(x)


# ---------------------------- Achievable grades -------------------------------
# Nominal Li & Swartz point, then a milder-conversion "premium B" grade
# (higher F10 / lower heat → higher xB3; the yield peak is not at maximum heat).
u_nom = plant.u_nom.copy()
u_premium = np.array([8.0, 8.0, 8.0, 11.0, 0.8, 3.0], dtype=float)

x_nom, y_nom = simulate_to_ss(u_nom, plant.x0_guess)
x_premium, y_premium = simulate_to_ss(u_premium, x_nom)

# ---------------------------- Time grid / references --------------------------
Ts = 1.0
sim_time = 1500
# Piecewise output setpoints (physical units). Disturbances do not change refs.
#   [0, 350):    settle / hold nominal grade
#   [350, 1000): premium xB3 grade (disturbances hit during this hold)
#   [1000, end): return to nominal grade
reference_ns = np.zeros((ny, sim_time))
reference_ns[:, :350] = y_nom.reshape(-1, 1)
reference_ns[:, 350:1000] = y_premium.reshape(-1, 1)
reference_ns[:, 1000:] = y_nom.reshape(-1, 1)
reference = scaler.transform(reference_ns.T).T

# Preferred economic input (nominal utilities / throughput), constant.
reference_u_ns = u_nom.copy()
reference_u = scalerU.transform(reference_u_ns.reshape(1, -1))[0]

# Unmeasured plant disturbances (applied in the closed-loop notebook).
# Times are sample indices (Ts = 1 s).
disturbances = [
    {"k": 600, "attr": "T10", "value": 310.0, "label": "cold feed T10: 313 -> 310 K"},
    {"k": 850, "attr": "xA10", "value": 0.97, "label": "lean feed xA10: 1.00 -> 0.97"},
]

# ---------------------------- Initial conditions ------------------------------
y_start_ns = y_nom.copy()
y_start = scaler.transform(y_start_ns.reshape(1, -1))
u_previous_ns = u_nom.copy()
u_previous = scalerU.transform(u_previous_ns.reshape(1, -1))[0]
x_start = x_nom.copy()

# ---------------------------- Observer / controller ---------------------------
nd = ny
P0 = 1.0
Q = 0.1
Qd = 0.1
R = 0.3

N = 60
# Quality (xB3) is the primary CV; temperatures are regulated but softer.
Qy_te = np.diag([1.0, 1.0, 1.0, 20.0])
Qu_te = np.diag([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]) * 0

Qy = np.diag([2.0, 2.0, 2.0, 15.0])
Qu = np.diag([0.2, 0.2, 0.2, 0.5, 0.5, 0.5])
Qdu = np.diag([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])

ident = np.load((DATA_DIR / "cstr_separator_ident.npz").as_posix(), allow_pickle=True)
noise_sigma = (
    np.array(ident["noise_sigma"], dtype=float)
    if "noise_sigma" in ident.files
    else 0.01 * np.array([12.0, 12.0, 12.0, 0.08])
)

sim_setup = {
    "y_start": y_start,
    "u_previous": u_previous,
    "y_start_ns": y_start_ns,
    "u_previous_ns": u_previous_ns,
    "x_start": x_start,
    "P0": P0,
    "Q": Q,
    "Qd": Qd,
    "R": R,
    "N": N,
    "Qy": Qy,
    "Qu": Qu,
    "Qdu": Qdu,
    "Qy_te": Qy_te,
    "Qu_te": Qu_te,
    "u_min": u_min,
    "u_max": u_max,
    "y_min": y_min,
    "y_max": y_max,
    "u_min_ns": u_min_ns,
    "u_max_ns": u_max_ns,
    "y_min_ns": y_min_ns,
    "y_max_ns": y_max_ns,
    "sim_time": sim_time,
    "Ts": Ts,
    "reference": reference,
    "reference_ns": reference_ns,
    "reference_u": reference_u,
    "reference_u_ns": reference_u_ns,
    "disturbances": disturbances,
    "noise_sigma": noise_sigma * 0,
    "y_names": y_names,
    "u_names": u_names,
    "notes": (
        "T10 (−3 K) and xA10 (−0.03) feed upsets; both grades remain "
        "reachable inside the Q/F identification box."
    ),
}

out_path = os.path.join(os.path.dirname(__file__), "sim_setup.pkl")
joblib.dump(sim_setup, out_path)
print("Wrote", out_path)
print("Nominal y:", np.round(y_nom, 4))
print("Premium y:", np.round(y_premium, 4))
print("Disturbances:", disturbances)
