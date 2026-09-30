"""Simulation setup for N4SID offset-free MPC of the square 4x4 CSTR-separator.

Industrial scenario (grade campaign + unmeasured feed upsets)
-----------------------------------------------------------
CVs (same as identification): T1, T2, T3, xB3.
MVs: Q1, Q2, Q3, F10 (F20 and Fr are fixed at their nominal values inside
the plant model; they are not manipulated).

The campaign is a bottoms-quality (xB3) grade change around the Li & Swartz
nominal point, with temperatures held as operating constraints. Two plant
disturbances that the controller does not measure are then applied:

  1. Cold fresh feed: modest T10 drop (preheat / utility drift).
  2. Slightly leaner fresh feed: modest xA10 drop (tank switch / impurity).

With only four manipulated inputs (no recycle Fr, no F20) the heaters have
less authority to reject a feed upset than in the 6-input plant, so the
upsets used here are softened relative to ``CSTRSeparator/Control/setup.py``
(T10: 313 -> 311 K instead of 310 K; xA10: 1.00 -> 0.98 instead of 0.97).
``feasibility_check()`` below solves the nonlinear steady-state inverse for
every phase of the campaign and prints the required input and its margin to
the identification box, so this setup is self-verifying: with the softened
upsets every phase is reachable (worst case Q1 ~= 19.1 kJ/s at the premium
grade under both upsets, vs. bound 25 kJ/s). The original 310 K / 0.97
upsets would saturate Q1 during [850, 1000) and leave a steady-state T1
offset of about 0.1 K -- i.e. infeasible for this square, reduced-input
plant.

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
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
REPO_ROOT = HERE.parent.parent
SRC = REPO_ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.append(SRC.as_posix())

import models  # noqa: E402

# ---------------------------- Plant / scalers ---------------------------------
plant = models.CSTRSeparator4x4()
y_names = list(plant.y_names)  # T1, T2, T3, xB3
u_names = list(plant.u_names)  # Q1, Q2, Q3, F10
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


def steady_state_inverse(
    y_target: np.ndarray,
    u0: np.ndarray,
    x0: np.ndarray,
    disturbance_attrs: dict | None = None,
    n_steps: int = 400,
    Ts: float = 1.0,
):
    """Find u (within the MV box) whose nonlinear steady state matches y_target.

    ``disturbance_attrs`` temporarily overrides plant attributes (e.g.
    ``T10``, ``xA10``) while solving, then restores them.
    """
    disturbance_attrs = disturbance_attrs or {}
    old = {k: getattr(plant, k) for k in disturbance_attrs}
    for k, v in disturbance_attrs.items():
        setattr(plant, k, v)
    try:
        y_scale = np.array([1.0, 1.0, 1.0, 0.01])

        def residual(u):
            _, y = simulate_to_ss(u, x0, n_steps=n_steps, Ts=Ts)
            return (y - y_target) / y_scale

        result = least_squares(
            residual, u0, bounds=(u_min_ns, u_max_ns), xtol=1e-12, ftol=1e-12
        )
        x_ss, y_ss = simulate_to_ss(result.x, x0, n_steps=n_steps, Ts=Ts)
    finally:
        for k, v in old.items():
            setattr(plant, k, v)
    return result.x, y_ss, x_ss, float(np.max(np.abs(result.fun)))


def feasibility_check(verbose: bool = True) -> list[dict]:
    """Verify every campaign phase is reachable inside the MV box.

    Returns a list of per-phase dicts with the required steady-state input,
    the resulting output, and the margin (in physical units) to each bound.
    Prints a summary table when ``verbose``.
    """
    phases = [
        ("nominal, no disturbance", y_nom, u_nom.copy(), {}),
        ("premium, no disturbance", y_premium, u_premium.copy(), {}),
        ("premium, T10 disturbance", y_premium, u_premium.copy(), {"T10": 311.0}),
        (
            "premium, T10 + xA10 disturbances",
            y_premium,
            u_premium.copy(),
            {"T10": 311.0, "xA10": 0.98},
        ),
        (
            "nominal, T10 + xA10 disturbances",
            y_nom,
            u_nom.copy(),
            {"T10": 311.0, "xA10": 0.98},
        ),
    ]
    rows = []
    for label, y_target, u0, dist in phases:
        u_req, y_ss, _, resid = steady_state_inverse(y_target, u0, x_nom, dist)
        margin_lo = u_req - u_min_ns
        margin_hi = u_max_ns - u_req
        feasible = bool(np.all(margin_lo >= -1e-6) and np.all(margin_hi >= -1e-6) and resid < 1e-2)
        rows.append(
            {
                "label": label,
                "disturbance": dist,
                "y_target": y_target,
                "u_required": u_req,
                "y_achieved": y_ss,
                "margin_to_min": margin_lo,
                "margin_to_max": margin_hi,
                "residual": resid,
                "feasible": feasible,
            }
        )
        if verbose:
            print(f"[{label}]  disturbance={dist}")
            print(f"  u_required = {np.round(u_req, 3)}  ({u_names})")
            print(f"  y_achieved = {np.round(y_ss, 4)}  (target {np.round(y_target, 4)})")
            print(f"  margin to [min,max] = {np.round(np.minimum(margin_lo, margin_hi), 3)}")
            print(f"  feasible = {feasible}  (residual={resid:.4g})")
    if verbose:
        n_bad = sum(not r["feasible"] for r in rows)
        print(
            f"\nfeasibility_check: {len(rows) - n_bad}/{len(rows)} phases reachable "
            f"inside the identification box."
        )
    return rows


# ---------------------------- Achievable grades -------------------------------
# Nominal Li & Swartz point, then a milder-conversion "premium B" grade
# (higher F10 / lower heat -> higher xB3; the yield peak is not at maximum
# heat). u_premium reproduces the same y_premium as the 6-input plant's
# premium grade, now realized with F20/Fr fixed at their nominal values.
u_nom = plant.u_nom.copy()
u_premium = np.array([6.619, 8.544, 8.499, 11.224], dtype=float)

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
# Times are sample indices (Ts = 1 s). Softened vs. the 6-input campaign
# (313->310 K, 1.00->0.97) because the square 4x4 plant has two fewer
# manipulated inputs (no Fr, no F20) to reject the same upset; see
# ``feasibility_check()`` above.
disturbances = [
    {"k": 600, "attr": "T10", "value": 311.0, "label": "cold feed T10: 313 -> 311 K"},
    {"k": 850, "attr": "xA10", "value": 0.98, "label": "lean feed xA10: 1.00 -> 0.98"},
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
# Square plant (nu == ny): Qu_te must stay off, otherwise the target QP
# trades away xB3 offset-free tracking for input cost and steady-state
# offset reappears (there is no spare input to absorb it).
Qu_te = np.diag([0.2, 0.2, 0.2, 0.5]) * 0

Qy = np.diag([2.0, 2.0, 2.0, 15.0])
Qu = np.diag([0.2, 0.2, 0.2, 0.5])
Qdu = np.diag([0.5, 0.5, 0.5, 1.0])

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
        "Square 4x4 plant (Q1, Q2, Q3, F10; F20/Fr fixed). Softened T10 "
        "(-2 K) and xA10 (-0.02) feed upsets so both grades remain "
        "reachable inside the Q/F10 identification box; see "
        "feasibility_check() in this file."
    ),
}

out_path = os.path.join(os.path.dirname(__file__), "sim_setup.pkl")
joblib.dump(sim_setup, out_path)
print("Wrote", out_path)
print("Nominal y:", np.round(y_nom, 4))
print("Premium y:", np.round(y_premium, 4))
print("Disturbances:", disturbances)

if __name__ == "__main__":
    print("\nRunning feasibility_check() over the campaign phases:\n")
    feasibility_check(verbose=True)
