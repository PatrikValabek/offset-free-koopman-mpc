#!/usr/bin/env python3
"""
Offset-free MPC (PaK) migrated from `control/offset_free_PaK.ipynb`.

- Loads A, B, C, scalers, and sim setup
- Builds block-diagonalized representation
- Runs target estimation + MPC closed-loop simulation
- Saves plots under repository `figures/`
- Prints the objective line exactly as in the notebook:
  "Closed-loop objective function value: <value>"

Note: Requires `src/helper` and `src/models` to be importable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Torch and cvxpy are used indirectly by helper code (problem definitions)
import torch  # noqa: F401
import cvxpy as cp  # noqa: F401


# --------------------------- Repository paths ---------------------------------

def get_repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def get_figures_dir(repo_root: Path) -> Path:
    figures = repo_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    return figures


# Ensure `src` is on PYTHONPATH
REPO_ROOT = get_repo_root()
SRC_PATH = REPO_ROOT / 'src'
if SRC_PATH.as_posix() not in sys.path:
    sys.path.append(SRC_PATH.as_posix())

import helper  # type: ignore
import models  # type: ignore


# ------------------------------ IO utilities ----------------------------------

def load_data_matrices(data_dir: Path, append: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.load((data_dir / f"A{append}.npy").as_posix())
    B = np.load((data_dir / f"B{append}.npy").as_posix())
    C = np.load((data_dir / f"C{append}.npy").as_posix())
    return A, B, C


def load_scalers(data_dir: Path):
    scaler = joblib.load((data_dir / 'scaler.pkl').as_posix())
    scalerU = joblib.load((data_dir / 'scalerU.pkl').as_posix())
    return scaler, scalerU


def load_sim_setup(repo_root: Path):
    # Prefer control/sim_setup.pkl if present, else control_python/sim_setup.pkl, else local
    control_path = repo_root / 'control' / 'sim_setup.pkl'
    control_py_path = repo_root / 'control_python' / 'sim_setup.pkl'
    if control_path.exists():
        return joblib.load(control_path.as_posix())
    if control_py_path.exists():
        return joblib.load(control_py_path.as_posix())
    return joblib.load('sim_setup.pkl')


# ------------------------------ Main routine ----------------------------------

def main() -> None:
    data_dir = REPO_ROOT / 'data'
    figures_dir = get_figures_dir(REPO_ROOT)

    # Load A, B, C using the notebook's suffix
    append = "_parsimK"
    A, B, C = load_data_matrices(data_dir, append)

    nz, nu = B.shape
    ny = C.shape[0]

    # Disturbance model
    F = np.eye(ny)
    nd = F.shape[1]

    # Real block-diagonalization of A (as in the notebook)
    T_real, A_block = helper.ident.real_block_diagonalize(A)

    # Transform inputs/outputs into the block-diagonal basis
    A = A_block
    B = inv(T_real) @ B
    C = C @ T_real

    # Load scalers
    scaler, scalerU = load_scalers(data_dir)

    # Define same physical model as in notebook
    A1 = 1
    A2 = 0.5
    k1 = 0.5
    k2 = 0.8
    TwoTanks = models.TwoTanks(True, A1, A2, k1, k2)

    # Load simulation setup
    loaded_setup = load_sim_setup(REPO_ROOT)

    # Initial conditions and KF setup
    y_start = loaded_setup['y_start']
    y_setpoint = loaded_setup['reference'][:, 0]
    u_previous = loaded_setup['u_previous']

    z_est_ = np.hstack(((np.linalg.pinv(C) @ y_start.T).T, np.zeros((1, nd))))

    P0 = loaded_setup['P0'][:nz + nd, :nz + nd]
    Q = loaded_setup['Q'][:nz + nd, :nz + nd]
    R = loaded_setup['R'][:nz + nd, :nz + nd]

    A_ = np.block([
        [A, np.zeros((nz, nd))],
        [np.zeros((nd, nz)), np.eye(nd)],
    ])
    B_ = np.vstack([
        B,
        np.zeros((nd, nu)),
    ])
    C_ = np.hstack([
        C, np.eye(nd),
    ])

    KF = helper.KF(A_, B_, C_, z_est_, P0, Q, R)

    # Target calculation
    target_estimation = helper.TargetEstimation(A, B, C)
    z_s, y_s = target_estimation.get_target(z_est_[:, nz:], y_setpoint)
    # Keep `zs` as reference
    z_ref = z_s

    # MPC problem
    mpc = helper.MPC(A, B, C)
    u_opt = mpc.get_u_optimal(z_est_[:, :nz], z_est_[:, nz:], u_previous, z_ref)

    # Closed-loop simulation
    sim_time = int(loaded_setup['sim_time'])
    Ts = 1

    z_sim = np.zeros((nz + nd, sim_time + 1))
    y_sim = np.zeros((ny, sim_time + 1))
    u_sim = np.zeros((nu, sim_time))
    ys_sim = np.zeros((ny, sim_time + 1))
    zs_sim = np.zeros((nz, sim_time + 1))

    y_sim_descaled = np.zeros((ny, sim_time + 1))
    u_sim_descaled = np.zeros((nu, sim_time))

    z_sim[:, 0] = z_est_.flatten()
    y_sim[:, 0] = y_start.flatten()
    ys_sim[:, 0] = y_s
    zs_sim[:, 0] = z_s
    u_prev = target_estimation.u_s.value

    # Initialize descaled first step for completeness
    u_sim_descaled[:, 0] = scalerU.inverse_transform(u_sim[:, 0].reshape(1, -1))[0]
    y_sim_descaled[:, 0] = scaler.inverse_transform(y_sim[:, 0].reshape(1, -1))[0]

    for k in range(0, sim_time):
        # MPC step
        u_opt = mpc.get_u_optimal(z_sim[:nz, k], z_sim[nz:, k], u_prev, zs_sim[:, k])
        u_sim[:, k] = u_opt
        u_sim_descaled[:, k] = scalerU.inverse_transform(u_sim[:, k].reshape(1, -1))[0]

        # Plant step in non-scaled domain via TwoTanks
        y_sim_descaled[:, k + 1] = TwoTanks.step(
            y_sim_descaled[:, k], u_sim_descaled[:, k].reshape(1, -1), Ts
        )
        # Scale back to internal domain
        y_sim[:, k + 1] = scaler.transform(y_sim_descaled[:, k + 1].reshape(1, -1))[0]

        # State estimation
        z_sim[:, k + 1] = KF.step(u_sim[:, k], y_sim[:, k]).flatten()

        # Target estimation with current disturbance estimate and reference
        zs_sim[:, k + 1], ys_sim[:, k + 1] = target_estimation.get_target(
            z_sim[nz:, k + 1], loaded_setup["reference"][:, k]
        )

        u_prev = u_sim[:, k]

    # Compute closed-loop objective (exactly like notebook cell)
    objective_value = 0.0
    Qy = loaded_setup["Qy"]
    Qu = loaded_setup["Qu"]

    for k in range(0, sim_time):
        y_diff = y_sim[:, k] - loaded_setup["reference"][:, k]
        u_diff = u_sim[:, k] - (u_sim[:, k - 1] if k > 0 else np.zeros_like(u_sim[:, k]))
        objective_value += float(y_diff.T @ Qy @ y_diff)
        objective_value += float(u_diff.T @ Qu @ u_diff)

    print(f"Closed-loop objective function value: {objective_value}")

    # Plots (saved to figures/)
    fig = plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(y_sim_descaled[0, :], label='h1')
    plt.plot(y_sim_descaled[1, :], label='h2')
    plt.plot(scaler.inverse_transform(ys_sim.T).T[0,:], color='red', linestyle='--', label='target h1')
    plt.plot(scaler.inverse_transform(ys_sim.T).T[1,:], color='red', linestyle=':', label='target h2')
    plt.title('K-MPC (Block-Diag C LTI) Simulation')
    plt.xlabel('Time step')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)
    fig.tight_layout()
    fig.savefig((figures_dir / 'PaK_states.png').as_posix(), dpi=200)

    fig2 = plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(u_sim_descaled[0, max(0, 50):], label='q1')
    plt.plot(u_sim_descaled[1, max(0, 50):], label='q2')
    plt.xlabel('Time step')
    plt.ylabel('Input')
    plt.legend()
    plt.grid(True)
    fig2.tight_layout()
    fig2.savefig((figures_dir / 'PaK_inputs.png').as_posix(), dpi=200)


if __name__ == "__main__":
    main()
