#!/usr/bin/env python3
"""
Offset-free MPC (CT) migrated from `control/offset_free_CT.ipynb`.

- Loads A, B, C, trained Koopman-related modules, scalers, and sim setup
- Builds transformed representation with T_real
- Runs target estimation + MPC closed-loop simulation
- Saves plots under repository `figures/`
- Prints the objective line exactly as in the notebook:
  "Closed-loop objective function value: <value>"

Requires `src/helper` and `src/models` to be importable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Optional deps used in helper/model stack
import torch  # noqa: F401
import cvxpy as cp  # noqa: F401

# neuromancer symbols are imported in notebook; here they are required indirectly via helper/models loading
from neuromancer.modules import blocks  # type: ignore
from neuromancer.system import Node, System  # type: ignore
from neuromancer.problem import Problem  # type: ignore
from neuromancer.loss import PenaltyLoss  # type: ignore


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

def load_matrices_C_variant(data_dir: Path, matrix_C: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.load((data_dir / f"A_C_{matrix_C}.npy").as_posix())
    B = np.load((data_dir / f"B_C_{matrix_C}.npy").as_posix())
    C = np.load((data_dir / f"C_C_{matrix_C}.npy").as_posix())
    return A, B, C


def load_scalers(data_dir: Path):
    scaler = joblib.load((data_dir / 'scaler.pkl').as_posix())
    scalerU = joblib.load((data_dir / 'scalerU.pkl').as_posix())
    return scaler, scalerU


def load_sim_setup(repo_root: Path):
    control_path = repo_root / 'control' / 'sim_setup.pkl'
    control_py_path = repo_root / 'control_python' / 'sim_setup.pkl'
    if control_path.exists():
        return joblib.load(control_path.as_posix())
    if control_py_path.exists():
        return joblib.load(control_py_path.as_posix())
    return joblib.load('sim_setup.pkl')


# ------------------------------ CT helpers ------------------------------------

def build_encoders_decoders(ny: int, nz: int, nu: int, matrix_C: bool):
    layers = [20, 40, 60]
    layers_dec = [60, 40, 20]

    # output encoder f_y
    f_y = blocks.MLP(
        ny,
        nz,
        bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=layers,
    )
    encode_Y0 = Node(f_y, ['Y0'], ['x'], name='encoder_Y0')
    encode_Y = Node(f_y, ['Y'], ['x_latent'], name='encoder_Y')

    # input encoder f_u representing B
    f_u = torch.nn.Linear(nu, nz, bias=False)
    encode_U = Node(f_u, ['U'], ['u_latent'], name='encoder_U')

    # decoder
    if not matrix_C:
        f_y_inv = blocks.MLP(
            nz, ny, bias=True, linear_map=torch.nn.Linear, nonlin=torch.nn.ELU, hsizes=layers_dec
        )
    elif matrix_C:
        f_y_inv = torch.nn.Linear(nz, ny, bias=False)
    else:
        raise ValueError('matrix_C must be boolean')
    decode_y = Node(f_y_inv, ['x'], ['yhat'], name='decoder_y')

    # Koopman linear K
    K = torch.nn.Linear(nz, nz, bias=False)
    Koopman = Node(helper.PredictionWControl(K), ['x', 'u_latent'], ['x'], name='K')

    dynamics_model = System([Koopman], name='Koopman', nsteps=1)
    nodes = [encode_Y0, encode_Y, encode_U, dynamics_model, decode_y]
    loss = PenaltyLoss([], constraints=[])
    problem = Problem(nodes, loss)
    return problem


def get_x_from_y(problem, y: np.ndarray) -> np.ndarray:
    x = problem.nodes[0]({"Y0": torch.from_numpy(y).float()})
    return x["x"].detach().numpy().reshape(1, -1).T


def get_y_from_x(problem, x: np.ndarray) -> np.ndarray:
    y = problem.nodes[4]({"x": torch.from_numpy(x.T).float()})
    return y["yhat"].detach().numpy().reshape(1, -1)


# ------------------------------ Main routine ----------------------------------

def main() -> None:
    data_dir = REPO_ROOT / 'data'
    figures_dir = get_figures_dir(REPO_ROOT)

    # Load matrices for matrix_C=False as in notebook
    matrix_C = False
    A, B, C = load_matrices_C_variant(data_dir, matrix_C)

    nz, nu = B.shape
    ny = C.shape[0]

    # Disturbance model dimension
    F = np.eye(ny)
    nd = F.shape[1]

    # Block diagonalization
    T_real, A_block = helper.ident.real_block_diagonalize(A)

    # Apply transformation as in notebook (Cell 6)
    A = inv(T_real) @ A @ T_real
    B = inv(T_real) @ B
    C = C @ T_real

    # Koopman enc/dec + problem and load weights
    problem = build_encoders_decoders(ny, nz, nu, matrix_C)
    problem.load_state_dict(torch.load((data_dir / f'model_C_{matrix_C}.pth').as_posix()), strict=False)

    # Load scalers
    scaler, scalerU = load_scalers(data_dir)

    # Plant model
    A1 = 1
    A2 = 0.5
    k1 = 0.5
    k2 = 0.8
    TwoTanks = models.TwoTanks(True, A1, A2, k1, k2)

    # Sim setup
    loaded_setup = load_sim_setup(REPO_ROOT)

    y_start = loaded_setup['y_start']
    y_setpoint = loaded_setup['reference'][:, 0]
    u_previous = loaded_setup['u_previous']

    # Initial state estimate includes disturbance
    z_est_ = np.hstack(((inv(T_real) @ get_x_from_y(problem, y_start)).T, np.zeros((1, nd))))

    P0 = loaded_setup['P0']
    Q = loaded_setup['Q']
    R = loaded_setup['R']

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

    # Target calc
    target_estimation = helper.TargetEstimation(A, B, C)
    z_s, y_s = target_estimation.get_target(z_est_[:, nz:], y_setpoint)
    z_ref = z_s

    # MPC problem
    mpc = helper.MPC(A, B, C)
    _ = mpc.get_u_optimal(z_est_[:, :nz], z_est_[:, nz:], u_previous, z_ref)

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

    u_sim_descaled[:, 0] = scalerU.inverse_transform(u_sim[:, 0].reshape(1, -1))[0]
    y_sim_descaled[:, 0] = scaler.inverse_transform(y_sim[:, 0].reshape(1, -1))[0]

    for k in range(0, sim_time):
        # MPC
        u_opt = mpc.get_u_optimal(z_sim[:nz, k], z_sim[nz:, k], u_prev, zs_sim[:, k])
        u_sim[:, k] = u_opt
        u_sim_descaled[:, k] = scalerU.inverse_transform(u_sim[:, k].reshape(1, -1))[0]

        # Plant
        y_sim_descaled[:, k + 1] = TwoTanks.step(
            y_sim_descaled[:, k], u_sim_descaled[:, k].reshape(1, -1), Ts
        )
        y_sim[:, k + 1] = scaler.transform(y_sim_descaled[:, k + 1].reshape(1, -1))[0]

        # Estimation
        z_sim[:, k + 1] = KF.step(u_sim[:, k], y_sim[:, k]).flatten()

        # Target update
        zs_sim[:, k + 1], ys_sim[:, k + 1] = target_estimation.get_target(
            z_sim[nz:, k + 1], loaded_setup["reference"][:, k]
        )
        u_prev = u_sim[:, k]

    # Objective (exact phrase preserved)
    objective_value = 0.0
    state_error_cost = 0.0
    control_increment_cost = 0.0
    for k in range(sim_time):
        y_diff = y_sim[:, k] - loaded_setup["reference"][:, k]
        # Match notebook behavior: always use k-1 (wraps to last element at k=0)
        u_diff = u_sim[:, k] - u_sim[:, k - 1]
        y_term = float(y_diff.T @ loaded_setup["Qy"] @ y_diff)
        u_term = float(u_diff.T @ loaded_setup["Qu"] @ u_diff)
        state_error_cost += y_term
        control_increment_cost += u_term
        objective_value += y_term + u_term

    print(f"Closed-loop objective function value: {objective_value}")
    print(f"State error term: {state_error_cost}")
    print(f"Control increment term: {control_increment_cost}")

    # Plots saved to figures/
    fig = plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(y_sim_descaled[0, :], label='h1')
    plt.plot(y_sim_descaled[1, :], label='h2')
    plt.plot(scaler.inverse_transform(ys_sim.T).T[0, :], color='red', linestyle='--', label='target h1')
    plt.plot(scaler.inverse_transform(ys_sim.T).T[1, :], color='red', linestyle=':', label='target h2')
    plt.title('K-MPC (Block-Diag C LTI) Simulation')
    plt.xlabel('Time step')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)
    fig.tight_layout()
    fig.savefig((figures_dir / 'CT_states.png').as_posix(), dpi=200)

    fig2 = plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(u_sim_descaled[0, max(0, 50):], label='q1')
    plt.plot(u_sim_descaled[1, max(0, 50):], label='q2')
    plt.xlabel('Time step')
    plt.ylabel('Input')
    plt.legend()
    plt.grid(True)
    fig2.tight_layout()
    fig2.savefig((figures_dir / 'CT_inputs.png').as_posix(), dpi=200)


if __name__ == "__main__":
    main()
