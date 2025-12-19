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

def get_repo_root() -> Path:
    # In a notebook, __file__ is not defined; use cwd as a fallback
    return Path.cwd().resolve().parent

# # Ensure `src` is on PYTHONPATH
REPO_ROOT = get_repo_root()
SRC_PATH = REPO_ROOT / 'src'
if SRC_PATH.as_posix() not in sys.path:
    sys.path.append(SRC_PATH.as_posix())

import helper  # type: ignore
import models  # type: ignore

def build_encoders_decoders(ny: int, nz: int, nu: int, matrix_C: bool):
    cons =3
    layers = [6*cons,12*cons,18*cons]
    layers_dec  = [18*cons,12*cons,6*cons]

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

def get_y(x: np.ndarray) -> np.ndarray:
    y = problem.nodes[4]({'x': torch.from_numpy(x.T).float()})
    return y['yhat'].detach().numpy().reshape(1, -1)


data_dir = REPO_ROOT / '../data'

def load():
    global A, B, C, loaded_setup, nz, nu, ny, nd, T_real, A_block, A_transformed, A_backtransformed, problem, scaler, scalerU, y_start, y_start_ns, reference, y_setpoint, u_previous, u_previous_ns, P0, Q, R, A_, B_, C_, EKF, target_estimation, mpc, Qy
    matrix_C = False
    A = np.load(f"../data/A_C_{matrix_C}_Ts.npy")
    B = np.load(f"../data/B_C_{matrix_C}_Ts.npy")
    C = np.load(f"../data/C_C_{matrix_C}.npy")

    loaded_setup = joblib.load("sim_setup.pkl")


    nz, nu = B.shape
    ny = C.shape[0]
    nd = ny

    # Q = np.eye(nz + nd) * loaded_setup['Q']
    Q = np.block([
        [np.eye(nz) * loaded_setup['Q'],  np.zeros((nz, nd))],   # Trust state model
        [np.zeros((nd, nz)), np.eye(nd) * 1.0]      # Disturbance adapts fast
    ])
    R = np.eye(ny) * loaded_setup['R']
    P0 = np.eye(nz + nd) * loaded_setup['P0']
    Qy = loaded_setup['Qy']

# Block diagonalization
    T_real, A_block = helper.ident.real_block_diagonalize(A)

    # Transform A to check
    A_transformed = inv(T_real) @ A @ T_real
    print("Close to block diagonal?", np.allclose(A_block, A_transformed, atol=1e-6))

    # Backtransform A_block to verify it equals A
    A_backtransformed = T_real @ A_block @ inv(T_real)
    print("Backtransformation equals original A?", np.allclose(A, A_backtransformed, atol=1e-6))
# T_real = np.eye(nz)  # Use identity for now, as A is already block diagonal

# Apply transformation as in notebook (Cell 6)
    A = inv(T_real) @ A @ T_real
    B = inv(T_real) @ B
    C = C @ T_real

# Koopman enc/dec + problem and load weights
    problem = build_encoders_decoders(ny, nz, nu, matrix_C)
    problem.load_state_dict(torch.load('../data/model_C_' + str(matrix_C) + '.pth'), strict=False)

    scaler = joblib.load('../data/scaler.pkl')
    scalerU = joblib.load('../data/scalerU.pkl')

    y_start = loaded_setup['y_start']
    y_start_ns = loaded_setup.get('y_start_ns')
    reference = loaded_setup.get('reference')
    y_setpoint = loaded_setup['reference'][:, 0]
    u_previous = loaded_setup['u_previous']
    u_previous_ns = loaded_setup.get('u_previous_ns')

# Initial state estimate includes disturbance
    z_est_ = np.hstack(((inv(T_real) @ get_x_from_y(problem, y_start)).T, np.zeros((1, nd))))

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
    EKF = helper.EKF(A_, B_, z_est_, P0, problem, Q, R, 3, T_real)
    target_estimation = helper.TaylorTargetEstimation(A, B)
    mpc = helper.TaylorMPC(A, B)

def tests():
    J = helper.evaluate_jacobian(
        problem.nodes[4],
        torch.from_numpy(T_real @ EKF.x[:, :nz].T).float().flatten(),
    ) @ T_real
    z_s, y_s, u_s = target_estimation.get_target( EKF.x[:, nz:], y_setpoint, get_y(T_real @ EKF.x[0, :nz]), EKF.x[0, :nz], J)
    z_ref_prev = z_s
    print(z_ref_prev)
    Qz = J.T @ Qy @ J
    Qz_psd = Qz + 1e-8 * np.eye(Qz.shape[0])   
    mpc.build_problem(Qz_psd)
    _ = mpc.get_u_optimal(EKF.x[0, :nz], EKF.x[:, nz:], u_previous, z_ref_prev, get_y(T_real @ z_s), z_s, J)

def next_optimal_input(previous_input, measurement, step):
    previous_input = np.array([previous_input])
    measurement = np.array([measurement])
    step = int(step)
    u_prev = scalerU.transform(previous_input.reshape(1, -1))[0]
    y_prev = scaler.transform(measurement.reshape(1, -1))[0]
    z_sim = EKF.step(u_prev, y_prev).flatten()

    y_setpoint = reference[:, step]

    J = helper.evaluate_jacobian(
        problem.nodes[4],
        torch.from_numpy(T_real @ z_sim[:nz]).float().flatten(),
    ) @ T_real
    print(get_y(T_real @ z_sim[:nz]))
    print(y_prev)
    z_s, y_s, u_s = target_estimation.get_target(
        z_sim[nz:], 
        y_setpoint, 
        get_y(T_real @ z_sim[:nz]), 
        z_sim[:nz], 
        J
    )
    print(z_sim[nz:])
    Qz = J.T @ Qy @ J
    Qz_psd = Qz + 1e-8 * np.eye(Qz.shape[0])
    mpc.build_problem(Qz_psd)

    z_ref = z_s
    u_opt = mpc.get_u_optimal(
        z_sim[:nz],
        z_sim[nz:],
        u_prev,
        z_ref,
        get_y(T_real @ z_sim[:nz]),
        z_sim[:nz],
        J,
    )

    u_opt = scalerU.inverse_transform(u_opt.reshape(1, -1))[0]
    y_s = scaler.inverse_transform(y_s.reshape(1, -1))[0]
    u_s = scalerU.inverse_transform(u_s.reshape(1, -1))[0].flatten()
    return y_s, u_opt, u_s
    

