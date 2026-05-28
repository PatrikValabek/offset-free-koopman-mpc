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
    cons = 2
    layers = [4*cons,8*cons,16*cons]
    layers_dec  = [16*cons,8*cons,4*cons]


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
    global A, B, C, loaded_setup, nz, nu, ny, nd, T_real, A_block, A_transformed, A_backtransformed, problem, scaler, scalerU, y_start, y_start_ns, reference, y_setpoint, u_previous, u_previous_ns, P0, Q, R, A_, B_, C_, TVKF, target_estimation, mpc, Qy, Bd, Cd, u_sp, zs_lin, J_mpc
    matrix_C = False
    A = np.load(f"../data/A_C_{matrix_C}.npy")
    B = np.load(f"../data/B_C_{matrix_C}.npy")
    C = np.load(f"../data/C_C_{matrix_C}.npy")

    loaded_setup = joblib.load("sim_setup.pkl")


    nz, nu = B.shape
    ny = C.shape[0]
    nd = ny

    Qd = np.eye(nd) * loaded_setup['Qd']
    Q = np.block([
        [np.eye(nz) * loaded_setup['Q'],  np.zeros((nz, nd))],   # Trust state model
        [np.zeros((nd, nz)), Qd]      # Disturbance adapts slow
    ])
    R = np.eye(ny) * loaded_setup['R']
    P0 = np.eye(nz + nd) * loaded_setup['P0']
    Qy = loaded_setup['Qy']
    Cd = np.eye(ny)
    Bd = np.zeros((nz, nd))

# Block diagonalization
    T_real, A_block = helper.ident.real_block_diagonalize(A)

    # Transform A to check
    A_transformed = inv(T_real) @ A @ T_real
    print("Close to block diagonal?", np.allclose(A_block, A_transformed, atol=1e-6))

    # Backtransform A_block to verify it equals A
    A_backtransformed = T_real @ A_block @ inv(T_real)
    print("Backtransformation equals original A?", np.allclose(A, A_backtransformed, atol=1e-6))

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
    u_sp = loaded_setup['reference_u']

# Initial state estimate includes disturbance
    z_est_ = np.hstack(((inv(T_real) @ get_x_from_y(problem, y_start)).T, np.zeros((1, nd))))

    A_ = np.block([
        [A, Bd],
        [np.zeros((nd, nz)), np.eye(nd)],
    ])
    B_ = np.vstack([
        B,
        np.zeros((nd, nu)),
    ])
    C_ = np.hstack([
        C, Cd,
    ])

    TVKF = helper.TVKF(A_, B_, C_, z_est_, P0, Q, R)
    target_estimation = helper.TaylorTargetEstimation(
        A, B, loaded_setup['Qy'], loaded_setup['Qu'] * 0, Bd, Cd
    )
    mpc = helper.TaylorMPC(A, B, loaded_setup['Qy'], loaded_setup['Qu'], loaded_setup['Qdu'], Bd, Cd)

    xf0 = np.asarray(z_est_).flatten()
    J_mpc = helper.evaluate_jacobian(
        problem.nodes[4],
        torch.from_numpy(T_real @ xf0[:nz]).float().flatten(),
    ) @ T_real
    z_s0, _y0, _u0 = target_estimation.get_target(
        xf0[nz:], y_setpoint, u_sp, get_y(T_real @ xf0[:nz]), xf0[:nz], J_mpc
    )
    zs_lin = np.asarray(z_s0).flatten().copy()
    J_mpc = helper.evaluate_jacobian(
        problem.nodes[4],
        torch.from_numpy(T_real @ zs_lin).float().flatten(),
    ) @ T_real

def tests():
    global zs_lin, J_mpc
    xf = np.asarray(TVKF.x).flatten()
    J_mpc = helper.evaluate_jacobian(
        problem.nodes[4],
        torch.from_numpy(T_real @ xf[:nz]).float().flatten(),
    ) @ T_real
    z_s, y_s, u_s = target_estimation.get_target(
        xf[nz:], y_setpoint, u_sp, get_y(T_real @ xf[:nz]), xf[:nz], J_mpc
    )
    zs_lin = np.asarray(z_s).flatten().copy()
    J_mpc = helper.evaluate_jacobian(
        problem.nodes[4],
        torch.from_numpy(T_real @ zs_lin).float().flatten(),
    ) @ T_real
    print(zs_lin)
    Qz = J_mpc.T @ Qy @ J_mpc
    Qz_psd = Qz + 1e-8 * np.eye(Qz.shape[0])
    mpc.build_problem()
    _ = mpc.get_u_optimal(
        xf[:nz],
        xf[nz:],
        u_previous,
        zs_lin,
        u_s,
        get_y(T_real @ zs_lin),
        zs_lin,
        J_mpc,
        Qz_psd,
    )

def next_optimal_input(previous_input, measurement, step):
    global zs_lin, J_mpc
    previous_input = np.array([previous_input])
    measurement = np.array([measurement])
    step = int(step)
    u_prev = scalerU.transform(previous_input.reshape(1, -1))[0]
    y_now = scaler.transform(measurement.reshape(1, -1))[0]

    z_sim = np.asarray(TVKF.x).flatten()
    y_setpoint = reference[:, step]

    z_s, y_s, u_s = target_estimation.get_target(
        z_sim[nz:],
        y_setpoint,
        u_sp,
        get_y(T_real @ zs_lin),
        zs_lin,
        J_mpc,
    )

    Qz = J_mpc.T @ Qy @ J_mpc
    Qz_psd = Qz + 1e-8 * np.eye(Qz.shape[0])

    z_ref = np.asarray(z_s).flatten()
    u_opt = mpc.get_u_optimal(
        z_sim[:nz],
        z_sim[nz:],
        u_prev,
        z_ref,
        u_s,
        get_y(T_real @ zs_lin),
        zs_lin,
        J_mpc,
        Qz_psd,
    )

    zs_new = np.asarray(z_s).flatten().copy()
    J_mpc = helper.evaluate_jacobian(
        problem.nodes[4],
        torch.from_numpy(T_real @ zs_new).float().flatten(),
    ) @ T_real
    C_ = np.hstack([J_mpc, Cd])
    _ = TVKF.step(
        u_opt,
        y_now,
        get_y(T_real @ zs_new[:nz]),
        zs_new[:nz],
        J_mpc,
        C_,
    )
    zs_lin = zs_new

    print(z_sim[nz:])
    u_opt = scalerU.inverse_transform(u_opt.reshape(1, -1))[0]
    y_s = scaler.inverse_transform(y_s.reshape(1, -1))[0]
    u_s = scalerU.inverse_transform(u_s.reshape(1, -1))[0].flatten()
    return y_s, u_opt, u_s
