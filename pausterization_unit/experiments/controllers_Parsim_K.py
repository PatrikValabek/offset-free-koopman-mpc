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



def get_x_from_y(y: np.ndarray) -> np.ndarray:
    x = np.linalg.pinv(C) @ y.T
    return x.reshape(1, -1).T


data_dir = REPO_ROOT / '../data'

def load():
    global A, B, C, loaded_setup, nz, nu, ny, nd, T_real, A_block, A_transformed, A_backtransformed, problem, scaler, scalerU, y_start, y_start_ns, reference, y_setpoint, u_previous, u_previous_ns, P0, Q, R, A_, B_, C_, KF, target_estimation, mpc
    matrix_C = True
    A = np.load(f"../data/A_parsimK_Ts.npy")
    B = np.load(f"../data/B_parsimK_Ts.npy")
    C = np.load(f"../data/C_parsimK.npy")

    loaded_setup = joblib.load("sim_setup.pkl")


    nz, nu = B.shape
    ny = C.shape[0]
    nd = ny

    Q = np.eye(nz + nd) * loaded_setup['Q']
    R = np.eye(ny) * loaded_setup['R']
    P0 = np.eye(nz + nd) * loaded_setup['P0']


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

    scaler = joblib.load('../data/scaler.pkl')
    scalerU = joblib.load('../data/scalerU.pkl')

    y_start = loaded_setup['y_start']
    y_start_ns = loaded_setup.get('y_start_ns')
    reference = loaded_setup.get('reference')
    y_setpoint = loaded_setup['reference'][:, 0]
    u_previous = loaded_setup['u_previous']
    u_previous_ns = loaded_setup.get('u_previous_ns')

# Initial state estimate includes disturbance
    z_est_ = np.hstack(((get_x_from_y(y_start)).T, np.zeros((1, nd))))

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
    target_estimation = helper.TargetEstimation(A, B, C)
    mpc = helper.MPC(A, B, C)

def tests():
    z_s, y_s, u_s = target_estimation.get_target(KF.x[:, nz:], y_setpoint)
    z_ref = z_s
    print(z_ref)
    _ = mpc.get_u_optimal(KF.x[:, :nz], KF.x[:, nz:], u_previous, z_ref)

def next_optimal_input(previous_input, measurement, step):
    previous_input = np.array([previous_input])
    measurement = np.array([measurement])
    step = int(step)
    u_prev = scalerU.transform(previous_input.reshape(1, -1))[0]
    y_prev = scaler.transform(measurement.reshape(1, -1))[0]
    _ = KF.step(u_prev, y_prev)
    y_setpoint = reference[:, step]
    z_s, y_s, u_s = target_estimation.get_target(KF.x[nz:], y_setpoint)
    z_ref = z_s
    u_opt = mpc.get_u_optimal(KF.x[:nz], KF.x[nz:], u_prev, z_ref)
    u_opt = scalerU.inverse_transform(u_opt.reshape(1, -1))[0]
    y_s = scaler.inverse_transform(y_s.reshape(1, -1))[0]
    u_s = scalerU.inverse_transform(u_s.reshape(1, -1))[0].flatten()
    return y_s, u_opt, u_s
    

