"""Headless CT (linear Koopman, matrix_C=True) closed-loop evaluation.

Mirrors ``CT.ipynb``: KF + TargetEstimation + linear MPC on the current
``sim_setup.pkl`` (same Qy/Qu/Qdu, horizon, disturbances, references).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from numpy.linalg import inv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SRC = REPO_ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.append(SRC.as_posix())
if HERE.as_posix() not in sys.path:
    sys.path.append(HERE.as_posix())

os.environ.setdefault("SIM_SETUP_PATH", str(HERE / "sim_setup.pkl"))

import helper  # noqa: E402
import plant_inference  # noqa: E402


def maybe_block_diagonalize(A, B, C, use_block_diag: bool):
    if not use_block_diag:
        return A, B, C, np.eye(A.shape[0])
    T_real, _ = helper.real_block_diagonalize(A)
    if not np.isfinite(T_real).all() or np.linalg.cond(T_real) > 1e10:
        raise RuntimeError("block-diagonalizing transformation is ill-conditioned")
    A_t = inv(T_real) @ A @ T_real
    B_t = inv(T_real) @ B
    C_t = C @ T_real
    return A_t, B_t, C_t, T_real


def closed_loop_of(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    *,
    use_block_diag: bool = True,
    sim_setup: dict | None = None,
    quiet: bool = True,
) -> dict:
    """Run the CT notebook loop and return the closed-loop objective.

    Objective is identical to the notebook: scaled tracking + Δu + u-setpoint
    costs with ``Qy``, ``Qu``, ``Qdu`` from ``setup.py``.
    """
    import joblib

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    A, B, C, _ = maybe_block_diagonalize(A, B, C, use_block_diag)

    setup_path = os.environ.get("SIM_SETUP_PATH", str(HERE / "sim_setup.pkl"))
    loaded = sim_setup if sim_setup is not None else joblib.load(setup_path)

    nz, nu = B.shape
    ny = C.shape[0]
    nd = ny
    Ts = float(loaded["Ts"])
    sim_time = int(loaded["sim_time"])

    plant_inference.init(x0=loaded["x_start"], Ts=Ts)
    plant_inference.reset(loaded["x_start"], Ts=Ts)

    y_start = np.asarray(loaded["y_start"]).reshape(1, -1)
    reference = np.asarray(loaded["reference"], dtype=float)
    u_sp = np.asarray(loaded["reference_u"], dtype=float).reshape(-1)
    scaler = plant_inference.scaler
    scalerU = plant_inference.scalerU

    z_est_ = np.hstack(((np.linalg.pinv(C) @ y_start.T).T, np.zeros((1, nd))))
    P0 = np.eye(nz + nd) * loaded["P0"]
    Q = np.block(
        [
            [np.eye(nz) * loaded["Q"], np.zeros((nz, nd))],
            [np.zeros((nd, nz)), np.eye(nd) * loaded["Qd"]],
        ]
    )
    R = np.eye(ny) * loaded["R"]
    Cd = np.eye(ny)
    Bd = np.zeros((nz, nd))
    A_ = np.block([[A, Bd], [np.zeros((nd, nz)), np.eye(nd)]])
    B_ = np.vstack([B, np.zeros((nd, nu))])
    C_ = np.hstack([C, Cd])

    KF = helper.KF(A_, B_, C_, z_est_, P0, Q, R)
    target_estimation = helper.TargetEstimation(A, B, C, loaded["Qy"], loaded["Qu_te"], Bd, Cd)
    z_s, y_s, u_s = target_estimation.get_target(z_est_[:, nz:], reference[:, 0], u_sp)
    Qx = C.T @ loaded["Qy"] @ C + 2e-8 * np.eye(nz)
    mpc = helper.MPC(A, B, C, loaded["Qy"], loaded["Qu"], loaded["Qdu"], Bd, Cd)
    mpc.build_problem(Qx)
    _ = mpc.get_u_optimal(z_est_[:, :nz], z_est_[:, nz:], u_s, u_s, z_s)

    z_sim = np.zeros((nz + nd, sim_time + 1))
    y_sim = np.zeros((ny, sim_time + 1))
    u_sim = np.zeros((nu, sim_time))
    us_sim = np.zeros((nu, sim_time))
    z_sim[:, 0] = z_est_.flatten()
    y_sim[:, 0] = y_start.flatten()
    u_prev = np.asarray(u_s).reshape(-1)

    dist_by_k = {int(d["k"]): d for d in loaded["disturbances"]}
    noise_sigma = np.asarray(loaded["noise_sigma"], dtype=float)
    rng = np.random.default_rng(0)

    iterator = range(sim_time)
    if not quiet:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="CT closed-loop", ncols=80)

    for k in iterator:
        if k in dist_by_k:
            d = dist_by_k[k]
            plant_inference.apply_disturbance(d["attr"], d["value"])
        zs, ys, us = target_estimation.get_target(z_sim[nz:, k], reference[:, k], u_sp)
        us_sim[:, k] = us
        u_opt = mpc.get_u_optimal(z_sim[:nz, k], z_sim[nz:, k], us, u_prev, zs)
        u_sim[:, k] = u_opt
        plant_inference.y_plus(u_sim[:, k])
        y_true_ns = plant_inference.measure_ns()
        noise = rng.normal(0.0, noise_sigma)
        y_meas_ns = y_true_ns + noise
        y_sim[:, k + 1] = scaler.transform(y_meas_ns.reshape(1, -1))[0]
        z_sim[:, k + 1] = KF.step(u_sim[:, k], y_sim[:, k + 1]).flatten()
        u_prev = u_sim[:, k]

    Qy, Qu, Qdu = loaded["Qy"], loaded["Qu"], loaded["Qdu"]
    objective = state_err = du_cost = u_sp_cost = 0.0
    for k in range(sim_time):
        y_diff = y_sim[:, k] - reference[:, k]
        prev_u = u_sim[:, k - 1] if k > 0 else u_sim[:, k]
        u_diff = u_sim[:, k] - prev_u
        u_error = us_sim[:, k] - u_sim[:, k]
        y_term = float(y_diff.T @ Qy @ y_diff)
        u_term = float(u_diff.T @ Qdu @ u_diff)
        u_sp_term = float(u_error.T @ Qu @ u_error)
        state_err += y_term
        du_cost += u_term
        u_sp_cost += u_sp_term
        objective += y_term + u_term + u_sp_term

    return {
        "objective": float(objective),
        "state_error_cost": float(state_err),
        "control_increment_cost": float(du_cost),
        "control_setpoint_cost": float(u_sp_cost),
        "rho_A": float(np.max(np.abs(np.linalg.eigvals(A)))),
    }


def evaluate_or_penalty(A, B, C, use_block_diag: bool = True) -> dict:
    try:
        return closed_loop_of(A, B, C, use_block_diag=use_block_diag, quiet=True)
    except Exception as exc:
        return {
            "objective": 1e6,
            "error": str(exc),
            "state_error_cost": float("nan"),
            "control_increment_cost": float("nan"),
            "control_setpoint_cost": float("nan"),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default=str(HERE.parent / "data" / "A_cstr_separator_C_True.npy"))
    parser.add_argument("--no-block-diag", action="store_true")
    args = parser.parse_args()
    data = Path(args.a).parent
    stem = Path(args.a).name.replace("A_", "")
    A = np.load(args.a)
    B = np.load(data / f"B_{stem}")
    C = np.load(data / f"C_{stem}")
    out = closed_loop_of(A, B, C, use_block_diag=not args.no_block_diag, quiet=False)
    print(out)
