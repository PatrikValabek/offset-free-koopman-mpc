"""Closed-loop weight sweep: original Qy/Qu, Qy×10, Qu×10.

Linear MPCs (N4SID, CT, T2D2, T3D3) share a 3-process pool; NMPC uses a
separate 3-process pool. Both pools use spawn so Gurobi / IPOPT objects are
never forked or shared. Each run logs the closed-loop objective, timings, and
the trajectories that the notebooks plot.

Existing notebook objectives (original weights):
  N4SID 201.63337681139214
  CT    174.87413990534156
  T2D2  162.33210881761514
  T3D3  160.93149415976228
  NMPC  145.62960989639097
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DATA_DIR = HERE.parent / "data"
SRC = REPO_ROOT / "src"

NOTEBOOK_OF = {
    "N4SID": 201.63337681139214,
    "CT": 174.87413990534156,
    "T2D2": 162.33210881761514,
    "T3D3": 160.93149415976228,
    "NMPC": 145.62960989639097,
}
LINEAR_CONTROLLERS = ("N4SID", "CT", "T2D2", "T3D3")
SETUP_NAMES = ("original", "qy_x10", "qy_x5", "qy_x2", "qy_x0.5", "qu_x10", "qu_x5")
TAYLOR_PARAMS = {"encoder_depth": 2, "width_mult": 1.0, "nonlin": "elu"}


def _ensure_paths():
    if SRC.as_posix() not in sys.path:
        sys.path.append(SRC.as_posix())
    if HERE.as_posix() not in sys.path:
        sys.path.append(HERE.as_posix())
    os.chdir(HERE)
    os.environ["SIM_SETUP_PATH"] = str(HERE / "sim_setup.pkl")


def isolate_worker(job_id: str, tmp_root: str):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    tmp = Path(tmp_root) / f"proc_{os.getpid()}_{job_id}"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    _ensure_paths()
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def apply_setup(base: dict, name: str) -> dict:
    loaded = copy.deepcopy(base)
    if name == "original":
        pass
    elif name.startswith("qy_x"):
        loaded["Qy"] = float(name[4:]) * np.asarray(base["Qy"], dtype=float)
    elif name.startswith("qu_x"):
        loaded["Qu"] = float(name[4:]) * np.asarray(base["Qu"], dtype=float)
    else:
        raise ValueError(f"unknown setup {name!r}")
    loaded["setup_name"] = name
    return loaded


def _solver_time(problem) -> float:
    st = getattr(problem, "solver_stats", None)
    if st is None:
        return float("nan")
    t = getattr(st, "solve_time", None)
    return float(t) if t is not None else float("nan")


def _time_solve(fn, problem=None):
    t0 = time.perf_counter()
    out = fn()
    wall = time.perf_counter() - t0
    return out, wall, _solver_time(problem)


def accumulate_of(y_sim, u_sim, us_sim, reference, Qy, Qu, Qdu, sim_time):
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
    }


def _nanstat(x, fn):
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0 or np.all(~np.isfinite(x)):
        return float("nan")
    return float(fn(x))


def _timing_summary(mpc_wall, mpc_solver, te_wall, wall_s, sim_time):
    mpc_wall = np.asarray(mpc_wall, dtype=float)
    mpc_solver = np.asarray(mpc_solver, dtype=float)
    te_wall = np.asarray(te_wall, dtype=float)
    return {
        "wall_s": float(wall_s),
        "mpc_wall_sum_s": float(np.nansum(mpc_wall)),
        "mpc_wall_mean_s": _nanstat(mpc_wall, np.nanmean),
        "mpc_wall_max_s": _nanstat(mpc_wall, np.nanmax),
        "mpc_solver_sum_s": float(np.nansum(mpc_solver)),
        "mpc_solver_mean_s": _nanstat(mpc_solver, np.nanmean),
        "mpc_solver_max_s": _nanstat(mpc_solver, np.nanmax),
        "target_wall_sum_s": float(np.nansum(te_wall)),
        "n_steps": int(sim_time),
        "mpc_wall_s": mpc_wall,
        "mpc_solver_s": mpc_solver,
        "target_wall_s": te_wall,
    }


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_run(out_dir: Path, summary: dict, traj: dict | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in summary.items() if k not in ("mpc_wall_s", "mpc_solver_s", "target_wall_s")}
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(slim), indent=2))
    if traj is not None:
        np.savez_compressed(out_dir / "trajectories.npz", **traj)


def _block_diagonalize(A, B, C):
    import helper
    from numpy.linalg import inv

    T_real, _ = helper.real_block_diagonalize(A)
    if not np.isfinite(T_real).all() or np.linalg.cond(T_real) > 1e10:
        raise RuntimeError("block-diagonalizing transformation is ill-conditioned")
    A_t = inv(T_real) @ A @ T_real
    B_t = inv(T_real) @ B
    C_t = C @ T_real
    return A_t, B_t, C_t, T_real


def run_linear_c(controller: str, loaded: dict):
    import helper
    import plant_inference

    stem = {
        "N4SID": "cstr_separator_sippy_hpo",
        "CT": "cstr_separator_C_cl_hpo",
    }[controller]
    A = np.load(DATA_DIR / f"A_{stem}.npy").astype(float)
    B = np.load(DATA_DIR / f"B_{stem}.npy").astype(float)
    C = np.load(DATA_DIR / f"C_{stem}.npy").astype(float)
    A, B, C, _ = _block_diagonalize(A, B, C)

    nz, nu = B.shape
    ny = C.shape[0]
    nd = ny
    Ts = float(loaded["Ts"])
    sim_time = int(loaded["sim_time"])
    session = plant_inference.PlantSession(loaded["x_start"], Ts=Ts)
    scaler, scalerU = session.scaler, session.scalerU

    y_start = np.asarray(loaded["y_start"]).reshape(1, -1)
    reference = np.asarray(loaded["reference"], dtype=float)
    u_sp = np.asarray(loaded["reference_u"], dtype=float).reshape(-1)
    u_previous = np.asarray(loaded["u_previous"], dtype=float).reshape(-1)
    qd_scale = 10.0 if controller == "N4SID" else 1.0

    z_est_ = np.hstack(((np.linalg.pinv(C) @ y_start.T).T, np.zeros((1, nd))))
    P0 = np.eye(nz + nd) * loaded["P0"]
    Q = np.block(
        [
            [np.eye(nz) * loaded["Q"], np.zeros((nz, nd))],
            [np.zeros((nd, nz)), np.eye(nd) * loaded["Qd"] * qd_scale],
        ]
    )
    R = np.eye(ny) * loaded["R"]
    Cd = np.eye(ny)
    Bd = np.zeros((nz, nd))
    A_ = np.block([[A, Bd], [np.zeros((nd, nz)), np.eye(nd)]])
    B_ = np.vstack([B, np.zeros((nd, nu))])
    C_ = np.hstack([C, Cd])
    kf = helper.KF(A_, B_, C_, z_est_, P0, Q, R)
    target = helper.TargetEstimation(A, B, C, loaded["Qy"], loaded["Qu_te"], Bd, Cd)
    z_s, y_s, u_s = target.get_target(z_est_[:, nz:], reference[:, 0], u_sp)
    Qx = C.T @ loaded["Qy"] @ C + 2e-8 * np.eye(nz)
    mpc = helper.MPC(A, B, C, loaded["Qy"], loaded["Qu"], loaded["Qdu"], Bd, Cd)
    mpc.build_problem(Qx)
    _ = mpc.get_u_optimal(z_est_[:, :nz], z_est_[:, nz:], u_s, u_previous, z_s)

    z_sim = np.zeros((nz + nd, sim_time + 1))
    y_sim = np.zeros((ny, sim_time + 1))
    u_sim = np.zeros((nu, sim_time))
    us_sim = np.zeros((nu, sim_time))
    ys_sim = np.zeros((ny, sim_time + 1))
    zs_sim = np.zeros((nz, sim_time + 1))
    y_sim_ns = np.zeros((ny, sim_time + 1))
    y_true_ns = np.zeros((ny, sim_time + 1))
    u_sim_ns = np.zeros((nu, sim_time))
    x_sim = np.zeros((9, sim_time + 1))
    z_sim[:, 0] = z_est_.flatten()
    y_sim[:, 0] = y_start.flatten()
    y_sim_ns[:, 0] = np.asarray(loaded["y_start_ns"]).flatten()
    y_true_ns[:, 0] = session.measure_ns()
    x_sim[:, 0] = session.x.copy()
    ys_sim[:, 0] = y_s
    zs_sim[:, 0] = z_s
    u_prev = np.asarray(u_s).reshape(-1)
    dist_by_k = {int(d["k"]): d for d in loaded["disturbances"]}
    noise_sigma = np.asarray(loaded["noise_sigma"], dtype=float)
    rng = np.random.default_rng(0)
    mpc_wall = np.zeros(sim_time)
    mpc_solver = np.zeros(sim_time)
    te_wall = np.zeros(sim_time)

    for k in range(sim_time):
        if k in dist_by_k:
            d = dist_by_k[k]
            session.apply_disturbance(d["attr"], d["value"])
        (zs_sim[:, k], ys_sim[:, k], us_sim[:, k]), te_wall[k], _ = _time_solve(
            lambda: target.get_target(z_sim[nz:, k], reference[:, k], u_sp),
            problem=target.te,
        )
        u_opt, mpc_wall[k], mpc_solver[k] = _time_solve(
            lambda: mpc.get_u_optimal(z_sim[:nz, k], z_sim[nz:, k], us_sim[:, k], u_prev, zs_sim[:, k]),
            problem=mpc.mpc,
        )
        u_sim[:, k] = u_opt
        u_sim_ns[:, k] = scalerU.inverse_transform(u_sim[:, k].reshape(1, -1))[0]
        session.y_plus(u_sim[:, k])
        y_true_ns[:, k + 1] = session.measure_ns()
        x_sim[:, k + 1] = session.x.copy()
        y_meas_ns = y_true_ns[:, k + 1] + rng.normal(0.0, noise_sigma)
        y_sim[:, k + 1] = scaler.transform(y_meas_ns.reshape(1, -1))[0]
        y_sim_ns[:, k + 1] = y_meas_ns
        z_sim[:, k + 1] = kf.step(u_sim[:, k], y_sim[:, k + 1]).flatten()
        u_prev = u_sim[:, k]

    y_hat = C @ z_sim[:nz] + z_sim[nz:]
    return _pack_linear_result(
        loaded, session, C, z_sim, y_sim, u_sim, us_sim, ys_sim, zs_sim,
        y_sim_ns, y_true_ns, u_sim_ns, x_sim, y_hat, mpc_wall, mpc_solver, te_wall,
    )


def run_taylor(controller: str, loaded: dict):
    import torch
    import helper
    import plant_inference
    from numpy.linalg import inv
    from taylor_closed_loop import rebuild_problem

    stem = "cstr_separator_C_noc_hpo"
    A = np.load(DATA_DIR / f"A_{stem}.npy").astype(float)
    B = np.load(DATA_DIR / f"B_{stem}.npy").astype(float)
    C_ls = np.load(DATA_DIR / f"C_{stem}.npy").astype(float)
    nz, nu = B.shape
    ny = C_ls.shape[0]
    nd = ny
    params = dict(TAYLOR_PARAMS)
    params["nz"] = nz
    state = torch.load(DATA_DIR / f"model_{stem}.pth", map_location="cpu")
    problem = rebuild_problem(params, state, ny, nu)
    A, B, C, T_real = _block_diagonalize(A, B, C_ls)

    def get_x(y):
        with torch.no_grad():
            x = problem.nodes[0]({"Y0": torch.from_numpy(np.asarray(y, dtype=np.float32))})
        return x["x"].detach().numpy().reshape(1, -1).T

    def get_y(x):
        with torch.no_grad():
            y = problem.nodes[4]({"x": torch.from_numpy(np.asarray(x, dtype=np.float32).T)})
        return y["yhat"].detach().numpy().reshape(1, -1)

    def jacobian_at(z):
        x_lp = torch.from_numpy((T_real @ np.asarray(z, dtype=float).reshape(-1)).astype(np.float32))
        return helper.evaluate_jacobian(problem.nodes[4], x_lp) @ T_real

    Ts = float(loaded["Ts"])
    sim_time = int(loaded["sim_time"])
    session = plant_inference.PlantSession(loaded["x_start"], Ts=Ts)
    scaler, scalerU = session.scaler, session.scalerU
    y_start = np.asarray(loaded["y_start"]).reshape(1, -1)
    reference = np.asarray(loaded["reference"], dtype=float)
    u_sp = np.asarray(loaded["reference_u"], dtype=float).reshape(-1)
    u_previous = np.asarray(loaded["u_previous"], dtype=float).reshape(-1)

    z_est_ = np.hstack(((inv(T_real) @ get_x(y_start)).T, np.zeros((1, nd))))
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
    tvkf = helper.TVKF(A_, B_, C_, z_est_, P0, Q, R)
    J = jacobian_at(z_est_[0, :nz])
    y_lp = get_y(T_real @ z_est_[0, :nz])
    target = helper.TaylorTargetEstimation(A, B, loaded["Qy"], loaded["Qu_te"], Bd, Cd)
    z_s, y_s, u_s = target.get_target(z_est_[:, nz:], reference[:, 0], u_sp, y_lp, z_est_[0, :nz], J)
    mode = controller.lower()
    if mode == "t2d2":
        J = jacobian_at(z_s)
        y_lp_init = get_y(T_real @ z_s)
        z_k_init = z_s
    else:
        y_lp_init = y_lp
        z_k_init = z_est_[0, :nz]
    Qz_psd = J.T @ loaded["Qy"] @ J + 1e-8 * np.eye(nz)
    mpc = helper.TaylorMPC(A, B, loaded["Qy"], loaded["Qu"], loaded["Qdu"], Bd, Cd)
    mpc.build_problem()
    _ = mpc.get_u_optimal(
        z_est_[0, :nz], z_est_[:, nz:], u_previous, z_s, u_s, y_lp_init, z_k_init, J, Qz_psd
    )

    z_sim = np.zeros((nz + nd, sim_time + 1))
    y_sim = np.zeros((ny, sim_time + 1))
    u_sim = np.zeros((nu, sim_time))
    us_sim = np.zeros((nu, sim_time))
    ys_sim = np.zeros((ny, sim_time + 1))
    zs_sim = np.zeros((nz, sim_time + 1))
    y_sim_ns = np.zeros((ny, sim_time + 1))
    y_true_ns = np.zeros((ny, sim_time + 1))
    u_sim_ns = np.zeros((nu, sim_time))
    x_sim = np.zeros((9, sim_time + 1))
    z_sim[:, 0] = z_est_.flatten()
    y_sim[:, 0] = y_start.flatten()
    y_sim_ns[:, 0] = np.asarray(loaded["y_start_ns"]).flatten()
    y_true_ns[:, 0] = session.measure_ns()
    x_sim[:, 0] = session.x.copy()
    ys_sim[:, 0] = y_s
    zs_sim[:, 0] = z_s
    u_prev = np.asarray(u_s).reshape(-1)
    Qy = loaded["Qy"]
    dist_by_k = {int(d["k"]): d for d in loaded["disturbances"]}
    noise_sigma = np.asarray(loaded["noise_sigma"], dtype=float)
    rng = np.random.default_rng(0)
    mpc_wall = np.zeros(sim_time)
    mpc_solver = np.zeros(sim_time)
    te_wall = np.zeros(sim_time)
    if mode == "t2d2":
        J = jacobian_at(zs_sim[:, 0])
    else:
        J = jacobian_at(z_sim[:nz, 0])

    for k in range(sim_time):
        if k in dist_by_k:
            d = dist_by_k[k]
            session.apply_disturbance(d["attr"], d["value"])
        if mode == "t2d2":
            idx_prev = max(k - 1, 0)
            y_lp = get_y(T_real @ zs_sim[:, idx_prev])
            z_lin = zs_sim[:, idx_prev]
        else:
            J = jacobian_at(z_sim[:nz, k])
            y_lp = get_y(T_real @ z_sim[:nz, k])
            z_lin = z_sim[:nz, k]
        (zs_sim[:, k], ys_sim[:, k], us_sim[:, k]), te_wall[k], _ = _time_solve(
            lambda: target.get_target(
                z_sim[nz:, k], reference[:, k], u_sp, y_lp, z_lin, J
            ),
            problem=target.te,
        )
        Qz_psd = J.T @ Qy @ J + 1e-8 * np.eye(nz)
        z_ref = zs_sim[:, k]
        us_k = us_sim[:, k]
        u_opt, mpc_wall[k], mpc_solver[k] = _time_solve(
            lambda: mpc.get_u_optimal(
                z_sim[:nz, k], z_sim[nz:, k], u_prev, z_ref, us_k, y_lp, z_lin, J, Qz_psd
            ),
            problem=mpc.mpc,
        )
        u_sim[:, k] = u_opt
        u_sim_ns[:, k] = scalerU.inverse_transform(u_sim[:, k].reshape(1, -1))[0]
        session.y_plus(u_sim[:, k])
        y_true_ns[:, k + 1] = session.measure_ns()
        x_sim[:, k + 1] = session.x.copy()
        y_meas_ns = y_true_ns[:, k + 1] + rng.normal(0.0, noise_sigma)
        y_sim[:, k + 1] = scaler.transform(y_meas_ns.reshape(1, -1))[0]
        y_sim_ns[:, k + 1] = y_meas_ns
        if mode == "t2d2":
            J = jacobian_at(zs_sim[:, k])
            y_obs = get_y(T_real @ zs_sim[:nz, k])
            z_obs = zs_sim[:nz, k]
        else:
            y_obs = y_lp
            z_obs = z_sim[:nz, k]
        C_k = np.hstack([J, Cd])
        z_sim[:, k + 1] = tvkf.step(u_sim[:, k], y_sim[:, k + 1], y_obs, z_obs, J, C_k).flatten()
        u_prev = u_sim[:, k]

    y_hat = np.zeros((ny, sim_time + 1))
    for k in range(sim_time + 1):
        y_hat[:, k] = get_y(T_real @ z_sim[:nz, k]).ravel() + z_sim[nz:, k]
    return _pack_linear_result(
        loaded, session, C_ls, z_sim, y_sim, u_sim, us_sim, ys_sim, zs_sim,
        y_sim_ns, y_true_ns, u_sim_ns, x_sim, y_hat, mpc_wall, mpc_solver, te_wall,
    )


def _pack_linear_result(
    loaded, session, C, z_sim, y_sim, u_sim, us_sim, ys_sim, zs_sim,
    y_sim_ns, y_true_ns, u_sim_ns, x_sim, y_hat, mpc_wall, mpc_solver, te_wall,
):
    scaler, scalerU = session.scaler, session.scalerU
    sim_time = int(loaded["sim_time"])
    reference = np.asarray(loaded["reference"], dtype=float)
    of = accumulate_of(y_sim, u_sim, us_sim, reference, loaded["Qy"], loaded["Qu"], loaded["Qdu"], sim_time)
    of_orig = accumulate_of(
        y_sim, u_sim, us_sim, reference,
        loaded["_Qy0"], loaded["_Qu0"], loaded["Qdu"], sim_time,
    )
    ys_sim_ns = scaler.inverse_transform(ys_sim.T).T
    us_sim_ns = scalerU.inverse_transform(us_sim.T).T
    y_hat_ns = scaler.inverse_transform(y_hat.T).T
    u_sp = np.asarray(loaded["reference_u"], dtype=float).reshape(-1)
    nz = z_sim.shape[0] - y_sim.shape[0]
    traj = {
        "y_true_ns": y_true_ns,
        "y_sim_ns": y_sim_ns,
        "y_hat_ns": y_hat_ns,
        "reference_ns": np.asarray(loaded["reference_ns"], dtype=float),
        "ys_sim_ns": ys_sim_ns,
        "u_sim_ns": u_sim_ns,
        "us_sim_ns": us_sim_ns,
        "u_sp_ns": scalerU.inverse_transform(u_sp.reshape(1, -1))[0],
        "u_min_ns": np.asarray(loaded["u_min_ns"], dtype=float),
        "u_max_ns": np.asarray(loaded["u_max_ns"], dtype=float),
        "d_est": z_sim[nz:],
        "x_sim": x_sim,
        "y_sim": y_sim,
        "u_sim": u_sim,
        "us_sim": us_sim,
        "ys_sim": ys_sim,
        "zs_sim": zs_sim,
        "z_sim": z_sim,
        "y_hat": y_hat,
        "mpc_wall_s": mpc_wall,
        "mpc_solver_s": mpc_solver,
        "target_wall_s": te_wall,
    }
    timing = _timing_summary(mpc_wall, mpc_solver, te_wall, 0.0, sim_time)
    return of, of_orig, traj, timing, u_sim


def run_nmpc(loaded: dict):
    import nmpc_cstr
    import plant_inference

    scaler_dummy = plant_inference.PlantSession(loaded["x_start"], Ts=float(loaded["Ts"]))
    scaler, scalerU = scaler_dummy.scaler, scaler_dummy.scalerU
    Ts = float(loaded["Ts"])
    sim_time = int(loaded["sim_time"])
    ny, nu, nx = len(loaded["y_names"]), len(loaded["u_names"]), nmpc_cstr.NX
    session = plant_inference.PlantSession(loaded["x_start"], Ts=Ts)
    feed = nmpc_cstr.FeedParams()
    current_ref = nmpc_cstr.CurrentRef()
    target = nmpc_cstr.TargetSelector(loaded, scaler, scalerU, feed, Ts=Ts)
    model = nmpc_cstr.build_model(Ts=Ts)
    mpc = nmpc_cstr.build_mpc(model, loaded, scaler, scalerU, feed, current_ref)
    x0 = np.asarray(loaded["x_start"], dtype=float).reshape(-1)
    u0 = np.asarray(loaded["u_previous_ns"], dtype=float).reshape(-1)
    P0, Q_ekf, R_ekf = nmpc_cstr.default_ekf_covariances(loaded)
    ekf = nmpc_cstr.DiscreteEKF(x0, P0, Q_ekf, R_ekf, feed, Ts=Ts)
    mpc.x0 = x0
    mpc.u0 = u0
    mpc.set_initial_guess()

    y_sim = np.zeros((ny, sim_time + 1))
    y_sim_ns = np.zeros((ny, sim_time + 1))
    y_true_ns = np.zeros((ny, sim_time + 1))
    u_sim = np.zeros((nu, sim_time))
    u_sim_ns = np.zeros((nu, sim_time))
    x_sim = np.zeros((nx, sim_time + 1))
    x_hat = np.zeros((nx, sim_time + 1))
    ys_sim = np.zeros((ny, sim_time))
    us_sim = np.zeros((nu, sim_time))
    xs_sim = np.zeros((nx, sim_time))
    y_true_ns[:, 0] = session.measure_ns()
    y_sim_ns[:, 0] = y_true_ns[:, 0]
    y_sim[:, 0] = scaler.transform(y_sim_ns[:, 0].reshape(1, -1))[0]
    x_sim[:, 0] = session.x.copy()
    x_hat[:, 0] = ekf.x.copy()
    dist_by_k = {int(d["k"]): d for d in loaded["disturbances"]}
    noise_sigma = np.asarray(loaded["noise_sigma"], dtype=float)
    reference_ns = np.asarray(loaded["reference_ns"], dtype=float)
    u_sp_ns = np.asarray(loaded["reference_u_ns"], dtype=float).reshape(-1)
    rng = np.random.default_rng(0)
    u_prev_ns = u0.copy()
    mpc_wall = np.zeros(sim_time)
    mpc_solver = np.full(sim_time, np.nan)
    te_wall = np.zeros(sim_time)

    for k in range(sim_time):
        if k in dist_by_k:
            d = dist_by_k[k]
            session.apply_disturbance(d["attr"], d["value"])
            feed.set(d["attr"], d["value"])
        (xs_sim[:, k], ys_sim[:, k], us_sim[:, k]), te_wall[k], _ = _time_solve(
            lambda: target.get_target(reference_ns[:, k], u_sp_ns, x_guess=ekf.x, u_guess=u_prev_ns)
        )
        current_ref.set(ys_sim[:, k], us_sim[:, k])
        t0 = time.perf_counter()
        u_opt = np.asarray(mpc.make_step(ekf.x), dtype=float).reshape(-1)
        mpc_wall[k] = time.perf_counter() - t0
        u_sim_ns[:, k] = u_opt
        u_sim[:, k] = scalerU.transform(u_opt.reshape(1, -1))[0]
        session.y_plus(u_sim[:, k])
        y_true_ns[:, k + 1] = session.measure_ns()
        x_sim[:, k + 1] = session.x.copy()
        y_meas_ns = y_true_ns[:, k + 1] + rng.normal(0.0, noise_sigma)
        y_sim_ns[:, k + 1] = y_meas_ns
        y_sim[:, k + 1] = scaler.transform(y_meas_ns.reshape(1, -1))[0]
        x_hat[:, k + 1] = ekf.step(u_opt, y_meas_ns)
        u_prev_ns = u_opt

    us_sim_scaled = scalerU.transform(us_sim.T).T
    reference = np.asarray(loaded["reference"], dtype=float)
    of = accumulate_of(y_sim, u_sim, us_sim_scaled, reference, loaded["Qy"], loaded["Qu"], loaded["Qdu"], sim_time)
    of_orig = accumulate_of(
        y_sim, u_sim, us_sim_scaled, reference, loaded["_Qy0"], loaded["_Qu0"], loaded["Qdu"], sim_time
    )
    y_hat_ns = x_hat[[2, 5, 8, 7], :]
    traj = {
        "y_true_ns": y_true_ns,
        "y_sim_ns": y_sim_ns,
        "y_hat_ns": y_hat_ns,
        "reference_ns": reference_ns,
        "ys_sim_ns": ys_sim,
        "u_sim_ns": u_sim_ns,
        "us_sim_ns": us_sim,
        "u_sp_ns": u_sp_ns,
        "u_min_ns": np.asarray(loaded["u_min_ns"], dtype=float),
        "u_max_ns": np.asarray(loaded["u_max_ns"], dtype=float),
        "x_sim": x_sim,
        "x_hat": x_hat,
        "xs_sim": xs_sim,
        "y_sim": y_sim,
        "u_sim": u_sim,
        "us_sim": us_sim_scaled,
        "ys_sim": scaler.transform(ys_sim.T).T,
        "mpc_wall_s": mpc_wall,
        "target_wall_s": te_wall,
    }
    timing = _timing_summary(mpc_wall, mpc_solver, te_wall, 0.0, sim_time)
    return of, of_orig, traj, timing, u_sim


def worker(payload: dict) -> dict:
    job_id = payload["job_id"]
    isolate_worker(job_id, payload["tmp_root"])
    controller = payload["controller"]
    setup_name = payload["setup_name"]
    t_wall0 = time.perf_counter()
    result = {
        "controller": controller,
        "setup_name": setup_name,
        "job_id": job_id,
        "pid": os.getpid(),
        "error": None,
    }
    try:
        base = joblib.load(HERE / "sim_setup.pkl")
        if payload.get("sim_time") is not None:
            base["sim_time"] = int(payload["sim_time"])
        loaded = apply_setup(base, setup_name)
        loaded["_Qy0"] = np.asarray(base["Qy"], dtype=float)
        loaded["_Qu0"] = np.asarray(base["Qu"], dtype=float)
        print(f"[start] {controller} {setup_name} pid={os.getpid()}", flush=True)
        if controller in ("N4SID", "CT"):
            of, of_orig, traj, timing, u_sim = run_linear_c(controller, loaded)
        elif controller in ("T2D2", "T3D3"):
            of, of_orig, traj, timing, u_sim = run_taylor(controller, loaded)
        elif controller == "NMPC":
            of, of_orig, traj, timing, u_sim = run_nmpc(loaded)
        else:
            raise ValueError(controller)
        timing["wall_s"] = time.perf_counter() - t_wall0
        summary = {
            **of,
            "objective_original_weights": of_orig["objective"],
            **{k: v for k, v in timing.items() if not k.endswith("_s") or k in (
                "wall_s", "mpc_wall_sum_s", "mpc_wall_mean_s", "mpc_wall_max_s",
                "mpc_solver_sum_s", "mpc_solver_mean_s", "mpc_solver_max_s",
                "target_wall_sum_s",
            )},
            "controller": controller,
            "setup_name": setup_name,
            "Qy_diag": np.diag(np.asarray(loaded["Qy"])).tolist(),
            "Qu_diag": np.diag(np.asarray(loaded["Qu"])).tolist(),
            "sim_time": int(loaded["sim_time"]),
            "notebook_of": NOTEBOOK_OF.get(controller) if setup_name == "original" else None,
            "pid": os.getpid(),
        }
        out_dir = Path(payload["out_dir"]) / f"{controller}_{setup_name}"
        if payload.get("tag"):
            out_dir = Path(payload["out_dir"]) / f"{controller}_{setup_name}_{payload['tag']}"
        if payload.get("save", True):
            traj_save = dict(traj)
            traj_save["mpc_wall_s"] = timing["mpc_wall_s"]
            traj_save["mpc_solver_s"] = timing["mpc_solver_s"]
            traj_save["target_wall_s"] = timing["target_wall_s"]
            save_run(out_dir, summary, traj_save)
        result.update(summary)
        result["u_sim"] = np.asarray(u_sim, dtype=float)
        print(
            f"[done] {controller} {setup_name} OF={of['objective']:.6g} "
            f"wall={timing['wall_s']:.1f}s mpc_mean={timing['mpc_wall_mean_s']:.4f}s",
            flush=True,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["wall_s"] = time.perf_counter() - t_wall0
        print(f"[fail] {controller} {setup_name} {result['error']}", flush=True)
        print(result["traceback"], flush=True)
        if payload.get("save", True):
            err_dir = Path(payload["out_dir"]) / f"{controller}_{setup_name}"
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "summary.json").write_text(json.dumps(_json_safe(result), indent=2))
    return result


def _submit(ex, payloads):
    return [ex.submit(worker, p) for p in payloads]


def _payload(controller, setup_name, out_dir, tmp_root, sim_time=None, save=True, tag=""):
    return {
        "controller": controller,
        "setup_name": setup_name,
        "out_dir": str(out_dir),
        "tmp_root": str(tmp_root),
        "sim_time": sim_time,
        "save": save,
        "tag": tag,
        "job_id": f"{controller}_{setup_name}_{tag or 'run'}",
    }


def _finite(x):
    return x is not None and np.isfinite(x)


def run_smoke(out_root: Path, linear_steps: int = 8, nmpc_steps: int = 3) -> bool:
    print("=" * 72, flush=True)
    print("SMOKE: spawn processes, private Gurobi env / IPOPT, no shared solvers", flush=True)
    tmp_root = out_root / "_tmp"
    smoke_dir = out_root / "_smoke"
    ctx = get_context("spawn")
    ok = True
    notes = []

    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
        futs = _submit(
            ex,
            [
                _payload("N4SID", "original", smoke_dir, tmp_root, sim_time=linear_steps, save=True, tag="a"),
                _payload("N4SID", "original", smoke_dir, tmp_root, sim_time=linear_steps, save=True, tag="b"),
            ],
        )
        lin = [f.result() for f in futs]
    for r in lin:
        if r.get("error") or not _finite(r.get("objective")):
            ok = False
            notes.append(f"linear copy failed: {r.get('error')}")
    if ok and len(lin) == 2:
        du = np.max(np.abs(lin[0]["u_sim"] - lin[1]["u_sim"]))
        dJ = abs(lin[0]["objective"] - lin[1]["objective"])
        notes.append(f"N4SID twin max|Δu|={du:.3e} ΔJ={dJ:.3e}")
        if du > 1e-3 or dJ > 1e-3:
            ok = False
            notes.append("N4SID parallel copies diverged")

    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
        futs = _submit(
            ex,
            [
                _payload("NMPC", "original", smoke_dir, tmp_root, sim_time=nmpc_steps, save=True, tag="a"),
                _payload("NMPC", "original", smoke_dir, tmp_root, sim_time=nmpc_steps, save=True, tag="b"),
            ],
        )
        nl = [f.result() for f in futs]
    for r in nl:
        if r.get("error") or not _finite(r.get("objective")):
            ok = False
            notes.append(f"NMPC copy failed: {r.get('error')}")
    if len(nl) == 2 and all(_finite(r.get("objective")) for r in nl) and all(not r.get("error") for r in nl):
        du = np.max(np.abs(nl[0]["u_sim"] - nl[1]["u_sim"]))
        dJ = abs(nl[0]["objective"] - nl[1]["objective"])
        rel = dJ / max(abs(nl[0]["objective"]), 1e-9)
        notes.append(f"NMPC twin max|Δu|={du:.3e} ΔJ={dJ:.3e} rel={rel:.3e}")
        if du > 1e-2 or rel > 1e-3:
            ok = False
            notes.append("NMPC parallel copies diverged")

    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as lin_ex, ProcessPoolExecutor(
        max_workers=1, mp_context=ctx
    ) as nl_ex:
        f_lin = lin_ex.submit(
            worker, _payload("CT", "original", smoke_dir, tmp_root, sim_time=linear_steps, save=True, tag="mix")
        )
        f_nl = nl_ex.submit(
            worker, _payload("NMPC", "qy_x10", smoke_dir, tmp_root, sim_time=nmpc_steps, save=True, tag="mix")
        )
        mix = [f_lin.result(), f_nl.result()]
    for r in mix:
        if r.get("error") or not _finite(r.get("objective")):
            ok = False
            notes.append(f"mixed linear+NMPC failed: {r.get('controller')} {r.get('error')}")
        else:
            notes.append(f"mixed {r['controller']} OF={r['objective']:.6g}")

    report = {"ok": ok, "notes": notes, "linear": _json_safe([{k: v for k, v in r.items() if k != "u_sim"} for r in lin]),
              "nmpc": _json_safe([{k: v for k, v in r.items() if k != "u_sim"} for r in nl])}
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / "smoke.json").write_text(json.dumps(report, indent=2))
    print("SMOKE notes:", *notes, sep="\n  ", flush=True)
    print("SMOKE", "PASS" if ok else "FAIL", flush=True)
    print("=" * 72, flush=True)
    return ok


def _run_key(r: dict):
    return (r.get("controller"), r.get("setup_name"))


def write_master_summary(out_root: Path, rows: list[dict], merge_existing: bool = True):
    by_key = {}
    if merge_existing:
        existing = out_root / "summary.json"
        if existing.exists():
            try:
                prev = json.loads(existing.read_text())
                for r in prev.get("runs", []):
                    by_key[_run_key(r)] = r
            except Exception:
                pass
    for r in rows:
        slim = {k: v for k, v in r.items() if k != "u_sim"}
        by_key[_run_key(slim)] = slim
    merged = list(by_key.values())
    payload = {
        "notebook_baseline_original": NOTEBOOK_OF,
        "runs": _json_safe(merged),
    }
    (out_root / "summary.json").write_text(json.dumps(payload, indent=2))
    cols = [
        "controller", "setup_name", "objective", "objective_original_weights",
        "state_error_cost", "control_increment_cost", "control_setpoint_cost",
        "wall_s", "mpc_wall_mean_s", "mpc_wall_max_s", "mpc_solver_mean_s", "error",
    ]
    order = {name: i for i, name in enumerate(SETUP_NAMES)}
    ctrl_order = {name: i for i, name in enumerate((*LINEAR_CONTROLLERS, "NMPC"))}
    merged.sort(key=lambda r: (order.get(r.get("setup_name"), 99), ctrl_order.get(r.get("controller"), 99)))
    lines = ["\t".join(cols)]
    for r in merged:
        lines.append("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols))
    (out_root / "summary.tsv").write_text("\n".join(lines) + "\n")


def run_full(out_root: Path, setups: tuple[str, ...] | None = None):
    setups = tuple(setups) if setups else SETUP_NAMES
    tmp_root = out_root / "_tmp"
    ctx = get_context("spawn")
    linear_payloads = [
        _payload(c, s, out_root, tmp_root) for s in setups for c in LINEAR_CONTROLLERS
    ]
    nmpc_payloads = [_payload("NMPC", s, out_root, tmp_root) for s in setups]
    print(
        f"FULL: {len(linear_payloads)} linear jobs (3 CPUs) + {len(nmpc_payloads)} NMPC jobs (3 CPUs)",
        flush=True,
    )
    rows = []
    with ProcessPoolExecutor(max_workers=3, mp_context=ctx) as lin_ex, ProcessPoolExecutor(
        max_workers=3, mp_context=ctx
    ) as nl_ex:
        futs = _submit(lin_ex, linear_payloads) + _submit(nl_ex, nmpc_payloads)
        for fut in as_completed(futs):
            rows.append(fut.result())
            write_master_summary(out_root, rows)
    write_master_summary(out_root, rows)
    n_err = sum(1 for r in rows if r.get("error"))
    print(f"FULL done: {len(rows)} runs, {n_err} errors. Summary: {out_root / 'summary.tsv'}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results" / "weight_sweep"))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--linear-smoke-steps", type=int, default=8)
    parser.add_argument("--nmpc-smoke-steps", type=int, default=3)
    parser.add_argument(
        "--setups",
        nargs="+",
        default=None,
        help="Subset of setups to run (e.g. qy_x5 qu_x5). Default: all.",
    )
    args = parser.parse_args()
    _ensure_paths()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if not (HERE / "sim_setup.pkl").exists():
        raise SystemExit("sim_setup.pkl missing; run python setup.py first")
    if not args.skip_smoke:
        if not run_smoke(out_root, args.linear_smoke_steps, args.nmpc_smoke_steps):
            raise SystemExit("smoke test failed; not starting the full sweep")
    if args.smoke_only:
        return
    run_full(out_root, setups=tuple(args.setups) if args.setups else None)


if __name__ == "__main__":
    main()
