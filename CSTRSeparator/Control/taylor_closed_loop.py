"""Headless T2D2 / T3D3 closed-loop evaluation for the noC Koopman autoencoder.

Mirrors ``T2D2.ipynb`` and ``T3D3.ipynb``: TVKF + TaylorTargetEstimation +
TaylorMPC, always real-block-diagonalized (A, B, C). Each call uses its own
plant instance so parallel trials do not share plant state.
"""

from __future__ import annotations

import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import torch
from numpy.linalg import inv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SRC = REPO_ROOT / "src"
DATA_DIR = HERE.parent / "data"
if SRC.as_posix() not in sys.path:
    sys.path.append(SRC.as_posix())
if HERE.as_posix() not in sys.path:
    sys.path.append(HERE.as_posix())

os.environ.setdefault("SIM_SETUP_PATH", str(HERE / "sim_setup.pkl"))

import helper  # noqa: E402
import models  # noqa: E402


class PlantSession:
    """Private CSTR–separator stepper (not the module-global plant_inference)."""

    def __init__(self, x0, Ts: float = 1.0):
        self.plant = models.CSTRSeparator()
        self.scaler = joblib.load(DATA_DIR / "scaler_cstr_separator.pkl")
        self.scalerU = joblib.load(DATA_DIR / "scalerU_cstr_separator.pkl")
        self.ts = float(Ts)
        self.x = np.asarray(x0, dtype=float).reshape(-1)
        self.reset_disturbances()

    def reset(self, x0, Ts=None):
        if Ts is not None:
            self.ts = float(Ts)
        self.x = np.asarray(x0, dtype=float).reshape(-1)
        self.reset_disturbances()

    def reset_disturbances(self):
        self.plant.T10 = 313.0
        self.plant.T20 = 313.0
        self.plant.xA10 = 1.0
        self.plant.xB10 = 0.0
        self.plant.xA20 = 1.0
        self.plant.xB20 = 0.0

    def apply_disturbance(self, attr: str, value: float):
        setattr(self.plant, attr, float(value))

    def measure_ns(self):
        return self.plant.measure(self.x)

    def y_plus(self, u_scaled):
        u_ns = self.scalerU.inverse_transform(np.asarray(u_scaled, dtype=float).reshape(1, -1))
        self.x = np.asarray(self.plant.step(self.x, u_ns, self.ts), dtype=float).reshape(-1)
        y_ns = self.measure_ns().reshape(1, -1)
        return self.scaler.transform(y_ns)[0]


def block_diagonalize(A, B, C):
    T_real, _ = helper.real_block_diagonalize(A)
    if not np.isfinite(T_real).all() or np.linalg.cond(T_real) > 1e10:
        raise RuntimeError("block-diagonalizing transformation is ill-conditioned")
    A_t = inv(T_real) @ A @ T_real
    B_t = inv(T_real) @ B
    C_t = C @ T_real
    return A_t, B_t, C_t, T_real


def rebuild_problem(params: dict, state_dict, ny: int, nu: int):
    problem, _, _, _ = helper.build_koopman_problem(
        ny=ny,
        nu=nu,
        nz=int(params["nz"]),
        matrix_C=False,
        encoder_depth=int(params["encoder_depth"]),
        width_mult=float(params["width_mult"]),
        nonlin=str(params["nonlin"]),
        loss_weights=helper.LossWeights(),
        nsteps=1,
    )
    problem.load_state_dict(state_dict)
    problem.eval()
    return problem


def _decoder_maps(problem, T_real):
    def get_x(y: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = problem.nodes[0]({"Y0": torch.from_numpy(np.asarray(y, dtype=np.float32))})
        return x["x"].detach().numpy().reshape(1, -1).T

    def get_y(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            y = problem.nodes[4]({"x": torch.from_numpy(np.asarray(x, dtype=np.float32).T)})
        return y["yhat"].detach().numpy().reshape(1, -1)

    def jacobian_at(z: np.ndarray) -> np.ndarray:
        x_lp = torch.from_numpy((T_real @ np.asarray(z, dtype=float).reshape(-1)).astype(np.float32))
        return helper.evaluate_jacobian(problem.nodes[4], x_lp) @ T_real

    return get_x, get_y, jacobian_at


def _accumulate_of(y_sim, u_sim, us_sim, reference, Qy, Qu, Qdu, sim_time):
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


def closed_loop_taylor(
    A: np.ndarray,
    B: np.ndarray,
    C_ls: np.ndarray,
    problem,
    *,
    mode: str,
    sim_setup: dict | None = None,
    quiet: bool = True,
) -> dict:
    """Run T2D2 (linearize at previous target) or T3D3 (linearize at current z)."""
    if mode not in ("t2d2", "t3d3"):
        raise ValueError(f"mode must be t2d2 or t3d3, got {mode!r}")

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C_ls = np.asarray(C_ls, dtype=float)
    A, B, C, T_real = block_diagonalize(A, B, C_ls)
    get_x, get_y, jacobian_at = _decoder_maps(problem, T_real)

    setup_path = os.environ.get("SIM_SETUP_PATH", str(HERE / "sim_setup.pkl"))
    loaded = sim_setup if sim_setup is not None else joblib.load(setup_path)

    nz, nu = B.shape
    ny = C_ls.shape[0]
    nd = ny
    Ts = float(loaded["Ts"])
    sim_time = int(loaded["sim_time"])
    session = PlantSession(loaded["x_start"], Ts=Ts)

    y_start = np.asarray(loaded["y_start"]).reshape(1, -1)
    reference = np.asarray(loaded["reference"], dtype=float)
    u_sp = np.asarray(loaded["reference_u"], dtype=float).reshape(-1)
    u_previous = np.asarray(loaded["u_previous"], dtype=float).reshape(-1)
    scaler = session.scaler

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
    target_estimation = helper.TaylorTargetEstimation(
        A, B, loaded["Qy"], loaded["Qu_te"], Bd, Cd
    )
    z_s, y_s, u_s = target_estimation.get_target(
        z_est_[:, nz:], reference[:, 0], u_sp, y_lp, z_est_[0, :nz], J
    )
    if mode == "t2d2":
        J = jacobian_at(z_s)
    Qz_psd = J.T @ loaded["Qy"] @ J + 1e-8 * np.eye(nz)
    mpc = helper.TaylorMPC(A, B, loaded["Qy"], loaded["Qu"], loaded["Qdu"], Bd, Cd)
    mpc.build_problem()
    z_ref = z_s
    y_lp_init = y_lp if mode == "t3d3" else get_y(T_real @ z_s)
    z_k_init = z_est_[0, :nz] if mode == "t3d3" else z_s
    _ = mpc.get_u_optimal(
        z_est_[0, :nz],
        z_est_[:, nz:],
        u_previous,
        z_ref,
        u_s,
        y_lp_init,
        z_k_init,
        J,
        Qz_psd,
    )

    z_sim = np.zeros((nz + nd, sim_time + 1))
    y_sim = np.zeros((ny, sim_time + 1))
    u_sim = np.zeros((nu, sim_time))
    us_sim = np.zeros((nu, sim_time))
    zs_sim = np.zeros((nz, sim_time + 1))
    z_sim[:, 0] = z_est_.flatten()
    y_sim[:, 0] = y_start.flatten()
    zs_sim[:, 0] = z_s
    u_prev = np.asarray(u_s).reshape(-1)
    Qy = loaded["Qy"]
    dist_by_k = {int(d["k"]): d for d in loaded["disturbances"]}
    noise_sigma = np.asarray(loaded["noise_sigma"], dtype=float)
    rng = np.random.default_rng(0)

    if mode == "t2d2":
        J = jacobian_at(zs_sim[:, 0])
    else:
        J = jacobian_at(z_sim[:nz, 0])

    iterator = range(sim_time)
    if not quiet:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc=f"{mode.upper()} closed-loop", ncols=80)

    for k in iterator:
        if k in dist_by_k:
            d = dist_by_k[k]
            session.apply_disturbance(d["attr"], d["value"])

        if mode == "t2d2":
            idx_prev = max(k - 1, 0)
            y_lp = get_y(T_real @ zs_sim[:, idx_prev])
            zs_sim[:, k], _, us_sim[:, k] = target_estimation.get_target(
                z_sim[nz:, k],
                reference[:, k],
                u_sp,
                y_lp,
                zs_sim[:, idx_prev],
                J,
            )
            z_lin = zs_sim[:, idx_prev]
        else:
            J = jacobian_at(z_sim[:nz, k])
            y_lp = get_y(T_real @ z_sim[:nz, k])
            zs_sim[:, k], _, us_sim[:, k] = target_estimation.get_target(
                z_sim[nz:, k],
                reference[:, k],
                u_sp,
                y_lp,
                z_sim[:nz, k],
                J,
            )
            z_lin = z_sim[:nz, k]

        Qz_psd = J.T @ Qy @ J + 1e-8 * np.eye(nz)
        u_opt = mpc.get_u_optimal(
            z_sim[:nz, k],
            z_sim[nz:, k],
            u_prev,
            zs_sim[:, k],
            us_sim[:, k],
            y_lp,
            z_lin,
            J,
            Qz_psd,
        )
        u_sim[:, k] = u_opt
        session.y_plus(u_sim[:, k])
        y_true_ns = session.measure_ns()
        noise = rng.normal(0.0, noise_sigma)
        y_meas_ns = y_true_ns + noise
        y_sim[:, k + 1] = scaler.transform(y_meas_ns.reshape(1, -1))[0]

        if mode == "t2d2":
            J = jacobian_at(zs_sim[:, k])
            C_k = np.hstack([J, Cd])
            z_sim[:, k + 1] = tvkf.step(
                u_sim[:, k],
                y_sim[:, k + 1],
                get_y(T_real @ zs_sim[:nz, k]),
                zs_sim[:nz, k],
                J,
                C_k,
            ).flatten()
        else:
            C_k = np.hstack([J, Cd])
            z_sim[:, k + 1] = tvkf.step(
                u_sim[:, k],
                y_sim[:, k + 1],
                y_lp,
                z_sim[:nz, k],
                J,
                C_k,
            ).flatten()
        u_prev = u_sim[:, k]

    out = _accumulate_of(
        y_sim, u_sim, us_sim, reference, loaded["Qy"], loaded["Qu"], loaded["Qdu"], sim_time
    )
    out["rho_A"] = float(np.max(np.abs(np.linalg.eigvals(A))))
    out["mode"] = mode
    return out


def evaluate_t2t3(
    A,
    B,
    C,
    params: dict,
    state_dict,
    *,
    parallel: bool = False,
) -> dict:
    """Run T2D2 and T3D3; objective is the sum of the two notebook OFs.

    Sequential by default: Gurobi's default environment is not safe to share
    across two Taylor MPC solves at once (HPO was scoring many 1e6 penalties).
    """
    ny = int(np.asarray(C).shape[0])
    nu = int(np.asarray(B).shape[1])

    def _run(mode: str) -> dict:
        problem = rebuild_problem(params, state_dict, ny, nu)
        return closed_loop_taylor(A, B, C, problem, mode=mode, quiet=True)

    try:
        if parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f2 = pool.submit(_run, "t2d2")
                f3 = pool.submit(_run, "t3d3")
                t2 = f2.result()
                t3 = f3.result()
        else:
            t2 = _run("t2d2")
            t3 = _run("t3d3")
    except Exception as exc:
        print(f"T2T3 eval failed: {exc}", flush=True)
        traceback.print_exc()
        return {
            "objective": 1e6,
            "error": str(exc),
            "t2d2": float("nan"),
            "t3d3": float("nan"),
            "state_error_cost": float("nan"),
            "control_increment_cost": float("nan"),
        }

    return {
        "objective": float(t2["objective"] + t3["objective"]),
        "t2d2": float(t2["objective"]),
        "t3d3": float(t3["objective"]),
        "t2d2_tracking": float(t2["state_error_cost"]),
        "t3d3_tracking": float(t3["state_error_cost"]),
        "t2d2_du": float(t2["control_increment_cost"]),
        "t3d3_du": float(t3["control_increment_cost"]),
        "rho_A": float(t2.get("rho_A", t3.get("rho_A", float("nan")))),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Score T2D2+T3D3 on saved noC weights.")
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Directory with A/B/C_cstr_separator_C_False.npy and the pth.",
    )
    parser.add_argument(
        "--params-json",
        default="",
        help="Architecture JSON (nz, encoder_depth, width_mult, nonlin). "
        "Defaults to example_training10 / noC ident winner.",
    )
    args = parser.parse_args()
    data = Path(args.data_dir)
    A = np.load(data / "A_cstr_separator_C_False.npy")
    B = np.load(data / "B_cstr_separator_C_False.npy")
    C = np.load(data / "C_cstr_separator_C_False.npy")
    state = torch.load(data / "model_cstr_separator_C_False.pth", map_location="cpu")
    if args.params_json:
        params = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
    else:
        params = {
            "nz": int(A.shape[0]),
            "encoder_depth": 2,
            "width_mult": 1.0,
            "nonlin": "elu",
        }
    params["nz"] = int(A.shape[0])
    out = evaluate_t2t3(A, B, C, params, state, parallel=False)
    print(
        f"T2D2+T3D3={out['objective']:.6g}  T2D2={out.get('t2d2')}  "
        f"T3D3={out.get('t3d3')}  rho(A)={out.get('rho_A')}"
    )
    if out.get("error"):
        print("error:", out["error"])
