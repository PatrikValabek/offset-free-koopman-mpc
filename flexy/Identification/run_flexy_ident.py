#!/usr/bin/env python3
"""Run FlexBend N4SID + Koopman C/noC identification and the joint test comparison."""

from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(1)

IDENT_DIR = Path(__file__).resolve().parent
DATA_DIR = IDENT_DIR.parent / "data"
SRC_PATH = IDENT_DIR.parent.parent / "src"
sys.path.insert(0, str(IDENT_DIR))
sys.path.insert(0, str(SRC_PATH))

from data_utils import NSTEPS, TS, load_flexy_splits, save_flexy_npz  # noqa: E402
from helper.koopman import (  # noqa: E402
    LossWeights,
    TrainConfig,
    build_koopman_problem,
    extract_matrices,
    get_data_loaders,
    rollout_predictions,
    train_koopman,
)


def ss_lsim(A, B, C, D, U, x0):
    n = U.shape[1]
    y = np.zeros((C.shape[0], n))
    x = np.asarray(x0, dtype=float).reshape(-1)
    for k in range(n):
        uk = U[:, k]
        y[:, k] = C @ x + D @ uk
        x = A @ x + B @ uk
    return y


def identify_n4sid(train_s: dict) -> None:
    import sippy_unipi.functionsetSIM as fsetSIM
    import sippy_unipi.OLSims_methods as olm
    from sippy_unipi.model import SS_Model

    def _vn_mat_dot(y, yest):
        eps = np.asarray(y).reshape(-1) - np.asarray(yest).reshape(-1)
        return float(np.dot(eps, eps) / max(eps.size, 1))

    fsetSIM.Vn_mat = _vn_mat_dot
    olm.Vn_mat = _vn_mat_dot

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ident = SS_Model._identify(
            train_s["Y"].T.copy(),
            train_s["U"].T.copy(),
            "N4SID",
            3,
            f=20,
            p=20,
            threshold=0.0,
            D_required=False,
            A_stability=True,
            B_recalc=False,
        )
    rho = float(np.max(np.abs(np.linalg.eigvals(ident.A))))
    print(f"N4SID A {ident.A.shape} rho(A)={rho:.4f}")
    np.save(DATA_DIR / "A_flexy_n4sid.npy", ident.A)
    np.save(DATA_DIR / "B_flexy_n4sid.npy", ident.B)
    np.save(DATA_DIR / "C_flexy_n4sid.npy", ident.C)
    np.save(DATA_DIR / "D_flexy_n4sid.npy", ident.D)


def train_one_koopman(
    *,
    matrix_c: bool,
    nz: int,
    nonlin: str,
    loss_weights: LossWeights,
    train_loader,
    dev_loader,
    eval_data,
    train_data,
    scaler,
    ny: int,
    nu: int,
    nsteps: int,
    bs: int,
) -> None:
    tag = "True" if matrix_c else "False"
    print(f"\n=== Koopman matrix_C={matrix_c} nz={nz} ===")
    torch.manual_seed(42)
    np.random.seed(42)
    problem, _, _, _ = build_koopman_problem(
        ny=ny,
        nu=nu,
        nz=nz,
        matrix_C=matrix_c,
        encoder_depth=2,
        width_mult=1.0,
        nonlin=nonlin,
        loss_weights=loss_weights,
        nsteps=nsteps,
    )
    train_cfg = TrainConfig(
        nsteps=nsteps, bs=bs, lr=1e-3, epochs=2000, warmup=100, patience=300
    )
    best_model, _ = train_koopman(
        problem, train_loader, dev_loader, eval_data, train_cfg, verbose=False
    )
    torch.save(best_model, DATA_DIR / f"model_flexy_C_{tag}.pth")
    a, b, c = extract_matrices(problem, matrix_c, train_full_dict=train_data)
    np.save(DATA_DIR / f"A_flexy_C_{tag}.npy", a)
    np.save(DATA_DIR / f"B_flexy_C_{tag}.npy", b)
    np.save(DATA_DIR / f"C_flexy_C_{tag}.npy", c)
    rho = float(np.max(np.abs(np.linalg.eigvals(a))))
    print(f"saved model_flexy_C_{tag}.pth A {a.shape} rho(A)={rho:.4f}")


def load_deep_koopman():
    state = torch.load(
        DATA_DIR / "model_flexy_C_False.pth", map_location="cpu", weights_only=False
    )
    nz = int(state["nodes.3.nodes.0.callable.K.weight"].shape[0])
    problem, _, _, _ = build_koopman_problem(
        ny=1,
        nu=1,
        nz=nz,
        matrix_C=False,
        encoder_depth=2,
        width_mult=1.0,
        nonlin="elu",
        loss_weights=LossWeights(),
        nsteps=4,
    )
    problem.load_state_dict(state)
    problem.eval()
    return problem


def rollout_linear(a, b, c, d, y, u, scaler, scaler_u):
    y_s = scaler.transform(y)
    u_s = scaler_u.transform(u)
    x0 = np.linalg.pinv(c) @ y_s[:1].T
    yhat_s = ss_lsim(a, b, c, d, u_s.T, x0)
    return scaler.inverse_transform(yhat_s.T)


def rollout_dk(problem, y, u, scaler, scaler_u):
    y_s = torch.tensor(scaler.transform(y)[None], dtype=torch.float32)
    u_s = torch.tensor(scaler_u.transform(u)[None], dtype=torch.float32)
    data = {"Y": y_s, "Y0": y_s[:, 0:1, :], "U": u_s}
    true_s, pred_s = rollout_predictions(problem, data)
    return scaler.inverse_transform(true_s), scaler.inverse_transform(pred_s)


def compare(test_exps, scaler, scaler_u) -> None:
    a_c = np.load(DATA_DIR / "A_flexy_C_True.npy")
    b_c = np.load(DATA_DIR / "B_flexy_C_True.npy")
    c_c = np.load(DATA_DIR / "C_flexy_C_True.npy")
    d_c = np.zeros((1, 1))
    a_n = np.load(DATA_DIR / "A_flexy_n4sid.npy")
    b_n = np.load(DATA_DIR / "B_flexy_n4sid.npy")
    c_n = np.load(DATA_DIR / "C_flexy_n4sid.npy")
    d_n = np.load(DATA_DIR / "D_flexy_n4sid.npy")
    problem_dk = load_deep_koopman()

    rows = []
    fig, axs = plt.subplots(len(test_exps), 1, figsize=(10, 3.2 * len(test_exps)))
    if len(test_exps) == 1:
        axs = [axs]
    for ax, exp in zip(axs, test_exps):
        y, u = exp["Y"], exp["U"]
        plant_dk, pred_dk = rollout_dk(problem_dk, y, u, scaler, scaler_u)
        pred_c = rollout_linear(a_c, b_c, c_c, d_c, y, u, scaler, scaler_u)
        pred_n = rollout_linear(a_n, b_n, c_n, d_n, y, u, scaler, scaler_u)
        n = min(plant_dk.shape[0], pred_c.shape[0] - 1, pred_n.shape[0] - 1, y.shape[0] - 1)
        plant = y[1 : 1 + n, 0]
        pred_dk_y = pred_dk[:n, 0]
        pred_c_y = pred_c[1 : 1 + n, 0]
        pred_n_y = pred_n[1 : 1 + n, 0]
        t = np.arange(n) * TS
        ax.plot(t, plant, color="k", lw=1.2, label="plant")
        ax.plot(t, pred_dk_y, "-.", color="#8B0000", lw=1.6, label="Deep Koopman")
        ax.plot(t, pred_c_y, "--", color="#FFA500", lw=1.2, label="Koopman linear C")
        ax.plot(t, pred_n_y, ":", color="#008000", lw=1.2, label="N4SID")
        ax.set_ylabel("y")
        ax.set_title(f"pch{exp['id']}: {exp['label']}")
        ax.grid(True)
        mae = {
            "Deep Koopman": float(np.mean(np.abs(plant - pred_dk_y))),
            "Koopman linear C": float(np.mean(np.abs(plant - pred_c_y))),
            "N4SID": float(np.mean(np.abs(plant - pred_n_y))),
        }
        rows.append((exp["id"], mae))
        print(f"pch{exp['id']} MAE {mae}")
    axs[0].legend(loc="best")
    axs[-1].set_xlabel("t [s]")
    fig.tight_layout()
    out = DATA_DIR / "ident_test_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"{'pch':<8} {'Deep Koopman':>14} {'linear C':>14} {'N4SID':>14}")
    mean = {k: 0.0 for k in rows[0][1]}
    for exp_id, mae in rows:
        print(
            f"{exp_id:<8} {mae['Deep Koopman']:14.4f} "
            f"{mae['Koopman linear C']:14.4f} {mae['N4SID']:14.4f}"
        )
        for k, v in mae.items():
            mean[k] += v
    n_e = len(rows)
    print(
        f"{'mean':<8} {mean['Deep Koopman']/n_e:14.4f} "
        f"{mean['Koopman linear C']/n_e:14.4f} {mean['N4SID']/n_e:14.4f}"
    )


def main() -> None:
    experiments, train_exps, dev_exps, test_exps, train, dev, test = load_flexy_splits()
    save_flexy_npz()
    print(
        f"loaded train {train['Y'].shape} dev {dev['Y'].shape} test {test['Y'].shape} "
        f"ids train={[e['id'] for e in train_exps]} "
        f"dev={[e['id'] for e in dev_exps]} test={[e['id'] for e in test_exps]}"
    )

    scaler = StandardScaler().fit(train["Y"])
    scaler_u = StandardScaler().fit(train["U"])
    joblib.dump(scaler, DATA_DIR / "scaler_flexy.pkl")
    joblib.dump(scaler_u, DATA_DIR / "scalerU_flexy.pkl")

    train_s = {"Y": scaler.transform(train["Y"]), "U": scaler_u.transform(train["U"])}
    identify_n4sid(train_s)

    nsteps = NSTEPS
    bs = 16
    train_loader, dev_loader, _, train_data = get_data_loaders(
        train, dev, test, nsteps, bs, scaler, scaler_u
    )
    _, _, dev_eval, _ = get_data_loaders(train, dev, dev, nsteps, bs, scaler, scaler_u)
    ny, nu = 1, 1
    train_one_koopman(
        matrix_c=True,
        nz=4,
        nonlin="gelu",
        loss_weights=LossWeights(
            y_loss=10.0, x_loss=10.0, onestep_loss=1.0, reconstruction_loss=5.0
        ),
        train_loader=train_loader,
        dev_loader=dev_loader,
        eval_data=dev_eval,
        train_data=train_data,
        scaler=scaler,
        ny=ny,
        nu=nu,
        nsteps=nsteps,
        bs=bs,
    )
    train_one_koopman(
        matrix_c=False,
        nz=6,
        nonlin="elu",
        loss_weights=LossWeights(
            y_loss=10.0, x_loss=1.0, onestep_loss=1.0, reconstruction_loss=20.0
        ),
        train_loader=train_loader,
        dev_loader=dev_loader,
        eval_data=dev_eval,
        train_data=train_data,
        scaler=scaler,
        ny=ny,
        nu=nu,
        nsteps=nsteps,
        bs=bs,
    )
    compare(test_exps, scaler, scaler_u)


if __name__ == "__main__":
    main()
