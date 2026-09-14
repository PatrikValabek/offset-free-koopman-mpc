#!/usr/bin/env python3
"""Open-loop test-split accuracy of the identified CSTR-separator models.

Errors are taken against the noise-free plant outputs ``Y_clean`` stored in the
npz, so the reported numbers are model errors and not measurement noise. Every
model is initialised from the first noise-free test sample and then run in free
open loop over the whole test split (no state feedback), driven by the recorded
inputs.

Prints per-output MAE / RMSE in physical units plus aggregates on the
standardised outputs, which is the quantity the identification searches
minimised.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")

import joblib
import numpy as np
import torch

IDENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = IDENT_DIR.parent.parent
DATA_DIR = IDENT_DIR.parent / "data"
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helper.koopman import LossWeights, build_koopman_problem  # noqa: E402

N_TRAIN = 12960
N_DEV = 2880

# Deep Koopman (nonlinear decoder): closed-loop HPO winner noC_t2t3_v2.
DK = {
    "stem": "cstr_separator_C_noc_hpo",
    "encoder_depth": 2,
    "width_mult": 1.0,
    "nonlin": "elu",
}
# Koopman with linear C: closed-loop HPO winner C_cl_qu.
LINC_STEM = "cstr_separator_C_cl_hpo"
# Linear subspace baselines.
SIPPY_STEMS = {
    "N4SID (order 17, CL HPO)": "cstr_separator_sippy_hpo",
    "PARSIM-K (order 13, notebook)": "cstr_separator_parsimK",
}


def ss_lsim(A, B, C, D, U, x0):
    """Process-form free-run rollout. ``U`` is ``(nu, N)``."""
    n = U.shape[1]
    y = np.zeros((C.shape[0], n))
    x = np.asarray(x0, dtype=float).reshape(-1)
    for k in range(n):
        uk = U[:, k]
        y[:, k] = C @ x + D @ uk
        x = A @ x + B @ uk
    return y


def load_deep_koopman(ny: int, nu: int):
    state = torch.load(
        DATA_DIR / f"model_{DK['stem']}.pth", map_location="cpu", weights_only=False
    )
    nz = int(state["nodes.3.nodes.0.callable.K.weight"].shape[0])
    problem, _, _, _ = build_koopman_problem(
        ny=ny,
        nu=nu,
        nz=nz,
        matrix_C=False,
        encoder_depth=DK["encoder_depth"],
        width_mult=DK["width_mult"],
        nonlin=DK["nonlin"],
        loss_weights=LossWeights(),
        nsteps=4,
    )
    problem.load_state_dict(state)
    problem.eval()
    return problem, nz


def deep_koopman_rollout(problem, y_scaled, u_scaled):
    ny = y_scaled.shape[1]
    y = torch.tensor(y_scaled[None, :, :], dtype=torch.float32)
    u = torch.tensor(u_scaled[None, :, :], dtype=torch.float32)
    problem.nodes[3].nsteps = y_scaled.shape[0]
    with torch.no_grad():
        out = problem.step({"Y": y, "Y0": y[:, 0:1, :], "U": u})
    return out["yhat"][:, 1:-1, :].detach().numpy().reshape(-1, ny).T


def main() -> None:
    torch.set_num_threads(1)
    data = np.load(DATA_DIR / "cstr_separator_ident.npz", allow_pickle=True)
    y_clean = np.asarray(data["Y_clean"], dtype=float)
    u = np.asarray(data["U"], dtype=float)
    y_names = [str(s) for s in data["y_names"]]
    ny, nu = y_clean.shape[1], u.shape[1]

    i0 = N_TRAIN + N_DEV
    y_test = y_clean[i0:]
    u_test = u[i0:]

    scaler = joblib.load(DATA_DIR / "scaler_cstr_separator.pkl")
    scaler_u = joblib.load(DATA_DIR / "scalerU_cstr_separator.pkl")
    y_scaled = scaler.transform(y_test)
    u_scaled = scaler_u.transform(u_test)

    problem, nz_dk = load_deep_koopman(ny, nu)
    pred_dk_s = deep_koopman_rollout(problem, y_scaled, u_scaled)

    n = min(pred_dk_s.shape[1], y_test.shape[0] - 1, max(u_scaled.shape[0] - 1, 1))
    y0 = y_scaled[1:2].T
    u_eval = u_scaled[1 : 1 + n].T

    preds = {f"Deep Koopman (nz = {nz_dk})": pred_dk_s[:, :n]}

    a_c = np.load(DATA_DIR / f"A_{LINC_STEM}.npy")
    b_c = np.load(DATA_DIR / f"B_{LINC_STEM}.npy")
    c_c = np.load(DATA_DIR / f"C_{LINC_STEM}.npy")
    preds[f"Koopman linear C (nz = {a_c.shape[0]})"] = ss_lsim(
        a_c,
        b_c,
        c_c,
        np.zeros((c_c.shape[0], b_c.shape[1])),
        u_eval,
        np.linalg.pinv(c_c) @ y0,
    )

    for label, stem in SIPPY_STEMS.items():
        a = np.load(DATA_DIR / f"A_{stem}.npy")
        b = np.load(DATA_DIR / f"B_{stem}.npy")
        c = np.load(DATA_DIR / f"C_{stem}.npy")
        d = np.load(DATA_DIR / f"D_{stem}.npy")
        preds[label] = ss_lsim(a, b, c, d, u_eval, np.linalg.pinv(c) @ y0)

    plant = y_test[1 : 1 + n]
    plant_s = y_scaled[1 : 1 + n]

    print(f"test split: {n} samples ({n / int(data['step_time']):.0f} input steps)")
    print("errors against the noise-free plant outputs Y_clean\n")
    header = f"{'model':<30}" + "".join(f"{name:>12}" for name in y_names)
    for metric in ("MAE", "RMSE"):
        print(f"{metric} [physical units]")
        print(header)
        for label, pred_s in preds.items():
            pred = scaler.inverse_transform(pred_s.T)
            err = plant - pred
            val = (
                np.mean(np.abs(err), axis=0)
                if metric == "MAE"
                else np.sqrt(np.mean(err**2, axis=0))
            )
            print(f"{label:<30}" + "".join(f"{v:12.4f}" for v in val))
        print()

    print(f"{'model':<30}{'sum MAE [-]':>14}{'mean RMSE [-]':>16}")
    for label, pred_s in preds.items():
        err = plant_s - pred_s.T
        print(
            f"{label:<30}"
            f"{np.sum(np.mean(np.abs(err), axis=0)):14.4f}"
            f"{np.mean(np.sqrt(np.mean(err**2, axis=0))):16.4f}"
        )


if __name__ == "__main__":
    main()
