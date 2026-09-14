#!/usr/bin/env python3
"""Paper figure: test-set comparison of Deep Koopman, linear C, and N4SID (HPO).

Tunables are in the block below. Re-run this file after edits.

The plotted window is N_INPUT_STEPS consecutive piecewise-constant excitations
from the identification test split (each of length step_time samples in the npz).
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
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

IDENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = IDENT_DIR.parent.parent
DATA_DIR = IDENT_DIR.parent / "data"
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helper.koopman import LossWeights, build_koopman_problem  # noqa: E402

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
N_INPUT_STEPS = 4  # number of identification input steps to show
START_INPUT_STEP = 0  # 0 = beginning of the test split
SHOW_LEGEND = False  # paper default: name colours in the caption

# elsarticle preprint, 12 pt, one column: 384 pt textwidth, figures at 0.95\linewidth
BODY_PT = 12
TICK_PT = 10
TEXT_IN = 384.0 / 72.0
INCLUDE_FRAC = 0.95
FIG_W_IN = INCLUDE_FRAC * TEXT_IN  # match \includegraphics[width=0.95\linewidth]
FIG_H_IN = 6.2

COLOR_PLANT = "#000000"
COLOR_DK = "#8B0000"  # Deep Koopman
COLOR_LINC = "#FFA500"  # Koopman linear C
COLOR_N4SID = "#008000"  # N4SID
LW_PLANT = 1.2
LW_DK = 1.8
LW_BASE = 1.2

N_TRAIN = 12960
N_DEV = 2880

PATH_DATA = DATA_DIR / "cstr_separator_ident.npz"
PATH_SCALER = DATA_DIR / "scaler_cstr_separator.pkl"
PATH_SCALER_U = DATA_DIR / "scalerU_cstr_separator.pkl"
PATH_DK = DATA_DIR / "model_cstr_separator_C_noc_hpo.pth"
# Architecture of the noC_t2t3_v2 winner (nz inferred from the checkpoint).
DK_ENCODER_DEPTH = 2
DK_WIDTH_MULT = 1.0
DK_NONLIN = "elu"
PATH_LINC_A = DATA_DIR / "A_cstr_separator_C_cl_hpo.npy"
PATH_LINC_B = DATA_DIR / "B_cstr_separator_C_cl_hpo.npy"
PATH_LINC_C = DATA_DIR / "C_cstr_separator_C_cl_hpo.npy"
PATH_N4SID_A = DATA_DIR / "A_cstr_separator_sippy_hpo.npy"
PATH_N4SID_B = DATA_DIR / "B_cstr_separator_sippy_hpo.npy"
PATH_N4SID_C = DATA_DIR / "C_cstr_separator_sippy_hpo.npy"
PATH_N4SID_D = DATA_DIR / "D_cstr_separator_sippy_hpo.npy"
OUT_PDF = REPO_ROOT / "document" / "figures" / "ident_test_comparison.pdf"

YLABELS = (
    r"$T_1$ [K]",
    r"$T_2$ [K]",
    r"$T_3$ [K]",
    r"$x_{\mathrm{B}3}$ [-]",
)


def ss_lsim(A, B, C, D, U, x0):
    """Process-form rollout. U is (nu, N)."""
    N = U.shape[1]
    y = np.zeros((C.shape[0], N))
    x = np.asarray(x0, dtype=float).reshape(-1)
    for k in range(N):
        uk = U[:, k]
        y[:, k] = C @ x + D @ uk
        x = A @ x + B @ uk
    return y


def load_deep_koopman(ny: int, nu: int):
    state = torch.load(PATH_DK, map_location="cpu", weights_only=False)
    nz = int(state["nodes.3.nodes.0.callable.K.weight"].shape[0])
    problem, _, _, _ = build_koopman_problem(
        ny=ny,
        nu=nu,
        nz=nz,
        matrix_C=False,
        encoder_depth=DK_ENCODER_DEPTH,
        width_mult=DK_WIDTH_MULT,
        nonlin=DK_NONLIN,
        loss_weights=LossWeights(),
        nsteps=4,
    )
    problem.load_state_dict(state)
    problem.eval()
    return problem


def deep_koopman_rollout(problem, Y_scaled, U_scaled):
    """Open-loop decoder rollout in scaled space. Y, U are (N, ny) / (N, nu)."""
    ny = Y_scaled.shape[1]
    n_t = Y_scaled.shape[0]
    y = torch.tensor(Y_scaled[None, :, :], dtype=torch.float32)
    u = torch.tensor(U_scaled[None, :, :], dtype=torch.float32)
    data = {"Y": y, "Y0": y[:, 0:1, :], "U": u}
    problem.nodes[3].nsteps = n_t
    with torch.no_grad():
        out = problem.step(data)
    pred = out["yhat"][:, 1:-1, :].detach().numpy().reshape(-1, ny).T
    return pred


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.labelsize": BODY_PT,
            "xtick.labelsize": TICK_PT,
            "ytick.labelsize": TICK_PT,
            "legend.fontsize": TICK_PT,
            "axes.titlesize": BODY_PT,
            "axes.linewidth": 0.6,
            "lines.linewidth": LW_BASE,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    torch.set_num_threads(1)
    apply_style()

    data = np.load(PATH_DATA, allow_pickle=True)
    y_clean = np.asarray(data["Y_clean"], dtype=float)
    u = np.asarray(data["U"], dtype=float)
    ts = float(np.asarray(data["Ts"]))
    step_time = int(np.asarray(data["step_time"]))
    y_names = list(data["y_names"])
    ny, nu = y_clean.shape[1], u.shape[1]

    i0 = N_TRAIN + N_DEV
    n_test = y_clean.shape[0] - i0
    n_test_steps = n_test // step_time
    if START_INPUT_STEP < 0 or START_INPUT_STEP + N_INPUT_STEPS > n_test_steps:
        raise ValueError(
            f"Need START_INPUT_STEP + N_INPUT_STEPS <= {n_test_steps} "
            f"(got {START_INPUT_STEP} + {N_INPUT_STEPS})"
        )

    k0 = i0 + START_INPUT_STEP * step_time
    n_plot = N_INPUT_STEPS * step_time
    sl = slice(k0, k0 + n_plot)
    y_plant = y_clean[sl]
    u_win = u[sl]

    scaler = joblib.load(PATH_SCALER)
    scaler_u = joblib.load(PATH_SCALER_U)
    y_scaled = scaler.transform(y_plant)
    u_scaled = scaler_u.transform(u_win)

    problem = load_deep_koopman(ny, nu)
    pred_dk_s = deep_koopman_rollout(problem, y_scaled, u_scaled)

    n_pred = pred_dk_s.shape[1]
    # Align with the notebook: linear models start at the first compared sample
    # and use U[:, 1:1+N]. Plant is plotted on the same samples.
    n = min(n_pred, y_plant.shape[0] - 1, max(u_scaled.shape[0] - 1, 1))
    y0 = y_scaled[1 : 2].T
    u_eval = u_scaled[1 : 1 + n].T

    a_c = np.load(PATH_LINC_A)
    b_c = np.load(PATH_LINC_B)
    c_c = np.load(PATH_LINC_C)
    d_c = np.zeros((c_c.shape[0], b_c.shape[1]))
    pred_linc_s = ss_lsim(a_c, b_c, c_c, d_c, u_eval, np.linalg.pinv(c_c) @ y0)

    a_p = np.load(PATH_N4SID_A)
    b_p = np.load(PATH_N4SID_B)
    c_p = np.load(PATH_N4SID_C)
    d_p = np.load(PATH_N4SID_D)
    pred_par_s = ss_lsim(a_p, b_p, c_p, d_p, u_eval, np.linalg.pinv(c_p) @ y0)

    plant = y_plant[1 : 1 + n]
    pred_dk = scaler.inverse_transform(pred_dk_s[:, :n].T)
    pred_linc = scaler.inverse_transform(pred_linc_s[:, :n].T)
    pred_par = scaler.inverse_transform(pred_par_s[:, :n].T)
    t = np.arange(n) * ts

    fig, axes = plt.subplots(ny, 1, figsize=(FIG_W_IN, FIG_H_IN), sharex=True)
    series = (
        (plant, COLOR_PLANT, "-", LW_PLANT, "Plant"),
        (pred_dk, COLOR_DK, "-.", LW_DK, "Deep Koopman"),
        (pred_linc, COLOR_LINC, "--", LW_BASE, "Koopman linear $C$"),
        (pred_par, COLOR_N4SID, ":", LW_BASE, "N4SID"),
    )
    for i, ax in enumerate(axes):
        for y, color, ls, lw, label in series:
            ax.plot(t, y[:, i], color=color, linestyle=ls, linewidth=lw, label=label, zorder=3)
        ax.set_ylabel(YLABELS[i], fontsize=BODY_PT)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
        ax.grid(True)
        ax.set_xlim(-0.02 * t[-1], t[-1])
        if SHOW_LEGEND and i == 0:
            ax.legend(frameon=False, loc="best")

    axes[-1].set_xlabel(r"Time $t$ [s]", fontsize=BODY_PT)
    fig.tight_layout(pad=0.3)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf")
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")
    print(f"window: test input steps {START_INPUT_STEP}..{START_INPUT_STEP + N_INPUT_STEPS - 1}  "
          f"({n} samples, {y_names})")


if __name__ == "__main__":
    main()
