#!/usr/bin/env python3
"""Wall-clock cost of identifying the three final CSTR-separator models.

Retrains the two Koopman models with the hyperparameters of their closed-loop
search winners and re-identifies the order-17 N4SID model, reporting elapsed
time and dev/test MAE so the numbers can be checked against the stored models.
Nothing is written to ``../data``.
"""

from __future__ import annotations

import os
import sys
import time
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

import numpy as np
import torch

IDENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = IDENT_DIR.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(IDENT_DIR) not in sys.path:
    sys.path.insert(0, str(IDENT_DIR))

from helper.koopman import TrainConfig  # noqa: E402
from hpo_common import (  # noqa: E402
    HPOConfig,
    TrialParams,
    configure_threading,
    load_dataset,
    run_trial,
)

# Closed-loop search winners: hpo_results/C_cl_qu and hpo_results/noC_t2t3_v2.
LINC = TrialParams(
    nz=19,
    encoder_depth=1,
    width_mult=0.5,
    nonlin="gelu",
    y_loss_w=1.1386935333924078,
    x_loss_w=2.147233715422615,
    recon_loss_w=4.836534015133088,
    nsteps=120,
    bs=40,
    lr=0.0009935205073738306,
    seed=5365,
)
DK = TrialParams(
    nz=26,
    encoder_depth=2,
    width_mult=1.0,
    nonlin="elu",
    y_loss_w=17.37815816725602,
    x_loss_w=1.5941204307461116,
    recon_loss_w=2.284353778107768,
    nsteps=120,
    bs=40,
    lr=0.0008014083896549183,
    seed=3250,
)
TRAIN_CFG = TrainConfig(epochs=4000, warmup=100, patience=300)

# hpo_sippy_cl.py trial 62.
N4SID_ORDER = 17
N4SID_SS_F = 57


def time_koopman(label: str, params: TrialParams, matrix_C: bool, data) -> None:
    train, dev, test, scaler, scalerU, y_names, _ = data
    cfg = HPOConfig(
        matrix_C=matrix_C,
        experiment_name="timing",
        study_name="timing",
        variant="timing",
    )
    t0 = time.perf_counter()
    result = run_trial(
        params, cfg, train, dev, test, scaler, scalerU, y_names, TRAIN_CFG
    )
    elapsed = time.perf_counter() - t0
    print(
        f"{label}: {elapsed:.1f} s for {result['trainer_epochs']} epochs, "
        f"dev MAE sum {result['dev_mae_sum']:.4g}, "
        f"test MAE sum {result['test_mae_sum']:.4g}",
        flush=True,
    )


def time_n4sid(data) -> None:
    from sippy_unipi.model import SS_Model

    from hpo_sippy_cl import patch_sippy_vn_mat, test_mae_physical

    patch_sippy_vn_mat()
    train, _, test, scaler, scalerU, _, _ = data
    y_train = scaler.transform(train["Y"]).T.copy()
    u_train = scalerU.transform(train["U"]).T.copy()
    y_test = scaler.transform(test["Y"]).T.copy()
    u_test = scalerU.transform(test["U"]).T.copy()

    t0 = time.perf_counter()
    ident = SS_Model._identify(
        y_train,
        u_train,
        "N4SID",
        N4SID_ORDER,
        f=N4SID_SS_F,
        p=N4SID_SS_F,
        threshold=0.0,
        D_required=False,
        A_stability=False,
    )
    elapsed = time.perf_counter() - t0
    rho = float(np.max(np.abs(np.linalg.eigvals(np.asarray(ident.A, dtype=float)))))
    print(
        f"N4SID order {N4SID_ORDER}: {elapsed:.1f} s, rho(A) = {rho:.4f}, "
        f"test MAE sum {test_mae_physical(ident, u_test, y_test, scaler):.4g}",
        flush=True,
    )


def main() -> None:
    configure_threading(1)
    torch.set_num_threads(1)
    data = load_dataset()
    time_n4sid(data)
    time_koopman("Koopman linear C (nz = 19)", LINC, True, data)
    time_koopman("Deep Koopman (nz = 26)", DK, False, data)


if __name__ == "__main__":
    main()
