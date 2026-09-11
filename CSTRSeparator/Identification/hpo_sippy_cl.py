#!/usr/bin/env python3
"""SIPPY N4SID / PARSIM-K HPO scored with the N4SID.ipynb closed-loop OF.

Identifies linear SS models the same way as ``sippy.ipynb`` (scaled train
split, ``SS_fixed_order``, ``D = 0``) and minimizes the closed-loop cost
printed in ``CSTRSeparator/Control/N4SID.ipynb`` (same ``sim_setup.pkl``,
block-diagonal A, KF + target QP + linear MPC).

Search couples the state order ``n`` to SIPPY horizons: ``SS_f = n + f_extra``
with ``SS_f > n`` (SIPPY requires the future horizon to exceed the order).
PARSIM-K also searches the past horizon ``SS_p = p_over_f * SS_f``.

One-core default (``n_jobs=1``, one BLAS thread). Stop with
``hpo_results/sippy_cl/STOP`` or by killing the PID.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.trial import TrialState

IDENT_DIR = Path(__file__).resolve().parent
CONTROL_DIR = IDENT_DIR.parent / "Control"
DATA_DIR = IDENT_DIR.parent / "data"
os.environ.setdefault("SIM_SETUP_PATH", str(CONTROL_DIR / "sim_setup.pkl"))
if str(CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROL_DIR))

from hpo_common import (  # noqa: E402
    HPOConfig,
    RESULTS_DIR,
    configure_threading,
    load_dataset,
    setup_mlflow,
)
from ct_closed_loop import evaluate_or_penalty  # noqa: E402

HPO_CFG = HPOConfig(
    matrix_C=False,
    experiment_name="sippy_cstr_separator_cl",
    study_name="sippy_cl",
    variant="sippy_cl",
)

USE_BLOCK_DIAG = True
PENALTY = 1e6

# SIPPY: future horizon must exceed the model order.
N_MIN, N_MAX = 4, 20
F_EXTRA_CHOICES = [5, 10, 15, 20, 27, 40]  # 27 → n=13, SS_f=40 (notebook)
P_OVER_F_CHOICES = [1.0, 1.5, 2.0]
CENTERING_CHOICES = ["None", "MeanVal", "InitVal"]

# Notebook N4SID winner used in N4SID.ipynb (order 13, SS_f=40).
NOTEBOOK_SEED = {
    "n": 13,
    "f_extra": 27,
    "centering": "None",
}

TRIAL_LOG_COLUMNS = [
    "trial",
    "method",
    "n",
    "SS_f",
    "SS_p",
    "f_extra",
    "p_over_f",
    "centering",
    "SS_A_stability",
    "SS_PK_B_reval",
    "rho",
    "n_ident",
    "test_mae",
    "cl_of",
    "cl_tracking",
    "cl_du",
    "error",
]


@dataclass
class SippyParams:
    method: str
    n: int
    f_extra: int
    centering: str
    SS_A_stability: bool = True
    p_over_f: float = 1.0
    SS_PK_B_reval: bool = False

    @property
    def SS_f(self) -> int:
        return int(self.n + self.f_extra)

    @property
    def SS_p(self) -> int:
        if self.method == "PARSIM-K":
            return int(round(self.SS_f * float(self.p_over_f)))
        return self.SS_f


def patch_sippy_vn_mat() -> None:
    """NumPy 2.x emits false overflow warnings for SIPPY's 1-D ``a @ a``."""

    def _vn_mat_dot(y, yest):
        eps = np.asarray(y).reshape(-1) - np.asarray(yest).reshape(-1)
        return float(np.dot(eps, eps) / max(eps.size, 1))

    import sippy_unipi.functionsetSIM as fsetSIM
    import sippy_unipi.OLSims_methods as olm
    import sippy_unipi.Parsim_methods as psm

    fsetSIM.Vn_mat = _vn_mat_dot
    olm.Vn_mat = _vn_mat_dot
    psm.Vn_mat = _vn_mat_dot


def sample_sippy_params(trial: optuna.Trial) -> SippyParams:
    method = trial.suggest_categorical("method", ["PARSIM-K", "N4SID"])
    n = trial.suggest_int("n", N_MIN, N_MAX)
    f_extra = trial.suggest_categorical("f_extra", F_EXTRA_CHOICES)
    centering = trial.suggest_categorical("centering", CENTERING_CHOICES)
    if method == "PARSIM-K":
        p_over_f = trial.suggest_categorical("p_over_f", P_OVER_F_CHOICES)
        b_reval = trial.suggest_categorical("SS_PK_B_reval", [False, True])
        return SippyParams(
            method=method,
            n=int(n),
            f_extra=int(f_extra),
            centering=str(centering),
            p_over_f=float(p_over_f),
            SS_PK_B_reval=bool(b_reval),
        )
    stability = trial.suggest_categorical("SS_A_stability", [True, False])
    return SippyParams(
        method=method,
        n=int(n),
        f_extra=int(f_extra),
        centering=str(centering),
        SS_A_stability=bool(stability),
    )


def _as_signals_by_time(arr: np.ndarray) -> np.ndarray:
    x = np.array(arr, dtype=float, copy=True)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D array, got {x.shape}")
    if x.shape[0] > x.shape[1]:
        x = x.T
    return x


def apply_centering(y: np.ndarray, u: np.ndarray, centering: str) -> tuple[np.ndarray, np.ndarray]:
    """Match SIPPY ``centering`` before calling ``SS_Model._identify``."""
    y = _as_signals_by_time(y)
    u = _as_signals_by_time(u)
    if centering == "InitVal":
        y = y - y[:, :1]
        u = u - u[:, :1]
    elif centering == "MeanVal":
        y = y - y.mean(axis=1, keepdims=True)
        u = u - u.mean(axis=1, keepdims=True)
    return y, u


def identify_sippy(y: np.ndarray, u: np.ndarray, params: SippyParams):
    """Identify A,B,C,D with the kmpc ``sippy_unipi`` SS API.

    The public ``system_identification`` wrapper in this SIPPY version does not
    forward ``SS_A_stability`` (it is passed as ``B_reval`` by position). Call
    ``SS_Model._identify`` with keywords so N4SID stability and PARSIM-K B
    re-evaluation actually take effect. Method name is ``PARSIM_K`` here
    (older SIPPY used ``PARSIM-K``).
    """
    from sippy_unipi.model import SS_Model

    y_c, u_c = apply_centering(y, u, params.centering)
    method = "PARSIM_K" if params.method == "PARSIM-K" else "N4SID"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=".*encountered in matmul",
        )
        ident = SS_Model._identify(
            y_c,
            u_c,
            method,
            int(params.n),
            f=int(params.SS_f),
            p=int(params.SS_p),
            threshold=0.0,
            D_required=False,
            A_stability=bool(params.SS_A_stability) if method == "N4SID" else False,
            B_recalc=bool(params.SS_PK_B_reval) if method == "PARSIM-K" else False,
        )
    return ident


def test_mae_physical(ident, U_test: np.ndarray, Y_test: np.ndarray, scaler) -> float:
    from sippy_unipi import functionsetSIM as fsetSIM

    x0 = np.linalg.pinv(ident.C) @ Y_test[:, :1]
    _, y_hat = fsetSIM.SS_lsim_process_form(
        ident.A, ident.B, ident.C, ident.D, U_test, x0
    )
    err = Y_test - y_hat
    mae_phys = np.mean(np.abs(err), axis=1) * scaler.scale_
    return float(np.sum(mae_phys))


def ident_is_healthy(ident, expected_n: int) -> str | None:
    A = np.asarray(getattr(ident, "A", np.array([[0.0]])), dtype=float)
    B = np.asarray(getattr(ident, "B", np.array([[0.0]])), dtype=float)
    C = np.asarray(getattr(ident, "C", np.array([[0.0]])), dtype=float)
    if A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
        return "A/B/C are not 2-D"
    if A.shape[0] != A.shape[1] or A.shape[0] < 1:
        return f"bad A shape {A.shape}"
    if A.shape[0] != expected_n:
        return f"identified n={A.shape[0]} != requested {expected_n}"
    if B.shape[0] != A.shape[0] or C.shape[1] != A.shape[0]:
        return f"incompatible shapes A{A.shape} B{B.shape} C{C.shape}"
    if not (np.isfinite(A).all() and np.isfinite(B).all() and np.isfinite(C).all()):
        return "non-finite A/B/C"
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    if not np.isfinite(rho):
        return "non-finite rho(A)"
    return None


def params_to_log(params: SippyParams) -> dict[str, Any]:
    return {
        "method": params.method,
        "n": params.n,
        "SS_f": params.SS_f,
        "SS_p": params.SS_p,
        "f_extra": params.f_extra,
        "p_over_f": params.p_over_f if params.method == "PARSIM-K" else "",
        "centering": params.centering,
        "SS_A_stability": params.SS_A_stability if params.method == "N4SID" else "",
        "SS_PK_B_reval": params.SS_PK_B_reval if params.method == "PARSIM-K" else "",
    }


def append_trial_log(trial_number: int, params: SippyParams, result: dict[str, Any]) -> None:
    path = RESULTS_DIR / HPO_CFG.variant / "trials.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        **params_to_log(params),
        "trial": trial_number,
        "rho": result.get("rho", ""),
        "n_ident": result.get("n_ident", ""),
        "test_mae": result.get("test_mae", ""),
        "cl_of": result.get("cl_of", ""),
        "cl_tracking": result.get("cl_tracking", ""),
        "cl_du": result.get("cl_du", ""),
        "error": (result.get("error") or "").replace("\t", " ").replace("\n", " ")[:200],
    }

    def _fmt(key: str) -> str:
        val = values.get(key, "")
        if val is None or val == "":
            return ""
        if isinstance(val, float) and not np.isfinite(val):
            return ""
        if isinstance(val, float):
            return f"{val:.6g}"
        return str(val)

    row = "\t".join(_fmt(c) for c in TRIAL_LOG_COLUMNS) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.tell() == 0:
            f.write("\t".join(TRIAL_LOG_COLUMNS) + "\n")
        f.write(row)
        f.flush()
    print(
        f"trial {trial_number:03d}  {params.method}  n={params.n}  "
        f"SS_f={params.SS_f}  SS_p={params.SS_p}  centering={params.centering}  "
        f"->  CL OF={_fmt('cl_of')}  rho={_fmt('rho')}  test MAE={_fmt('test_mae')}",
        flush=True,
    )


def save_best_trial(
    out_dir: Path,
    trial_number: int,
    params: SippyParams,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    result: dict[str, Any],
) -> None:
    snap = out_dir / "best"
    snap.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / "best.lock"
    with open(lock_path, "a", encoding="utf-8") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        winner_path = out_dir / "cl_winner.json"
        if winner_path.is_file():
            prev = json.loads(winner_path.read_text(encoding="utf-8"))
            prev_of = float(prev.get("cl", {}).get("objective", float("inf")))
            if float(result["cl_of"]) >= prev_of:
                return
        np.save(snap / "A.npy", A)
        np.save(snap / "B.npy", B)
        np.save(snap / "C.npy", C)
        np.save(snap / "D.npy", D)
        np.save(DATA_DIR / "A_cstr_separator_sippy_hpo.npy", A)
        np.save(DATA_DIR / "B_cstr_separator_sippy_hpo.npy", B)
        np.save(DATA_DIR / "C_cstr_separator_sippy_hpo.npy", C)
        np.save(DATA_DIR / "D_cstr_separator_sippy_hpo.npy", D)
        meta = {
            "trial": trial_number,
            "params": asdict(params) | {"SS_f": params.SS_f, "SS_p": params.SS_p},
            "use_block_diag": USE_BLOCK_DIAG,
            "cl": {k: result[k] for k in ("cl_of", "cl_tracking", "cl_du", "rho") if k in result},
            "test_mae_sum": result.get("test_mae"),
            "n_ident": result.get("n_ident"),
        }
        meta["cl"] = {
            "objective": result.get("cl_of"),
            "state_error_cost": result.get("cl_tracking"),
            "control_increment_cost": result.get("cl_du"),
            "rho_A": result.get("rho"),
        }
        (snap / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        winner_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        archive = out_dir / f"best_trial_{trial_number:03d}"
        archive.mkdir(parents=True, exist_ok=True)
        np.save(archive / "A.npy", A)
        np.save(archive / "B.npy", B)
        np.save(archive / "C.npy", C)
        np.save(archive / "D.npy", D)
        (archive / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(
            f"Saved best trial {trial_number}  OF={result['cl_of']:.4g} "
            f"({params.method} n={params.n} SS_f={params.SS_f}).",
            flush=True,
        )


def enqueue_seed_trials(study: optuna.Study) -> None:
    """PARSIM-K then N4SID notebook seeds, then a small n-grid for each method."""
    study.enqueue_trial(
        {
            **NOTEBOOK_SEED,
            "method": "PARSIM-K",
            "p_over_f": 1.0,
            "SS_PK_B_reval": False,
        }
    )
    study.enqueue_trial(
        {
            **NOTEBOOK_SEED,
            "method": "N4SID",
            "SS_A_stability": True,
        }
    )
    for n in (7, 9, 11, 16, 20):
        study.enqueue_trial(
            {
                "method": "N4SID",
                "n": n,
                "f_extra": 27,
                "centering": "None",
                "SS_A_stability": True,
            }
        )
        study.enqueue_trial(
            {
                "method": "PARSIM-K",
                "n": n,
                "f_extra": 27,
                "centering": "None",
                "p_over_f": 1.0,
                "SS_PK_B_reval": False,
            }
        )


def fail_stale_running_trials(study: optuna.Study) -> int:
    n = 0
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        try:
            study.tell(trial.number, state=TrialState.FAIL, skip_if_finished=True)
            n += 1
        except Exception as exc:
            print(f"Could not fail stale trial {trial.number}: {exc}", flush=True)
    return n


def safe_best_value(study: optuna.Study) -> float | None:
    try:
        return float(study.best_value)
    except ValueError:
        return None


class StopIfRequested:
    def __init__(self, stop_path: Path, n_trials: int | None):
        self.stop_path = stop_path
        self.n_trials = n_trials

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self.stop_path.exists():
            print(f"Found {self.stop_path}; stopping after trial {trial.number}.", flush=True)
            study.stop()
            return
        if self.n_trials is None:
            return
        n_done = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
        if n_done >= self.n_trials:
            print(f"Reached {n_done} complete trials; stopping.", flush=True)
            study.stop()


def eval_existing_notebook_model() -> dict:
    A = np.load(DATA_DIR / "A_cstr_separator_parsimK.npy")
    B = np.load(DATA_DIR / "B_cstr_separator_parsimK.npy")
    C = np.load(DATA_DIR / "C_cstr_separator_parsimK.npy")
    print(
        f"Evaluating notebook N4SID  A {A.shape}  "
        f"rho={np.max(np.abs(np.linalg.eigvals(A))):.4f}",
        flush=True,
    )
    cl = evaluate_or_penalty(A, B, C, use_block_diag=USE_BLOCK_DIAG)
    print(
        f"existing parsimK files  OF={cl['objective']:.6g}  "
        f"tracking={cl.get('state_error_cost')}  du={cl.get('control_increment_cost')}",
        flush=True,
    )
    if cl.get("error"):
        print(f"error: {cl['error']}", flush=True)
    return cl


def run_identification_and_cl(
    params: SippyParams,
    y_train: np.ndarray,
    u_train: np.ndarray,
    y_test: np.ndarray,
    u_test: np.ndarray,
    scaler,
) -> tuple[float, dict[str, Any], np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    result: dict[str, Any] = {
        "cl_of": PENALTY,
        "cl_tracking": float("nan"),
        "cl_du": float("nan"),
        "test_mae": float("nan"),
        "rho": float("nan"),
        "n_ident": None,
        "error": None,
    }
    try:
        ident = identify_sippy(y_train, u_train, params)
    except Exception as exc:
        result["error"] = f"ident: {exc}"
        traceback.print_exc()
        return PENALTY, result, None, None, None, None

    bad = ident_is_healthy(ident, params.n)
    A = np.asarray(ident.A, dtype=float)
    B = np.asarray(ident.B, dtype=float)
    C = np.asarray(ident.C, dtype=float)
    D = np.asarray(ident.D, dtype=float)
    result["n_ident"] = int(A.shape[0]) if A.ndim == 2 else None
    if bad:
        result["error"] = bad
        return PENALTY, result, A, B, C, D

    result["rho"] = float(np.max(np.abs(np.linalg.eigvals(A))))
    try:
        result["test_mae"] = test_mae_physical(ident, u_test, y_test, scaler)
    except Exception as exc:
        result["error"] = f"mae: {exc}"

    cl = evaluate_or_penalty(A, B, C, use_block_diag=USE_BLOCK_DIAG)
    result["cl_of"] = float(cl["objective"])
    result["cl_tracking"] = cl.get("state_error_cost")
    result["cl_du"] = cl.get("control_increment_cost")
    if cl.get("error"):
        result["error"] = str(cl["error"])
    return float(result["cl_of"]), result, A, B, C, D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Complete-trial budget. Default 200. Stop earlier with the STOP file.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Optuna parallel trials. Default 1 (one-core budget).",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="BLAS threads per process. Default 1.",
    )
    parser.add_argument(
        "--eval-existing",
        action="store_true",
        help="Score N4SID.ipynb A/B/C_cstr_separator_parsimK.npy (no search).",
    )
    args = parser.parse_args()

    configure_threading(args.blas_threads)
    os.chdir(CONTROL_DIR)
    patch_sippy_vn_mat()

    if args.eval_existing:
        eval_existing_notebook_model()
        return

    train, _, test, scaler, scalerU, _, _ = load_dataset()
    y_train = scaler.transform(train["Y"]).T.copy()
    u_train = scalerU.transform(train["U"]).T.copy()
    y_test = scaler.transform(test["Y"]).T.copy()
    u_test = scalerU.transform(test["U"]).T.copy()
    print(
        f"SIPPY HPO data  y_train {y_train.shape}  u_train {u_train.shape}  "
        f"y_test {y_test.shape}",
        flush=True,
    )

    try:
        import mlflow
    except Exception:
        mlflow = None  # type: ignore[assignment]
    if mlflow is not None:
        setup_mlflow(HPO_CFG)

    out_dir = RESULTS_DIR / HPO_CFG.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_path = out_dir / "STOP"
    pid_path = out_dir / "search.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    db = out_dir / f"{HPO_CFG.study_name}.db"
    study = optuna.create_study(
        study_name=HPO_CFG.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=f"sqlite:///{db.resolve()}",
        load_if_exists=True,
    )
    n_stale = fail_stale_running_trials(study)
    if n_stale:
        print(f"Marked {n_stale} stale RUNNING trials as FAIL.", flush=True)
    if len(study.trials) == 0:
        enqueue_seed_trials(study)
        print(
            f"New study {HPO_CFG.study_name}: first trials PARSIM-K then N4SID "
            f"(notebook n=13, SS_f=40), n_jobs={args.n_jobs}, "
            f"n_trials={args.n_trials}.",
            flush=True,
        )
    else:
        print(
            f"Continuing {HPO_CFG.study_name} "
            f"({len(study.trials)} existing, best OF={safe_best_value(study)}).",
            flush=True,
        )

    def objective(trial: optuna.Trial) -> float:
        params = sample_sippy_params(trial)
        trial.set_user_attr("SS_f", params.SS_f)
        trial.set_user_attr("SS_p", params.SS_p)
        print(
            f"trial {trial.number} start  {params.method}  n={params.n}  "
            f"SS_f={params.SS_f}  SS_p={params.SS_p}  centering={params.centering}  "
            f"A_stab={params.SS_A_stability if params.method == 'N4SID' else '-'}  "
            f"B_reval={params.SS_PK_B_reval if params.method == 'PARSIM-K' else '-'}",
            flush=True,
        )
        of, result, A, B, C, D = run_identification_and_cl(
            params, y_train, u_train, y_test, u_test, scaler
        )
        if mlflow is not None:
            try:
                with mlflow.start_run(run_name=f"sippy_trial_{trial.number}"):
                    mlflow.log_params(
                        {
                            k: v
                            for k, v in (asdict(params) | {"SS_f": params.SS_f, "SS_p": params.SS_p}).items()
                            if v != "" and v is not None
                        }
                    )
                    mlflow.log_metric("cl_of", of)
                    if np.isfinite(result.get("test_mae", float("nan"))):
                        mlflow.log_metric("test_mae_sum", result["test_mae"])
                    if np.isfinite(result.get("rho", float("nan"))):
                        mlflow.log_metric("rho_A", result["rho"])
            except Exception as exc:
                print(f"MLflow log failed (trial {trial.number}): {exc}", flush=True)
        trial.set_user_attr("cl_of", of)
        trial.set_user_attr("test_mae_sum", result.get("test_mae"))
        trial.set_user_attr("rho", result.get("rho"))
        if result.get("error"):
            trial.set_user_attr("error", str(result["error"])[:300])
            print(f"trial {trial.number} error: {result['error']}", flush=True)
        append_trial_log(trial.number, params, result)
        if A is not None and of < PENALTY * 0.5:
            save_best_trial(out_dir, trial.number, params, A, B, C, D, result)
        return of

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=max(1, int(args.n_jobs)),
            show_progress_bar=False,
            callbacks=[StopIfRequested(stop_path, args.n_trials)],
            catch=(Exception,),
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Writing summary of completed trials.", flush=True)

    best_value = safe_best_value(study)
    if best_value is not None:
        print(f"Best CL OF: {best_value:.6g}")
        print(f"Best params: {study.best_params}")
    else:
        print("No completed trials.")
    summary = {
        "best_value": best_value,
        "best_params": study.best_params if best_value is not None else None,
        "n_trials": len(study.trials),
        "use_block_diag": USE_BLOCK_DIAG,
    }
    (out_dir / "stage_A_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
