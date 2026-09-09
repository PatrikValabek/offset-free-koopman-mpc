#!/usr/bin/env python3
"""Identify noC Koopman models (example_training10) and score T2D2 + T3D3.

Optuna minimizes OF_T2D2 + OF_T3D3 from the Control notebooks.
``use_block_diag`` is always True (same as the notebooks).

Training uses ``--n-jobs`` concurrent trials (default 4, one BLAS thread each).
Each finished model is scored with T2D2 and T3D3 in parallel (2 evals).
With 4 trials in the eval phase that is up to 8 closed-loop workers.

The actual best-trial weights and A/B/C are saved (no retrain).
Stop by creating ``hpo_results/noC_t2t3/STOP`` or killing the PID.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
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
    TrainConfig,
    TrialParams,
    append_trial_log,
    configure_threading,
    load_dataset,
    add_parallel_cli,
    run_trial,
    sample_trial_params,
    setup_mlflow,
)
from taylor_closed_loop import evaluate_t2t3  # noqa: E402

HPO_CFG = HPOConfig(
    matrix_C=False,
    experiment_name="koopman_cstr_separator_noC_t2t3",
    study_name="koopman_noC_t2t3_ep",
    variant="noC_t2t3",
)

EPOCH_CHOICES = [500, 1000, 1500, 2000, 3000, 4000]

# Identification winner from the noC MAE search (example_training10).
SEED_PARAMS = {
    "nz": 32,
    "encoder_depth": 2,
    "width_mult": 1.0,
    "nonlin": "elu",
    "y_loss_w": 13.295727018943847,
    "x_loss_w": 3.3437777809707585,
    "recon_loss_w": 1.7456648315750745,
    "nsteps": 120,
    "bs": 40,
    "lr": 0.0007530643085207749,
    "seed": 2843,
}


def enqueue_seed_trials(study: optuna.Study) -> None:
    for epochs in EPOCH_CHOICES:
        study.enqueue_trial({**SEED_PARAMS, "epochs": epochs})
    for nz in (8, 11, 16, 20, 24, 32):
        study.enqueue_trial({**SEED_PARAMS, "nz": nz, "seed": 2843 + nz, "epochs": 1500})


def enqueue_from_previous_tsv(study: optuna.Study, tsv_path: Path) -> int:
    """Re-queue round-1 configs that actually returned a CL OF, across the epoch grid."""
    if not tsv_path.is_file():
        return 0
    lines = tsv_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return 0
    hdr = lines[0].split("\t")

    def _to_params(row: dict, epochs: int) -> dict:
        return {
            "nz": int(float(row["nz"])),
            "encoder_depth": int(float(row["encoder_depth"])),
            "width_mult": float(row["width_mult"]),
            "nonlin": row["nonlin"],
            "y_loss_w": float(row["y_loss_w"]),
            "x_loss_w": float(row["x_loss_w"]),
            "recon_loss_w": float(row["recon_loss_w"]),
            "nsteps": int(float(row["nsteps"])),
            "bs": int(float(row["bs"])),
            "lr": float(row["lr"]),
            "seed": int(float(row["seed"])),
            "epochs": int(epochs),
        }

    n = 0
    for line in lines[1:]:
        row = dict(zip(hdr, line.split("\t")))
        try:
            of = float(row.get("cl_of", "nan"))
        except ValueError:
            continue
        if not np.isfinite(of) or of >= 1e5:
            continue
        for epochs in EPOCH_CHOICES:
            study.enqueue_trial(_to_params(row, epochs))
            n += 1
    return n


def fail_stale_running_trials(study: optuna.Study) -> int:
    n = 0
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        try:
            study.tell(trial.number, state=TrialState.FAIL, skip_if_finished=True)
            n += 1
        except Exception as exc:
            print(f"Could not fail stale trial {trial.number}: {exc}", flush=True)
    return n


def save_best_trial(out_dir: Path, trial_number: int, params: dict, result: dict, cl: dict) -> None:
    snap = out_dir / "best"
    snap.mkdir(parents=True, exist_ok=True)
    np.save(snap / "A.npy", result["A"])
    np.save(snap / "B.npy", result["B"])
    np.save(snap / "C.npy", result["C"])
    torch.save(result["best_model"], snap / "model.pth")
    meta = {
        "trial": trial_number,
        "params": params,
        "use_block_diag": True,
        "cl": {k: cl[k] for k in cl if k != "error"},
        "dev_mae_sum": result["dev_mae_sum"],
        "test_mae_sum": result["test_mae_sum"],
    }
    (snap / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "cl_winner.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    archive = out_dir / f"best_trial_{trial_number:03d}"
    archive.mkdir(parents=True, exist_ok=True)
    np.save(archive / "A.npy", result["A"])
    np.save(archive / "B.npy", result["B"])
    np.save(archive / "C.npy", result["C"])
    torch.save(result["best_model"], archive / "model.pth")
    (archive / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class StopIfRequested:
    def __init__(self, stop_path: Path):
        self.stop_path = stop_path

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self.stop_path.exists():
            print(f"Found {self.stop_path}; stopping after trial {trial.number}.", flush=True)
            study.stop()


def _print_cl(tag: str, cl: dict) -> None:
    print(
        f"{tag}  T2D2+T3D3={cl['objective']:.6g}  "
        f"T2D2={cl.get('t2d2')}  T3D3={cl.get('t3d3')}",
        flush=True,
    )
    if cl.get("error"):
        print(f"{tag}  error: {cl['error']}", flush=True)


def eval_existing_notebook_model() -> dict:
    """Score the files T2D2/T3D3 notebooks load (ident noC winner)."""
    A = np.load(DATA_DIR / "A_cstr_separator_C_False.npy")
    B = np.load(DATA_DIR / "B_cstr_separator_C_False.npy")
    C = np.load(DATA_DIR / "C_cstr_separator_C_False.npy")
    state = torch.load(DATA_DIR / "model_cstr_separator_C_False.pth", map_location="cpu")
    params = {**SEED_PARAMS, "nz": int(A.shape[0])}
    print(
        f"Evaluating notebook weights  A {A.shape}  rho={np.max(np.abs(np.linalg.eigvals(A))):.4f}",
        flush=True,
    )
    cl = evaluate_t2t3(A, B, C, params, state, parallel=False)
    _print_cl("existing noC", cl)
    return cl


def retrain_current_config(epochs: int, patience: int, warmup: int, verbose: bool) -> dict:
    """Train example_training10 / noC ident winner, then score T2D2+T3D3."""
    train, dev, test, scaler, scalerU, y_names, _ = load_dataset()
    params = TrialParams(**{k: v for k, v in SEED_PARAMS.items()})
    trial_cfg = TrainConfig(
        epochs=int(epochs),
        patience=int(patience),
        warmup=min(int(warmup), max(1, int(epochs) // 5)),
    )
    print(
        f"Retraining current config  nz={params.nz} seed={params.seed} "
        f"epochs={trial_cfg.epochs} patience={trial_cfg.patience}",
        flush=True,
    )
    result = run_trial(
        params,
        HPO_CFG,
        train,
        dev,
        test,
        scaler,
        scalerU,
        y_names,
        trial_cfg,
        verbose=verbose,
    )
    cl = evaluate_t2t3(
        result["A"],
        result["B"],
        result["C"],
        asdict(params),
        result["best_model"],
        parallel=False,
    )
    _print_cl("retrained current config", cl)
    print(
        f"dev MAE={result['dev_mae_sum']:.4g}  test MAE={result['test_mae_sum']:.4g}  "
        f"trainer_epochs={result['trainer_epochs']}",
        flush=True,
    )
    out_dir = RESULTS_DIR / HPO_CFG.variant / "current_config_retrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "A.npy", result["A"])
    np.save(out_dir / "B.npy", result["B"])
    np.save(out_dir / "C.npy", result["C"])
    torch.save(result["best_model"], out_dir / "model.pth")
    meta = {
        "params": {**asdict(params), "epochs": epochs},
        "cl": {k: cl[k] for k in cl if k != "error"},
        "error": cl.get("error"),
        "dev_mae_sum": result["dev_mae_sum"],
        "test_mae_sum": result["test_mae_sum"],
        "trainer_epochs": result["trainer_epochs"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote retrain snapshot to {out_dir}", flush=True)
    return cl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10_000,
        help="Upper bound; stop earlier with the STOP file or by killing the PID.",
    )
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--eval-existing",
        action="store_true",
        help="Score T2D2+T3D3 on the current notebook A/B/C/model (no search).",
    )
    parser.add_argument(
        "--retrain-current",
        action="store_true",
        help="Train the example_training10 config, then score T2D2+T3D3.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3000,
        help="Epoch budget for --retrain-current (notebook default 3000).",
    )
    add_parallel_cli(parser)
    parser.set_defaults(n_jobs=int(os.environ.get("HPO_N_JOBS", "4")))
    args = parser.parse_args()

    configure_threading(args.blas_threads)
    os.chdir(CONTROL_DIR)

    if args.eval_existing:
        eval_existing_notebook_model()
        return
    if args.retrain_current:
        retrain_current_config(args.epochs, args.patience, args.warmup, args.verbose)
        return

    train, dev, test, scaler, scalerU, y_names, _ = load_dataset()
    setup_mlflow(HPO_CFG)

    out_dir = RESULTS_DIR / HPO_CFG.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_path = out_dir / "STOP"
    if stop_path.exists():
        stop_path.unlink()
    prev_tsv = out_dir / "trials.tsv"
    if prev_tsv.is_file() and not (out_dir / "trials_before_epochs.tsv").is_file():
        prev_tsv.replace(out_dir / "trials_before_epochs.tsv")
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
        n_warm = enqueue_from_previous_tsv(study, out_dir / "trials_before_epochs.tsv")
        enqueue_seed_trials(study)
        print(
            f"New study {HPO_CFG.study_name}: queued {n_warm} prior configs "
            f"(epochs snapped to {EPOCH_CHOICES}) plus epoch-grid seeds.",
            flush=True,
        )
    else:
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        best = min((t.value for t in completed), default=None)
        print(
            f"Continuing {HPO_CFG.study_name} "
            f"({len(study.trials)} existing, best OF={best}).",
            flush=True,
        )

    best_lock = threading.Lock()
    try:
        best_of = [float(study.best_value)]
    except ValueError:
        best_of = [float("inf")]

    def objective(trial: optuna.Trial) -> float:
        params = sample_trial_params(trial)
        epoch_budget = trial.suggest_categorical("epochs", EPOCH_CHOICES)
        trial_cfg = TrainConfig(
            epochs=int(epoch_budget),
            patience=args.patience,
            warmup=min(args.warmup, max(1, int(epoch_budget) // 5)),
        )
        log_params = {
            **asdict(params),
            "epochs": int(epoch_budget),
            "use_block_diag": True,
        }
        with mlflow.start_run(run_name=f"t2t3_trial_{trial.number}"):
            try:
                mlflow.log_params(log_params)
            except Exception as exc:
                print(f"MLflow param log failed (trial {trial.number}): {exc}", flush=True)
            result = run_trial(
                params,
                HPO_CFG,
                train,
                dev,
                test,
                scaler,
                scalerU,
                y_names,
                trial_cfg,
                verbose=args.verbose,
            )
            cl = evaluate_t2t3(
                result["A"],
                result["B"],
                result["C"],
                asdict(params),
                result["best_model"],
                parallel=False,
            )
            if cl.get("error"):
                print(f"trial {trial.number} CL error: {cl['error']}", flush=True)
            try:
                mlflow.log_metric("dev_mae_sum", result["dev_mae_sum"])
                mlflow.log_metric("test_mae_sum", result["test_mae_sum"])
                mlflow.log_metric("cl_of", cl["objective"])
                if "t2d2" in cl:
                    mlflow.log_metric("cl_t2d2", cl["t2d2"])
                    mlflow.log_metric("cl_t3d3", cl["t3d3"])
            except Exception as exc:
                print(f"MLflow metric log failed (trial {trial.number}): {exc}", flush=True)
            trial.set_user_attr("dev_mae_sum", result["dev_mae_sum"])
            trial.set_user_attr("test_mae_sum", result["test_mae_sum"])
            trial.set_user_attr("cl_of", cl["objective"])
            trial.set_user_attr("cl_detail", json.dumps({k: cl[k] for k in cl if k != "error"}))
            result["use_block_diag"] = True
            result["cl_of"] = cl["objective"]
            result["t2d2"] = cl.get("t2d2")
            result["t3d3"] = cl.get("t3d3")
            append_trial_log(HPO_CFG, trial.number, log_params, result)
            with best_lock:
                if cl["objective"] < best_of[0]:
                    best_of[0] = float(cl["objective"])
                    save_best_trial(out_dir, trial.number, log_params, result, cl)
                    print(
                        f"Saved best trial {trial.number}  OF={cl['objective']:.4g} "
                        f"to {out_dir / 'best'} (notebook C_False files left unchanged).",
                        flush=True,
                    )
            return float(cl["objective"])

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
        callbacks=[StopIfRequested(stop_path)],
    )

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    if completed:
        best_trial = min(completed, key=lambda t: t.value)
        print(f"Best combined T2D2+T3D3 OF: {best_trial.value:.6g}")
        print(f"Best params: {best_trial.params}")
        summary = {
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "n_trials": len(study.trials),
            "use_block_diag": True,
        }
        (out_dir / "stage_A_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
