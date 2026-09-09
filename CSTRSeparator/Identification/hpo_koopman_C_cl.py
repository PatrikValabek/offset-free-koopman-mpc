#!/usr/bin/env python3
"""Identify linear-C Koopman models and score them with the CT closed-loop OF.

The Optuna objective is the same closed-loop cost printed in
``CSTRSeparator/Control/CT.ipynb``. Identification MAE is logged but is not
minimized. ``use_block_diag`` is always True (not searched).

Default: new study ``koopman_C_cl_bd``, 2 parallel jobs, run until Ctrl+C.
Logs: ``hpo_results/C_cl_bd/trials.tsv``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import asdict, replace
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
    save_final_model,
    setup_mlflow,
)
from ct_closed_loop import evaluate_or_penalty  # noqa: E402

HPO_CFG = HPOConfig(
    matrix_C=True,
    experiment_name="koopman_cstr_separator_C_cl",
    study_name="koopman_C_cl_bd",
    variant="C_cl_bd",
)

TARGET_OF = 250.0
USE_BLOCK_DIAG = True
EPOCH_GRID = [500, 1000, 1500, 2000, 3000, 4000]

# Identification params that previously won on MAE (C variant).
SEED_PARAMS = {
    "nz": 24,
    "encoder_depth": 2,
    "width_mult": 1.0,
    "nonlin": "gelu",
    "y_loss_w": 1.7519810798188438,
    "x_loss_w": 1.1184530191749238,
    "recon_loss_w": 3.459130306783038,
    "nsteps": 120,
    "bs": 40,
    "lr": 0.00037313513553171717,
    "seed": 2220,
}

# Closed-loop winner from the first intnz map search (trial 55).
BEST_CL_PARAMS = {
    "nz": 11,
    "encoder_depth": 2,
    "width_mult": 0.5,
    "nonlin": "gelu",
    "y_loss_w": 1.1687437202646689,
    "x_loss_w": 1.7642340669321424,
    "recon_loss_w": 4.703307756752022,
    "nsteps": 120,
    "bs": 40,
    "lr": 0.0006654235663016651,
    "seed": 4274,
}


def enqueue_seed_trials(study: optuna.Study) -> None:
    study.enqueue_trial({**SEED_PARAMS, "epochs": 4000})
    study.enqueue_trial({**BEST_CL_PARAMS, "epochs": 4000})
    for nz in (4, 5, 7, 8, 9, 11, 13, 15, 16, 20, 24, 32):
        study.enqueue_trial({**SEED_PARAMS, "nz": nz, "epochs": 4000})
    for epochs in EPOCH_GRID:
        study.enqueue_trial({**BEST_CL_PARAMS, "epochs": epochs})


def enqueue_epoch_budget_trials(study: optuna.Study) -> None:
    """Add epoch-budget trials for the known CL architecture if not already searched."""
    if any("epochs" in (t.params or {}) for t in study.trials):
        return
    for epochs in EPOCH_GRID:
        study.enqueue_trial({**BEST_CL_PARAMS, "epochs": epochs})


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


def save_abc_for_ct(A, B, C) -> None:
    np.save(DATA_DIR / "A_cstr_separator_C_True.npy", A)
    np.save(DATA_DIR / "B_cstr_separator_C_True.npy", B)
    np.save(DATA_DIR / "C_cstr_separator_C_True.npy", C)


class StopBelowTarget:
    def __init__(self, target: float):
        self.target = target

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value is not None and trial.value < self.target:
            print(f"Reached closed-loop OF {trial.value:.4g} < {self.target}; stopping.")
            study.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Trial budget. Default: run until interrupted (Ctrl+C).",
    )
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--target-of", type=float, default=TARGET_OF)
    parser.add_argument(
        "--stop-below-target",
        action="store_true",
        help="Stop the study as soon as a trial has CL OF below --target-of.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-final-retrain",
        action="store_true",
        default=True,
        help="Do not retrain the winner at the end (default).",
    )
    parser.add_argument(
        "--final-retrain",
        action="store_false",
        dest="no_final_retrain",
        help="Retrain the winner after the search stops.",
    )
    parser.add_argument(
        "--eval-existing",
        action="store_true",
        help="Only score the current A/B/C_cstr_separator_C_True.npy (no search).",
    )
    add_parallel_cli(parser)
    args = parser.parse_args()

    configure_threading(args.blas_threads)
    os.chdir(CONTROL_DIR)

    if args.eval_existing:
        A = np.load(DATA_DIR / "A_cstr_separator_C_True.npy")
        B = np.load(DATA_DIR / "B_cstr_separator_C_True.npy")
        C = np.load(DATA_DIR / "C_cstr_separator_C_True.npy")
        out = evaluate_or_penalty(A, B, C, use_block_diag=USE_BLOCK_DIAG)
        print(f"existing C_True  block_diag={USE_BLOCK_DIAG}  OF={out}")
        return

    train_cfg = TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
    )
    train, dev, test, scaler, scalerU, y_names, _ = load_dataset()
    setup_mlflow(HPO_CFG)

    out_dir = RESULTS_DIR / HPO_CFG.variant
    out_dir.mkdir(parents=True, exist_ok=True)
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
            f"New study {HPO_CFG.study_name}: block_diag={USE_BLOCK_DIAG}, "
            f"n_jobs={args.n_jobs}, n_trials={args.n_trials or 'unlimited'}.",
            flush=True,
        )
    else:
        enqueue_epoch_budget_trials(study)
        print(
            f"Continuing study {HPO_CFG.study_name} "
            f"({len(study.trials)} existing trials, best OF={safe_best_value(study)}).",
            flush=True,
        )

    best_lock = threading.Lock()
    _best = safe_best_value(study)
    best_of = [float(_best) if _best is not None else float("inf")]

    def objective(trial: optuna.Trial) -> float:
        params = sample_trial_params(trial)
        epoch_budget = trial.suggest_categorical("epochs", EPOCH_GRID)
        trial_train_cfg = replace(train_cfg, epochs=int(epoch_budget))
        use_block_diag = USE_BLOCK_DIAG
        with mlflow.start_run():
            mlflow.log_params(asdict(params))
            mlflow.log_param("use_block_diag", use_block_diag)
            mlflow.log_param("epoch_budget", epoch_budget)
            result = run_trial(
                params,
                HPO_CFG,
                train,
                dev,
                test,
                scaler,
                scalerU,
                y_names,
                trial_train_cfg,
                verbose=args.verbose,
            )
            cl = evaluate_or_penalty(
                result["A"], result["B"], result["C"], use_block_diag=use_block_diag
            )
            mlflow.log_metric("dev_mae_sum", result["dev_mae_sum"])
            mlflow.log_metric("test_mae_sum", result["test_mae_sum"])
            mlflow.log_metric("cl_of", cl["objective"])
            mlflow.log_metric("trainer_epochs", result["trainer_epochs"])
            trial.set_user_attr("dev_mae_sum", result["dev_mae_sum"])
            trial.set_user_attr("test_mae_sum", result["test_mae_sum"])
            trial.set_user_attr("cl_of", cl["objective"])
            trial.set_user_attr("cl_detail", json.dumps({k: cl[k] for k in cl if k != "error"}))
            result["use_block_diag"] = use_block_diag
            result["cl_of"] = cl["objective"]
            result["cl_tracking"] = cl.get("state_error_cost")
            result["cl_du"] = cl.get("control_increment_cost")
            log_params = {**asdict(params), "epochs": epoch_budget}
            append_trial_log(HPO_CFG, trial.number, log_params, result)
            with best_lock:
                if cl["objective"] < best_of[0]:
                    best_of[0] = float(cl["objective"])
                    save_abc_for_ct(result["A"], result["B"], result["C"])
                    torch.save(result["best_model"], DATA_DIR / "model_cstr_separator_C_True.pth")
                    meta = {
                        "trial": trial.number,
                        "params": {**asdict(params), "epochs": epoch_budget},
                        "use_block_diag": use_block_diag,
                        "cl": {k: cl[k] for k in cl if k != "error"},
                        "dev_mae_sum": result["dev_mae_sum"],
                        "test_mae_sum": result["test_mae_sum"],
                    }
                    (out_dir / "cl_winner.json").write_text(
                        json.dumps(meta, indent=2), encoding="utf-8"
                    )
                    print(
                        f"New best trial {trial.number}: CL OF={cl['objective']:.4g} "
                        "(saved A/B/C and model from this trial, not a retrain).",
                        flush=True,
                    )
            return float(cl["objective"])

    callbacks = [StopBelowTarget(args.target_of)] if args.stop_below_target else []
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
            callbacks=callbacks,
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
        "target_of": args.target_of,
        "hit_target": best_value is not None and best_value < args.target_of,
        "use_block_diag": USE_BLOCK_DIAG,
    }
    (out_dir / "stage_A_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if best_value is not None and not args.no_final_retrain:
        best = dict(study.best_params)
        best.pop("use_block_diag", None)
        params = TrialParams(**best)
        meta = save_final_model(
            HPO_CFG,
            params,
            train_cfg,
            output_prefix="model_cstr_separator_C_cl_optimized",
            verbose=args.verbose,
        )
        winner = run_trial(
            params, HPO_CFG, train, dev, test, scaler, scalerU, y_names, train_cfg
        )
        save_abc_for_ct(winner["A"], winner["B"], winner["C"])
        cl = evaluate_or_penalty(winner["A"], winner["B"], winner["C"], USE_BLOCK_DIAG)
        meta["cl"] = cl
        meta["use_block_diag"] = USE_BLOCK_DIAG
        (out_dir / "stage_C_final_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("Final retrain CL OF:", cl)
    elif args.no_final_retrain:
        print("Skipping final retrain (--no-final-retrain).")


if __name__ == "__main__":
    main()
