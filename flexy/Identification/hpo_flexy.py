#!/usr/bin/env python3
"""Flexy SISO Koopman HPO (linear C or nonlinear decoder).

Stop either search with ``flexy/data/hpo_results/STOP`` (both) or
``flexy/data/hpo_results/{C,noC}/STOP``. Status:

    python hpo_flexy.py --status
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

IDENT_DIR = Path(__file__).resolve().parent
FLEXY_DATA = IDENT_DIR.parent / "data"
REPO_ROOT = IDENT_DIR.parent.parent
CSTR_IDENT = REPO_ROOT / "CSTRSeparator" / "Identification"
RESULTS_DIR = FLEXY_DATA / "hpo_results"

sys.path.insert(0, str(IDENT_DIR))
sys.path.insert(0, str(CSTR_IDENT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import hpo_common as hc  # noqa: E402
import optuna  # noqa: E402
from optuna.trial import TrialState  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from data_utils import Y_NAMES, load_flexy_splits  # noqa: E402
from helper.koopman import TrainConfig  # noqa: E402

hc.RESULTS_DIR = RESULTS_DIR


def sample_flexy_params(trial: optuna.Trial) -> hc.TrialParams:
    """SISO-downscaled search relative to the CSTR HPO ranges."""
    return hc.TrialParams(
        nz=trial.suggest_int("nz", 2, 12),
        encoder_depth=trial.suggest_int("encoder_depth", 1, 3),
        width_mult=trial.suggest_categorical("width_mult", [0.5, 1.0, 2.0]),
        nonlin=trial.suggest_categorical("nonlin", ["relu", "elu", "gelu"]),
        y_loss_w=trial.suggest_float("y_loss_w", 1.0, 20.0, log=True),
        x_loss_w=trial.suggest_float("x_loss_w", 1.0, 20.0, log=True),
        recon_loss_w=trial.suggest_float("recon_loss_w", 0.5, 20.0, log=True),
        nsteps=trial.suggest_categorical("nsteps", [20, 40, 80]),
        bs=trial.suggest_categorical("bs", [8, 16, 32]),
        lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        seed=trial.suggest_int("seed", 0, 10_000),
    )


def load_flexy_dataset():
    _, _, _, _, train, dev, test = load_flexy_splits(trim_nsteps=20)
    scaler = StandardScaler().fit(train["Y"])
    scaler_u = StandardScaler().fit(train["U"])
    return train, dev, test, scaler, scaler_u, list(Y_NAMES), ["u"]


def hpo_cfg_for(variant: str) -> hc.HPOConfig:
    matrix_c = variant == "C"
    return hc.HPOConfig(
        matrix_C=matrix_c,
        experiment_name=f"koopman_flexy_{variant}",
        study_name=f"koopman_flexy_{variant}",
        variant=variant,
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


def safe_best(study: optuna.Study) -> float | None:
    try:
        return float(study.best_value)
    except ValueError:
        return None


class StopIfRequested:
    def __init__(self, stop_paths: list[Path]):
        self.stop_paths = stop_paths

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        for path in self.stop_paths:
            if path.exists():
                print(f"Found {path}; stopping after trial {trial.number}.", flush=True)
                study.stop()
                return


def maybe_save_best(hpo_cfg: hc.HPOConfig, trial_number: int, result: dict) -> None:
    import joblib
    import numpy as np
    import torch

    out_dir = hc.results_dir_for(hpo_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".best.lock"
    meta_path = out_dir / "best" / "meta.json"
    with open(lock_path, "a", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        prev_mae = None
        if meta_path.exists():
            prev_mae = json.loads(meta_path.read_text(encoding="utf-8")).get("dev_mae")
        if prev_mae is not None and result["dev_mae_sum"] >= float(prev_mae):
            return
        snap = out_dir / "best"
        snap.mkdir(parents=True, exist_ok=True)
        torch.save(result["best_model"], snap / "model.pth")
        np.save(snap / "A.npy", result["A"])
        np.save(snap / "B.npy", result["B"])
        np.save(snap / "C.npy", result["C"])
        (snap / "meta.json").write_text(
            json.dumps(
                {
                    "trial": trial_number,
                    "dev_mae": result["dev_mae_sum"],
                    "test_mae": result["test_mae_sum"],
                    "params": result["params"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def collect_status() -> dict:
    report = {"updated": datetime.now(timezone.utc).isoformat()}
    for variant in ("C", "noC"):
        hpo_cfg = hpo_cfg_for(variant)
        db = RESULTS_DIR / variant / f"{hpo_cfg.study_name}.db"
        block = {
            "variant": variant,
            "n_complete": 0,
            "dev_mae": None,
            "test_mae": None,
            "trial": None,
        }
        if db.exists():
            try:
                study = optuna.load_study(
                    study_name=hpo_cfg.study_name,
                    storage=f"sqlite:///{db.resolve()}",
                )
                complete = [
                    t
                    for t in study.trials
                    if t.state == TrialState.COMPLETE and t.value is not None
                ]
                block["n_complete"] = len(complete)
                block["n_trials"] = len(study.trials)
                if complete:
                    best = min(complete, key=lambda t: float(t.value))
                    block["dev_mae"] = float(best.value)
                    test_mae = best.user_attrs.get("test_mae_sum")
                    block["test_mae"] = (
                        float(test_mae) if test_mae is not None else None
                    )
                    block["trial"] = int(best.number)
            except Exception as exc:
                block["error"] = str(exc)
        report[variant] = block
    return report


def format_status(report: dict) -> str:
    def _mae(block: dict) -> str:
        val = block.get("dev_mae")
        n = block.get("n_complete", 0)
        if val is None:
            return f"pending (n={n})"
        test = block.get("test_mae")
        extra = f", test {float(test):.4g}" if test is not None else ""
        return f"{float(val):.4g}{extra} (n={n})"

    c = report.get("C", {})
    noc = report.get("noC", {})
    return f"C { _mae(c) }   noC { _mae(noc) }"


def run_search(args: argparse.Namespace) -> None:
    variant = args.variant
    hpo_cfg = hpo_cfg_for(variant)
    out_dir = hc.results_dir_for(hpo_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    hc.configure_threading(args.blas_threads)
    train, dev, test, scaler, scaler_u, y_names, _ = load_flexy_dataset()
    hc.setup_mlflow(hpo_cfg)
    study = hc.load_or_create_study(hpo_cfg)
    n_stale = fail_stale_running_trials(study)
    if n_stale:
        print(f"Marked {n_stale} stale RUNNING trials as FAIL.", flush=True)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
    )

    def objective(trial: optuna.Trial) -> float:
        params = sample_flexy_params(trial)
        result = hc.run_trial(
            params,
            hpo_cfg,
            train,
            dev,
            test,
            scaler,
            scaler_u,
            y_names,
            train_cfg,
            verbose=args.verbose,
        )
        trial.set_user_attr("test_mae_sum", result["test_mae_sum"])
        hc.append_trial_log(hpo_cfg, trial.number, asdict(params), result)
        maybe_save_best(hpo_cfg, trial.number, result)
        return result["dev_mae_sum"]

    stop_paths = [RESULTS_DIR / "STOP", out_dir / "STOP"]
    print(
        f"Flexy HPO {variant}: n_jobs={args.n_jobs}, blas={args.blas_threads}, "
        f"        until STOP at {stop_paths[0]} or {stop_paths[1]}",
        flush=True,
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=False,
        callbacks=[StopIfRequested(stop_paths)],
        catch=(Exception,),
    )
    print(f"Stopped {variant}. best dev MAE={safe_best(study)}  n={len(study.trials)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["C", "noC"])
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--n-trials", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.status:
        print(format_status(collect_status()))
        return
    if args.variant is None:
        parser.error("--variant is required unless --status")
    run_search(args)


if __name__ == "__main__":
    main()
