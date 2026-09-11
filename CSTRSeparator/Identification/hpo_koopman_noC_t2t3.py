#!/usr/bin/env python3
"""Identify noC Koopman models (example_training10) and score T2D2 + T3D3.

Optuna minimizes OF_T2D2 + OF_T3D3 from the Control notebooks.
``use_block_diag`` is always True (same as the notebooks).

Training uses ``--n-jobs`` concurrent **processes** (default 8, one BLAS thread
each). T2D2 and T3D3 are scored sequentially inside each process. Optuna's
thread pool is not used: CVXPY, Gurobi, and MLflow are not thread-safe.

The actual best-trial weights and A/B/C are saved (no retrain).
Previous ``noC_t2t3`` results are left untouched; this run writes
``hpo_results/noC_t2t3_v2``. Stop with ``hpo_results/noC_t2t3_v2/STOP``
or by killing the PID.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
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
    experiment_name="koopman_cstr_separator_noC_t2t3_v2",
    study_name="koopman_noC_t2t3_v2",
    variant="noC_t2t3_v2",
)

PREVIOUS_TSV = RESULTS_DIR / "noC_t2t3" / "trials.tsv"

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
    for epochs in (3000, 4000):
        study.enqueue_trial({**SEED_PARAMS, "epochs": epochs})


def _row_to_params(row: dict, epochs: int | None = None) -> dict:
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
        "epochs": int(epochs if epochs is not None else float(row["epochs"])),
    }


def enqueue_top_from_previous_tsv(study: optuna.Study, tsv_path: Path, top_k: int = 10) -> int:
    """Queue the previous-run winners under the new closed-loop weights."""
    if not tsv_path.is_file():
        return 0
    lines = tsv_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return 0
    hdr = lines[0].split("\t")
    ranked: list[tuple[float, dict]] = []
    for line in lines[1:]:
        row = dict(zip(hdr, line.split("\t")))
        try:
            of = float(row.get("cl_of", "nan"))
        except ValueError:
            continue
        if not np.isfinite(of) or of >= 1e5:
            continue
        ranked.append((of, row))
    ranked.sort(key=lambda item: item[0])

    seen: set[tuple] = set()
    n = 0
    for _, row in ranked:
        if n >= top_k:
            break
        params = _row_to_params(row)
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        study.enqueue_trial(params)
        n += 1
        # Epoch budget moved the previous OF a lot; also try the other long budget.
        alt = 3000 if params["epochs"] >= 4000 else 4000
        alt_params = {**params, "epochs": alt}
        alt_key = tuple(sorted(alt_params.items()))
        if alt_key not in seen:
            seen.add(alt_key)
            study.enqueue_trial(alt_params)
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


def make_storage(out_dir: Path) -> optuna.storages.RDBStorage:
    db = out_dir / f"{HPO_CFG.study_name}.db"
    return optuna.storages.RDBStorage(
        url=f"sqlite:///{db.resolve()}",
        engine_kwargs={"connect_args": {"timeout": 120}},
    )


def load_study(out_dir: Path) -> optuna.Study:
    return optuna.load_study(
        study_name=HPO_CFG.study_name,
        storage=make_storage(out_dir),
    )


def save_best_trial(out_dir: Path, trial_number: int, params: dict, result: dict, cl: dict) -> None:
    snap = out_dir / "best"
    snap.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / "best.lock"
    with open(lock_path, "a", encoding="utf-8") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        winner_path = out_dir / "cl_winner.json"
        if winner_path.is_file():
            prev = json.loads(winner_path.read_text(encoding="utf-8"))
            prev_of = float(prev.get("cl", {}).get("objective", float("inf")))
            if float(cl["objective"]) >= prev_of:
                return
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
        winner_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        archive = out_dir / f"best_trial_{trial_number:03d}"
        archive.mkdir(parents=True, exist_ok=True)
        np.save(archive / "A.npy", result["A"])
        np.save(archive / "B.npy", result["B"])
        np.save(archive / "C.npy", result["C"])
        torch.save(result["best_model"], archive / "model.pth")
        (archive / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(
            f"Saved best trial {trial_number}  OF={cl['objective']:.4g} "
            f"to {snap} (notebook C_False files left unchanged).",
            flush=True,
        )


class StopIfRequested:
    def __init__(self, stop_path: Path, n_trials: int):
        self.stop_path = stop_path
        self.n_trials = n_trials

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self.stop_path.exists():
            print(f"Found {self.stop_path}; stopping after trial {trial.number}.", flush=True)
            study.stop()
            return
        n_done = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
        if n_done >= self.n_trials:
            print(f"Reached {n_done} complete trials; worker stopping.", flush=True)
            study.stop()


def run_worker(args: argparse.Namespace, out_dir: Path, stop_path: Path) -> None:
    """One process, one Optuna thread: train then sequential T2D2+T3D3."""
    train, dev, test, scaler, scalerU, y_names, _ = load_dataset()
    setup_mlflow(HPO_CFG)
    study = load_study(out_dir)

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
            with mlflow.start_run(run_name=f"t2t3_trial_{trial.number}"):
                mlflow.log_params(log_params)
                mlflow.log_metric("dev_mae_sum", result["dev_mae_sum"])
                mlflow.log_metric("test_mae_sum", result["test_mae_sum"])
                mlflow.log_metric("cl_of", cl["objective"])
                if cl.get("t2d2") is not None and np.isfinite(cl["t2d2"]):
                    mlflow.log_metric("cl_t2d2", cl["t2d2"])
                    mlflow.log_metric("cl_t3d3", cl["t3d3"])
        except Exception as exc:
            print(f"MLflow log failed (trial {trial.number}): {exc}", flush=True)
        trial.set_user_attr("dev_mae_sum", result["dev_mae_sum"])
        trial.set_user_attr("test_mae_sum", result["test_mae_sum"])
        trial.set_user_attr("cl_of", cl["objective"])
        trial.set_user_attr("cl_detail", json.dumps({k: cl[k] for k in cl if k != "error"}))
        result["use_block_diag"] = True
        result["cl_of"] = cl["objective"]
        result["t2d2"] = cl.get("t2d2")
        result["t3d3"] = cl.get("t3d3")
        append_trial_log(HPO_CFG, trial.number, log_params, result)
        save_best_trial(out_dir, trial.number, log_params, result, cl)
        return float(cl["objective"])

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=1,
        show_progress_bar=False,
        callbacks=[StopIfRequested(stop_path, args.n_trials)],
    )


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
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    add_parallel_cli(parser)
    parser.set_defaults(n_jobs=int(os.environ.get("HPO_N_JOBS", "8")))
    args = parser.parse_args()

    configure_threading(args.blas_threads)
    os.chdir(CONTROL_DIR)

    if args.eval_existing:
        eval_existing_notebook_model()
        return
    if args.retrain_current:
        retrain_current_config(args.epochs, args.patience, args.warmup, args.verbose)
        return

    out_dir = RESULTS_DIR / HPO_CFG.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_path = out_dir / "STOP"

    if args.worker:
        run_worker(args, out_dir, stop_path)
        return

    if stop_path.exists():
        stop_path.unlink()

    study = optuna.create_study(
        study_name=HPO_CFG.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=make_storage(out_dir),
        load_if_exists=True,
    )
    n_stale = fail_stale_running_trials(study)
    if n_stale:
        print(f"Marked {n_stale} stale RUNNING trials as FAIL.", flush=True)
    if len(study.trials) == 0:
        n_warm = enqueue_top_from_previous_tsv(study, PREVIOUS_TSV, top_k=10)
        enqueue_seed_trials(study)
        print(
            f"New study {HPO_CFG.study_name} in {out_dir}: queued top {n_warm} "
            f"configs from {PREVIOUS_TSV} (plus 3000/4000 twins) and ident seeds.",
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

    n_jobs = max(1, int(args.n_jobs))
    print(f"Launching {n_jobs} Optuna worker processes (n_jobs=1 each).", flush=True)
    worker_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--n-trials",
        str(args.n_trials),
        "--patience",
        str(args.patience),
        "--warmup",
        str(args.warmup),
        "--n-jobs",
        "1",
        "--blas-threads",
        str(args.blas_threads),
    ]
    if args.verbose:
        worker_cmd.append("--verbose")
    procs = [subprocess.Popen(worker_cmd) for _ in range(n_jobs)]

    def _shutdown(signum, _frame):
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    rc = 0
    for proc in procs:
        proc.wait()
        if proc.returncode:
            rc = proc.returncode
    study = load_study(out_dir)
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
    if rc:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
