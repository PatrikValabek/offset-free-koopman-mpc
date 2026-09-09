#!/usr/bin/env python3
"""Shared utilities for Koopman HPO scripts (C and noC variants)."""

from __future__ import annotations

import fcntl
import json
import os

# macOS OpenMP workaround: allow two libomp copies (NumPy + PyTorch).
# Thread counts: HPO_BLAS_THREADS (default 1). Parallel trials: HPO_N_JOBS.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_BLAS_THREADS = os.environ.get("HPO_BLAS_THREADS", "1")
for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, _BLAS_THREADS)

import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import mlflow
import numpy as np
import optuna
import torch
from sklearn.preprocessing import StandardScaler

IDENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = IDENT_DIR.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
DATA_PATH = IDENT_DIR / "../data/cstr_separator_ident.npz"
RESULTS_DIR = IDENT_DIR / "../data/hpo_results"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from helper.koopman import (  # noqa: E402
    LossWeights,
    TrainConfig,
    build_koopman_problem,
    evaluate_mae_physical,
    extract_matrices,
    get_data_loaders,
    load_cstr_separator_data,
    train_koopman,
)


_BLAS_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_threading(blas_threads: int | None = None) -> int:
    """OpenMP workaround + thread counts for this process (and future children).

    ``KMP_DUPLICATE_LIB_OK=TRUE`` is the macOS fix for dual libomp.
    Prefer ``n_jobs>1`` (separate processes) over many BLAS threads in one
    process. If both are >1, you will oversubscribe CPUs.
    """
    n = int(os.environ.get("HPO_BLAS_THREADS", "1") if blas_threads is None else blas_threads)
    n = max(1, n)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["HPO_BLAS_THREADS"] = str(n)
    for key in _BLAS_ENV_KEYS:
        os.environ[key] = str(n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, min(n, 4)))
    except RuntimeError:
        pass
    return n


def add_parallel_cli(parser) -> None:
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("HPO_N_JOBS", "8")),
        help="Parallel Optuna trials (process pool). -1 = all CPUs. Default 8.",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=int(os.environ.get("HPO_BLAS_THREADS", "1")),
        help="PyTorch/BLAS threads per trial. Use 1 when --n-jobs>1.",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class HPOConfig:
    matrix_C: bool
    experiment_name: str
    study_name: str
    variant: str  # "C" or "noC"


@dataclass
class TrialParams:
    nz: int
    encoder_depth: int
    width_mult: float
    nonlin: str
    y_loss_w: float
    x_loss_w: float
    recon_loss_w: float
    nsteps: int
    bs: int
    lr: float
    seed: int = 42


TRIAL_LOG_COLUMNS = [
    "trial",
    "nz",
    "encoder_depth",
    "width_mult",
    "nonlin",
    "nsteps",
    "bs",
    "lr",
    "y_loss_w",
    "x_loss_w",
    "recon_loss_w",
    "seed",
    "epochs",
    "dev_mae",
    "test_mae",
]

CL_TRIAL_LOG_COLUMNS = TRIAL_LOG_COLUMNS + [
    "use_block_diag",
    "cl_of",
    "cl_tracking",
    "cl_du",
]

T2T3_TRIAL_LOG_COLUMNS = TRIAL_LOG_COLUMNS + [
    "use_block_diag",
    "cl_of",
    "t2d2",
    "t3d3",
]


def trial_log_path(hpo_cfg: HPOConfig) -> Path:
    return results_dir_for(hpo_cfg) / "trials.tsv"


def trial_log_columns(result: dict[str, Any]) -> list[str]:
    if "t2d2" in result:
        return T2T3_TRIAL_LOG_COLUMNS
    if "cl_of" in result:
        return CL_TRIAL_LOG_COLUMNS
    return TRIAL_LOG_COLUMNS


def _fmt(value: Any, spec: str | None = None) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if spec is None:
        return str(value)
    return f"{float(value):{spec}}"


def format_trial_row(trial_number: int, params: dict, result: dict[str, Any]) -> str:
    values = {
        "trial": trial_number,
        "nz": params.get("nz"),
        "encoder_depth": params.get("encoder_depth"),
        "width_mult": params.get("width_mult"),
        "nonlin": params.get("nonlin"),
        "nsteps": params.get("nsteps"),
        "bs": params.get("bs"),
        "lr": _fmt(params.get("lr"), ".4g"),
        "y_loss_w": _fmt(params.get("y_loss_w"), ".3g"),
        "x_loss_w": _fmt(params.get("x_loss_w"), ".3g"),
        "recon_loss_w": _fmt(params.get("recon_loss_w"), ".3g"),
        "seed": params.get("seed"),
        "epochs": params.get("epochs", result.get("trainer_epochs", "")),
        "dev_mae": _fmt(result.get("dev_mae_sum"), ".4g"),
        "test_mae": _fmt(result.get("test_mae_sum"), ".4g"),
        "use_block_diag": result.get("use_block_diag", params.get("use_block_diag", "")),
        "cl_of": _fmt(result.get("cl_of"), ".4g"),
        "cl_tracking": _fmt(result.get("cl_tracking"), ".4g"),
        "cl_du": _fmt(result.get("cl_du"), ".4g"),
        "t2d2": _fmt(result.get("t2d2"), ".4g"),
        "t3d3": _fmt(result.get("t3d3"), ".4g"),
    }
    return "\t".join(str(values[c]) for c in trial_log_columns(result))


def format_trial_pretty(trial_number: int, params: dict, result: dict[str, Any]) -> str:
    settings = (
        f"trial {trial_number:03d}  "
        f"nz={params.get('nz')} depth={params.get('encoder_depth')} "
        f"width={params.get('width_mult')} {params.get('nonlin')}  "
        f"nsteps={params.get('nsteps')} bs={params.get('bs')} "
        f"epochs={params.get('epochs', result.get('trainer_epochs', ''))} "
        f"lr={_fmt(params.get('lr'), '.3g')}  "
        f"y/x/recon={_fmt(params.get('y_loss_w'), '.2g')}/"
        f"{_fmt(params.get('x_loss_w'), '.2g')}/"
        f"{_fmt(params.get('recon_loss_w'), '.2g')}"
    )
    if "t2d2" in result:
        return (
            f"{settings}  block_diag={result.get('use_block_diag')}  "
            f"->  T2D2+T3D3={_fmt(result.get('cl_of'), '.4g')}  "
            f"T2D2={_fmt(result.get('t2d2'), '.4g')}  "
            f"T3D3={_fmt(result.get('t3d3'), '.4g')}  "
            f"(dev MAE={_fmt(result.get('dev_mae_sum'), '.4g')})"
        )
    if "cl_of" in result:
        return (
            f"{settings}  block_diag={result.get('use_block_diag')}  "
            f"->  CL OF={_fmt(result.get('cl_of'), '.4g')}  "
            f"tracking={_fmt(result.get('cl_tracking'), '.4g')}  "
            f"du={_fmt(result.get('cl_du'), '.4g')}  "
            f"(dev MAE={_fmt(result.get('dev_mae_sum'), '.4g')})"
        )
    return (
        f"{settings}  "
        f"->  dev MAE={_fmt(result.get('dev_mae_sum'), '.4g')}  "
        f"test MAE={_fmt(result.get('test_mae_sum'), '.4g')}"
    )


def append_trial_log(hpo_cfg: HPOConfig, trial_number: int, params: dict, result: dict[str, Any]) -> None:
    path = trial_log_path(hpo_cfg)
    columns = trial_log_columns(result)
    header = "\t".join(columns) + "\n"
    row = format_trial_row(trial_number, params, result) + "\n"
    pretty = format_trial_pretty(trial_number, params, result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.tell() == 0:
            f.write(header)
        f.write(row)
        f.flush()
    print(pretty, flush=True)


def results_dir_for(hpo_cfg: HPOConfig) -> Path:
    out = RESULTS_DIR / hpo_cfg.variant
    out.mkdir(parents=True, exist_ok=True)
    return out


def storage_uri(hpo_cfg: HPOConfig) -> str:
    db_path = results_dir_for(hpo_cfg) / f"{hpo_cfg.study_name}.db"
    return f"sqlite:///{db_path.resolve()}"


def load_dataset(data_path: Path = DATA_PATH):
    _, train, dev, test, y_names, u_names = load_cstr_separator_data(str(data_path))
    scaler = StandardScaler().fit(train["Y"])
    scalerU = StandardScaler().fit(train["U"])
    return train, dev, test, scaler, scalerU, y_names, u_names


def sample_trial_params(trial: optuna.Trial) -> TrialParams:
    return TrialParams(
        nz=trial.suggest_int("nz", 4, 32),
        encoder_depth=trial.suggest_int("encoder_depth", 1, 3),
        width_mult=trial.suggest_categorical("width_mult", [0.5, 1.0, 2.0]),
        nonlin=trial.suggest_categorical("nonlin", ["relu", "elu", "gelu"]),
        y_loss_w=trial.suggest_float("y_loss_w", 1.0, 20.0, log=True),
        x_loss_w=trial.suggest_float("x_loss_w", 1.0, 20.0, log=True),
        recon_loss_w=trial.suggest_float("recon_loss_w", 0.5, 5.0, log=True),
        nsteps=trial.suggest_categorical("nsteps", [20, 40, 80, 120]),
        bs=trial.suggest_categorical("bs", [40, 80, 160]),
        lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        seed=trial.suggest_int("seed", 0, 10_000),
    )


def make_eval_dict(sim_dict: dict, nsteps: int, scaler, scalerU) -> dict:
    _, _, eval_dict, _ = get_data_loaders(
        sim_dict,
        sim_dict,
        sim_dict,
        nsteps,
        bs=1,
        scaler=scaler,
        scalerU=scalerU,
    )
    return eval_dict


def run_trial(
    params: TrialParams,
    hpo_cfg: HPOConfig,
    train: dict,
    dev: dict,
    test: dict,
    scaler,
    scalerU,
    y_names: list[str],
    train_cfg: TrainConfig,
    verbose: bool = False,
) -> dict[str, Any]:
    set_seed(params.seed)
    ny = train["Y"].shape[1]
    nu = train["U"].shape[1]

    train_loader, dev_loader, test_dict, train_full_dict = get_data_loaders(
        train,
        dev,
        test,
        params.nsteps,
        params.bs,
        scaler,
        scalerU,
    )
    dev_eval_dict = make_eval_dict(dev, params.nsteps, scaler, scalerU)

    loss_weights = LossWeights(
        y_loss=params.y_loss_w,
        x_loss=params.x_loss_w,
        onestep_loss=1.0,
        reconstruction_loss=params.recon_loss_w,
    )
    cfg = TrainConfig(
        nsteps=params.nsteps,
        bs=params.bs,
        lr=params.lr,
        epochs=train_cfg.epochs,
        warmup=train_cfg.warmup,
        patience=train_cfg.patience,
    )

    problem, K, f_u, f_y_inv = build_koopman_problem(
        ny=ny,
        nu=nu,
        nz=params.nz,
        matrix_C=hpo_cfg.matrix_C,
        encoder_depth=params.encoder_depth,
        width_mult=params.width_mult,
        nonlin=params.nonlin,
        loss_weights=loss_weights,
        nsteps=params.nsteps,
    )

    best_model, trainer = train_koopman(
        problem,
        train_loader,
        dev_loader,
        dev_eval_dict,
        cfg,
        verbose=verbose,
    )

    dev_metrics = evaluate_mae_physical(problem, dev_eval_dict, scaler, y_names)
    test_metrics = evaluate_mae_physical(problem, test_dict, scaler, y_names)
    A = K.weight.detach().numpy()
    B = f_u.weight.detach().numpy()
    if hpo_cfg.matrix_C:
        C = f_y_inv.weight.detach().numpy()
    else:
        problem.nodes[3].nsteps = train_full_dict["Y"].shape[1]
        train_outputs = problem.step(train_full_dict)
        Y = train_full_dict["Y"].reshape(-1, ny).detach().numpy()
        Z = train_outputs["x"][:, :-1, :].detach().numpy().reshape(-1, params.nz)
        C, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
        C = C.T

    return {
        "params": asdict(params),
        "best_model": best_model,
        "dev_mae_sum": dev_metrics["mae_sum"],
        "dev_mae_mean": dev_metrics["mae_mean"],
        "dev_mae_per_output": dev_metrics["mae_per_output"],
        "test_mae_sum": test_metrics["mae_sum"],
        "test_mae_mean": test_metrics["mae_mean"],
        "test_mae_per_output": test_metrics["mae_per_output"],
        "A": A,
        "B": B,
        "C": C,
        "trainer_epochs": trainer.current_epoch,
    }


def create_objective(
    hpo_cfg: HPOConfig,
    train: dict,
    dev: dict,
    test: dict,
    scaler,
    scalerU,
    y_names: list[str],
    train_cfg: TrainConfig,
    verbose: bool = False,
):
    def objective(trial: optuna.Trial) -> float:
        params = sample_trial_params(trial)
        with mlflow.start_run(nested=True):
            mlflow.log_params(asdict(params))
            mlflow.log_param("matrix_C", hpo_cfg.matrix_C)
            mlflow.log_param("variant", hpo_cfg.variant)
            result = run_trial(
                params,
                hpo_cfg,
                train,
                dev,
                test,
                scaler,
                scalerU,
                y_names,
                train_cfg,
                verbose=verbose,
            )
            mlflow.log_metric("dev_mae_sum", result["dev_mae_sum"])
            mlflow.log_metric("dev_mae_mean", result["dev_mae_mean"])
            mlflow.log_metric("test_mae_sum", result["test_mae_sum"])
            mlflow.log_metric("test_mae_mean", result["test_mae_mean"])
            mlflow.log_metric("trainer_epochs", result["trainer_epochs"])
            for i, v in enumerate(result["dev_mae_per_output"]):
                mlflow.log_metric(f"dev_mae_{y_names[i]}", float(v))
            trial.set_user_attr("test_mae_sum", result["test_mae_sum"])
            append_trial_log(hpo_cfg, trial.number, asdict(params), result)
            return result["dev_mae_sum"]

    return objective


def load_or_create_study(hpo_cfg: HPOConfig) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    return optuna.create_study(
        study_name=hpo_cfg.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_uri(hpo_cfg),
        load_if_exists=True,
    )


def setup_mlflow(hpo_cfg: HPOConfig) -> None:
    mlruns = (RESULTS_DIR / "mlruns").resolve()
    mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns.as_uri())
    mlflow.set_experiment(hpo_cfg.experiment_name)


def run_stage_a(
    hpo_cfg: HPOConfig,
    n_trials: int,
    train_cfg: TrainConfig,
    verbose: bool = False,
    n_jobs: int = 1,
    blas_threads: int = 1,
) -> optuna.Study:
    configure_threading(blas_threads)
    train, dev, test, scaler, scalerU, y_names, _ = load_dataset()
    setup_mlflow(hpo_cfg)

    study = load_or_create_study(hpo_cfg)
    with mlflow.start_run(run_name=f"{hpo_cfg.study_name}_stage_A"):
        mlflow.log_param("variant", hpo_cfg.variant)
        mlflow.log_param("matrix_C", hpo_cfg.matrix_C)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("n_jobs", n_jobs)
        mlflow.log_param("blas_threads", blas_threads)
        objective = create_objective(
            hpo_cfg,
            train,
            dev,
            test,
            scaler,
            scalerU,
            y_names,
            train_cfg,
            verbose=verbose,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )

    summary = {
        "study_name": hpo_cfg.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    summary_path = results_dir_for(hpo_cfg) / "stage_A_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Stage A complete. Best dev MAE sum={study.best_value:.6g}")
    print(f"Summary: {summary_path}")
    return study


def top_trials(study: optuna.Study, n: int = 3) -> list[optuna.trial.FrozenTrial]:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    complete.sort(key=lambda t: t.value)
    return complete[:n]


def run_stage_b(
    hpo_cfg: HPOConfig,
    top_k: int,
    seeds: list[int],
    train_cfg: TrainConfig,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    configure_threading()
    study = load_or_create_study(hpo_cfg)
    train, dev, test, scaler, scalerU, y_names, _ = load_dataset()

    base_trials = top_trials(study, top_k)
    if not base_trials:
        raise RuntimeError("No completed trials found for Stage B")

    refined = []
    for trial in base_trials:
        base = dict(trial.params)
        seed_scores = []
        for seed in seeds:
            params = TrialParams(**{**base, "seed": seed})
            result = run_trial(
                params,
                hpo_cfg,
                train,
                dev,
                test,
                scaler,
                scalerU,
                y_names,
                train_cfg,
                verbose=verbose,
            )
            seed_scores.append(result)
        refined.append(
            {
                "params": base,
                "mean_dev_mae_sum": float(np.mean([r["dev_mae_sum"] for r in seed_scores])),
                "std_dev_mae_sum": float(np.std([r["dev_mae_sum"] for r in seed_scores])),
                "mean_test_mae_sum": float(np.mean([r["test_mae_sum"] for r in seed_scores])),
                "seed_runs": [
                    {
                        "seed": seeds[i],
                        "dev_mae_sum": seed_scores[i]["dev_mae_sum"],
                        "test_mae_sum": seed_scores[i]["test_mae_sum"],
                    }
                    for i in range(len(seeds))
                ],
            }
        )

    refined.sort(key=lambda r: r["mean_dev_mae_sum"])
    out_path = results_dir_for(hpo_cfg) / "stage_B_refined.json"
    out_path.write_text(json.dumps(refined, indent=2), encoding="utf-8")
    print(f"Stage B complete. Best mean dev MAE sum={refined[0]['mean_dev_mae_sum']:.6g}")
    print(f"Results: {out_path}")
    return refined


def run_stage_c(
    hpo_cfg: HPOConfig,
    train_cfg: TrainConfig,
    output_prefix: str,
    verbose: bool = False,
) -> dict[str, Any]:
    refine_path = results_dir_for(hpo_cfg) / "stage_B_refined.json"
    if refine_path.exists():
        refined = json.loads(refine_path.read_text(encoding="utf-8"))
        best_params = refined[0]["params"]
    else:
        study = load_or_create_study(hpo_cfg)
        best_params = study.best_params

    params = TrialParams(**best_params)
    return save_final_model(
        hpo_cfg,
        params,
        train_cfg,
        output_prefix=output_prefix,
        verbose=verbose,
    )


def save_final_model(
    hpo_cfg: HPOConfig,
    params: TrialParams,
    train_cfg: TrainConfig,
    output_prefix: str,
    verbose: bool = False,
) -> dict[str, Any]:
    configure_threading()
    train, dev, test, scaler, scalerU, y_names, u_names = load_dataset()
    result = run_trial(
        params,
        hpo_cfg,
        train,
        dev,
        test,
        scaler,
        scalerU,
        y_names,
        train_cfg,
        verbose=verbose,
    )

    data_dir = IDENT_DIR / "../data"
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result["best_model"], data_dir / f"{output_prefix}.pth")
    np.save(data_dir / f"A_{output_prefix}.npy", result["A"])
    np.save(data_dir / f"B_{output_prefix}.npy", result["B"])
    np.save(data_dir / f"C_{output_prefix}.npy", result["C"])
    joblib.dump(scaler, data_dir / "scaler_cstr_separator.pkl")
    joblib.dump(scalerU, data_dir / "scalerU_cstr_separator.pkl")

    meta = {
        "variant": hpo_cfg.variant,
        "matrix_C": hpo_cfg.matrix_C,
        "params": asdict(params),
        "dev_mae_sum": result["dev_mae_sum"],
        "test_mae_sum": result["test_mae_sum"],
        "dev_mae_by_name": {
            name: float(v)
            for name, v in zip(y_names, result["dev_mae_per_output"])
        },
        "test_mae_by_name": {
            name: float(v)
            for name, v in zip(y_names, result["test_mae_per_output"])
        },
        "y_names": y_names,
        "u_names": u_names,
    }
    meta_path = data_dir / f"{output_prefix}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    final_meta_path = results_dir_for(hpo_cfg) / "stage_C_final_meta.json"
    final_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved final model artifacts with prefix {output_prefix}")
    return meta
