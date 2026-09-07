#!/usr/bin/env python3
"""Optuna HPO for Koopman model with MLP decoder (matrix_C=False, autoencoder shape)."""

from __future__ import annotations

import argparse

from hpo_common import HPOConfig, TrainConfig, run_stage_a, run_stage_b, run_stage_c

HPO_CFG = HPOConfig(
    matrix_C=False,
    experiment_name="koopman_cstr_separator_noC",
    study_name="koopman_noC",
    variant="noC",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["A", "B", "C", "all"],
        default="all",
        help="HPO stage to run",
    )
    parser.add_argument("--n-trials", type=int, default=40, help="Stage A trial count")
    parser.add_argument("--top-k", type=int, default=3, help="Stage B top configs")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds for Stage B refinement",
    )
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
    )

    if args.stage in ("A", "all"):
        print("=== Stage A: broad search (noC / matrix_C=False) ===")
        run_stage_a(HPO_CFG, args.n_trials, train_cfg, verbose=args.verbose)

    if args.stage in ("B", "all"):
        print("=== Stage B: seed refinement (noC) ===")
        run_stage_b(HPO_CFG, args.top_k, args.seeds, train_cfg, verbose=args.verbose)

    if args.stage in ("C", "all"):
        print("=== Stage C: final model (noC) ===")
        run_stage_c(
            HPO_CFG,
            train_cfg,
            output_prefix="model_cstr_separator_noC_optimized",
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
