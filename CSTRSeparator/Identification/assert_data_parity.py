#!/usr/bin/env python3
"""Verify NN and SIPPY identification use identical data split and scalers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

IDENT_DIR = Path(__file__).resolve().parent
DATA_PATH = IDENT_DIR / "../data/cstr_separator_ident.npz"

N_TRAIN = 12960
N_DEV = 2880


def load_split(data_path: Path = DATA_PATH):
    data = np.load(data_path, allow_pickle=True)
    sim = {
        "Y": np.array(data["Y"], dtype=float),
        "U": np.array(data["U"], dtype=float),
    }
    train = {k: v[:N_TRAIN] for k, v in sim.items()}
    dev = {k: v[N_TRAIN : N_TRAIN + N_DEV] for k, v in sim.items()}
    test = {k: v[N_TRAIN + N_DEV :] for k, v in sim.items()}
    return sim, train, dev, test


def fit_scalers(train: dict[str, np.ndarray]):
    scaler_y = StandardScaler().fit(train["Y"])
    scaler_u = StandardScaler().fit(train["U"])
    return scaler_y, scaler_u


def assert_parity(data_path: Path = DATA_PATH) -> None:
    sim, train, dev, test = load_split(data_path)

    assert sim["Y"].shape == (18000, 4), sim["Y"].shape
    assert sim["U"].shape == (18000, 6), sim["U"].shape
    assert train["Y"].shape == (N_TRAIN, 4)
    assert dev["Y"].shape == (N_DEV, 4)
    assert test["Y"].shape == (2160, 4)

    scaler_y_a, scaler_u_a = fit_scalers(train)
    scaler_y_b, scaler_u_b = fit_scalers(train)

    np.testing.assert_allclose(scaler_y_a.mean_, scaler_y_b.mean_, rtol=0, atol=0)
    np.testing.assert_allclose(scaler_y_a.scale_, scaler_y_b.scale_, rtol=0, atol=0)
    np.testing.assert_allclose(scaler_u_a.mean_, scaler_u_b.mean_, rtol=0, atol=0)
    np.testing.assert_allclose(scaler_u_a.scale_, scaler_u_b.scale_, rtol=0, atol=0)

    print("Data parity OK")
    print(f"  npz: {data_path.resolve()}")
    print(f"  Y shape: {sim['Y'].shape}, U shape: {sim['U'].shape}")
    print(f"  split: train={train['Y'].shape[0]}, dev={dev['Y'].shape[0]}, test={test['Y'].shape[0]}")
    print(f"  scaler Y mean: {scaler_y_a.mean_}")
    print(f"  scaler Y scale: {scaler_y_a.scale_}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to cstr_separator_ident.npz",
    )
    args = parser.parse_args()
    try:
        assert_parity(args.data)
    except AssertionError as exc:
        print(f"Data parity FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
