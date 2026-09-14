"""Load FlexBend SISO step tests from flexy/data/*.mat into train/dev/test splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

IDENT_DIR = Path(__file__).resolve().parent
DATA_DIR = IDENT_DIR.parent / "data"
REPO_ROOT = IDENT_DIR.parent.parent

# pch1–6 are Ts = 0.05 s. pch7–9 are Ts = 0.01 s and are resampled to 0.05 s.
TS = 0.05
TS_FINE = 0.01
SKIP_S = 1.0  # drop the dead-sensor / calibration startup
NSTEPS = 40  # 2 s windows; experiments are trimmed to a multiple of this

EXPERIMENTS = {
    1: {"file": "pch1.mat", "var": "pch1", "label": "40 -> 80", "ts": 0.05},
    2: {"file": "pch2.mat", "var": "pch2", "label": "50 -> 80", "ts": 0.05},
    3: {"file": "pch3.mat", "var": "pch3", "label": "20 -> 40", "ts": 0.05},
    4: {"file": "pch4.mat", "var": "pch4", "label": "30 -> 10", "ts": 0.05},
    5: {"file": "pch5.mat", "var": "pch5", "label": "70 -> 30", "ts": 0.05},
    6: {"file": "pch6.mat", "var": "pch6", "label": "40 -> 90", "ts": 0.05},
    7: {"file": "pch7.mat", "var": "pch7", "label": "40 -> 80", "ts": 0.01},
    8: {"file": "pch8.mat", "var": "pch8", "label": "50 -> 80", "ts": 0.01},
    9: {"file": "pch9.mat", "var": "pch9", "label": "35 -> 55", "ts": 0.01},
}

TRAIN_IDS = (1, 2, 3, 4, 5)
DEV_IDS = (6,)
TEST_IDS = (7, 8, 9)

Y_NAMES = ["y"]
U_NAMES = ["u"]


def _resample_to_ts(t: np.ndarray, y: np.ndarray, u: np.ndarray, ts_src: float, ts_dst: float):
    if np.isclose(ts_src, ts_dst):
        return t, y, u
    factor = int(round(ts_dst / ts_src))
    if not np.isclose(factor * ts_src, ts_dst):
        raise ValueError(f"Cannot resample {ts_src} s to {ts_dst} s")
    return t[::factor], y[::factor], u[::factor]


def load_experiment(
    exp_id: int,
    data_dir: Path | None = None,
    trim_nsteps: int | None = NSTEPS,
) -> dict:
    meta = EXPERIMENTS[exp_id]
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    raw = loadmat(data_dir / meta["file"], squeeze_me=False)
    if meta["var"] not in raw:
        raise KeyError(f"{meta['var']} not in {meta['file']}: {list(raw)}")
    arr = np.asarray(raw[meta["var"]], dtype=float)
    t, y, u = arr[:, 0], arr[:, 1], arr[:, 2]
    t, y, u = _resample_to_ts(t, y, u, meta["ts"], TS)
    skip_n = int(round(SKIP_S / TS))
    t, y, u = t[skip_n:], y[skip_n:], u[skip_n:]
    if trim_nsteps is not None:
        n = (len(y) // trim_nsteps) * trim_nsteps
        t, y, u = t[:n], y[:n], u[:n]
    return {
        "id": exp_id,
        "label": meta["label"],
        "t": t.reshape(-1),
        "Y": y.reshape(-1, 1),
        "U": u.reshape(-1, 1),
        "Ts": TS,
    }


def _stack(exps: list[dict]) -> dict:
    return {
        "Y": np.vstack([e["Y"] for e in exps]),
        "U": np.vstack([e["U"] for e in exps]),
    }


def load_flexy_splits(data_dir: Path | None = None, trim_nsteps: int | None = NSTEPS):
    """Return (experiments, train, dev, test) with Y/U as (N, 1)."""
    experiments = {
        i: load_experiment(i, data_dir=data_dir, trim_nsteps=trim_nsteps)
        for i in EXPERIMENTS
    }
    train_exps = [experiments[i] for i in TRAIN_IDS]
    dev_exps = [experiments[i] for i in DEV_IDS]
    test_exps = [experiments[i] for i in TEST_IDS]
    train = _stack(train_exps)
    dev = _stack(dev_exps)
    test = _stack(test_exps)
    return experiments, train_exps, dev_exps, test_exps, train, dev, test


def save_flexy_npz(path: Path | None = None, data_dir: Path | None = None) -> Path:
    experiments, train_exps, dev_exps, test_exps, train, dev, test = load_flexy_splits(
        data_dir=data_dir
    )
    path = Path(path) if path is not None else DATA_DIR / "flexy_ident.npz"
    np.savez(
        path,
        Y=np.vstack([train["Y"], dev["Y"], test["Y"]]),
        U=np.vstack([train["U"], dev["U"], test["U"]]),
        Y_train=train["Y"],
        U_train=train["U"],
        Y_dev=dev["Y"],
        U_dev=dev["U"],
        Y_test=test["Y"],
        U_test=test["U"],
        n_train=len(train["Y"]),
        n_dev=len(dev["Y"]),
        n_test=len(test["Y"]),
        Ts=TS,
        nsteps=NSTEPS,
        skip_s=SKIP_S,
        train_ids=np.array(TRAIN_IDS),
        dev_ids=np.array(DEV_IDS),
        test_ids=np.array(TEST_IDS),
        y_names=np.array(Y_NAMES),
        u_names=np.array(U_NAMES),
        experiments=np.array(experiments, dtype=object),
    )
    return path
