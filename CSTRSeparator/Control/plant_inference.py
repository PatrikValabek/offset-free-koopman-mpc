"""Nonlinear CSTR–separator plant stepper for closed-loop notebooks.

Analogous to PU2x2 ``baseline_inference``, but the plant is the ODE model
(``models.CSTRSeparator``), not a learned predictor. Inputs/outputs are
scaled the same way as identification (``scaler_cstr_separator`` /
``scalerU_cstr_separator``).
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "../.."))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

import models  # noqa: E402


def init(x0=None, Ts=1.0):
    """Instantiate the plant, scalers, and internal state. Call once."""
    global plant, scaler, scalerU, x, ts, y_names, u_names

    plant = models.CSTRSeparator()
    data_dir = os.path.join(_HERE, "..", "data")
    scaler = joblib.load(os.path.join(data_dir, "scaler_cstr_separator.pkl"))
    scalerU = joblib.load(os.path.join(data_dir, "scalerU_cstr_separator.pkl"))
    y_names = list(plant.y_names)
    u_names = list(plant.u_names)
    ts = float(Ts)
    x = np.asarray(plant.x0_guess if x0 is None else x0, dtype=float).reshape(-1)
    reset_disturbances()
    return plant


def reset(x0, Ts=None):
    """Reset the plant state and restore nominal feed parameters."""
    global x, ts
    if Ts is not None:
        ts = float(Ts)
    x = np.asarray(x0, dtype=float).reshape(-1)
    reset_disturbances()


def reset_disturbances():
    """Restore Li & Swartz nominal fresh-feed composition and temperature."""
    plant.T10 = 313.0
    plant.T20 = 313.0
    plant.xA10 = 1.0
    plant.xB10 = 0.0
    plant.xA20 = 1.0
    plant.xB20 = 0.0


def apply_disturbance(attr: str, value: float):
    """Change a plant parameter the controller does not measure."""
    if not hasattr(plant, attr):
        raise AttributeError(f"Unknown plant attribute {attr!r}")
    setattr(plant, attr, float(value))


def measure_ns():
    """Noise-free measurement in physical units, shape (ny,)."""
    return plant.measure(x)


def y_plus(u_scaled):
    """Advance one sample. ``u_scaled`` is in scalerU space; returns scaled y."""
    global x
    u_ns = scalerU.inverse_transform(np.asarray(u_scaled, dtype=float).reshape(1, -1))
    x = np.asarray(plant.step(x, u_ns, ts), dtype=float).reshape(-1)
    y_ns = measure_ns().reshape(1, -1)
    return scaler.transform(y_ns)[0]
