#!/usr/bin/env python3
"""Generate the identification dataset for the square 4x4 CSTR-separator.

Reproduces the record format of ``CSTRSeparator/data/cstr_separator_ident.npz``
for the reduced-input plant (manipulated inputs [Q1, Q2, Q3, F10]; F20 and Fr
held fixed at their nominal values inside the plant model).

Design (mirrors the 6-input dataset):
    - 100 random steps of 180 s each (18000 samples, Ts = 1 s).
    - Latent conversion-intent steps (``generate_cstr_yield_steps_4x4``) that
      push the plant across the xB3 yield peak, seeded with 42.
    - 1% (of the train-split range) Gaussian measurement noise added to the
      clean outputs, seeded with 42.

Run this file from this directory:
    python generate_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
REPO_ROOT = HERE.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import models  # noqa: E402
from helper.ident import generate_cstr_yield_steps_4x4  # noqa: E402

# ---------------------------- Design constants ---------------------------------
Ts = 1.0
STEP_TIME = 180
NO_STEPS = 100
SIM_LEN = STEP_TIME * NO_STEPS  # 18000
SEED = 42
NOISE_FRAC = 0.01
NOISE_SEED = 42
N_TRAIN = 12960
N_DEV = 2880
N_TRAIN_NOISE = N_TRAIN  # noise scale calibrated on the train split only

# ---------------------------- Plant / inputs ------------------------------------
plant = models.CSTRSeparator4x4()
y_names = list(plant.y_names)  # T1, T2, T3, xB3
u_names = list(plant.u_names)  # Q1, Q2, Q3, F10

U = generate_cstr_yield_steps_4x4(STEP_TIME, NO_STEPS, plant.mv_constraints, seed=SEED)
assert U.shape == (SIM_LEN, 4)

# ---------------------------- Open-loop simulation -------------------------------
x0 = plant.x0_guess.copy()
X = np.zeros((SIM_LEN, 9))
Y_clean = np.zeros((SIM_LEN, 4))
U_plant = np.zeros((SIM_LEN, 6))

x = x0.copy()
for k in range(SIM_LEN):
    u_row = U[k : k + 1, :]
    x = plant.step(x, u_row, Ts)
    X[k, :] = x
    Y_clean[k, :] = plant.measure(x)
    U_plant[k, :4] = U[k, :]
    U_plant[k, 4] = plant.F20_fixed
    U_plant[k, 5] = plant.Fr_fixed

    if (k + 1) % 2000 == 0:
        print(f"Simulated {k + 1}/{SIM_LEN} steps", flush=True)

# ---------------------------- Measurement noise -----------------------------------
noise_sigma = NOISE_FRAC * (
    Y_clean[:N_TRAIN_NOISE].max(axis=0) - Y_clean[:N_TRAIN_NOISE].min(axis=0)
)
rng = np.random.default_rng(NOISE_SEED)
Y = Y_clean + rng.normal(size=Y_clean.shape) * noise_sigma

print("Measurement noise sigma (1% of train-range):")
for name, s in zip(y_names, noise_sigma):
    print(f"  {name}: sigma={s:.4g}")

# ---------------------------- Save -------------------------------------------------
out_path = DATA_DIR / "cstr_separator_ident.npz"
DATA_DIR.mkdir(parents=True, exist_ok=True)
np.savez(
    out_path,
    Y=Y,
    Y_clean=Y_clean,
    U=U,
    X=X,
    U_plant=U_plant,
    Ts=Ts,
    step_time=STEP_TIME,
    no_steps=NO_STEPS,
    y_names=np.array(y_names),
    u_names=np.array(u_names),
    x0=x0,
    u_nom=plant.u_nom,
    mv_constraints=plant.mv_constraints,
    noise_frac=NOISE_FRAC,
    noise_sigma=noise_sigma,
    noise_seed=NOISE_SEED,
    n_train_noise=N_TRAIN_NOISE,
)
print("Wrote", out_path)
print("Y", Y.shape, y_names)
print("U", U.shape, u_names)
print("U_plant", U_plant.shape, ["Q1", "Q2", "Q3", "F10", "F20", "Fr"])

# ---------------------------- Quick-look figure -------------------------------------
part = 2000
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for i, name in enumerate(y_names):
    axes[0].plot(Y[:part, i], label=name)
axes[0].set_ylabel("Outputs")
axes[0].set_title("Measured outputs (first %d samples)" % part)
axes[0].legend()

for i, name in enumerate(u_names):
    axes[1].plot(U[:part, i], label=name)
axes[1].set_xlabel("Time steps [s]")
axes[1].set_ylabel("Inputs")
axes[1].set_title("Manipulated inputs (F20, Fr fixed)")
axes[1].legend()
plt.tight_layout()
fig_path = HERE / "generate_data_preview.png"
plt.savefig(fig_path, dpi=150)
print("Wrote", fig_path)
