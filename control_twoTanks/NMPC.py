"""NMPC script for two-tank system.

This script builds a do-mpc model and controller, runs a simulation, saves
figures into a repository-level `figures/` directory, and prints the
closed-loop objective value.
"""

from pathlib import Path
from typing import Dict, Tuple

import do_mpc
from casadi import SX
import matplotlib.pyplot as plt
import numpy as np
import joblib


# ----------------------------- IO utilities ---------------------------------

def load_sim_setup(sim_setup_path: Path) -> Dict:
    """Load simulation setup dictionary saved by `setup.py`.

    Parameters
    ----------
    sim_setup_path: Path
        Path to the `sim_setup.pkl` file.

    Returns
    -------
    Dict
        Simulation setup dictionary with keys like N, Qy, Qu, reference_ns, etc.
    """
    return joblib.load(sim_setup_path.as_posix())


def get_figures_dir(repo_root: Path) -> Path:
    """Return (and create) the `figures/` directory under repository root."""
    figures_dir = repo_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def as_float(value) -> float:
    """Return a native float extracted from scalars/arrays safely (no deprecation warnings)."""
    arr = np.asarray(value)
    if arr.size == 0:
        return float('nan')
    return arr.reshape(-1)[0].item()


# ------------------------------ Model setup ---------------------------------

def build_model() -> do_mpc.model.Model:
    """Build and return the continuous two-tank model used by do-mpc."""
    model = do_mpc.model.Model('continuous')

    # States
    h1 = model.set_variable(var_type='_x', var_name='h1')
    h2 = model.set_variable(var_type='_x', var_name='h2')

    # Inputs
    u1 = model.set_variable(var_type='_u', var_name='u1')
    u2 = model.set_variable(var_type='_u', var_name='u2')

    # Physical parameters (dimensionless placeholders)
    A1 = 1
    A2 = 0.5
    k1 = 0.5
    k2 = 0.8

    delta_h = h1 - h2
    flow12 = k1 * SX.sign(delta_h) * SX.sqrt(SX.fabs(delta_h))

    # Dynamics
    dh1_dt = (-flow12 + u1) / A1
    dh2_dt = (flow12 - k2 * (h2 ** 0.5) + u2) / A2

    model.set_rhs('h1', dh1_dt)
    model.set_rhs('h2', dh2_dt)

    model.setup()
    return model


def build_mpc(model: do_mpc.model.Model, loaded_setup: Dict, reference_column: np.ndarray) -> do_mpc.controller.MPC:
    """Build the MPC controller with objective and constraints.

    Parameters
    ----------
    model: do_mpc.model.Model
        The process model.
    loaded_setup: Dict
        Simulation setup dict (contains N, Qy, Qu, etc.).
    reference_column: np.ndarray
        Reference vector for [h1, h2] at current step (non-scaled domain).
    """
    mpc = do_mpc.controller.MPC(model)

    mpc.set_param(
        n_horizon=int(loaded_setup["N"]),
        t_step=1.0,
        state_discretization='collocation',
        collocation_type='radau',
        collocation_deg=2,
        collocation_ni=2,
        store_full_solution=True,
        nlpsol_opts={
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
        },
    )

    # Tracking objective in non-scaled domain
    weight_h1 = loaded_setup["Qy"][0, 0] / 0.79368273
    weight_h2 = loaded_setup["Qy"][1, 1] / 0.52258528
    h1_ref = reference_column[0]
    h2_ref = reference_column[1]

    mterm = (weight_h1 * (model.x['h1'] - h1_ref) ** 2 + weight_h2 * (model.x['h2'] - h2_ref) ** 2) * 0
    lterm = weight_h1 * (model.x['h1'] - h1_ref) ** 2 + weight_h2 * (model.x['h2'] - h2_ref) ** 2
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Small penalty on input moves
    mpc.set_rterm(u1=loaded_setup["Qu"][0, 0] / 0.02223645, u2=loaded_setup["Qu"][1, 1] / 0.08454784)

    # Input constraints in non-scaled domain
    mpc.bounds['lower', '_u', 'u1'] = 0.0
    mpc.bounds['upper', '_u', 'u1'] = 0.5
    mpc.bounds['lower', '_u', 'u2'] = 0.0
    mpc.bounds['upper', '_u', 'u2'] = 1.0

    mpc.setup()
    return mpc


def build_simulator(model: do_mpc.model.Model) -> do_mpc.simulator.Simulator:
    """Build the do-mpc simulator."""
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=1)
    simulator.setup()
    return simulator


# ---------------------------- Simulation core -------------------------------

def run_closed_loop(
    model: do_mpc.model.Model,
    loaded_setup: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the closed-loop simulation and return time, states, and inputs histories.

    Returns
    -------
    time_history: (T+1,)
    y_history: (2, T+1)  non-scaled states [h1; h2]
    u_history: (2, T+1)  applied inputs [u1; u2]
    """
    simulator = build_simulator(model)
    mpc = build_mpc(model, loaded_setup, reference_column=loaded_setup["reference_ns"][:, 0])

    x0 = loaded_setup["y_start_ns"]
    u0 = loaded_setup['u_previous_ns']
    mpc.x0 = x0
    mpc.u0 = u0
    simulator.x0 = x0
    mpc.set_initial_guess()

    n_steps = int(loaded_setup["sim_time"])

    time_history = [0]
    h1_history = [as_float(x0[0])]
    h2_history = [as_float(x0[1])]
    u_history = [u0.flatten().tolist()]

    for k in range(1, n_steps + 1):
        if k % 100 == 0 and k < n_steps:
            mpc = build_mpc(model, loaded_setup, reference_column=loaded_setup["reference_ns"][:, k])
            x0 = np.array([h1_history[-1], h2_history[-1]])
            mpc.x0 = x0
            mpc.u0 = np.array(u_history[-1])
            simulator.x0 = x0
            mpc.set_initial_guess()

        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)

        time_history.append(k)
        h1_history.append(as_float(x0[0]))
        h2_history.append(as_float(x0[1]))
        u_history.append(u0.flatten().tolist())
        x0 = y_next

    time_arr = np.asarray(time_history)
    y_arr = np.vstack([np.asarray(h1_history), np.asarray(h2_history)])
    u_arr = np.asarray(u_history).T
    return time_arr, y_arr, u_arr


# ------------------------------ Plotting ------------------------------------

def plot_states(figures_dir: Path, time_arr: np.ndarray, y_arr: np.ndarray, loaded_setup: Dict) -> None:
    """Plot and save states vs references."""
    n_steps = time_arr[-1]
    fig = plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(y_arr[0, :], label='h1')
    plt.plot(y_arr[1, :], label='h2')
    plt.plot(loaded_setup["reference_ns"][0, :n_steps+1], 'r--', label='h1 ref')
    plt.plot(loaded_setup["reference_ns"][1, :n_steps+1], 'r--', label='h2 ref')
    plt.xlabel('Time step')
    plt.ylabel('Output')
    plt.title('NMPC Simulation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(figures_dir / 'NMPC_states.png', dpi=200)


def plot_inputs(figures_dir: Path, time_arr: np.ndarray, u_arr: np.ndarray) -> None:
    """Plot and save input trajectories."""
    fig = plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(u_arr[0, :], label='q1')
    plt.plot(u_arr[1, :], label='q2')
    plt.xlabel('Time step')
    plt.ylabel('Inputs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(figures_dir / 'NMPC_inputs.png', dpi=200)


# ----------------------------- Evaluation -----------------------------------

def compute_closed_loop_objective(
    y_history: np.ndarray,
    u_history: np.ndarray,
    loaded_setup: Dict,
    data_dir: Path,
) -> float:
    """Compute the closed-loop objective value using scaled variables.

    Uses Qy on output tracking error and Qu on input increments.
    """
    scaler = joblib.load((data_dir / 'scaler.pkl').as_posix())
    scalerU = joblib.load((data_dir / 'scalerU.pkl').as_posix())

    y_sim = scaler.transform(y_history.T).T
    u_sim = scalerU.transform(u_history.T).T

    Qy = loaded_setup["Qy"]
    Qu = loaded_setup["Qu"]

    n_steps = min(500, y_history.shape[1])
    objective_value = 0.0
    y_term = 0.0
    u_term = 0.0
    for k in range(n_steps):
        y_diff = y_sim[:, k] - loaded_setup["reference"][:, k]
        u_diff = u_sim[:, k] - (u_sim[:, k - 1] if k > 0 else u_sim[:, k])
        y_term += float(y_diff.T @ Qy @ y_diff)
        u_term += float(u_diff.T @ Qu @ u_diff)
        objective_value = y_term + u_term
    return objective_value, y_term, u_term


# --------------------------------- Main -------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    sim_setup_path = repo_root / 'control' / 'sim_setup.pkl' if not (repo_root / 'control_python' / 'sim_setup.pkl').exists() else (repo_root / 'control_python' / 'sim_setup.pkl')
    if not sim_setup_path.exists():
        # Fallback to current working directory
        sim_setup_path = Path('sim_setup.pkl')

    loaded_setup = load_sim_setup(sim_setup_path)

    model = build_model()
    time_arr, y_arr, u_arr = run_closed_loop(model, loaded_setup)

    figures_dir = get_figures_dir(repo_root)
    plot_states(figures_dir, time_arr, y_arr, loaded_setup)
    plot_inputs(figures_dir, time_arr, u_arr)

    data_dir = repo_root / 'data'
    objective_value, y_term, u_term = compute_closed_loop_objective(y_arr, u_arr, loaded_setup, data_dir)
    print(f"Closed-loop objective function value: {objective_value}")
    # print u and y terms
    print(f"  - State tracking term: {y_term}")
    print(f"  - Input increment term: {u_term}")


if __name__ == "__main__":
    main()

