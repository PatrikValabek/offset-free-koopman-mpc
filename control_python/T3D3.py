import os
import sys
import time
import numpy as np
from numpy.linalg import inv
import joblib
import matplotlib.pyplot as plt
import torch

# External libs used by the notebook logic
import cvxpy as cp  # noqa: F401  (imported to match notebook environment)
from neuromancer.system import Node, System
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks


def main() -> None:
    # Resolve project paths (repo root assumed to be parent of this script's CWD)
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)

    figures_dir = os.path.join(project_root, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Local imports from the project
    import models  # type: ignore
    import helper  # type: ignore

    # -----------------------------
    # Load matrices, model, scalers
    # -----------------------------
    matrix_C = False

    A = np.load(os.path.join(project_root, 'data', f'A_C_{matrix_C}.npy'))
    B = np.load(os.path.join(project_root, 'data', f'B_C_{matrix_C}.npy'))
    C = np.load(os.path.join(project_root, 'data', f'C_C_{matrix_C}.npy'))

    nz, nu = B.shape
    ny = C.shape[0]

    # disturbance matrix F and dimension nd
    F = np.eye(ny)
    nd = F.shape[1]

    # Block-diagonalize A using helper
    T_real, A_block = helper.ident.real_block_diagonalize(A)

    # Transform A to check
    A_transformed = inv(T_real) @ A @ T_real
    print('Close to block diagonal?', np.allclose(A_block, A_transformed, atol=1e-6))

    # Backtransform A_block to verify it equals A
    A_backtransformed = T_real @ A_block @ inv(T_real)
    print('Backtransformation equals original A?', np.allclose(A, A_backtransformed, atol=1e-6))

    # Save sparsity pattern figure instead of showing
    plt.figure()
    plt.imshow(np.abs(A_transformed) > 1e-6, cmap='gray')
    plt.title('Nonzero pattern in Schur form R')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'A_transformed_pattern.png'), dpi=200)
    plt.close()

    # Apply similarity transform to A, B, C
    A = A_block
    B = inv(T_real) @ B
    C = C @ T_real

    # -----------------------------
    # Build Koopman model components
    # -----------------------------
    layers = [20, 40, 60]
    layers_dec = [60, 40, 20]

    # Output encoder f_y
    f_y = blocks.MLP(
        ny,
        nz,
        bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=layers,
    )
    encode_Y0 = Node(f_y, ['Y0'], ['x'], name='encoder_Y0')
    encode_Y = Node(f_y, ['Y'], ['x_latent'], name='encoder_Y')

    # Input encoder f_u representing B in linear system
    f_u = torch.nn.Linear(nu, nz, bias=False)
    encode_U = Node(f_u, ['U'], ['u_latent'], name='encoder_U')

    # State decoder f_y_inv
    if not matrix_C:
        f_y_inv = blocks.MLP(
            nz,
            ny,
            bias=True,
            linear_map=torch.nn.Linear,
            nonlin=torch.nn.ELU,
            hsizes=layers_dec,
        )
    elif matrix_C:
        f_y_inv = torch.nn.Linear(nz, ny, bias=False)
    else:
        raise ValueError('matrix_C must be boolean')

    decode_y = Node(f_y_inv, ['x'], ['yhat'], name='decoder_y')

    # Linear map K for latent dynamics
    K = torch.nn.Linear(nz, nz, bias=False)

    # Symbolic Koopman model with control inputs
    Koopman = Node(helper.PredictionWControl(K), ['x', 'u_latent'], ['x'], name='K')
    dynamics_model = System([Koopman], name='Koopman', nsteps=1)

    nodes = [encode_Y0, encode_Y, encode_U, dynamics_model, decode_y]
    loss = PenaltyLoss([], constraints=[])
    problem = Problem(nodes, loss)

    problem.load_state_dict(
        torch.load(os.path.join(project_root, 'data', f'model_C_{matrix_C}.pth')),
        strict=False,
    )

    # -----------------------------
    # Load scalers and plant model
    # -----------------------------
    scaler = joblib.load(os.path.join(project_root, 'data', 'scaler.pkl'))
    scalerU = joblib.load(os.path.join(project_root, 'data', 'scalerU.pkl'))

    # TwoTanks physical model
    A1 = 1
    A2 = 0.5
    k1 = 0.5
    k2 = 0.8
    TwoTanks = models.TwoTanks(True, A1, A2, k1, k2)

    # Helper conversion utilities
    def get_x(y: np.ndarray) -> np.ndarray:
        x = problem.nodes[0]({'Y0': torch.from_numpy(y).float()})
        return x['x'].detach().numpy().reshape(1, -1).T

    def get_y(x: np.ndarray) -> np.ndarray:
        y = problem.nodes[4]({'x': torch.from_numpy(x.T).float()})
        return y['yhat'].detach().numpy().reshape(1, -1)

    # -----------------------------
    # Observer - Kalman filter setup
    # -----------------------------
    loaded_setup = joblib.load(os.path.join(os.getcwd(), 'sim_setup.pkl'))

    y_start = loaded_setup['y_start']
    y_setpoint = loaded_setup['reference'][:, 0]
    u_previous = loaded_setup['u_previous']

    z_est_ = np.hstack(((inv(T_real) @ get_x(y_start)).T, np.zeros((1, nd))))
    P0 = loaded_setup['P0']
    Q = loaded_setup['Q']  # process noise
    Rm = loaded_setup['R']  # measurement noise

    A_ = np.block([
        [A, np.zeros((nz, nd))],
        [np.zeros((nd, nz)), np.eye(nd)],
    ])
    B_ = np.vstack([
        B,
        np.zeros((nd, nu)),
    ])

    EKF = helper.EKF(A_, B_, z_est_, P0, problem, Q, Rm, 2, T_real)
    # EKF = helper.EKF_C(A_, B_, C, z_est_, P0, problem, Q, Rm, 2, T_real)

    # -----------------------------
    # Target calculation (T3): linearize at current estimate \hat z
    # -----------------------------
    target_estimation = helper.TaylorTargetEstimation(A, B)

    J_hat = helper.evaluate_jacobian(
        problem.nodes[4],
        torch.from_numpy(T_real @ z_est_[0, :nz]).float(),
    ) @ T_real

    z_s, y_s = target_estimation.get_target(
        z_est_[:, nz:], y_setpoint, get_y(T_real @ z_est_[0, :nz]), z_est_[0, :nz], J_hat
    )
    print(target_estimation.te.status)
    print('Optimal y:', scaler.inverse_transform(y_s.reshape(1, -1)))
    print('Desired y:', scaler.inverse_transform(y_setpoint.reshape(1, -1)))
    print('Optimal u:', scalerU.inverse_transform(target_estimation.u_s.value.reshape(1, -1)))
    z_ref = z_s

    # -----------------------------
    # MPC problem formulation (D3): linearize dynamics/output at current estimate \hat z
    # -----------------------------
    Qy = loaded_setup['Qy']
    J = helper.evaluate_jacobian(
        problem.nodes[4],
        torch.from_numpy(T_real @ z_est_[0, :nz]).float(),
    ) @ T_real
    Qz = J.T @ Qy @ J
    Qz_psd = Qz + 1e-8 * np.eye(Qz.shape[0])

    mpc = helper.TaylorMPC(A, B)
    mpc.build_problem(Qz_psd)
    u_opt = mpc.get_u_optimal(
        z_est_[0, :nz], z_est_[:, nz:], u_previous, z_ref, get_y(T_real @ z_est_[0, :nz]), z_est_[0, :nz], J
    )
    print(u_opt)
    print(mpc.mpc.status)

    # -----------------------------
    # Closed-loop simulation
    # -----------------------------
    sim_time = loaded_setup['sim_time']
    Ts = 1
    z_sim = np.zeros((nz + nd, sim_time + 1))
    y_sim = np.zeros((ny, sim_time + 1))
    u_sim = np.zeros((nu, sim_time))
    ys_sim = np.zeros((ny, sim_time + 1))
    zs_sim = np.zeros((nz, sim_time + 1))
    us_sim = np.zeros((nu, sim_time + 1))

    total_time_target = 0.0
    total_time_mpc = 0.0

    # Initial target refinement at current estimate (T3)
    start_time_target = time.time()
    z_s, y_s = target_estimation.get_target(
        z_est_[:, nz:], y_setpoint, get_y(T_real @ z_est_[0, :nz]), z_est_[0, :nz], J_hat
    )
    end_time_target = time.time()
    total_time_target += end_time_target - start_time_target

    y_sim_descaled = np.zeros((ny, sim_time + 1))
    u_sim_descaled = np.zeros((nu, sim_time))

    z_sim[:, 0] = z_est_.flatten()
    y_sim[:, 0] = y_start.flatten()
    ys_sim[:, 0] = y_s
    zs_sim[:, 0] = z_s
    u_prev = target_estimation.u_s.value
    u_sim_descaled[:, 0] = scalerU.inverse_transform(u_sim[:, 0].reshape(1, -1))[0]
    y_sim_descaled[:, 0] = scaler.inverse_transform(y_sim[:, 0].reshape(1, -1))[0]

    for k in range(sim_time):
        y_setpoint = loaded_setup['reference'][:, k]

        # D3T3: linearize at current estimate z_sim[:nz, k]
        J_t = helper.evaluate_jacobian(
            problem.nodes[4],
            torch.from_numpy(T_real @ z_sim[:nz, k]).float(),
        ) @ T_real
        J = J_t
        
        # T3: target update, linearize at current estimate \hat z_{k+1}
        start_time_target = time.time()
        zs_sim[:, k + 1], ys_sim[:, k + 1] = target_estimation.get_target(
            z_sim[nz:, k + 1], 
            y_setpoint, 
            get_y(T_real @ z_sim[:nz, k]), 
            z_sim[:nz, k], 
            J
        )
        end_time_target = time.time()
        total_time_target += end_time_target - start_time_target

        Qz = J.T @ Qy @ J
        Qz_psd = Qz + 1e-8 * np.eye(Qz.shape[0])
        mpc.build_problem(Qz_psd)

        start_time_mpc = time.time()
        u_opt = mpc.get_u_optimal(
            z_sim[:nz, k],
            z_sim[nz:, k],
            u_prev,
            zs_sim[:, k],
            # D3: use h(\hat z_k) and linearization point \hat z_k
            get_y(T_real @ z_sim[:nz, k]),
            z_sim[:nz, k],
            J,
        )
        end_time_mpc = time.time()
        total_time_mpc += end_time_mpc - start_time_mpc

        u_sim[:, k] = u_opt
        u_sim_descaled[:, k] = scalerU.inverse_transform(u_sim[:, k].reshape(1, -1))[0]

        # simulate plant in physical space
        y_sim_descaled[:, k + 1] = TwoTanks.step(
            y_sim_descaled[:, k], u_sim_descaled[:, k].reshape(1, -1), Ts
        )
        y_sim[:, k + 1] = scaler.transform(y_sim_descaled[:, k + 1].reshape(1, -1))[0]

        # state estimation
        z_sim[:, k + 1] = EKF.step(u_sim[:, k], y_sim[:, k]).flatten()

        u_prev = u_sim[:, k]

    print(f'Total time spent in target estimation: {total_time_target:.4f} seconds')
    print(f'Total time spent in MPC solve: {total_time_mpc:.4f} seconds')

    # -----------------------------
    # Plots and objective (up to the specified line)
    # -----------------------------
    # Outputs plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(y_sim_descaled[0, 0:], label='h1')
    plt.plot(y_sim_descaled[1, 0:], label='h2')
    plt.plot(
        scaler.inverse_transform(ys_sim.T).T[0, :], color='red', linestyle='--', label='target h1'
    )
    plt.plot(
        scaler.inverse_transform(ys_sim.T).T[1, :], color='red', linestyle=':', label='target h2'
    )
    plt.xlabel('Time step')
    plt.ylabel('Output')
    plt.title('TK-MPC (T3D3) Simulation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'T3D3_states.png'), dpi=200)
    plt.close()

    # Inputs plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(u_sim_descaled[0, :], label='q1')
    plt.plot(u_sim_descaled[1, :], label='q2')
    plt.xlabel('Time step')
    plt.ylabel('Input')
    plt.title('TK-MPC (T3D3) Inputs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'T3D3_inputs.png'), dpi=200)
    plt.close()

    # Closed-loop objective value
    Qu = loaded_setup['Qu']
    objective_value = 0.0
    state_error_cost = 0.0
    control_increment_cost = 0.0
    sim_time_eval = 500
    for k in range(sim_time_eval):
        y_diff = y_sim[:, k] - loaded_setup['reference'][:, k]
        u_diff = u_sim[:, k] - u_sim[:, k - 1]
        y_term = float(y_diff.T @ Qy @ y_diff)
        u_term = float(u_diff.T @ Qu @ u_diff)
        state_error_cost += y_term
        control_increment_cost += u_term
        objective_value += y_term + u_term

    print(f"Closed-loop objective function value: {objective_value}")
    print(f"State error term: {state_error_cost}")
    print(f"Control increment term: {control_increment_cost}")


if __name__ == '__main__':
    main()


