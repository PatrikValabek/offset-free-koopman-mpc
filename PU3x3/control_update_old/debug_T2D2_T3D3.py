"""
Diagnostic script for T2D2 and T3D3 control notebooks.
Tests: encoder/decoder roundtrip, get_y correctness, Jacobian accuracy,
target estimation, and closed-loop simulation.
"""
import sys
import os
from pathlib import Path

import numpy as np
from numpy.linalg import inv
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromancer.modules import blocks
from neuromancer.system import Node, System
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
src_path = os.path.join(project_root, '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))

identification_path = os.path.join(project_root, 'Identification')
sys.path.insert(0, os.path.abspath(identification_path))

import helper
import baseline_inference

# ─── Config ──────────────────────────────────────────────────────────────
matrix_C = False  # T2D2/T3D3 use nonlinear decoder

# ─── Load matrices ───────────────────────────────────────────────────────
A_orig = np.load("../data/A_C_False.npy")
B_orig = np.load("../data/B_C_False.npy")
C_lstsq = np.load("../data/C_C_False.npy")

nz, nu = B_orig.shape
ny = C_lstsq.shape[0]
print(f"Model dims: ny={ny}, nz={nz}, nu={nu}")

# ─── T_real transformation ──────────────────────────────────────────────
T_real, A_block = helper.ident.real_block_diagonalize(A_orig)
A = A_block
B = inv(T_real) @ B_orig
C = C_lstsq @ T_real

print(f"T_real condition number: {np.linalg.cond(T_real):.2f}")

# ─── Build model architecture ───────────────────────────────────────────
cons = 5
layers = [4*cons, 8*cons, 16*cons]
layers_dec = [16*cons, 8*cons, 4*cons]

f_y = blocks.MLP(ny, nz, bias=True, linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ELU, hsizes=layers)
encode_Y0 = Node(f_y, ['Y0'], ['x'], name='encoder_Y0')
encode_Y = Node(f_y, ['Y'], ['x_latent'], name='encoder_Y')

f_u = torch.nn.Linear(nu, nz, bias=False)
encode_U = Node(f_u, ['U'], ['u_latent'], name='encoder_U')


class mELU(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return -F.elu(-x, alpha=self.alpha)

f_y_inv = blocks.MLP(nz, ny, bias=True, linear_map=torch.nn.Linear,
                     nonlin=mELU, hsizes=layers_dec)
decode_y = Node(f_y_inv, ['x'], ['yhat'], name='decoder_y')

K = torch.nn.Linear(nz, nz, bias=False)
Koopman = Node(helper.PredictionWControl(K), ['x', 'u_latent'], ['x'], name='K')
dynamics_model = System([Koopman], name='Koopman', nsteps=1)

nodes = [encode_Y0, encode_Y, encode_U, dynamics_model, decode_y]
loss = PenaltyLoss([], constraints=[])
problem = Problem(nodes, loss)

# ─── Load weights ────────────────────────────────────────────────────────
state_dict = torch.load(os.path.join(project_root, 'data', f'model_C_{matrix_C}.pth'),
                        map_location='cpu')
problem.load_state_dict(state_dict, strict=True)
problem.eval()
print("Model weights loaded successfully (strict=True)")

# ─── Load scalers ────────────────────────────────────────────────────────
scaler = joblib.load(os.path.join(project_root, 'data', 'scaler.pkl'))
scalerU = joblib.load(os.path.join(project_root, 'data', 'scalerU.pkl'))
scaler_baseline = joblib.load(os.path.join(project_root, 'data', 'scaler_baseline.pkl'))
scalerU_baseline = joblib.load(os.path.join(project_root, 'data', 'scalerU_baseline.pkl'))

loaded_setup = joblib.load('sim_setup.pkl')

print(f"\nScaler means (4-feat): {scaler.mean_}")
print(f"ScalerU means (MPC):  {scalerU.mean_}")
print(f"Scaler_baseline means (3-feat): {scaler_baseline.mean_}")
print(f"ScalerU_baseline means: {scalerU_baseline.mean_}")


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS  (matching training notebook's convention)
# ═══════════════════════════════════════════════════════════════════════
def get_x(y: np.ndarray) -> np.ndarray:
    """Encode scaled y → original latent x. Input (1, ny) or (ny,)."""
    y_t = torch.from_numpy(np.atleast_2d(y)).float()
    out = problem.nodes[0]({'Y0': y_t})
    return out['x'].detach().numpy().reshape(-1, 1)  # (nz, 1) column


def get_y(x: np.ndarray) -> np.ndarray:
    """Decode original latent x → scaled y.
    CRITICAL: ensure 2-D input (1, nz) to match training convention."""
    x_col = np.asarray(x, dtype=np.float64).reshape(-1, 1)  # (nz, 1)
    x_row = torch.from_numpy(x_col.T).float()               # (1, nz)
    out = problem.nodes[4]({'x': x_row})
    return out['yhat'].detach().numpy().reshape(1, -1)        # (1, ny)


def get_y_vec(x: np.ndarray) -> np.ndarray:
    """Same as get_y but returns (ny,) 1-D."""
    return get_y(x).flatten()


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Shape consistency — 1-D vs 2-D input to decoder
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 1: Shape consistency check (1-D vs 2-D decoder input)")
print("="*70)

z_test = np.random.randn(nz)
x_orig = T_real @ z_test  # back to original coords

# 1-D input (how control notebooks were doing it)
x_1d = torch.from_numpy(x_orig).float()         # (nz,)
y_1d = problem.nodes[4]({'x': x_1d})['yhat'].detach().numpy()

# 2-D input (how training notebook does it)
x_2d = torch.from_numpy(x_orig.reshape(1, -1)).float()  # (1, nz)
y_2d = problem.nodes[4]({'x': x_2d})['yhat'].detach().numpy()

print(f"  1-D input shape: {x_1d.shape} → output: {y_1d.shape}, values: {y_1d.flatten()[:4]}")
print(f"  2-D input shape: {x_2d.shape} → output: {y_2d.shape}, values: {y_2d.flatten()[:4]}")
print(f"  Max abs diff: {np.max(np.abs(y_1d.flatten() - y_2d.flatten())):.2e}")

if np.max(np.abs(y_1d.flatten() - y_2d.flatten())) > 1e-5:
    print("  *** WARNING: 1-D and 2-D inputs give DIFFERENT results! ***")
else:
    print("  OK: Both shapes give identical results.")


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Encoder → Decoder roundtrip
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 2: Encoder-Decoder roundtrip")
print("="*70)

y_start_ns = loaded_setup['y_start_ns']
y_start_scaled = scaler.transform(y_start_ns.reshape(1, -1))

print(f"  y_start physical: {y_start_ns}")
print(f"  y_start scaled:   {y_start_scaled}")

x_encoded = get_x(y_start_scaled)   # (nz, 1) in original coords
y_decoded = get_y(x_encoded)        # (1, ny) in scaled space
y_decoded_ns = scaler.inverse_transform(y_decoded)

print(f"  Encoded x (original coords): {x_encoded.flatten()[:4]}...")
print(f"  Decoded y (scaled):  {y_decoded.flatten()}")
print(f"  Decoded y (physical): {y_decoded_ns.flatten()}")
print(f"  Roundtrip error (physical): {y_start_ns.flatten() - y_decoded_ns.flatten()}")
print(f"  Roundtrip max error: {np.max(np.abs(y_start_ns.flatten() - y_decoded_ns.flatten())):.4f}")


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: get_y vs C @ z comparison
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 3: get_y(T_real @ z) vs C @ z comparison")
print("="*70)

z_transformed = inv(T_real) @ x_encoded.flatten()  # transformed coords
y_from_get_y = get_y(T_real @ z_transformed)        # should equal get_y(x_encoded)
y_from_C = C @ z_transformed                        # linear approximation

print(f"  get_y (scaled):    {y_from_get_y.flatten()}")
print(f"  C @ z (scaled):    {y_from_C}")
print(f"  Difference:        {y_from_get_y.flatten() - y_from_C}")
print(f"  get_y (physical):  {scaler.inverse_transform(y_from_get_y).flatten()}")
print(f"  C @ z (physical):  {scaler.inverse_transform(y_from_C.reshape(1,-1)).flatten()}")


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Jacobian accuracy
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 4: Jacobian accuracy (finite difference check)")
print("="*70)

x0_orig = T_real @ z_transformed  # (nz,) in original coords
J_autograd = helper.evaluate_jacobian(
    problem.nodes[4], torch.from_numpy(x0_orig).float()
)
J_with_T = J_autograd @ T_real  # Jacobian in transformed coords

eps = 1e-5
J_fd = np.zeros((ny, nz))
y0 = get_y_vec(x0_orig)
for j in range(nz):
    x_pert = x0_orig.copy()
    x_pert[j] += eps
    J_fd[:, j] = (get_y_vec(x_pert) - y0) / eps

print(f"  Max abs error (autograd vs FD): {np.max(np.abs(J_autograd - J_fd)):.2e}")
if np.max(np.abs(J_autograd - J_fd)) > 1e-3:
    print("  *** WARNING: Jacobian mismatch! ***")


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Decoder at zero latent state
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 5: Decoder at zero latent state (bias check)")
print("="*70)

z_zero = np.zeros(nz)
y_at_zero = get_y(T_real @ z_zero)  # T_real @ 0 = 0
y_at_zero_ns = scaler.inverse_transform(y_at_zero)
print(f"  get_y(0) scaled:   {y_at_zero.flatten()}")
print(f"  get_y(0) physical: {y_at_zero_ns.flatten()}")
print(f"  C @ 0:             {(C @ z_zero)}")


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: Target estimation check
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 6: Target estimation")
print("="*70)

nd = ny
F_mat = np.eye(ny)
Bd = np.zeros((nz, nd))
Cd = np.eye(ny)

y_start = loaded_setup['y_start']
y_setpoint = loaded_setup['reference'][:, 0]
u_sp = loaded_setup['reference_u'][:, 0]

z_est_orig = get_x(y_start)  # (nz, 1)
z_est_transformed = inv(T_real) @ z_est_orig.flatten()
z_est_ = np.hstack(((inv(T_real) @ z_est_orig).T, np.zeros((1, nd))))

# Compute Jacobian at current estimate
J_te = helper.evaluate_jacobian(
    problem.nodes[4],
    torch.from_numpy(T_real @ z_est_[0, :nz]).float(),
) @ T_real

# get_y at current estimate
y_current = get_y(T_real @ z_est_[0, :nz])

print(f"  y_setpoint (scaled): {y_setpoint}")
print(f"  y_setpoint (phys):   {scaler.inverse_transform(y_setpoint.reshape(1,-1)).flatten()}")
print(f"  y_current via get_y (scaled): {y_current.flatten()}")
print(f"  y_current via get_y (phys):   {scaler.inverse_transform(y_current).flatten()}")
print(f"  y_current via C @ z (scaled): {C @ z_est_[0, :nz]}")
print(f"  y_current via C @ z (phys):   {scaler.inverse_transform((C @ z_est_[0,:nz]).reshape(1,-1)).flatten()}")

# Taylor linearization check: y ≈ y_lp + J(z - z_lp)
z_lp = z_est_[0, :nz]
y_lp = y_current.flatten()
y_taylor_at_lp = y_lp + J_te @ (z_lp - z_lp)  # should equal y_lp
print(f"\n  Taylor at linearization point: {y_taylor_at_lp}")
print(f"  get_y at same point:           {y_lp}")
print(f"  Match: {np.allclose(y_taylor_at_lp, y_lp)}")

# Run target estimation
target_estimation = helper.TaylorTargetEstimation(
    A, B, loaded_setup["Qy_te"], loaded_setup["Qu_te"], Bd, Cd
)

z_s, y_s, u_s = target_estimation.get_target(
    z_est_[:, nz:], y_setpoint, u_sp, y_current, z_est_[0, :nz], J_te
)

print(f"\n  Target z_s: {z_s[:3]}...")
print(f"  Target y_s (scaled): {y_s}")
print(f"  Target y_s (phys): {scaler.inverse_transform(y_s.reshape(1,-1)).flatten()}")
print(f"  Target u_s (scaled): {u_s}")
print(f"  Target u_s (phys): {scalerU.inverse_transform(u_s.reshape(1,-1)).flatten()}")

# Verify: does get_y(T_real @ z_s) match y_s?
y_at_zs_gety = get_y(T_real @ z_s)
y_at_zs_taylor = y_current.flatten() + J_te @ (z_s - z_est_[0, :nz])

print(f"\n  get_y(T_real @ z_s) (scaled): {y_at_zs_gety.flatten()}")
print(f"  get_y(T_real @ z_s) (phys):   {scaler.inverse_transform(y_at_zs_gety).flatten()}")
print(f"  Taylor at z_s (scaled):        {y_at_zs_taylor}")
print(f"  Taylor at z_s (phys):          {scaler.inverse_transform(y_at_zs_taylor.reshape(1,-1)).flatten()}")
print(f"  y_s from solver (phys):        {scaler.inverse_transform(y_s.reshape(1,-1)).flatten()}")

diff_gety_vs_taylor = np.abs(y_at_zs_gety.flatten() - y_at_zs_taylor)
print(f"\n  |get_y - Taylor| at z_s: {diff_gety_vs_taylor}")
if np.max(diff_gety_vs_taylor) > 0.5:
    print("  *** LARGE mismatch: Taylor linearization is poor at z_s! ***")
    print("  This means z_s is far from the linearization point z_lp.")
    print(f"  ||z_s - z_lp||: {np.linalg.norm(z_s - z_est_[0,:nz]):.4f}")


# ═══════════════════════════════════════════════════════════════════════
# TEST 7: Multiple Jacobian re-linearizations for target
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 7: Iterative target estimation with re-linearization")
print("="*70)

z_lp_iter = z_est_[0, :nz].copy()
for it in range(10):
    J_iter = helper.evaluate_jacobian(
        problem.nodes[4],
        torch.from_numpy(T_real @ z_lp_iter).float(),
    ) @ T_real
    y_lp_iter = get_y(T_real @ z_lp_iter)

    z_s_iter, y_s_iter, u_s_iter = target_estimation.get_target(
        z_est_[:, nz:], y_setpoint, u_sp, y_lp_iter, z_lp_iter, J_iter
    )

    y_at_zs_true = get_y(T_real @ z_s_iter)
    err = np.max(np.abs(y_at_zs_true.flatten() - y_s_iter))
    y_phys = scaler.inverse_transform(y_at_zs_true).flatten()

    print(f"  Iter {it}: y_s(phys)={scaler.inverse_transform(y_s_iter.reshape(1,-1)).flatten()}, "
          f"get_y(phys)={y_phys}, |taylor-true|_max={err:.4f}, "
          f"||z_s - z_lp||={np.linalg.norm(z_s_iter - z_lp_iter):.4f}")

    z_lp_iter = z_s_iter.copy()


# ═══════════════════════════════════════════════════════════════════════
# TEST 8: Quick closed-loop (50 steps) to check stability
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 8: Quick closed-loop simulation (50 steps)")
print("="*70)

sim_steps = 50
z_sim = np.zeros((nz + nd, sim_steps + 1))
y_sim = np.zeros((ny, sim_steps + 1))
u_sim = np.zeros((nu, sim_steps))
us_sim = np.zeros((nu, sim_steps + 1))
ys_sim = np.zeros((ny, sim_steps + 1))
zs_sim = np.zeros((nz, sim_steps + 1))

A_ = np.block([[A, Bd], [np.zeros((nd, nz)), np.eye(nd)]])
B_ = np.vstack([B, np.zeros((nd, nu))])
C_ = np.hstack([C, Cd])

z_est_init = np.hstack(((inv(T_real) @ get_x(y_start)).T, np.zeros((1, nd))))
z_sim[:, 0] = z_est_init.flatten()
y_sim[:, 0] = y_start.flatten()

P0 = np.eye(nz + nd) * loaded_setup['P0']
Q_kf = np.block([
    [np.eye(nz) * 0.1, np.zeros((nz, nd))],
    [np.zeros((nd, nz)), np.eye(nd) * 0.1]
])
R_kf = np.eye(ny) * 100
TVKF = helper.TVKF(A_, B_, C_, z_est_init, P0, Q_kf, R_kf)

z_s_init = z_sim[:nz, 0]
J_init = helper.evaluate_jacobian(
    problem.nodes[4], torch.from_numpy(T_real @ z_s_init).float()
) @ T_real

z_s = z_s_init.copy()
u_prev = loaded_setup['u_previous'].flatten()
Qy = loaded_setup['Qy']

mpc = helper.TaylorMPC(A, B, Qy, loaded_setup["Qu"], loaded_setup["Qdu"], Bd, Cd)
mpc.build_problem()

# Initial target
z_s, y_s, u_s = target_estimation.get_target(
    z_sim[nz:, 0], loaded_setup['reference'][:, 0],
    loaded_setup['reference_u'][:, 0],
    get_y(T_real @ z_s), z_s, J_init
)
zs_sim[:, 0] = z_s
ys_sim[:, 0] = y_s
us_sim[:, 0] = u_s

J = J_init.copy()

# Initialize baseline plant
y_start_ns = loaded_setup['y_start_ns']
y_start_baseline = scaler_baseline.transform(y_start_ns[:, :3].reshape(1, -1))
baseline_inference.get_x(y_start_baseline)

from tqdm import trange
for k in trange(sim_steps, desc="Closed-loop"):
    y_setpoint_k = loaded_setup['reference'][:, min(k, loaded_setup['reference'].shape[1]-1)]
    u_sp_k = loaded_setup['reference_u'][:, min(k, loaded_setup['reference_u'].shape[1]-1)]
    idx_prev = max(k - 1, 0)

    # Target
    zs_sim[:, k], ys_sim[:, k], us_sim[:, k] = target_estimation.get_target(
        z_sim[nz:, k], y_setpoint_k, u_sp_k,
        get_y(T_real @ zs_sim[:, idx_prev]), zs_sim[:, idx_prev], J
    )

    Qz = J.T @ Qy @ J
    Qz_psd = Qz + 1e-8 * np.eye(nz)

    # MPC
    try:
        u_opt = mpc.get_u_optimal(
            z_sim[:nz, k], z_sim[nz:, k], u_prev,
            zs_sim[:, k], us_sim[:, k],
            get_y(T_real @ zs_sim[:, idx_prev]), zs_sim[:, idx_prev],
            J, Qz_psd
        )
    except RuntimeError as e:
        print(f"  MPC failed at step {k}: {e}")
        break

    u_sim[:, k] = u_opt
    u_ns = scalerU.inverse_transform(u_opt.reshape(1, -1))[0]

    # Simulate plant
    u_bl = scalerU_baseline.transform(u_ns.reshape(1, -1))[0]
    y_bl = baseline_inference.y_plus(u_bl)
    y_bl_ns = scaler_baseline.inverse_transform(y_bl.reshape(1, -1))[0]
    L_ns = 10.0 ** ((y_bl_ns[0] - 40.0) / 5.0)
    y_full_ns = np.array([y_bl_ns[0], y_bl_ns[1], y_bl_ns[2], L_ns])
    y_sim[:, k+1] = scaler.transform(y_full_ns.reshape(1, -1))[0]

    # Recompute Jacobian
    J = helper.evaluate_jacobian(
        problem.nodes[4], torch.from_numpy(T_real @ zs_sim[:, k]).float()
    ) @ T_real
    C_k = np.hstack([J, Cd])

    # State estimation
    z_sim[:, k+1] = TVKF.step(
        u_sim[:, k], y_sim[:, k+1],
        get_y(T_real @ zs_sim[:nz, k]), zs_sim[:nz, k], J, C_k
    ).flatten()

    u_prev = u_sim[:, k]

    if k < 5 or k % 10 == 0:
        y_phys = scaler.inverse_transform(y_sim[:, k+1].reshape(1, -1)).flatten()
        ref_phys = scaler.inverse_transform(y_setpoint_k.reshape(1, -1)).flatten()
        d_est = z_sim[nz:, k+1]
        print(f"  k={k}: y={y_phys}, ref={ref_phys}, d_est={d_est}, u_ns={u_ns}")

print("\n--- Simulation complete ---")
y_final_phys = scaler.inverse_transform(y_sim[:, -1].reshape(1, -1)).flatten()
ref_final_phys = scaler.inverse_transform(
    loaded_setup['reference'][:, min(sim_steps-1, loaded_setup['reference'].shape[1]-1)].reshape(1, -1)
).flatten()
print(f"Final y (phys): {y_final_phys}")
print(f"Final ref (phys): {ref_final_phys}")
print(f"Final disturbance estimate: {z_sim[nz:, -1]}")
