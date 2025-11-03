# Taylor-Based MPC Variants for CSTR System

This directory contains different implementations of Taylor-based Model Predictive Control (MPC) for the CSTR (Continuous Stirred Tank Reactor) system. The variants differ in where they linearize the system for target estimation and MPC optimization.

## Nomenclature

The file naming follows the pattern `TxDy.py` where:

- **Tx** refers to the Target estimation linearization strategy
- **Dy** refers to the Dynamics/MPC linearization strategy

### Target Estimation Linearization (T)

- **T2**: Target estimation linearizes at **previous target** `zs_sim[:, k-1]`
- **T3**: Target estimation linearizes at **current state estimate** `z_sim[:nz, k]`

### MPC Linearization (D)

- **D2**: MPC linearizes at **previous target** `zs_sim[:, k-1]`
- **D3**: MPC linearizes at **current state estimate** `z_sim[:nz, k]`
- **D4**: MPC linearizes at **current target** `zs_sim[:, k]`

## Available Variants

### T2 Variants (Target at Previous Target)

1. **T2D2.py** ✅

   - Target estimation: Previous target
   - MPC: Previous target
   - Both use the same linearization point

2. **T2D3.py** ✅

   - Target estimation: Previous target
   - MPC: Current state estimate
   - Different linearization points for target and MPC

3. **T2D4.py** ✅
   - Target estimation: Previous target
   - MPC: Current target
   - MPC uses the newly computed target

### T3 Variants (Target at Current State)

4. **T3D2.py** ✅

   - Target estimation: Current state estimate
   - MPC: Previous target
   - Different linearization points

5. **T3D3.py** ✅

   - Target estimation: Current state estimate
   - MPC: Current state estimate
   - Both use the same linearization point (most consistent)

6. **T3D4.py** ✅
   - Target estimation: Current state estimate
   - MPC: Current target
   - MPC uses the target computed at current state

## Key Implementation Details

### Common Features

- **Plant Model**: `CSTRSeriesRecycle()` - Full nonlinear CSTR series with recycle
- **Network Architecture**:
  - Encoder: `[40, 80, 120]` hidden layers
  - Decoder: `[120, 80, 40]` hidden layers
- **Observer**: Extended Kalman Filter (EKF) with disturbance estimation
- **Transformation**: Block-diagonal transformation of system matrix A
- **Optimization**: CVXPY with Gurobi solver

### Linearization Details

Each variant computes Jacobians at different points:

```python
# T2 variants: Target estimation at previous target
J_ref = evaluate_jacobian(problem.nodes[4], T_real @ zs_sim[:, k-1]) @ T_real
zs[:, k], ys[:, k] = target_estimation.get_target(..., J_ref)

# T3 variants: Target estimation at current state
J_ref = evaluate_jacobian(problem.nodes[4], T_real @ z_sim[:nz, k]) @ T_real
zs[:, k], ys[:, k] = target_estimation.get_target(..., J_ref)

# D2: MPC at previous target
J = evaluate_jacobian(problem.nodes[4], T_real @ zs_sim[:, k-1]) @ T_real

# D3: MPC at current state
J = evaluate_jacobian(problem.nodes[4], T_real @ z_sim[:nz, k]) @ T_real

# D4: MPC at current target
J = evaluate_jacobian(problem.nodes[4], T_real @ zs_sim[:, k]) @ T_real
```

## Running the Scripts

All scripts should be run from the `control_cstr` directory:

```bash
cd /path/to/offset-free-koopman-mpc/control_cstr
conda activate kmpc

# Run any variant
python T2D2.py
python T2D3.py
python T2D4.py
python T3D2.py
python T3D3.py
python T3D4.py
```

## Output

Each script generates:

1. **Console output**:

   - Objective function value
   - State tracking cost
   - Input increment cost
   - Computation times

2. **Figures** (saved to `../figures/`):
   - `TxDy_states.png`: 8 CSTR states vs. reference
   - `TxDy_inputs.png`: 4 control inputs over time

## Expected Behavior

- **T2D2**: Most conservative, uses oldest linearization
- **T3D3**: Most consistent, uses current state for both
- **T2D4**: Progressive update, target uses new linearization
- **T3D4**: Most aggressive, both use latest information

Performance may vary depending on:

- Initial conditions
- Reference trajectory
- Tuning parameters (Q, R matrices)
- System nonlinearity

## Comparison with control_python

The `control_python` directory contains similar implementations for the Two-Tanks system with smaller networks `[20, 40, 60]`. The CSTR variants in `control_cstr` use:

- Larger neural networks
- CSTR plant instead of Two-Tanks
- Same linearization strategies
