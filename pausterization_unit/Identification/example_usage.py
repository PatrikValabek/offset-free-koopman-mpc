"""
Example usage of the Koopman model inference interface.

This script demonstrates how to:
1. Initialize the model
2. Encode output measurements
3. Decode latent states
4. Predict next steps with control inputs
5. Run multi-step simulations
"""

import numpy as np
import baseline_inference as model

def example_basic_usage():
    """Example 1: Basic initialization and single-step prediction."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Initialize the model (must be called first)
    model.init()
    print()
    
    # Set initial condition from measurement
    y_current = np.array([[75.5],   # T1 in °C
                          [80.2],   # T2 in °C
                          [78.9]])  # T4 in °C
    
    print(f"Current output measurement: {y_current.flatten()}")
    
    # Encode to latent space
    model.get_x(y_current)
    print(f"Latent state shape: {model.x.shape}")
    print(f"Latent state (first 5 dims): {model.x[:5, 0]}")
    print()
    
    # Decode back to output space (verify encoding)
    y_reconstructed = model.get_y(model.x)
    print(f"Reconstructed output: {y_reconstructed.flatten()}")
    print(f"Reconstruction error: {np.linalg.norm(y_current - y_reconstructed):.6f}")
    print()
    
    # Apply control input
    u_control = np.array([[0.5],   # u1
                          [0.3],   # u2
                          [0.7]])  # u3
    
    print(f"Control input: {u_control.flatten()}")
    
    # Predict next step
    y_next = model.y_plus(u_control)
    print(f"Predicted next output: {y_next.flatten()}")
    print(f"Output change: {(y_next - y_current).flatten()}")
    print()


def example_multi_step_prediction():
    """Example 2: Multi-step prediction with constant control."""
    print("=" * 60)
    print("Example 2: Multi-Step Prediction")
    print("=" * 60)
    
    # Reset to initial condition
    y_initial = np.array([[75.0], [80.0], [78.0]])
    model.get_x(y_initial)
    
    # Constant control input
    u_constant = np.array([[0.2], [0.4], [0.5]])
    
    # Simulate 20 steps ahead
    n_steps = 20
    predictions = [y_initial]
    
    print(f"Initial output: {y_initial.flatten()}")
    print(f"Constant control: {u_constant.flatten()}")
    print()
    
    for k in range(n_steps):
        y_k = model.y_plus(u_constant)
        predictions.append(y_k.copy())
    
    predictions = np.hstack(predictions)  # Shape: [3, n_steps+1]
    
    print(f"After {n_steps} steps:")
    print(f"  T1: {predictions[0, 0]:.2f} -> {predictions[0, -1]:.2f}")
    print(f"  T2: {predictions[1, 0]:.2f} -> {predictions[1, -1]:.2f}")
    print(f"  T4: {predictions[2, 0]:.2f} -> {predictions[2, -1]:.2f}")
    print()


def example_varying_control():
    """Example 3: Multi-step prediction with varying control."""
    print("=" * 60)
    print("Example 3: Varying Control Sequence")
    print("=" * 60)
    
    # Reset to initial condition
    y_initial = np.array([[75.0], [80.0], [78.0]])
    model.get_x(y_initial)
    
    # Create varying control sequence (sinusoidal)
    n_steps = 50
    time = np.arange(n_steps)
    u_sequence = np.zeros((3, n_steps))
    u_sequence[0, :] = 0.3 + 0.2 * np.sin(2 * np.pi * time / 20)  # u1
    u_sequence[1, :] = 0.5 + 0.1 * np.cos(2 * np.pi * time / 15)  # u2
    u_sequence[2, :] = 0.4 + 0.15 * np.sin(2 * np.pi * time / 25) # u3
    
    # Simulate forward
    predictions = [y_initial]
    for k in range(n_steps):
        u_k = u_sequence[:, k].reshape(-1, 1)
        y_k = model.y_plus(u_k)
        predictions.append(y_k.copy())
    
    predictions = np.hstack(predictions)  # Shape: [3, n_steps+1]
    
    print(f"Simulated {n_steps} steps with sinusoidal control inputs")
    print(f"Output statistics:")
    print(f"  T1: min={predictions[0, :].min():.2f}, max={predictions[0, :].max():.2f}, final={predictions[0, -1]:.2f}")
    print(f"  T2: min={predictions[1, :].min():.2f}, max={predictions[1, :].max():.2f}, final={predictions[1, -1]:.2f}")
    print(f"  T4: min={predictions[2, :].min():.2f}, max={predictions[2, :].max():.2f}, final={predictions[2, -1]:.2f}")
    print()


def example_access_matrices():
    """Example 4: Access the A and B matrices for MPC."""
    print("=" * 60)
    print("Example 4: Accessing System Matrices")
    print("=" * 60)
    
    # A and B matrices are available after init()
    print(f"A matrix (Koopman operator):")
    print(f"  Shape: {model.A.shape}")
    print(f"  Norm: {np.linalg.norm(model.A):.4f}")
    print(f"  Spectral radius: {np.max(np.abs(np.linalg.eigvals(model.A))):.4f}")
    print()
    
    print(f"B matrix (input encoder):")
    print(f"  Shape: {model.B.shape}")
    print(f"  Norm: {np.linalg.norm(model.B):.4f}")
    print()
    
    # These matrices can be used in MPC formulations
    print("These matrices define the linear dynamics in latent space:")
    print("  x[k+1] = A @ x[k] + B @ u[k]")
    print("  y[k] = decoder(x[k])")
    print()


def example_comparison_with_zero_control():
    """Example 5: Compare predictions with and without control."""
    print("=" * 60)
    print("Example 5: Effect of Control Input")
    print("=" * 60)
    
    # Initial condition
    y_initial = np.array([[75.0], [80.0], [78.0]])
    
    # Scenario 1: No control
    model.get_x(y_initial)
    u_zero = np.array([[0.0], [0.0], [0.0]])
    y_no_control = model.y_plus(u_zero)
    
    # Scenario 2: With control
    model.get_x(y_initial)
    u_with_control = np.array([[0.5], [0.5], [0.5]])
    y_with_control = model.y_plus(u_with_control)
    
    print(f"Initial state: {y_initial.flatten()}")
    print()
    print(f"No control input: {u_zero.flatten()}")
    print(f"  Next output: {y_no_control.flatten()}")
    print()
    print(f"With control: {u_with_control.flatten()}")
    print(f"  Next output: {y_with_control.flatten()}")
    print()
    print(f"Control effect (difference): {(y_with_control - y_no_control).flatten()}")
    print()


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_multi_step_prediction()
    example_varying_control()
    example_access_matrices()
    example_comparison_with_zero_control()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

