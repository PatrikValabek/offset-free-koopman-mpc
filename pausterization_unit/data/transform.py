"""
Transform state space matrices from Ts=1s to Ts=10s sampling time.

This script loads A, B, C matrices for three systems:
- C_False (A_C_False.npy, B_C_False.npy, C_C_False.npy)
- C_True (A_C_True.npy, B_C_True.npy, C_C_True.npy)
- parsimK (A_parsimK.npy, B_parsimK.npy, C_parsimK.npy)

And transforms them from discrete-time with Ts=1s to Ts=10s.
The C matrix remains unchanged as it's not affected by sampling time.
"""

import numpy as np
from scipy.linalg import logm, expm, inv
from pathlib import Path


def discrete_to_continuous(Ad, Bd, Ts):
    """
    Convert discrete-time state space to continuous-time.
    
    For discrete system: x[k+1] = Ad*x[k] + Bd*u[k] at sampling time Ts
    Returns continuous system: dx/dt = Ac*x + Bc*u
    
    Args:
        Ad: Discrete-time A matrix
        Bd: Discrete-time B matrix
        Ts: Current sampling time
        
    Returns:
        Ac: Continuous-time A matrix
        Bc: Continuous-time B matrix
    """
    # Compute continuous-time A matrix using matrix logarithm
    # Ad = exp(Ac * Ts), so Ac = (1/Ts) * log(Ad)
    Ac = (1.0 / Ts) * logm(Ad)
    
    # Compute continuous-time B matrix
    # For zero-order hold discretization: Bd = integral(exp(Ac*t), 0 to Ts) * Bc
    # This gives: Bc = Ac * inv(Ad - I) * Bd (if Ad - I is invertible)
    I = np.eye(Ad.shape[0])
    Ad_minus_I = Ad - I
    
    # Check if Ad - I is invertible
    if np.linalg.cond(Ad_minus_I) < 1e12:
        Bc = Ac @ inv(Ad_minus_I) @ Bd
    else:
        # Alternative method using matrix exponential series
        # Bc = inv(integral(exp(Ac*t), 0 to Ts)) * Bd
        # For small Ts or when Ad - I is singular, use numerical integration approach
        # Approximate: Bc ≈ (1/Ts) * inv(Ad) * Bd for stable systems
        # More robust: use pseudo-inverse
        Bc = Ac @ np.linalg.pinv(Ad_minus_I) @ Bd
    
    return Ac, Bc


def continuous_to_discrete(Ac, Bc, Ts):
    """
    Convert continuous-time state space to discrete-time.
    
    For continuous system: dx/dt = Ac*x + Bc*u
    Returns discrete system: x[k+1] = Ad*x[k] + Bd*u[k] at sampling time Ts
    
    Args:
        Ac: Continuous-time A matrix
        Bc: Continuous-time B matrix
        Ts: Target sampling time
        
    Returns:
        Ad: Discrete-time A matrix
        Bd: Discrete-time B matrix
    """
    # Compute discrete-time A matrix using matrix exponential
    # Ad = exp(Ac * Ts)
    Ad = expm(Ac * Ts)
    
    # Compute discrete-time B matrix
    # For zero-order hold: Bd = integral(exp(Ac*t), 0 to Ts) * Bc
    # This gives: Bd = inv(Ac) * (Ad - I) * Bc (if Ac is invertible)
    I = np.eye(Ac.shape[0])
    Ad_minus_I = Ad - I
    
    # Check if Ac is invertible
    if np.linalg.cond(Ac) < 1e12:
        Bd = inv(Ac) @ Ad_minus_I @ Bc
    else:
        # Alternative method: use numerical integration or pseudo-inverse
        # For singular Ac, use pseudo-inverse
        Bd = np.linalg.pinv(Ac) @ Ad_minus_I @ Bc
    
    return Ad, Bd


def transform_sampling_time(Ad_old, Bd_old, Ts_old, Ts_new):
    """
    Transform discrete-time state space matrices from Ts_old to Ts_new.
    
    Args:
        Ad_old: Discrete-time A matrix at Ts_old
        Bd_old: Discrete-time B matrix at Ts_old
        Ts_old: Original sampling time
        Ts_new: New sampling time
        
    Returns:
        Ad_new: Discrete-time A matrix at Ts_new
        Bd_new: Discrete-time B matrix at Ts_new
    """
    # Convert to continuous-time
    Ac, Bc = discrete_to_continuous(Ad_old, Bd_old, Ts_old)
    
    # Convert back to discrete-time with new sampling time
    Ad_new, Bd_new = continuous_to_discrete(Ac, Bc, Ts_new)
    
    return Ad_new, Bd_new


def transform_system(system_name, data_dir, Ts_old=1.0, Ts_new=10.0):
    """
    Transform a complete system's matrices from Ts_old to Ts_new.
    
    Args:
        system_name: Name of the system (e.g., 'C_False', 'C_True', 'parsimK')
        data_dir: Directory containing the .npy files
        Ts_old: Original sampling time (default: 1.0s)
        Ts_new: New sampling time (default: 10.0s)
    """
    data_dir = Path(data_dir)
    
    # Construct file paths based on system name
    if system_name in ['C_False', 'C_True']:
        A_path = data_dir / f'A_{system_name}.npy'
        B_path = data_dir / f'B_{system_name}.npy'
        C_path = data_dir / f'C_{system_name}.npy'
        A_new_path = data_dir / f'A_{system_name}_Ts.npy'
        B_new_path = data_dir / f'B_{system_name}_Ts.npy'
        C_new_path = data_dir / f'C_{system_name}_Ts.npy'
    elif system_name == 'parsimK':
        A_path = data_dir / 'A_parsimK.npy'
        B_path = data_dir / 'B_parsimK.npy'
        C_path = data_dir / 'C_parsimK.npy'
        A_new_path = data_dir / 'A_parsimK_Ts.npy'
        B_new_path = data_dir / 'B_parsimK_Ts.npy'
        C_new_path = data_dir / 'C_parsimK_Ts.npy'
    else:
        raise ValueError(f"Unknown system name: {system_name}")
    
    # Load matrices
    print(f"\nTransforming system: {system_name}")
    print(f"  Loading from: {A_path.name}, {B_path.name}, {C_path.name}")
    
    A_old = np.load(A_path)
    B_old = np.load(B_path)
    C_old = np.load(C_path)
    
    print(f"  Original A shape: {A_old.shape}, B shape: {B_old.shape}, C shape: {C_old.shape}")
    
    # Transform A and B matrices (C remains unchanged)
    A_new, B_new = transform_sampling_time(A_old, B_old, Ts_old, Ts_new)
    C_new = C_old.copy()  # C matrix is not affected by sampling time
    
    print(f"  New A shape: {A_new.shape}, B shape: {B_new.shape}, C shape: {C_new.shape}")
    print(f"  Sampling time: {Ts_old}s -> {Ts_new}s")
    
    # Save transformed matrices
    np.save(A_new_path, A_new)
    np.save(B_new_path, B_new)
    np.save(C_new_path, C_new)
    
    print(f"  Saved to: {A_new_path.name}, {B_new_path.name}, {C_new_path.name}")
    
    # Verify transformation by checking eigenvalues
    eigenvals_old = np.linalg.eigvals(A_old)
    eigenvals_new = np.linalg.eigvals(A_new)
    print(f"  Original A eigenvalues (max abs): {np.max(np.abs(eigenvals_old)):.6f}")
    print(f"  New A eigenvalues (max abs): {np.max(np.abs(eigenvals_new)):.6f}")
    
    return A_new, B_new, C_new


def main():
    """
    Main function to transform all three systems.
    """
    # Define paths
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir  # Data files are in the same directory as this script
    
    # Define sampling times
    Ts_old = 1.0  # Current sampling time (1 second)
    Ts_new = 10.0  # New sampling time (10 seconds)
    
    print("=" * 60)
    print("State Space Matrix Transformation")
    print("=" * 60)
    print(f"Transforming from Ts = {Ts_old}s to Ts = {Ts_new}s")
    print(f"Data directory: {data_dir}")
    print("=" * 60)
    
    # Define the three systems explicitly
    systems = ['C_False', 'C_True', 'parsimK']
    
    # Transform each system
    for system_name in systems:
        try:
            transform_system(system_name, data_dir, Ts_old, Ts_new)
        except FileNotFoundError as e:
            print(f"  ERROR: Could not find files for system {system_name}")
            print(f"  {e}")
        except Exception as e:
            print(f"  ERROR: Failed to transform system {system_name}")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Transformation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

