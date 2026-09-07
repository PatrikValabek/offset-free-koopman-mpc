import numpy as np
from numpy.linalg import eig, inv

def real_block_diagonalize(A):
    """
    Perform real block diagonalization of a square matrix A.

    This function computes a real transformation matrix T_real and a block-diagonal
    matrix A_block such that:
        A_block = T_real^(-1) * A * T_real
    where A_block is block-diagonal with real eigenvalues as 1x1 blocks and complex
    conjugate eigenvalues as 2x2 real blocks.

    Args:
        A (numpy.ndarray): A square matrix of shape (n, n).

    Returns:
        tuple:
            - T_real (numpy.ndarray): The real transformation matrix of shape (n, n).
            - A_block (numpy.ndarray): The block-diagonalized matrix of shape (n, n).

    Example:
        >>> A = np.array([[1, -1], [1, 1]])
        >>> T_real, A_block = real_block_diagonalize(A)
        >>> print(T_real)
        >>> print(A_block)
    """
    eigvals, eigvecs = eig(A)
    n = A.shape[0]

    T_real_cols = []
    block_sizes = []
    block_matrices = []

    used = set()

    for i in range(n):
        if i in used:
            continue

        λ = eigvals[i]
        v = eigvecs[:, i]

        if np.isreal(λ):
            # Real eigenvalue
            T_real_cols.append(np.real(v))
            block_sizes.append(1)
            block_matrices.append(np.array([[np.real(λ)]]))
        else:
            # Complex eigenvalue → find conjugate
            λ_conj = np.conj(λ)
            for j in range(i + 1, n):
                if j not in used and np.isclose(eigvals[j], λ_conj):
                    v_conj = eigvecs[:, j]
                    # Construct real 2D basis
                    v1 = np.real(v)
                    v2 = np.imag(v)
                    T_real_cols.append(v1)
                    T_real_cols.append(v2)

                    a = np.real(λ)
                    b = np.imag(λ)
                    block_sizes.append(2)
                    block_matrices.append(np.array([[a, b], [-b, a]]))
                    used.add(j)
                    break
        used.add(i)

    # Assemble T_real
    T_real = np.column_stack(T_real_cols)

    # Assemble A_block from blocks
    A_block = np.zeros((n, n))
    idx = 0
    for size, block in zip(block_sizes, block_matrices):
        A_block[idx:idx+size, idx:idx+size] = block
        idx += size

    return T_real, A_block

def generate_cstr_yield_steps(step_time: int, how_many: int, constraints, seed: int = 42):
    """
    Random steps that push the CSTR–separator across the xB3 yield peak.

    constraints rows: [Q1, Q2, Q3, F10, F20, Fr], each [min, max].
    A latent conversion intent s∈[0,1] sets low F10 + high Q + high Fr
    (over-conversion) versus high F10 + low Q + low Fr (under-conversion).
    F20 is drawn independently in its box.
    """
    rng = np.random.default_rng(seed)
    cons = np.asarray(constraints, dtype=float)
    n_u = cons.shape[0]
    steps = np.zeros((how_many * step_time, n_u))
    q_lo, q_hi = cons[0]
    f10_lo, f10_hi = cons[3]
    f20_lo, f20_hi = cons[4]
    fr_lo, fr_hi = cons[5]
    q_jitter = 0.25 * (q_hi - q_lo)
    fr_jitter = 0.15 * (fr_hi - fr_lo)

    for i in range(how_many):
        s = rng.uniform(0.0, 1.0)
        f10 = f10_lo + (1.0 - s) * (f10_hi - f10_lo)
        q_center = q_lo + s * (q_hi - q_lo)
        fr_center = fr_lo + s * (fr_hi - fr_lo)
        q1 = np.clip(q_center + rng.uniform(-q_jitter, q_jitter), q_lo, q_hi)
        q2 = np.clip(q_center + rng.uniform(-q_jitter, q_jitter), q_lo, q_hi)
        q3 = np.clip(q_center + rng.uniform(-q_jitter, q_jitter), q_lo, q_hi)
        fr = np.clip(fr_center + rng.uniform(-fr_jitter, fr_jitter), fr_lo, fr_hi)
        f20 = rng.uniform(f20_lo, f20_hi)
        steps[i * step_time : (i + 1) * step_time, :] = [q1, q2, q3, f10, f20, fr]
    return steps


def generate_steps(step_time: float, how_many: int, constraints: list, seed: int = 42):
    """
    Get the time points where the range changes. Range should be a numpy array representing a range of values. In left column min, in right column max.
    """
    np.random.seed(seed)
    
    step_changes = np.zeros((how_many*step_time, constraints.shape[0]))
    for i in range(how_many):
        for j in range(constraints.shape[0]):
            step_changes[i*step_time:(i+1)*step_time,j] = np.random.uniform(low=constraints[j,0], high=constraints[j,1])
        
    return step_changes