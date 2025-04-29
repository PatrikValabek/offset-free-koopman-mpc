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