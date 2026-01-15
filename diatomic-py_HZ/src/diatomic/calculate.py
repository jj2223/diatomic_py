from diatomic import log_time
import diatomic.operators as operators
import hamiltonian_builder as hb # 假设这是我们上一节定义的构建器
import numpy as np
import warnings
from operators import HalfInt

@log_time
def solve_system(hamiltonians, num_diagonals=None):
    """
    Solves for the eigenenergies and eigenstates of given Hamiltonian(s over 1
    parameter).

    This function diagonalizes the input Hamiltonian(s) to find the corresponding
    eigenenergies and eigenstates. It supports both single Hamiltonian and a set
    of Hamiltonians (e.g., varying with time or another parameter).

    It takes care to make sure the eigenenergies and eigenstates vary smoothly with
    respect to the varying parameter, unlike `np.eigh`.

    Args:
        hamiltonians (np.ndarray): A single Hamiltonian matrix or an array of
            Hamiltonian matrices.
        num_diagonals (int | None, optional): The number of local swaps above and below
            to consider when smoothing eigenfunctions. Changing this to a number can
            speed up smoothing calculations for large numbers of eigenstates. If None,
            compare all eigenstates to all other eigenstates when smoothing (safest).

    Returns:
        tuple: (eigenenergies, eigenstates).

    Raises:
        ValueError: If the input Hamiltonian has more than three dimensions.
    """
    eigenenergies_raw, eigenstates_raw = log_time(np.linalg.eigh)(hamiltonians)
    if hamiltonians.ndim == 2:
        return eigenenergies_raw, eigenstates_raw
    elif hamiltonians.ndim == 3:
        eigenenergies, eigenstates = sort_smooth(
            eigenenergies_raw, eigenstates_raw, num_diagonals=num_diagonals
        )
        return eigenenergies, eigenstates
    else:
        raise ValueError(
            "Too many dimensions, solve_system doesn't support smoothing"
            " eigenvalues over >1 parameters"
        )


def _matrix_prod_diagonal(A, B, d=0):
    """
    Computes the dth diagonal of the product of two matrices A and B,
    without computing the entire matrix product.

    Args:
        A (np.ndarray): The first matrix operand.
        B (np.ndarray): The second matrix operand.
        d (int, optional): The diagonal offset. Defaults to 0, which gives the
            main diagonal.

    Returns:
        np.ndarray: The computed diagonal elements as a one-dimensional array.
    """
    A_slice = slice(-d if d < 0 else None, -d if d > 0 else None)
    B_slice = slice(d if d > 0 else None, d if d < 0 else None)

    diag = np.einsum("...ab,...ba->...a", A[..., A_slice, :], B[..., B_slice])

    return diag


@log_time
def sort_smooth(eigvals, eigvecs, num_diagonals=None):
    """
    Smooths the eigenvalue trajectories as a function of an external parameter, rather
    than by energy order, as is the case with `np.eigh`.

    This function sorts the eigenvalues and eigenvectors to maintain continuity as the
    external parameter changes, preventing sudden jumps in the eigenvalue ordering.

    Args:
        eigvals (np.ndarray): Eigenvalues array with shape (steps, eigstate).
        eigvecs (np.ndarray): Eigenvectors array with shape (steps, basis, eigstate).
        num_diagonals (int | None, optional): The number of local swaps above and below
            to consider when smoothing eigenfunctions. Changing this to a number can
            speed up smoothing calculations for large numbers of eigenstates. If None,
            compare all eigenstates to all other eigenstates when smoothing (safest).

    Returns:
        tuple: The smoothed eigenvalues and eigenvectors.

    Raises:
        ValueError: If eigvecs does not have three dimensions.
    """
    if eigvecs.ndim != 3:
        raise ValueError("eigvecs is not a set of eigenvectors over one parameter.")

    param_step_count = eigvecs.shape[0]
    basis_size = eigvecs.shape[1]
    eigstate_count = eigvecs.shape[2]

    # Compute overlap matrix between (k)th and (k-1)th
    if num_diagonals is not None:
        k = min(num_diagonals, eigstate_count - 1)
        diagonals = np.zeros((param_step_count - 1, 2 * k + 1, eigstate_count))
        for diag_num in range(-k, k + 1):
            my_prod_diag = np.abs(
                _matrix_prod_diagonal(
                    eigvecs[:-1].swapaxes(-1, -2), eigvecs[1:].conj(), d=diag_num
                )
            )
            if diag_num > 0:
                diagonals[
                    :, k - diag_num, diag_num : diag_num + my_prod_diag.shape[1]
                ] = my_prod_diag
            else:
                diagonals[:, k - diag_num, 0 : my_prod_diag.shape[1]] = my_prod_diag

        best_overlaps = np.argmax(diagonals, axis=-2) - np.arange(
            k, k - eigstate_count, -1
        )
    else:
        overlap_matrices = np.abs(eigvecs[:-1].swapaxes(-1, -2) @ eigvecs[1:].conj())
        best_overlaps = np.argmax(overlap_matrices, axis=1)

    # Cumulative permutations
    integrated_permutations = np.empty((param_step_count, eigstate_count), dtype=int)
    integrated_permutations[-1] = np.arange(eigstate_count)

    for i in range(param_step_count - 2, -1, -1):
        integrated_permutations[i] = best_overlaps[i][integrated_permutations[i + 1]]

    # Rearrange to maintain continuity
    sorted_eigvals = eigvals[
        np.arange(param_step_count)[:, None], integrated_permutations
    ]
    sorted_eigvecs = eigvecs[
        np.arange(param_step_count)[:, None, None],
        np.arange(basis_size)[None, :, None],
        integrated_permutations[:, None, :],
    ]

    return sorted_eigvals, sorted_eigvecs


def _get_prior_repeat_indices(labels):
    """
    Given a numpy array, returns a list of 'prior repeat indices'. Each entry
    of the array that repeats gets an incrementing index on each subsequent occurrence.

    For example, given labels = [[1,1],[2,3],[1,1]], we see [1,1] repeats, and its
    second occurrence gets a 'prior repeat index' of 1. The output is [0,0,1].

    Args:
        labels (np.array): 2D array with labels.

    Returns:
        np.array: 1D array with 'prior repeat indices'.
    """

    double_labels = (2 * labels).astype(int)

    # Identify unique labels, get inverse indices and counts of labels
    _, inverse_indices, counts = np.unique(
        double_labels, return_inverse=True, return_counts=True, axis=0
    )

    # Initialize array for repeat indices
    repeat_indices = np.zeros(double_labels.shape[0], dtype=int)

    # For each unique label
    for idx, count in enumerate(counts):
        # Find all instances of the unique label
        instances = np.nonzero(inverse_indices == idx)[0]

        # Assign repeat indices in order of occurrence
        repeat_indices[instances] = np.arange(count)

    return repeat_indices


def _solve_quadratic(a, b, c):
    """Solve a quadratic equation a*x^2+b*x+c=0 .

    This is a simple function to solve the quadratic formula for x.
    Returns the most positive value of x supported.

    Args:
        a,b,c (floats) - coefficients in quadratic

    Returns:
        x (float) - maximum value of x supported by equation

    """
    x1 = (-b + np.sqrt((b**2) - (4 * (a * c)))) / (2 * a)
    x2 = (-b - np.sqrt((b**2) - (4 * (a * c)))) / (2 * a)
    return np.maximum(x1, x2)

# ==========================================
# 2. 新的量子数标记与属性计算
# ==========================================

def calculate_properties(basis: hb.CollisionBasis, eigenenergies, eigenstates, B_fields):
    """
    计算并分析本征态的物理属性，新增 F1, F2 量子数计算。
    """
    n_steps, n_basis, n_states = eigenstates.shape
    
    # 获取算符 (已扩展到全自旋空间 S1 x I1 x S2 x I2)
    S1, I1, S2, I2 = basis.get_spin_operators()
    sys = basis.sys
    
    # 定义扩展函数 (扩展到 L 空间)
    def expand_op(op_spin):
        blocks = []
        for L in basis.L_list:
            dim_L = int(2*L+1)
            blocks.append(np.kron(op_spin, np.eye(dim_L)))
        return hb.block_diag(*blocks)

    # --- 1. 计算磁矩 (保持不变) ---
    muB = hb.scipy.constants.physical_constants["Bohr magneton"][0]
    muN = hb.scipy.constants.physical_constants["nuclear magneton"][0]
    mu_op_spin = -1 * (
        sys.gS * muB * (S1['z'] + S2['z']) 
        - sys.gI[0] * muN * I1['z'] 
        - sys.gI[1] * muN * I2['z']
    )
    mu_op_total = expand_op(mu_op_spin)
    magnetic_moments = np.einsum('sin,ij,sjn->sn', eigenstates.conj(), mu_op_total, eigenstates).real

    # --- 2. 计算好量子数 M_tot (保持不变) ---
    M_op_spin = S1['z'] + I1['z'] + S2['z'] + I2['z']
    M_blocks = []
    for L in basis.L_list:
        mL_op = np.diag(np.arange(L, -L-1, -1)) 
        block = np.kron(M_op_spin, np.eye(len(mL_op))) + np.kron(np.eye(basis.dim_spin), mL_op)
        M_blocks.append(block)
    M_op_total = hb.block_diag(*M_blocks)
    m_tot_vals = np.einsum('sin,ij,sjn->sn', eigenstates.conj(), M_op_total, eigenstates).real
    m_tot_labels = np.array([[HalfInt(of=int(round(2*x))) for x in row] for row in m_tot_vals])

    # --- 3. 成分分析 (保持不变) ---
    S1_dot_S2 = hb.tensor_dot_vec(S1, S2)
    P_singlet_spin = 0.25 * np.eye(basis.dim_spin) - S1_dot_S2
    P_singlet_total = expand_op(P_singlet_spin)
    singlet_fraction = np.einsum('sin,ij,sjn->sn', eigenstates.conj(), P_singlet_total, eigenstates).real
    
    # --- 4. [新增] 计算 F1 和 F2 量子数 (用于低场标识) ---
    # F1 = S1 + I1
    F1_sq_op = (S1['x']+I1['x'])@(S1['x']+I1['x']) + \
               (S1['y']+I1['y'])@(S1['y']+I1['y']) + \
               (S1['z']+I1['z'])@(S1['z']+I1['z'])
    F1_z_op  = S1['z'] + I1['z']
    
    # F2 = S2 + I2
    F2_sq_op = (S2['x']+I2['x'])@(S2['x']+I2['x']) + \
               (S2['y']+I2['y'])@(S2['y']+I2['y']) + \
               (S2['z']+I2['z'])@(S2['z']+I2['z'])
    F2_z_op  = S2['z'] + I2['z']

    # 扩展到全空间
    F1_sq_total = expand_op(F1_sq_op)
    F1_z_total  = expand_op(F1_z_op)
    F2_sq_total = expand_op(F2_sq_op)
    F2_z_total  = expand_op(F2_z_op)

    # 计算期望值
    # F^2 本征值是 F(F+1)，我们需要反解出 F
    # F = sqrt(F_sq + 1/4) - 1/2
    f1_sq_vals = np.einsum('sin,ij,sjn->sn', eigenstates.conj(), F1_sq_total, eigenstates).real
    f1_vals = np.sqrt(f1_sq_vals + 0.25) - 0.5
    f1_z_vals = np.einsum('sin,ij,sjn->sn', eigenstates.conj(), F1_z_total, eigenstates).real
    
    f2_sq_vals = np.einsum('sin,ij,sjn->sn', eigenstates.conj(), F2_sq_total, eigenstates).real
    f2_vals = np.sqrt(f2_sq_vals + 0.25) - 0.5
    f2_z_vals = np.einsum('sin,ij,sjn->sn', eigenstates.conj(), F2_z_total, eigenstates).real

    # 转成 HalfInt 以便显示
    def to_halfint_arr(arr):
        return np.array([[HalfInt(of=int(round(2*x))) for x in row] for row in arr])

    return {
        "magnetic_moments": magnetic_moments,
        "m_tot": m_tot_labels,
        "singlet_fraction": singlet_fraction,
        # 新增返回
        "f1": to_halfint_arr(f1_vals),
        "mf1": to_halfint_arr(f1_z_vals),
        "f2": to_halfint_arr(f2_vals),
        "mf2": to_halfint_arr(f2_z_vals)
    }

def get_coupling_strength(basis: hb.CollisionBasis, R, state_idx_1, state_idx_2):
    """
    计算两个通道/基矢之间的耦合矩阵元大小 (用于分析)。
    """
    # 这是一个辅助函数，如果用户想知道具体的 <i|V|j>
    # 可以调用 hb.build_dipolar_coupling(basis, R) 得到矩阵 H_dip
    # 然后取 H_dip[i, j]
    pass