import numbers
import numpy as np
import scipy.constants
from scipy.linalg import block_diag
from scipy.special import sph_harm
from sympy.physics.wigner import wigner_3j

from diatomic import log_time

"""
Basis function labels and order
"""


class HalfInt:
    """
    Container to encapsulate half-integer spin & projection lengths. Values will be
    automatically downcast to an integer if the numerator of <of>/2 is an even number
    as soon as possible, even upon initialisation.
    """

    def __new__(cls, *, of):
        if not isinstance(of, numbers.Integral):
            raise TypeError("The argument 'of' must be an integer.")
        elif of % 2 == 0:
            return of // 2
        else:
            return super().__new__(cls)

    def __init__(self, *, of):
        self._double = of

    def _fraction_str(self):
        return f"({self._double}/2)"

    def __repr__(self):
        return f"({self._fraction_str()} : {self.__class__.__name__})"

    def __str__(self):
        return self._fraction_str()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._double == other._double
        elif isinstance(other, int):
            return self._double == 2 * other
        elif isinstance(other, float):
            return self.__float__() == other
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self._double < other._double
        elif isinstance(other, int):
            return self._double < 2 * other
        elif isinstance(other, float):
            return self.__float__() < other
        else:
            return NotImplemented

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self._double > other._double
        elif isinstance(other, int):
            return self._double > 2 * other
        elif isinstance(other, float):
            return self.__float__() > other
        else:
            return NotImplemented

    def __ge__(self, other):
        return self > other or self == other

    def __float__(self):
        return self._double / 2

    def __int__(self):
        return int(self._double / 2)

    def __abs__(self):
        return self.__class__(of=abs(self._double))

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(of=self._double + other._double)
        elif isinstance(other, int):
            return self.__class__(of=self._double + 2 * other)
        elif isinstance(other, float):
            return self.__float__() + other
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(of=self._double - other._double)
        elif isinstance(other, int):
            return self.__class__(of=self._double - 2 * other)
        elif isinstance(other, float):
            return self.__float__() - other
        else:
            return NotImplemented

    def __rsub__(self, other):
        return -(self - other)

    def __neg__(self):
        return self.__class__(of=-self._double)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.__float__() * float(other)
        elif isinstance(other, int):
            return self.__class__(of=self._double * other)
        elif isinstance(other, float):
            return self.__float__() * other
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other


# 复用你之前的 HalfInt 类，因为它处理量子数很方便
# (此处省略 HalfInt 类定义，假设已导入或在同一文件中)
from systems import AlkaliCollisionSystem  # 假设这是我们在上一步定义的系统类

# ==========================================
# 1. 基础算符构建工具
# ==========================================

def j_matrices(j_val):
    """
    生成角动量 J 的自旋矩阵: Jz, J+, J-, Jx, Jy, J_sq
    basis: |j, m> 从 m=+j 到 m=-j
    """
    d = int(2 * j_val + 1)
    m_vals = np.arange(j_val, -j_val - 1, -1)
    
    # Jz 是对角矩阵
    Jz = np.diag(m_vals)
    
    # J+ (Raising operator)
    # <m+1|J+|m> = sqrt(j(j+1) - m(m+1))
    J_plus = np.zeros((d, d))
    for i, m in enumerate(m_vals):
        if i > 0: # 目标态是 m_vals[i-1] = m+1
            m_target = m + 1
            val = np.sqrt(j_val*(j_val+1) - m*m_target)
            J_plus[i-1, i] = val
            
    J_minus = J_plus.T.conj()
    
    Jx = 0.5 * (J_plus + J_minus)
    Jy = -0.5j * (J_plus - J_minus)
    J_sq = (j_val * (j_val + 1)) * np.eye(d)
    
    return {"z": Jz, "+": J_plus, "-": J_minus, "x": Jx, "y": Jy, "sq": J_sq}

# ==========================================
# 2. 全空间算符生成器 (Kronecker Products)
# ==========================================

class CollisionBasis:
    """
    管理未耦合基矢的辅助类。
    Basis Order: S1 (x) I1 (x) S2 (x) I2 (x) L
    """
    def __init__(self, sys: AlkaliCollisionSystem, L_list: list[int]):
        self.sys = sys
        self.L_list = L_list # 例如 [0, 2] 表示 s波和 d波
        
        # 单粒子维数
        self.dim_S1 = int(2 * 0.5 + 1) # 电子自旋总是 1/2
        self.dim_I1 = int(2 * sys.Ii[0] + 1)
        self.dim_S2 = int(2 * 0.5 + 1)
        self.dim_I2 = int(2 * sys.Ii[1] + 1)
        
        # 自旋部分的维数 (S1 * I1 * S2 * I2)
        self.dim_spin = self.dim_S1 * self.dim_I1 * self.dim_S2 * self.dim_I2
        
        # 计算总维数 (包含 L)
        self.dim_total = 0
        self.channel_indices = {} # 记录每个 L 通道在哈密顿量中的起始位置
        
        current_idx = 0
        for L in L_list:
            dim_L = int(2 * L + 1)
            dim_block = self.dim_spin * dim_L
            self.channel_indices[L] = (current_idx, current_idx + dim_block)
            current_idx += dim_block
            self.dim_total += dim_block

    def get_spin_operators(self):
        """
        生成单纯自旋空间 (S1 x I1 x S2 x I2) 的算符字典。
        不包含 L 空间，用于构建原子内部相互作用。
        """
        # 1. 生成单粒子小矩阵
        ops_S1 = j_matrices(0.5)
        ops_I1 = j_matrices(self.sys.Ii[0])
        ops_S2 = j_matrices(0.5)
        ops_I2 = j_matrices(self.sys.Ii[1])
        
        # Identitiies
        eye_S1 = np.eye(self.dim_S1)
        eye_I1 = np.eye(self.dim_I1)
        eye_S2 = np.eye(self.dim_S2)
        eye_I2 = np.eye(self.dim_I2)
        
        # 2. 使用 Kronecker Product 扩展到全自旋空间
        # Order: S1 (x) I1 (x) S2 (x) I2
        
        def expand(o_s1, o_i1, o_s2, o_i2):
            return np.kron(o_s1, np.kron(o_i1, np.kron(o_s2, o_i2)))

        # S1 算符: S1 x 1 x 1 x 1
        S1 = {k: expand(op, eye_I1, eye_S2, eye_I2) for k, op in ops_S1.items()}
        # I1 算符: 1 x I1 x 1 x 1
        I1 = {k: expand(eye_S1, op, eye_S2, eye_I2) for k, op in ops_I1.items()}
        # S2 算符
        S2 = {k: expand(eye_S1, eye_I1, op, eye_I2) for k, op in ops_S2.items()}
        # I2 算符
        I2 = {k: expand(eye_S1, eye_I1, eye_S2, op) for k, op in ops_I2.items()}
        
        return S1, I1, S2, I2

# ==========================================
# 3. 哈密顿量构建函数
# ==========================================

def tensor_dot_vec(Vec1, Vec2):
    """ 计算两个矢量算符的点积: V1 . V2 = V1xV2x + V1yV2y + V1zV2z """
    return Vec1['x'] @ Vec2['x'] + Vec1['y'] @ Vec2['y'] + Vec1['z'] @ Vec2['z']

def build_atomic_hamiltonian(basis: CollisionBasis, B_field_z):
    """
    构建原子内部哈密顿量 (Hyperfine + Zeeman)。
    这对角于 L 空间，即与 L 无关。
    """
    S1, I1, S2, I2 = basis.get_spin_operators()
    sys = basis.sys
    
    # --- 单原子项 ---
    # H_atom1 = Ahf1 * I1.S1 + gS*B*S1z - gI1*B*I1z
    # 注意: gI 的符号定义。通常 H_Z = (gS muB Sz - gI muN Iz) B
    
    # 1. Hyperfine (Scalar)
    H_hf1 = sys.Ahf[0] * tensor_dot_vec(I1, S1)
    H_hf2 = sys.Ahf[1] * tensor_dot_vec(I2, S2)
    
    # 2. Zeeman
    muB = scipy.constants.physical_constants["Bohr magneton"][0]
    muN = scipy.constants.physical_constants["nuclear magneton"][0]
    h = scipy.constants.h # 如果 Ahf 是以 Hz 为单位，这里不需要 h；如果 Ahf 是焦耳，则不需要额外操作。
    # 假设 Ahf 是焦耳 (J)。
    
    # Zeeman Frequency in Joules
    # E = - mu . B = - (-g mu B S) . B = g mu B S B
    # 通常写为 H = (g_e mu_B S_z - g_N mu_N I_z) B_z
    H_z1 = (sys.gS * muB * S1['z'] - sys.gI[0] * muN * I1['z']) * B_field_z
    H_z2 = (sys.gS * muB * S2['z'] - sys.gI[1] * muN * I2['z']) * B_field_z
    
    H_spin = H_hf1 + H_hf2 + H_z1 + H_z2
    
    # --- 扩展到全空间 (包含 L) ---
    # 因为 H_atomic 不依赖于 L，它在每个 L 区块上都是一样的对角块
    # 使用 block_diag 构建大矩阵
    
    blocks = []
    for L in basis.L_list:
        dim_L = int(2*L + 1)
        # H_spin 对 L 是单位矩阵
        block = np.kron(H_spin, np.eye(dim_L))
        blocks.append(block)
        
    return block_diag(*blocks)

def build_exchange_hamiltonian(basis: CollisionBasis):
    """
    构建交换相互作用: V_ex = P_singlet * V_S + P_triplet * V_T
    """
    S1, I1, S2, I2 = basis.get_spin_operators()
    sys = basis.sys
    
    # 计算 S1 . S2
    S1_dot_S2 = tensor_dot_vec(S1, S2)
    
    # 投影算符 (对于两个自旋1/2)
    # P_singlet = 1/4 - S1.S2
    # P_triplet = 3/4 + S1.S2
    eye = np.eye(basis.dim_spin)
    P_singlet = 0.25 * eye - S1_dot_S2
    P_triplet = 0.75 * eye + S1_dot_S2
    
    # 这里的 V_S 和 V_T 实际上应该是径向波函数的重叠积分
    # 如果我们在做一个简单的能级图（束缚态能量已知），这里直接填入能量
    # 如果是在散射计算，这里应该是 V(R) 算符，但为了画能级图，我们假设是渐进值或特定 R 处的值
    # 假设 system 类里存的是势能值
    
    # 警告：对于画 Zeeman 图，这通常是相对于阈值的束缚能
    V_ex_spin = sys.potential_params.get('E_singlet', 0.0) * P_singlet + \
                sys.potential_params.get('E_triplet', 0.0) * P_triplet

    # 扩展到 L 空间
    blocks = []
    for L in basis.L_list:
        dim_L = int(2*L + 1)
        block = np.kron(V_ex_spin, np.eye(dim_L))
        blocks.append(block)
        
    return block_diag(*blocks)

# ==========================================
# 4. 偶极-偶极相互作用 (通道耦合的核心)
# ==========================================

def get_spherical_tensor_spin(basis):
    """
    构建自旋部分的秩2球张量 T^2_q(S)
    T^2 = [S1 x S2]^2
    """
    S1, _, S2, _ = basis.get_spin_operators()
    
    # 定义球基算符 S_plus, S_minus, S_z -> S_+1, S_-1, S_0
    # S_+1 = -1/sqrt(2) * (Sx + iSy) = -1/sqrt(2) * S_plus
    # S_-1 = +1/sqrt(2) * (Sx - iSy) = +1/sqrt(2) * S_minus
    # S_0  = Sz
    
    def get_sph_comp(S_op):
        return {
            +1: -1/np.sqrt(2) * S_op['+'],
            -1:  1/np.sqrt(2) * S_op['-'],
             0:  S_op['z']
        }
        
    S1q = get_sph_comp(S1)
    S2q = get_sph_comp(S2)
    
    # 构建复合张量 [S1 x S2]^2_q
    # T^2_Q = Sum_{q1, q2} <1 q1 1 q2 | 2 Q> S1_q1 S2_q2
    T2_spin = {}
    for Q in range(-2, 3):
        mat = np.zeros_like(S1['z'], dtype=complex)
        for q1 in [-1, 0, 1]:
            for q2 in [-1, 0, 1]:
                # Clebsch-Gordan: <1 q1 1 q2 | 2 Q>
                # wigner_3j conversion: <j1 m1 j2 m2 | J M> = (-1)^(j1-j2+M) sqrt(2J+1) ( j1 j2 J \\ m1 m2 -M )
                cg = float(wigner_3j(1, 1, 2, q1, q2, -Q)) * (-1)**(1-1+Q) * np.sqrt(5)
                # 注意：Sympy wigner_3j 的符号可能需要根据具体约定核对，通常 CG = (-1)^(j1-j2+m3) * sqrt(2j3+1) * 3j
                # 这里简单起见使用 standard computation
                if abs(cg) > 1e-10:
                    mat += cg * (S1q[q1] @ S2q[q2])
        T2_spin[Q] = mat
        
    return T2_spin

def build_dipolar_coupling(basis: CollisionBasis, R_dist: float = 1e-9):
    """
    构建磁偶极-磁偶极相互作用矩阵。
    V_dd = C_dd / R^3 * Sum_q (-1)^q T^2_q(Spin) * C^2_-q(Space)
    
    这个函数会填充不同 L 通道之间的非对角元。
    """
    sys = basis.sys
    prefactor = sys.dipole_prefactor / (R_dist**3) # J
    
    # 1. 获取自旋张量 T^2_q(Spin) (dim_spin x dim_spin)
    T2_spin = get_spherical_tensor_spin(basis)
    
    # 2. 初始化全哈密顿量
    H_total = np.zeros((basis.dim_total, basis.dim_total), dtype=complex)
    
    # 3. 遍历所有通道对 (L, L')
    for L_row in basis.L_list:
        for L_col in basis.L_list:
            
            # 选择规则：|L - L'| <= 2 <= L + L' 且 L+L' 为偶数 (宇称守恒)
            # 对于偶极耦合，实际上只有 L'=L, L'=L+2, L'=L-2
            if abs(L_row - L_col) > 2 or (L_row + L_col) % 2 != 0:
                continue
                
            # 计算空间矩阵元 <L m_L | C^2_-q | L' m_L'>
            # 这里 C^2_q = sqrt(4pi / 5) Y_2q
            # Reduced element <L || C^2 || L'>
            # formula: <L || C^k || L'> = (-1)^L * sqrt((2L+1)(2L'+1)) * 3j(L, k, L'; 0, 0, 0)
            reduced_mat_elem = (-1)**L_row * np.sqrt((2*L_row+1)*(2*L_col+1)) * \
                               float(wigner_3j(L_row, 2, L_col, 0, 0, 0))
            
            # 构建空间部分的矩阵 (dim_L_row x dim_L_col)
            # Sum_q (-1)^q T^2_q(Spin) (x) <Lm | C^2_-q | L'm'>
            
            # 获取对应的块索引
            idx_row_start, idx_row_end = basis.channel_indices[L_row]
            idx_col_start, idx_col_end = basis.channel_indices[L_col]
            
            dim_row = int(2*L_row+1)
            dim_col = int(2*L_col+1)
            
            # 初始化这个 L-L' 块
            block = np.zeros((basis.dim_spin * dim_row, basis.dim_spin * dim_col), dtype=complex)
            
            # 对 q 求和
            for q in range(-2, 3):
                # 空间部分矩阵
                space_mat = np.zeros((dim_row, dim_col))
                for m_row_idx, m_row_val in enumerate(range(L_row, -L_row-1, -1)):
                    for m_col_idx, m_col_val in enumerate(range(L_col, -L_col-1, -1)):
                        # Wigner-Eckart theorem:
                        # <L m | T^k_q | L' m'> = (-1)^(L-m) * (L k L' \\ -m q m') * <L || T^k || L'>
                        # 注意 C^2_-q 的 q 符号
                        target_q = -q
                        three_j = float(wigner_3j(L_row, 2, L_col, -m_row_val, target_q, m_col_val))
                        
                        val = (-1)**(L_row - m_row_val) * three_j * reduced_mat_elem
                        space_mat[m_row_idx, m_col_idx] = val
                
                # Tensor Product: Spin part (x) Space part
                # T^2_q(Spin) * (-1)^q
                spin_part = T2_spin[q] * ((-1)**q)
                
                block += np.kron(spin_part, space_mat)
            
            # 填入大矩阵
            H_total[idx_row_start:idx_row_end, idx_col_start:idx_col_end] += prefactor * block

    return H_total