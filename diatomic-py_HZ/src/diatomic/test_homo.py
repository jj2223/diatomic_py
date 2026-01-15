import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

from systems import AlkaliCollisionSystem
import hamiltonian_builder as hb
import calculate as calc
# 在 test_homo.py 顶部添加
from sympy.physics.quantum.cg import CG
# --- 单位 ---
h_planck = const.h
E_to_GHz = 1e-9 / h_planck
B_Gauss_to_Tesla = 1e-4
def build_coupled_transformation(basis):
    """
    构建从未耦合基 |mS, mI> 到耦合基 |F, mF> 的变换矩阵 U_atom。
    适用于单个原子。
    """
    sys = basis.sys
    I = sys.Ii[0] # 同核，取第一个即可
    S = 0.5
    
    # 1. 确定维数和量子数列表
    dim_atom = int((2*S + 1) * (2*I + 1))
    
    # 未耦合基列表 (mS, mI) - 注意顺序要与 hamiltonian_builder 保持一致
    # 假设顺序是 mS 慢变, mI 快变 (或者相反，需核对 get_quantum_numbers_from_index)
    # 我们之前的逻辑是: i_S1 * d_I1 + i_I1. 所以 S 是高位(慢), I 是低位(快)
    uncoupled_states = []
    for i_S in range(int(2*S+1)):
        for i_I in range(int(2*I+1)):
            mS = S - i_S # S, S-1 ...
            mI = I - i_I # I, I-1 ...
            uncoupled_states.append((mS, mI))
            
    # 耦合基列表 (F, mF)
    # F 从 I+S 到 |I-S|
    coupled_states = []
    F_values = [I + S, abs(I - S)] 
    # 排序 F，通常大的 F 在前还是在后取决于习惯，这里我们按 F 从大到小排
    F_values = sorted(F_values, reverse=True) 
    
    for F in F_values:
        # mF 从 F 到 -F
        for i_mF in range(int(2*F+1)):
            mF = F - i_mF
            coupled_states.append((F, mF))
            
    # 2. 构建矩阵 U (Rows: Uncoupled, Cols: Coupled)
    U = np.zeros((dim_atom, dim_atom), dtype=float)
    
    for c_idx, (F, mF) in enumerate(coupled_states):
        for u_idx, (mS, mI) in enumerate(uncoupled_states):
            # 只有 mS + mI == mF 时 CG 系数才非零
            if abs(mS + mI - mF) > 1e-9:
                continue
                
            # 计算 <S mS, I mI | F mF>
            # sympy 的 CG(j1, m1, j2, m2, j3, m3)
            cg_val = float(CG(S, mS, I, mI, F, mF).doit())
            U[u_idx, c_idx] = cg_val
            
    return U, coupled_states

def get_homonuclear_label(vec_full_at_zero_field, U_atom, coupled_states):
    """
    输入:
      vec_full: 全空间本征态向量 (dim_full,)
      U_atom: 单原子变换矩阵
      coupled_states: 单原子耦合基量子数列表 [(F, mF), ...]
    输出:
      标签字符串
    """
    # 1. 构建双原子变换矩阵 U_tot = U_atom (x) U_atom
    # 为了节省时间，这个 U_tot 应该在外面算好传进来，但为了逻辑清晰先写在这里
    # 实际调用时我们可以利用 Kronecker 积性质: 
    # vec_coupled = (U (x) U).T @ vec_uncoupled
    # 这等价于重塑矩阵后: Mat_coupled = U.T @ Mat_uncoupled @ U
    
    dim_atom = U_atom.shape[0]
    
    # 将向量重塑为 (dim_atom, dim_atom) 矩阵，对应 (Atom1, Atom2)
    # 注意 numpy reshape 是 C-order (行优先)，对应我们的 S1(x)I1(x)S2(x)I2 结构
    # 这里的结构是 (Atom1_Combined) x (Atom2_Combined)
    psi_matrix_unc = vec_full_at_zero_field.reshape((dim_atom, dim_atom))
    
    # 变换到耦合基: Psi_coup = U.T @ Psi_unc @ U
    psi_matrix_coup = U_atom.T @ psi_matrix_unc @ U_atom
    
    # 2. 找到最大分量
    # argmax 返回的是线性索引，需要转回 (idx_1, idx_2)
    flat_idx = np.argmax(np.abs(psi_matrix_coup))
    idx_1, idx_2 = np.unravel_index(flat_idx, (dim_atom, dim_atom))
    
    # 3. 获取量子数
    F1, mF1 = coupled_states[idx_1]
    F2, mF2 = coupled_states[idx_2]
    
    # 4. 对称化排序 (消除 |a,b> 和 |b,a> 的区别)
    # 规则: 优先 F 大的在前；F 相同时，mF 大的在前
    state1 = (F1, mF1)
    state2 = (F2, mF2)
    
    if state1 >= state2:
        fa, ma, fb, mb = F1, mF1, F2, mF2
    else:
        fa, ma, fb, mb = F2, mF2, F1, mF1
        
    return f"$|{fa}, {ma}; {fb}, {mb}\\rangle$"
def run_homonuclear_analysis():
    print("=== 同核双原子分子 (Rb87-Rb87) 塞曼能级计算 ===")
    
    # 1. 初始化
    sys = AlkaliCollisionSystem.from_preset("Rb87Rb87")
    basis = hb.CollisionBasis(sys, L_list=[0]) # 只看 s波
    
    print(f"系统: {sys.name} ({sys.statistics})")
    print(f"原始全基矢维数: {basis.dim_total}")
    
    # 2. 构建投影矩阵 P (只需一次)
    # P 的形状是 (64, 36)
    P_proj = hb.build_total_projection_matrix(basis)
    dim_sym = P_proj.shape[1]
    print(f"对称化后维数: {dim_sym} (预期: 36)")
    
    # 3. 扫描磁场并构建对称哈密顿量
    B_gauss = np.linspace(0.1, 1000, 400)
    B_tesla = B_gauss * B_Gauss_to_Tesla
    
    hamiltonians_sym = []
    
    print("构建与投影哈密顿量...")
    for B in B_tesla:
        # A. 构建原始大矩阵 (64x64)
        H_full = hb.build_atomic_hamiltonian(basis, B)
        
        # B. 投影到对称子空间 (36x36)
        # H_sym = P.T * H * P
        H_sym = P_proj.T @ H_full @ P_proj
        
        hamiltonians_sym.append(H_sym)
        
    hamiltonians_sym = np.array(hamiltonians_sym)
    
    # 4. 对角化 (解的是小矩阵 36x36)
    print("对角化...")
    eigvals, eigvecs_sym = calc.solve_system(hamiltonians_sym)
    
    # 5. [关键步骤] 恢复全空间波函数以计算属性
    # 我们需要计算 M_tot, Singlet Fraction 等。
    # 现有的 calculate_properties 是基于全基矢算符 (S1, I1...) 写的。
    # 所以我们将本征态投影回全空间: Psi_full = P * Psi_sym
    
    print("恢复全空间波函数...")
    # eigvecs_sym shape: (steps, dim_sym, states)
    # P shape: (dim_full, dim_sym)
    # 结果 shape: (steps, dim_full, states)
    # 使用 einsum 进行批量矩阵乘法: full = P @ sym
    eigvecs_full = np.einsum('fs,tse->tfe', P_proj, eigvecs_sym)
    
    # 6. 计算属性 (使用恢复后的波函数)
    print("计算属性...")
    props = calc.calculate_properties(basis, eigvals, eigvecs_full, B_tesla)
    # === 新增：准备标签生成器 ===
    print("构建耦合基变换矩阵...")
    U_atom, coupled_states_list = build_coupled_transformation(basis)
    # 7. 绘图 (和之前一样)
    print("绘图...")
    fig, ax = plt.subplots(figsize=(12, 8))
    E_plot = (eigvals - eigvals[0, 0]) * E_to_GHz
    
    target_mf = 0.0 
    
    for i in range(dim_sym): 
        # 1. 过滤 M_tot
        if props['m_tot'][0, i] != target_mf:
            continue
            
        # 2. 颜色
        s_val = np.mean(props['singlet_fraction'][:, i])
        c_red = np.array([0.8, 0.0, 0.0])
        c_blue = np.array([0.0, 0.0, 0.8])
        line_color = s_val * c_red + (1.0 - s_val) * c_blue
        
        ax.plot(B_gauss, E_plot[:, i], color=line_color, linewidth=1.5)
        
        # 3. 生成标签 (使用第 0 步/低场的本征态)
        # 获取第 0 步的全空间波函数向量
        vec_at_zero = eigvecs_full[0, :, i] # shape: (64,)
        
        # 调用我们的新函数获取标签
        label_str = get_homonuclear_label(vec_at_zero, U_atom, coupled_states_list)
        
        # 4. 标注
        ax.text(B_gauss[-1]+5, E_plot[-1, i], label_str, fontsize=8, color=line_color, verticalalignment='center')
            

    ax.set_title(f"Rb87-Rb87 Homonuclear Spectrum (L=0, Sym), M={target_mf}")
    ax.set_xlabel("Magnetic Field (G)")
    ax.set_ylabel("Energy (GHz)")
    ax.grid(True, alpha=0.3)
    plt.subplots_adjust(right=0.85)
    
    plt.savefig("spectrum_homonuclear.png", dpi=200)
    print("完成。图片保存为 spectrum_homonuclear.png")

if __name__ == "__main__":
    run_homonuclear_analysis()