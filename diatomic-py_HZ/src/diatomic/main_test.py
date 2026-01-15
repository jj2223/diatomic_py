import matplotlib
# 关键修改 1: 在导入 pyplot 之前设置后端为 'Agg' (Anti-Grain Geometry)，用于生成文件
matplotlib.use('Agg') 

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

# 导入你的模块
from systems import AlkaliCollisionSystem
import hamiltonian_builder as hb
import calculate as calc

# --- 单位转换辅助 ---
h_planck = const.h
E_to_MHz = 1e-6 / h_planck  # Joule -> MHz
E_to_GHz = 1e-9 / h_planck  # Joule -> GHz
B_Gauss_to_Tesla = 1e-4
def find_all_features(B_array, eigenvals, props, tolerance=1e-6):
    """
    同时查找真实交叉(Real Crossings)和避免交叉(Avoided Crossings)。
    
    Returns:
        list: [(B, E, i, j, type_str), ...]
        type_str 是 'real' 或 'avoided'
    """
    n_steps, n_states = eigenvals.shape
    features = []
    
    m_tot = props['m_tot'][0, :] # 获取所有态的 M (好量子数)

    for i in range(n_states):
        for j in range(i + 1, n_states):
            
            # 1. 区分类型
            m_i = m_tot[i]
            m_j = m_tot[j]
            
            delta_E = eigenvals[:, i] - eigenvals[:, j]
            
            # --- 情况 A: 不同 M_tot -> 寻找真实交叉 (变号) ---
            if m_i != m_j:
                sign = np.sign(delta_E)
                sign[sign == 0] = 1 
                diff_sign = np.diff(sign)
                crossing_indices = np.where(diff_sign != 0)[0]
                
                for idx in crossing_indices:
                    # 线性插值找零点
                    y1 = delta_E[idx]; y2 = delta_E[idx+1]
                    x1 = B_array[idx]; x2 = B_array[idx+1]
                    if x1 < tolerance: continue
                    
                    slope = (y2 - y1) / (x2 - x1)
                    b_feat = x1 - y1 / slope
                    
                    # 对应的能量
                    e_slope = (eigenvals[idx+1, i] - eigenvals[idx, i]) / (x2 - x1)
                    e_feat = eigenvals[idx, i] + e_slope * (b_feat - x1)
                    
                    features.append((b_feat, e_feat, i, j, 'real'))
            
            # --- 情况 B: 相同 M_tot -> 寻找避免交叉 (极小值) ---
            else:
                # 计算能量差的绝对值
                gap = np.abs(delta_E)
                
                # 寻找局部极小值: gap[k] < gap[k-1] 且 gap[k] < gap[k+1]
                # 排除端点，只找中间的极小值
                local_min_indices = np.where((gap[1:-1] < gap[0:-2]) & (gap[1:-1] < gap[2:]))[0] + 1
                
                for idx in local_min_indices:
                    x_pos = B_array[idx]
                    if x_pos < tolerance: continue
                    
                    # 阈值过滤：如果“避免交叉”离得太远（比如几 GHz），那就不算是有趣的特征
                    # 这里设一个宽松的阈值，比如 500 MHz (0.5e9 * h)
                    # 只有靠得比较近的才标记
                    if gap[idx] > (0.5e9 * h_planck): 
                        continue

                    # 记录位置 (直接取网格点，不再插值，因为极值点插值比较复杂且通常没必要)
                    b_feat = B_array[idx]
                    e_feat = (eigenvals[idx, i] + eigenvals[idx, j]) / 2.0 # 取两态中间位置
                    
                    features.append((b_feat, e_feat, i, j, 'avoided'))

    return features
def test_atomic_spectrum():
    print("=== 测试 1: 原子能级 (Breit-Rabi) 带 F1, F2 标识 ===")
    
    # 1. 初始化 (K40Rb87)
    try:
        sys = AlkaliCollisionSystem.from_preset("K40Rb87")
    except KeyError:
        sys = AlkaliCollisionSystem(name="Test", Ii=(1.5, 3.5), Ahf=(1e9*h_planck, 2e9*h_planck))
    
    basis = hb.CollisionBasis(sys, L_list=[0])
    
    # 2. 磁场扫描
    B_gauss = np.linspace(0.1, 1000, 400) 
    B_tesla = B_gauss * B_Gauss_to_Tesla
    
    hamiltonians = []
    for B in B_tesla:
        H_atom = hb.build_atomic_hamiltonian(basis, B_field_z=B)
        hamiltonians.append(H_atom)
    hamiltonians = np.array(hamiltonians)
    
    # 3. 求解
    print("正在对角化...")
    eigvals, eigvecs = calc.solve_system(hamiltonians)
    
    # 4. 计算属性 (包含 F1, F2)
    print("正在计算属性...")
    props = calc.calculate_properties(basis, eigvals, eigvecs, B_tesla)
    all_crossings = find_level_crossings(B_tesla, eigvals)

    # 5. 筛选条件
    target_mf = None  # 修改为你想要的 M_tot
    
    print("正在绘图...")
    plt.figure(figsize=(14, 10)) # 宽度加大，给右侧标签留空间
    
    E_plot = (eigvals - eigvals[0, 0]) * E_to_GHz
    
    # 用于记录y轴位置，防止标签重叠 (简单的防重叠逻辑)
    label_y_positions = [] 

    for i in range(basis.dim_total):
        this_state_m = props['m_tot'][0, i]
        
        # 过滤
        if target_mf is not None:
            if this_state_m != target_mf:
                continue 

        # --- 获取低场 (index=0) 的量子数用于标识 ---
        # 注意：K40 (Atom 1), Rb87 (Atom 2)
        f1 = props['f1'][0, i]
        mf1 = props['mf1'][0, i]
        f2 = props['f2'][0, i]
        mf2 = props['mf2'][0, i]
        
        # 构造标签字符串: |F1, mF1; F2, mF2>
        label_str = f"$|{f1}, {mf1}; {f2}, {mf2}\\rangle$"
        
# 获取单重态成分 (0 到 1 之间)
        singlet_frac = props['singlet_fraction'][:, i]
        s_val = np.mean(singlet_frac) # 取平均值作为该线条的颜色依据
        
        # ================= 修改颜色逻辑 =================
        
        # 方案：手动线性插值 (Blue -> Purple -> Red)
        # s_val = 1.0 -> 纯红 (1, 0, 0)
        # s_val = 0.0 -> 纯蓝 (0, 0, 1)
        # s_val = 0.5 -> 紫色 (0.5, 0, 0.5)
        
        # 定义端点颜色 (R, G, B)
        c_red  = np.array([0.8, 0.0, 0.0]) # 深红 (比纯红柔和一点)
        c_blue = np.array([0.0, 0.0, 0.8]) # 深蓝
        
        # 计算混合色
        line_color = s_val * c_red + (1.0 - s_val) * c_blue
        
        line, = plt.plot(B_gauss, E_plot[:, i], 
                         color=line_color,  # 使用计算出的颜色
                         linewidth=1.8,     #稍微加粗一点
                         alpha=0.9)         # 不透明度设高一点，防止变淡
        # --- 在最右侧添加文本标注 ---
        x_end = B_gauss[-1]
        y_end = E_plot[-1, i]
        
        # 简单的文本绘制
        plt.text(x_end + 10, y_end, label_str, 
                 fontsize=9, color=line_color, verticalalignment='center')
        
    # 绘制交叉点 (逻辑保持之前的一致)
    if all_crossings:
        xs, ys = [], []
        for b_t, e_j, idx_i, idx_j in all_crossings:
            m_i = props['m_tot'][0, idx_i]
            m_j = props['m_tot'][0, idx_j]
            # 严格过滤: 只显示可见线的交叉
            if target_mf is not None:
                 if m_i != target_mf or m_j != target_mf:
                    continue
            xs.append(b_t / B_Gauss_to_Tesla)
            ys.append((e_j - eigvals[0, 0]) * E_to_GHz)
        
        if xs:
            xs = np.array(xs)
            ys = np.array(ys)
            mask = (xs >= np.min(B_gauss)) & (xs <= np.max(B_gauss))
            plt.scatter(xs[mask], ys[mask], color='black', marker='x', s=60, zorder=10)

    title_str = f"{sys.name} Levels (L=0)"
    if target_mf is not None:
        title_str += f", $M_{{tot}} = {target_mf}$"
        
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Energy (GHz relative to ground)")
    plt.title(title_str)
    plt.grid(True, alpha=0.3)
    
    # 调整布局，给右侧标签留出空白
    plt.subplots_adjust(right=0.8) 
    
    filename = f"plot_breit_rabi_labeled_M{target_mf}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ 图片已保存: {filename}")
def test_dipolar_coupling():
    print("\n=== 测试 2: 偶极耦合矩阵元检查 ===")
    
    sys = AlkaliCollisionSystem.from_preset("Rb87Cs133")
    basis = hb.CollisionBasis(sys, L_list=[0, 2])
    print(f"基矢总维数 (L=0, 2): {basis.dim_total}")
    
    R_dist = 50 * const.physical_constants["Bohr radius"][0]
    H_dip = hb.build_dipolar_coupling(basis, R_dist)
    
    start_0, end_0 = basis.channel_indices[0]
    start_2, end_2 = basis.channel_indices[2]
    
    block_sd = H_dip[start_2:end_2, start_0:end_0]
    
    max_coupling = np.max(np.abs(block_sd))
    avg_coupling = np.mean(np.abs(block_sd))
    
    print(f"距离 R = 50 a0 处的偶极耦合:")
    print(f"s-d 耦合块最大矩阵元: {max_coupling:.4e} J ({max_coupling * E_to_MHz:.4f} MHz)")
    
    if max_coupling > 0:
        print("✅ 成功: 检测到耦合项。")
        
        # 保存矩阵稀疏图
        plt.figure(figsize=(8, 8))
        plt.spy(H_dip, markersize=1)
        plt.title("Hamiltonian Sparsity Pattern (Dipolar)")
        
        filename = "plot_sparsity.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✅ 矩阵结构图已保存为: {filename}")
    else:
        print("❌ 失败: s-d 耦合块全为零。")    
if __name__ == "__main__":
    test_atomic_spectrum()