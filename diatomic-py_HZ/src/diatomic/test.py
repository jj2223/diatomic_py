import matplotlib
# 设置后端为 Agg，适用于无屏幕的服务器环境
matplotlib.use('Agg') 

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from matplotlib.transforms import blended_transform_factory

# 导入你的核心模块
from systems import AlkaliCollisionSystem
import hamiltonian_builder as hb
import calculate as calc

# --- 物理常数与单位转换 ---
h_planck = const.h
E_to_MHz = 1e-6 / h_planck  # Joule -> MHz
E_to_GHz = 1e-9 / h_planck  # Joule -> GHz
B_Gauss_to_Tesla = 1e-4     # Gauss -> Tesla

# ==========================================
# 核心算法 1: 特征查找 (交叉 & 避免交叉)
# ==========================================
def find_all_features(B_array, eigenvals, props, tolerance=1e-6):
    """
    同时查找真实交叉(Real Crossings)和避免交叉(Avoided Crossings)。
    
    Args:
        B_array: 磁场数组 (Tesla)
        eigenvals: 平滑后的能级 (Joule)
        props: 属性字典 (用于获取 M_tot)
        tolerance: 忽略极低场 (Tesla)
        
    Returns:
        list: [(B, E, i, j, type_str), ...]
    """
    n_steps, n_states = eigenvals.shape
    features = []
    
    m_tot = props['m_tot'][0, :] # 获取所有态的 M (好量子数)

    # 遍历每一对能级
    for i in range(n_states):
        for j in range(i + 1, n_states):
            
            m_i = m_tot[i]
            m_j = m_tot[j]
            delta_E = eigenvals[:, i] - eigenvals[:, j]
            
            # --- 类型 A: 不同 M_tot -> 寻找真实交叉 (变号点) ---
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
                    
                    # 对应的能量 (对态 i 进行插值)
                    e_slope = (eigenvals[idx+1, i] - eigenvals[idx, i]) / (x2 - x1)
                    e_feat = eigenvals[idx, i] + e_slope * (b_feat - x1)
                    
                    features.append((b_feat, e_feat, i, j, 'real'))
            
            # --- 类型 B: 相同 M_tot -> 寻找避免交叉 (能隙极小值) ---
            else:
                gap = np.abs(delta_E)
                # 寻找局部极小值: gap[k] < gap[k-1] 且 gap[k] < gap[k+1]
                local_min_indices = np.where((gap[1:-1] < gap[0:-2]) & (gap[1:-1] < gap[2:]))[0] + 1
                
                for idx in local_min_indices:
                    x_pos = B_array[idx]
                    if x_pos < tolerance: continue
                    
                    # 阈值过滤：如果能隙太大 (比如 > 500 MHz)，通常不是我们要找的 Feshbach 相关特征
                    if gap[idx] > (0.5e9 * h_planck): 
                        continue

                    b_feat = B_array[idx]
                    e_feat = (eigenvals[idx, i] + eigenvals[idx, j]) / 2.0 # 取中间位置
                    
                    features.append((b_feat, e_feat, i, j, 'avoided'))

    return features

# ==========================================
# 核心算法 2: 绘图逻辑
# ==========================================
def plot_spectrum_for_mf(sys, basis, B_gauss, E_plot, props, target_mf, features, output_filename):
    """
    绘制特定 M_tot 的能级图，并标出相关的交叉(x)和避免交叉(o)
    """
    fig, ax = plt.subplots(figsize=(14, 9)) # 宽图，给标签留空间
    
    # --- 1. 绘制能级线 ---
    lines_drawn = 0
    for i in range(basis.dim_total):
        # 筛选 M_tot
        if props['m_tot'][0, i] != target_mf:
            continue
        lines_drawn += 1
            
        # 颜色计算: Red(Singlet) -> Purple(Mixed) -> Blue(Triplet)
        s_val = np.mean(props['singlet_fraction'][:, i])
        c_red = np.array([0.8, 0.0, 0.0])
        c_blue = np.array([0.0, 0.0, 0.8])
        line_color = s_val * c_red + (1.0 - s_val) * c_blue
        
        # 画线
        ax.plot(B_gauss, E_plot[:, i], color=line_color, linewidth=1.8, alpha=0.9)
        
        # 标签: |F1, mF1; F2, mF2>
        f1 = props['f1'][0, i]; mf1 = props['mf1'][0, i]
        f2 = props['f2'][0, i]; mf2 = props['mf2'][0, i]
        label_str = f"$|{f1}, {mf1}; {f2}, {mf2}\\rangle$"
        
        # 在最右侧标注
        ax.text(B_gauss[-1] + 5, E_plot[-1, i], label_str, 
                fontsize=9, color=line_color, verticalalignment='center')

    # --- 2. 标记特征点 (交叉 & 避免交叉) ---
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    b_min, b_max = np.min(B_gauss), np.max(B_gauss)
    
    # 筛选相关特征点
    relevant_features = []
    for feat in features:
        _, _, idx_i, idx_j, _ = feat
        m_i = props['m_tot'][0, idx_i]
        m_j = props['m_tot'][0, idx_j]
        # 只要有一个态属于当前 target_mf，就显示这个特征
        if m_i == target_mf or m_j == target_mf:
            relevant_features.append(feat)

    # 绘制特征
    for b_tesla, e_joule, _, _, f_type in relevant_features:
        b_g = b_tesla / B_Gauss_to_Tesla
        
        # 过滤可视范围
        if not (b_min <= b_g <= b_max): continue

        # 转换能量 (注意 E_plot 是相对能量，这里要做相应转换)
        # E_plot = (E_raw - E_ground) * scale
        e_ghz = (e_joule - props['ground_energy_val']) * E_to_GHz
        
        # 样式定义
        if f_type == 'real':
            # 真实交叉: 黑色虚线, X
            color = 'black'
            marker = 'x'
            ls = '--'
            lw = 0.8
        else: # avoided
            # 避免交叉: 红色点线, 空心圆
            color = 'red'
            marker = 'o'
            ls = ':'
            lw = 1.0

        # 画垂直参考线
        ax.axvline(x=b_g, color=color, linestyle=ls, linewidth=lw, alpha=0.6)
        
        # 画点
        if f_type == 'avoided':
            ax.scatter([b_g], [e_ghz], facecolors='none', edgecolors=color, marker=marker, s=60, zorder=10)
        else:
            ax.scatter([b_g], [e_ghz], color=color, marker=marker, s=40, zorder=10)
            
        # 在 X 轴下方标注数值
        ax.text(b_g, -0.02, f"{b_g:.1f}", transform=trans, 
                rotation=90, verticalalignment='top', horizontalalignment='center',
                fontsize=8, color=color)

    # --- 3. 图表设置 ---
    ax.set_xlabel("Magnetic Field (Gauss)")
    ax.set_ylabel("Energy (GHz relative to ground)")
    ax.set_title(f"{sys.name} Spectrum, $M_{{tot}}={target_mf}$ (x=Real, o=Avoided)")
    ax.grid(True, alpha=0.3)
    
    # 调整布局 (右侧留白给标签，底部留白给坐标字)
    plt.subplots_adjust(right=0.82, bottom=0.15)
    
    plt.savefig(output_filename, dpi=200)
    plt.close(fig)
    print(f"   -> 已保存: {output_filename} (含 {lines_drawn} 条能级)")

def plot_master_spectrum(sys, B_gauss, E_plot, props, features, output_filename):
    """
    绘制总览图，包含所有能级和交叉参考线
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    n_states = E_plot.shape[1]
    
    # 画所有线 (细淡)
    for i in range(n_states):
        s_val = np.mean(props['singlet_fraction'][:, i])
        c_red = np.array([0.8, 0.0, 0.0]); c_blue = np.array([0.0, 0.0, 0.8])
        line_color = s_val * c_red + (1.0 - s_val) * c_blue
        ax.plot(B_gauss, E_plot[:, i], color=line_color, linewidth=0.8, alpha=0.5)
        
    # 标出所有特征参考线
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    b_min, b_max = np.min(B_gauss), np.max(B_gauss)
    
    for b_tesla, _, _, _, f_type in features:
        b_g = b_tesla / B_Gauss_to_Tesla
        if not (b_min <= b_g <= b_max): continue
        
        color = 'red' if f_type == 'avoided' else 'black'
        ls = ':' if f_type == 'avoided' else '--'
        
        ax.axvline(x=b_g, color=color, linestyle=ls, linewidth=0.5, alpha=0.4)
        ax.text(b_g, -0.02, f"{b_g:.1f}", transform=trans, 
                rotation=90, verticalalignment='top', horizontalalignment='center',
                fontsize=6, color=color)

    ax.set_xlabel("Magnetic Field (Gauss)")
    ax.set_ylabel("Energy (GHz)")
    ax.set_title(f"{sys.name} Master Spectrum (All Levels)")
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)
    print(f"   -> 已保存总览图: {output_filename}")

# ==========================================
# 主流程
# ==========================================
def run_full_analysis():
    print("=== 开始全谱计算与自动化绘图流程 ===")
    
    # 1. 初始化系统
    try:
        sys = AlkaliCollisionSystem.from_preset("Rb87Rb87")
    except KeyError:
        sys = AlkaliCollisionSystem(name="Test", Ii=(1.5, 3.5), Ahf=(1e9*h_planck, 2e9*h_planck))
    
    # 只包含 L=0 (纯原子谱)
    basis = hb.CollisionBasis(sys, L_list=[0])
    
    # 2. 磁场设置 (高分辨率以捕获 Avoided Crossing)
    B_gauss = np.linspace(0.1, 1200, 600) 
    B_tesla = B_gauss * B_Gauss_to_Tesla
    
    # 3. 构建矩阵 (一次性)
    print(f"正在构建 {len(B_tesla)} 个哈密顿量矩阵...")
    hamiltonians = []
    for B in B_tesla:
        # 仅原子部分，无偶极耦合
        hamiltonians.append(hb.build_atomic_hamiltonian(basis, B_field_z=B))
    hamiltonians = np.array(hamiltonians)
    
    # 4. 对角化 (最耗时步骤)
    print("正在对角化 (Solve System)...")
    eigvals, eigvecs = calc.solve_system(hamiltonians)
    
    # 5. 计算属性
    print("正在计算量子数与属性...")
    props = calc.calculate_properties(basis, eigvals, eigvecs, B_tesla)
    
    # 记录基态能量用于归一化
    props['ground_energy_val'] = eigvals[0, 0]
    E_plot = (eigvals - eigvals[0, 0]) * E_to_GHz
    
    # 6. 查找所有特征 (Real & Avoided)
    print("正在分析能级特征 (Crossings & Avoided)...")
    features = find_all_features(B_tesla, eigvals, props)
    print(f"共发现 {len(features)} 个特征点。")
    
    # 7. 绘图循环
    # 获取所有唯一的 M_tot
    unique_mfs = np.unique(props['m_tot'][0, :])
    # 排序 (转成float排序)
    unique_mfs = sorted(unique_mfs, key=lambda x: float(x))
    
    print(f"\n检测到 {len(unique_mfs)} 个不同的 M_tot 子空间。开始绘图...")
    
    # A. 分别绘制每个 M_tot
    for mf in unique_mfs:
        mf_val = float(mf)
        fname = f"spectrum_M{mf_val:+.1f}.png"
        plot_spectrum_for_mf(sys, basis, B_gauss, E_plot, props, mf, features, fname)
        
    # B. 绘制总览图
    print("\n正在绘制总览图...")
    plot_master_spectrum(sys, B_gauss, E_plot, props, features, "spectrum_MASTER.png")
    
    print("\n=== 所有任务完成 ===")

if __name__ == "__main__":
    run_full_analysis()