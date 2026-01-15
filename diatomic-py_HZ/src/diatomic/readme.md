这是一份为您定制的 `README.md` 文档，涵盖了我们目前构建的整套双碱金属原子碰撞计算程序。
---

# Diatomic-Py: 双碱金属原子碰撞哈密顿量求解器

## 简介

这是一个用于计算双碱金属原子（异核或同核）在磁场中的能级结构（Zeeman Effect）、超精细结构（Hyperfine Structure）以及相互作用（Dipolar Coupling）的 Python 程序包。

该程序采用**未耦合基矢 (Uncoupled Basis)**  构建哈密顿量，能够处理：

1. **Breit-Rabi 图**：计算原子对在磁场下的能级分裂。
2. **同核对称性**：自动处理全同粒子（如 ）的波函数对称化。
3. **能级交叉分析**：自动识别真实交叉（Real Crossing）和避免交叉（Avoided Crossing/Feshbach Resonance）。
4. **物理属性分析**：计算本征态的单重态/三重态成分、磁矩、以及低场量子数标记 。

## 依赖库

* `numpy`: 矩阵运算
* `scipy`: 物理常数与线性代数求解
* `matplotlib`: 绘图
* `sympy`: 计算 Clebsch-Gordan 系数 (用于同核基矢变换)

## 模块说明

### 1. `systems.py` (物理系统定义)

定义了 `AlkaliCollisionSystem` 类，用于存储物理常数。

* **功能**：存储质量、核自旋 、超精细常数 、g因子、统计性质（Boson/Fermion）。
* **预设 (Presets)**：内置了 `Rb87Cs133`, `K40Rb87`, `Rb87Rb87` 等常用体系参数。
* **用法**：
```python
sys = AlkaliCollisionSystem.from_preset("Rb87Rb87")
# 系统会自动识别它是 Boson，需要对称化

```



### 2. `hamiltonian_builder.py` (核心构建器)

这是构建哈密顿矩阵的核心模块。

* **`CollisionBasis` 类**：管理基矢空间，处理自旋与轨道角动量  的直积与索引。
* **`build_atomic_hamiltonian(basis, B)`**：构建原子内部相互作用（超精细 + 塞曼）。
* **`build_dipolar_coupling(basis, R)`**：构建磁偶极-磁偶极相互作用（连接  波和  波）。
* **`build_total_projection_matrix(basis)`**：(针对同核) 构建投影矩阵 ，用于将哈密顿量投影到对称/反对称子空间。

### 3. `calculate.py` (求解与分析)

负责数值求解和后处理。

* **`solve_system(H)`**：对角化哈密顿量矩阵。包含 `sort_smooth` 算法，确保能级随磁场变化时连接平滑（绝热追踪）。
* **`calculate_properties(...)`**：计算本征态的物理属性：
* `singlet_fraction`: 单重态成分（用于区分颜色）。
* `m_tot`: 总投影量子数（好量子数）。
* `f1, mf1, f2, mf2`: 将高场本征态投影回零场，识别其原子物理起源。



### 4. `test.py` / `main_test.py` (主程序脚本)

执行计算流程的脚本。

* **流程**：初始化系统 -> 构建基矢 -> 扫描磁场 -> 对角化 -> 分析交叉 -> 绘图。
* **`find_all_features`**：自动查找能级交叉点（x）和最近邻点（o）。
* **绘图**：自动生成分  的能级图，并在右侧标记 。

---

## 快速上手

### 计算异核原子 (如 K40-Rb87)

```python
from system import AlkaliCollisionSystem
import hamiltonian_builder as hb
import calculate as calc

# 1. 加载系统
sys = AlkaliCollisionSystem.from_preset("K40Rb87")
# 2. 定义基矢 (仅 s 波)
basis = hb.CollisionBasis(sys, L_list=[0])
# 3. 构建矩阵 (B_field 是磁场数组)
H_list = [hb.build_atomic_hamiltonian(basis, b) for b in B_field]
# 4. 求解
eigvals, eigvecs = calc.solve_system(np.array(H_list))

```

### 计算同核原子 (如 Rb87-Rb87)

```python
# 1. 加载系统 (会自动识别 statistics='boson')
sys = AlkaliCollisionSystem.from_preset("Rb87Rb87")
basis = hb.CollisionBasis(sys, L_list=[0])

# 2. 获取对称化投影矩阵 P
P = hb.build_total_projection_matrix(basis)

# 3. 构建并投影
H_full = hb.build_atomic_hamiltonian(basis, b)
H_sym = P.T @ H_full @ P  # <--- 投影到对称子空间

# 4. 对角化小矩阵 H_sym
eigvals, eigvecs_sym = calc.solve_system(np.array([H_sym]))

```

---

## 关于 Python 中的 `@` 符号

在你看到的这一行代码中：

```python
H_sym = P_proj.T @ H_full @ P_proj

```

**`@` 是 Python 中的矩阵乘法运算符 (Matrix Multiplication Operator)。**


###  为什么要用它？
在同核原子计算中，`P` 是一个长方形矩阵（例如 ）。

* `P.T` 是 。
* `H_full` 是 。
* `P` 是 。

`P.T @ H_full @ P` 执行了两次矩阵乘法，最终得到一个  的小矩阵，这个矩阵就是剔除了反对称态、仅保留物理允许态的**对称化哈密顿量**。