import scipy.constants
from diatomic.operators import HalfInt
import numpy as np

# 物理常数
pi = scipy.constants.pi
h = scipy.constants.h
hbar = scipy.constants.hbar
muB = scipy.constants.physical_constants["Bohr magneton"][0]
muN = scipy.constants.physical_constants["nuclear magneton"][0]
mu0 = scipy.constants.mu_0
u_mass = scipy.constants.atomic_mass # 原子质量单位 kg

class AlkaliCollisionSystem:
    """
    描述双碱金属原子碰撞系统的物理参数容器。
    包含原子核自旋、超精细结构常数、塞曼效应参数以及相互作用强度系数。
    """

    def __init__(
        self,
        name: str = "Unknown",
        # --- 质量 (用于计算转动/离心势能) ---
        masses: tuple[float, float] = (0.0, 0.0), # (m1, m2) in kg
        
        # --- 原子核自旋 I1, I2 ---
        Ii: tuple[float, float] = (1.5, 1.5), 
        
        # --- 核 g 因子 (gI) ---
        # 定义: mu_nuc = gI * muN * I (注意符号，有时文献定义带负号)
        gI: tuple[float, float] = (0.0, 0.0),
        
        # --- 电子 g 因子 (gS) ---
        gS: float = 2.00231930436256,
        
        # --- 超精细结构常数 A_hf (单位: Joule) ---
        # H_hf = A * I \cdot S
        Ahf: tuple[float, float] = (0.0, 0.0),
        
        # --- 交换相互作用 (Exchange) 参数 ---
        # 用于描述单重态和三重态势能曲线的差异
        # 在简单模型中，这可能是散射长度(a_s, a_t)或者C6/C8/C12系数
        # 这里暂时留空，视具体势能函数需求而定
        potential_params: dict = {},
        statistics: str = 'distinguishable',
        # --- 二阶自旋轨道耦合强度 (SO2) ---
        # 这是一个唯象参数，通常表示为 H_SO ~ lambda(R) * Tensor_Operator
        # 这里可以存储前设常数
        so2_params: dict = {}
    ):
        self.name = name
        self.masses = masses
        self.mu = (masses[0] * masses[1]) / (masses[0] + masses[1]) if sum(masses) > 0 else 0.0
        
        self.Ii = Ii
        self.gI = gI
        self.gS = gS
        self.Ahf = Ahf
        self.potential_params = potential_params
        self.so2_params = so2_params
        self.statistics = statistics
# 判断是否为同核
        self.is_homonuclear = (statistics in ['boson', 'fermion'])
    @property
    def dipole_prefactor(self):
        """
        磁偶极-磁偶极相互作用的前置系数 (C_dd)。
        V_dd = C_dd / R^3 * Tensor_Op
        C_dd = (mu0 * (gS * muB)^2) / (4 * pi)
        注意：这里假设主要贡献来自电子自旋-电子自旋。
        如果是异核，应该是 gS1 * gS2。通常 gS1 approx gS2 approx 2。
        """
        # C_dd 因子 (J * m^3)
        return (mu0 * (self.gS * muB)**2) / (4 * pi)

    @classmethod
    def from_preset(cls, str_name):
        if str_name in cls.presets:
            data = cls.presets[str_name]
            # 计算真实质量 (amu -> kg)
            masses = (data["mass_amu"][0] * u_mass, data["mass_amu"][1] * u_mass)
            
            return cls(
                name=str_name,
                masses=masses,
                Ii=data["Ii"],
                gI=data["gI"],
                Ahf=data["Ahf"],
                
                # --- 必须添加这一行 ---
                # 从字典中读取 statistics，如果字典里没有，默认设为 distinguishable
                statistics=data.get("statistics", "distinguishable"), 
                # -------------------
                
                so2_params=data.get("so2_params", {})
            )
        else:
            raise KeyError(f"Preset {str_name} not found.")
    presets = {
        "Rb87Cs133": {
            # 数据来源参考: Gregory et al. (2021), Arndt (1997), Steck
            "mass_amu": (86.90918, 132.90545),
            "Ii": (1.5, 3.5), # 3/2, 7/2
            
            # 超精细常数 (Joule) = h * Frequency
            "Ahf": (
                3.4173413e9 * h, # Rb87 Ground State
                9.192631770e9 * h # Cs133 Ground State
            ),
            
            # 核g因子 (无量纲)
            # mu_Rb87 = 2.751 muN, I=1.5 -> gI = 2.751/1.5 = 1.834 (需核对Steck定义)
            # Steck Rb87: gI = -0.000995 muB/muN ??? 
            # 通常最好直接用 gI_factor * muN * B * I_z
            # 这里填入文献常用值，注意符号约定
            "gI": (0.000995, 0.000398), # 示例值，请务必核对正负号！
            
            "so2_params": {
                "magnitude": 0.0 # 需要填入具体文献中的二阶SO耦合常数
            }
        },
        "Na23K40": {
             "mass_amu": (22.989, 39.964),
             "Ii": (1.5, 4.0),
             "Ahf": (885.813e6 * h, -1.24358e9 * h), # K40是负的
             "gI": (0.0, 0.0) # 需查表
        },
        "K40Rb87": {
            # 质量 (amu)
            "mass_amu": (39.96399848, 86.909180527), 
            
            # 核自旋 I: K40 (I=4), Rb87 (I=3/2)
            "Ii": (4.0, 1.5), 
            
            # --- 关键修正: 超精细系数 A (Coefficients) ---
            # A_coeff = Splitting / (I + 0.5)
            # K40 Splitting ~ -1285.79 MHz -> A ~ -285.73 MHz (与Fortran一致)
            # Rb87 Splitting ~ 6834.68 MHz -> A ~ 3417.34 MHz (与Fortran一致)
            "Ahf": (
                -285.7308e6 * h,  # K40 (注意 Fortran 是 285730800)
                 3.4173413e9 * h   # Rb87 (注意 Fortran 是 3417341306)
            ),
            
            # --- 关键修正: 核 g 因子 (以 muN 为单位) ---
            # 转换公式: g(Python) = - g(Fortran) * (muB / muN)
            # 或者是直接查询标准表值
            # Fortran gK = 0.00017649 (muB单位) -> Python gK ~ -0.324 (muN单位)
            # Fortran gRb = -0.000995 (muB单位) -> Python gRb ~ 1.83 (muN单位)
            "gI": (
                -0.32413, # K40 (对应 Fortran 的 0.000176 * 1836, 且符号反转)
                 1.82723  # Rb87 (对应 Fortran 的 0.000995 * 1836, 且符号反转)
            ),
            
            # 精确电子 g 因子 (从 Fortran 移植)
            # 用于覆盖默认的 gS=2.0023...
            "gS_individual": (
                2.0022942,      # geK
                2.0023193043737 # geRb
            ),

            # 二阶自旋轨道耦合等 (如有需要)
            "so2_params": {}
        
        },
        "Rb87Rb87": {
            # 质量 (amu)
            "mass_amu": (86.909180527, 86.909180527), 
            
            # 核自旋 I: K40 (I=4), Rb87 (I=3/2)
            "Ii": (1.5, 1.5), 
            
            # --- 关键修正: 超精细系数 A (Coefficients) ---
            # A_coeff = Splitting / (I + 0.5)
            # K40 Splitting ~ -1285.79 MHz -> A ~ -285.73 MHz (与Fortran一致)
            # Rb87 Splitting ~ 6834.68 MHz -> A ~ 3417.34 MHz (与Fortran一致)
            "Ahf": (
                 3.4173413e9 * h,  # K40 (注意 Fortran 是 285730800)
                 3.4173413e9 * h   # Rb87 (注意 Fortran 是 3417341306)
            ),
            
            # --- 关键修正: 核 g 因子 (以 muN 为单位) ---
            # 转换公式: g(Python) = - g(Fortran) * (muB / muN)
            # 或者是直接查询标准表值
            # Fortran gK = 0.00017649 (muB单位) -> Python gK ~ -0.324 (muN单位)
            # Fortran gRb = -0.000995 (muB单位) -> Python gRb ~ 1.83 (muN单位)
            "gI": (
                 1.82723, # K40 (对应 Fortran 的 0.000176 * 1836, 且符号反转)
                 1.82723  # Rb87 (对应 Fortran 的 0.000995 * 1836, 且符号反转)
            ),
            
            # 精确电子 g 因子 (从 Fortran 移植)
            # 用于覆盖默认的 gS=2.0023...
            "gS_individual": (
                2.0023193043737,      # geK
                2.0023193043737 # geRb
            ),
            "statistics": "boson" ,# Rb87 是玻色子
            # 二阶自旋轨道耦合等 (如有需要)
            "so2_params": {}
        
        }
    }