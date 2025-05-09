import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


##############################################################################
# 1. 定义踩踏模式分布函数（可根据需要增加或修改）
##############################################################################

def mode_single_person(nx, ny, x_center, y_center, sigma_x, sigma_y, delta_y=0.0):
    """
    模式 1：单人行走(上楼/下楼合并)，脚步分布集中在台阶中部，
    并可通过 delta_y 调整前后缘的偏移量。

    参数：
    nx, ny     : 网格在 x、y 方向的离散点数
    x_center   : x方向中心
    y_center   : y方向中心
    sigma_x,y  : 高斯分布的标准差
    delta_y    : y方向偏移量，>0 表示更靠前缘，<0 表示更靠后缘

    返回：
    一个形状为 (nx, ny) 的二维数组，表示该模式在每个网格点上的“单位踩踏”分布。
    """
    x_idx = np.arange(nx)
    y_idx = np.arange(ny)
    xx, yy = np.meshgrid(x_idx, y_idx, indexing='ij')

    # 高斯分布核心
    dist_x = (xx - x_center) ** 2 / (2 * sigma_x ** 2)
    dist_y = (yy - (y_center + delta_y)) ** 2 / (2 * sigma_y ** 2)
    gauss = np.exp(-(dist_x + dist_y))

    # 为方便后续累加，可不做严格归一化，也可根据需要进行归一化
    return gauss


def mode_two_persons(nx, ny, x_left, x_right, y_center, sigma_x, sigma_y):
    """
    模式 2：双人并排行走，左右两侧对称高斯分布。

    参数：
    nx, ny       : 网格在 x、y 方向的离散点数
    x_left,right : 左、右人脚步的 x 方向中心
    y_center     : y 方向中心
    sigma_x,y    : 高斯分布的标准差

    返回：
    shape=(nx, ny) 的 2D 数组，表示双人并行的单位踩踏分布（双峰）。
    """
    x_idx = np.arange(nx)
    y_idx = np.arange(ny)
    xx, yy = np.meshgrid(x_idx, y_idx, indexing='ij')

    dist_left = np.exp(-(((xx - x_left) ** 2) / (2 * sigma_x ** 2) + (yy - y_center) ** 2 / (2 * sigma_y ** 2)))
    dist_right = np.exp(-(((xx - x_right) ** 2) / (2 * sigma_x ** 2) + (yy - y_center) ** 2 / (2 * sigma_y ** 2)))

    return dist_left + dist_right


##############################################################################
# 2. 构建多级台阶、多时间段的磨损模拟与拟合主函数
##############################################################################

class StairWearModel:
    """
    多级台阶磨损模型类，支持：
    - S 级台阶
    - 每级台阶均有 (nx, ny) 网格
    - M 种踩踏模式(模式的分布函数由用户传入)
    - 时间分段 L 段 (每段可以设定不同人流量、模式分布比例、是否修缮等)
    """

    def __init__(self,
                 S=2,  # 台阶级数
                 nx=50, ny=50,  # 网格规模
                 alpha_s=None,  # 每级台阶的材料磨损系数列表, shape=(S,)
                 repair_matrix=None  # 修缮系数, 若不为 None 则 shape=(S, L, nx, ny)
                 ):
        """
        初始化楼梯模型。

        参数：
        S           : 台阶级数
        nx, ny      : 每级台阶网格离散数
        alpha_s     : 数组，长度为 S；若为 None，后续可通过设定或拟合来补充
        repair_matrix : 四维数组 [S, L, nx, ny]，表示在每段时间开始时，对台阶 s 的每个网格点 (i,j) 进行的修复系数 rho_{s,l}(i,j)
                        如果没有修缮需求，可在 simulate() 时传入 None 或全 1 矩阵。
        """
        self.S = S
        self.nx = nx
        self.ny = ny
        if alpha_s is None:
            # 默认给一个初始值
            self.alpha_s = np.ones(S) * 0.02
        else:
            self.alpha_s = np.array(alpha_s, dtype=float)

        self.repair_matrix = repair_matrix  # 在simulate()时可再具体应用

        # 存储仿真或观测得到的磨损数据
        # d_observed[s, i, j] 表示第 s 级台阶，第 (i,j) 网格处的测量/仿真 磨损深度
        self.d_observed = None

    def simulate(self,
                 usage_modes,  # 一个列表，包含 M 种分布模式(每种模式是 shape=(nx, ny) 的numpy数组)
                 usage_schedule,  # usage_schedule[l]: dict，包含该时间段的 (daily_flow, proportion=[p1, p2, ...], duration等)
                 if_store=True
                 ):
        """
        根据给定的踩踏模式、时间分段使用策略，模拟 S 级台阶在 [0,T] 期间的磨损结果。

        参数：
        usage_modes    : list of numpy.array,
                         [mode1_dist, mode2_dist, ... , modeM_dist],
                         每个 modeX_dist shape=(nx, ny)
        usage_schedule : list of dict，长度为 L（时间分段数）。
                         每个字典包含：
                           {
                             'daily_flow': 1000,    # 每天多少人
                             'duration':   365,     # 天数
                             'proportion': [0.7, 0.3],  # 各模式在此段出现的比例(与 usage_modes 顺序对应)
                           }
        if_store       : 是否将结果存到 self.d_observed 里（模拟结果可视为“真值”）

        返回：
        d_simulated, shape=(S, nx, ny)，表示最终模拟得到的磨损深度
        """
        L = len(usage_schedule)
        M = len(usage_modes)

        # 初始化磨损矩阵
        d_simulated = np.zeros((self.S, self.nx, self.ny), dtype=float)

        # 分段累积
        for l in range(L):
            seg_info = usage_schedule[l]
            daily_flow = seg_info['daily_flow']
            duration = seg_info['duration']  # 单位：天，也可改为年
            proportion = seg_info['proportion']  # 各模式的比例

            # 每段时间开始时，若有修缮，则对 d_simulated 做 “重置”
            if self.repair_matrix is not None:
                # repair_matrix[s, l, i, j] 表示第 l 段开始时对台阶 s 的 (i,j) 网格点的磨损保留比
                # 修缮后: d_new = repair_matrix * d_old
                for s in range(self.S):
                    rho_slij = self.repair_matrix[s, l, :, :]  # shape=(nx, ny)
                    d_simulated[s] = d_simulated[s] * rho_slij

            # 计算该段时间内，总踩踏次数 = daily_flow * duration
            N_segment = daily_flow * duration

            # 对每个台阶 s，计算本段的增量磨损
            for s in range(self.S):
                # alpha_s 表示材料磨损系数
                alpha = self.alpha_s[s]

                # 组合所有踩踏模式
                delta_w_total = np.zeros((self.nx, self.ny), dtype=float)
                for m in range(M):
                    # C_m = N_segment * proportion[m]
                    # 单次踩踏的分布是 usage_modes[m]
                    # 增量 = alpha * C_m * usage_modes[m]
                    cm = N_segment * proportion[m]
                    delta_w_total += alpha * cm * usage_modes[m]

                # 将本段增量累加到 d_simulated[s]
                d_simulated[s] += delta_w_total

        if if_store:
            self.d_observed = d_simulated.copy()

        return d_simulated

    def least_squares_fit(self,
                          usage_modes,
                          usage_schedule,
                          d_target=None,
                          x0=None
                          ):
        """
        针对已知(或假设)的 usage_modes, usage_schedule，利用最小二乘方法反演:
          - 各台阶的 alpha_s (材料系数)
          - 各段时间的人流量 (daily_flow_l) 或模式比例
        等参数。这里演示简化：只拟合 alpha_s 和 各段的 daily_flow，假设模式比例固定已知。

        参数：
        usage_modes    : 同 simulate()
        usage_schedule : 同 simulate()，但其中 daily_flow 等可能是“待拟合”或“有先验”
        d_target       : shape=(S, nx, ny)，观测(或模拟)到的磨损数据。若 None，则默认使用 self.d_observed
        x0            : 初始猜测的参数向量

        返回：
        res : 优化结果对象，包含最优参数及残差信息等
        """
        if d_target is None:
            if self.d_observed is None:
                raise ValueError("没有可用的观测数据 d_target，请先进行提供真实观测值。")
            d_target = self.d_observed

        # 假设需要拟合的参数有：
        #   p = [alpha_s(0), alpha_s(1), ..., alpha_s(S-1),
        #        daily_flow_0, daily_flow_1, ..., daily_flow_(L-1)]
        L = len(usage_schedule)
        M = len(usage_modes)

        # 构建初值 x0
        if x0 is None:
            x0_alphas = self.alpha_s  # 当前 self.alpha_s
            x0_flows = [seg['daily_flow'] for seg in usage_schedule]
            x0 = np.concatenate([x0_alphas, x0_flows]).astype(float)

        # 为了便于最小二乘，需要一个“误差函数” residual_func(par)
        def residual_func(par):
            # par前S个是 alpha_s，后面L个是 daily_flow
            alpha_vals = par[:self.S]
            flows_vals = par[self.S: self.S + L]

            # 其他不变参数：duration, proportion
            # 先构造新的 usage_schedule_temp
            usage_schedule_temp = []
            for l in range(L):
                seg_info = usage_schedule[l].copy()
                seg_info['daily_flow'] = flows_vals[l]
                usage_schedule_temp.append(seg_info)

            # 更新 self.alpha_s
            alpha_old = self.alpha_s.copy()
            self.alpha_s = alpha_vals

            # 在不更改修缮矩阵的前提下，做一次 simulate
            d_model = self.simulate(usage_modes, usage_schedule_temp, if_store=False)

            # 恢复 alpha_s
            self.alpha_s = alpha_old

            # 计算残差
            res_arr = (d_model - d_target).ravel()  # 拉平为一维
            return res_arr

        # 用scipy的 least_squares 进行拟合
        res = least_squares(residual_func, x0, method='trf', xtol=1e-10, ftol=1e-10, gtol=1e-10)

        # 拟合完的结果解析
        par_opt = res.x
        alpha_opt = par_opt[:self.S]
        flow_opt = par_opt[self.S: self.S + L]

        # 更新对象属性
        self.alpha_s = alpha_opt
        for l in range(L):
            usage_schedule[l]['daily_flow'] = flow_opt[l]

        return res

    def plot_wear_2D(self, d_data=None, step_index=0, title=""):
        """
        可视化某一级台阶的 2D 磨损深度分布（使用 imshow）。

        参数：
        d_data    : shape=(S, nx, ny) 的数据，比如模拟或观测值。
        step_index: 要查看的台阶编号
        title     : 图像标题
        """
        if d_data is None:
            d_data = self.d_observed
        if d_data is None:
            raise ValueError("没有可视化数据，请先模拟或加载观测数据。")

        plt.figure(figsize=(6, 5))
        plt.imshow(d_data[step_index], origin='lower', cmap='viridis')
        plt.colorbar(label="Wear depth (mm)")
        plt.title(f"{title} - Step #{step_index}")
        plt.xlabel("Y direction")
        plt.ylabel("X direction")
        plt.show()

    def print_summary(self):
        """
        打印当前模型的关键参数（如 alpha_s）和(若存在)观测/模拟数据的统计信息。
        """
        print("========= StairWearModel Summary =========")
        print(f"Number of Steps S = {self.S}")
        print(f"Grid size (nx, ny) = ({self.nx}, {self.ny})")
        print("Alpha_s (material wear coefficients) =", self.alpha_s)
        if self.d_observed is not None:
            dmin = self.d_observed.min()
            dmax = self.d_observed.max()
            dmean = self.d_observed.mean()
            print(f"Observed/Simulated wear depth: min={dmin:.4f}, max={dmax:.4f}, mean={dmean:.4f}")
        else:
            print("No observed/simulated wear data stored yet.")
        print("==========================================")

    def differential_evolution_fit(self, usage_modes, usage_schedule, d_target=None):
        """
        使用差分进化算法优化模型参数。

        参数：
        usage_modes    : 踩踏模式分布函数列表
        usage_schedule : 时间分段策略
        d_target       : shape=(S, nx, ny) 的观测磨损深度

        返回：
        result : 差分进化的优化结果对象
        """
        if d_target is None:
            if self.d_observed is None:
                raise ValueError("没有可用的观测数据 d_target，请先进行模拟或提供真实观测值。")
            d_target = self.d_observed

        L = len(usage_schedule)
        M = len(usage_modes)

        # 参数范围设置
        alpha_bounds = [(0.005, 0.05)] * self.S  # 每级台阶的材料系数范围
        flow_bounds = [(100, 5000)] * L  # 每段时间的 daily_flow 范围
        bounds = alpha_bounds + flow_bounds  # 总参数范围

        # 定义目标函数
        def objective_func(par):
            alpha_vals = par[:self.S]
            flows_vals = par[self.S: self.S + L]

            # 构建新的 usage_schedule
            usage_schedule_temp = []
            for l in range(L):
                seg_info = usage_schedule[l].copy()
                seg_info['daily_flow'] = flows_vals[l]
                usage_schedule_temp.append(seg_info)

            # 更新 alpha_s
            alpha_old = self.alpha_s.copy()
            self.alpha_s = alpha_vals

            # 模拟预测值
            d_model = self.simulate(usage_modes, usage_schedule_temp, if_store=False)

            # 恢复 alpha_s
            self.alpha_s = alpha_old

            # 返回残差平方和
            return np.sum((d_model - d_target) ** 2)

        # 使用差分进化算法进行优化
        result = differential_evolution(objective_func, bounds, strategy='best1bin',
                                        maxiter=1000, tol=1e-7, seed=42)

        # 更新模型参数
        par_opt = result.x
        self.alpha_s = par_opt[:self.S]
        for l in range(L):
            usage_schedule[l]['daily_flow'] = par_opt[self.S + l]

        return result


from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import differential_evolution



# 可视化为 3D 图形
def plot_3d_wear(data, title="Wear Depth 3D", step_index=0, cmap='viridis'):
    """
    绘制三维磨损分布图。

    参数：
    data        : 3D 磨损数据，形状为 (S, nx, ny)
    title       : 图标题
    step_index  : 选择显示的台阶编号
    cmap        : 颜色映射
    """
    nx, ny = data.shape[1], data.shape[2]
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    zz = data[step_index]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xx, yy, zz, cmap=cmap, edgecolor='k', linewidth=0.1, alpha=0.8)
    fig.colorbar(surf, ax=ax, label='Wear Depth (mm)')
    ax.set_title(f"{title} - Step #{step_index}")
    ax.set_xlabel("X Direction")
    ax.set_ylabel("Y Direction")
    ax.set_zlabel("Wear Depth (mm)")
    plt.show()
##############################################################################
# 3. 主程序示例：如何使用上面类和函数
##############################################################################

if __name__ == "__main__":
    # 一、设置模型规模及参数
    S = 2  # 假设有 2 级台阶
    nx = 60
    ny = 60

    # 材料磨损系数初值(这里示例假设两级台阶材料相似，但数值略有差异)
    alpha_init = [0.02, 0.03]

    # 二、构造或假设踩踏模式
    # 比如只使用两种模式：单人行走 + 并排行走
    # 分布函数都在 (nx, ny) 网格上
    mode1 = mode_single_person(nx, ny, x_center=nx / 2, y_center=ny / 2,
                               sigma_x=10, sigma_y=10, delta_y=5.0)  # 偏前缘
    mode2 = mode_two_persons(nx, ny, x_left=nx / 3, x_right=2 * nx / 3,
                             y_center=ny / 2, sigma_x=8, sigma_y=10)
    usage_modes = [mode1, mode2]

    # 三、定义时间分段 (L=2 段)
    # 段1: 每天1000人, 持续 200天, 单人行走占 80%, 并排占 20%
    # 段2: 每天2000人, 持续 100天, 单人行走占 50%, 并排占 50%
    usage_schedule = [
        {
            'daily_flow': 1000,
            'duration': 200,
            'proportion': [0.8, 0.2]
        },
        {
            'daily_flow': 2000,
            'duration': 100,
            'proportion': [0.5, 0.5]
        }
    ]
    L = len(usage_schedule)

    # 四、修缮矩阵 repair_matrix[s, l, i, j]
    # 假设只在第2段开始时对 第1级台阶(s=0) 有一次简单修缮，其余台阶或时段不修缮
    # 这里演示：第 l=1 段开始(即第二段开始)将第1级台阶保留50%磨损，第二级台阶不修
    repair_mat = np.ones((S, L, nx, ny), dtype=float)  # 全部初始=1
    # 只对 s=0, l=1 的网格做 0.5
    repair_mat[0, 1, :, :] = 0.5

    # 五、构建模型
    model = StairWearModel(S=S, nx=nx, ny=ny, alpha_s=alpha_init, repair_matrix=repair_mat)
    model.print_summary()

    # 六、进行模拟，得到“真值”磨损分布
    d_sim = model.simulate(usage_modes, usage_schedule, if_store=True)
    print("模拟后的磨损数据统计：")
    model.print_summary()  # 会打印 updated 的统计

    # 可视化某个台阶的磨损结果
    model.plot_wear_2D(d_data=d_sim, step_index=0, title="Simulated Wear (Step 0)")
    model.plot_wear_2D(d_data=d_sim, step_index=1, title="Simulated Wear (Step 1)")

    # 七、(可选)在这里，你可以“假装”我们只知道 d_sim，是测量观测数据，然后通过最小二乘来反演 alpha_s 和 daily_flow
    #     当然实际中可以加噪声、或者使用真实测量数据替换 d_sim
    # 先在 d_sim 基础上加一点模拟噪声
    noise_level = 0.01
    d_noisy = d_sim + noise_level * np.random.randn(*d_sim.shape) * d_sim.max()

    # 反演：假设对 usage_schedule 的 proportion 已经比较确定；我们只拟合 alpha_s 与 daily_flow。
    # 构造一个初值(与真实值偏差较大，以测试拟合)
    x0_alpha = [0.01, 0.04]  # 材料系数初猜
    x0_flow = [500, 3000]  # daily_flow 初猜
    x0 = np.array(x0_alpha + x0_flow)

    # 将 d_noisy 当作“观测数据”
    model.d_observed = d_noisy

    # 执行差分进化拟合
    res = model.differential_evolution_fit(usage_modes, usage_schedule, d_target=d_noisy)

    print("\n差分进化拟合完成：")
    print("优化结束标志：", res.message)
    print("拟合后参数：", res.x)
    print("最终残差范数：", np.linalg.norm(res.fun))

    # 拟合后的模型参数
    model.print_summary()

    # 八、使用拟合后的参数再模拟一下
    d_fit = model.simulate(usage_modes, usage_schedule, if_store=False)

    # 对比观测和拟合
    residual = d_fit - d_noisy
    mse = np.mean(residual ** 2)
    print(f"拟合后 MSE = {mse:.6f}")

    # 可视化 拟合结果 与 观测数据 的对比（仅演示 step=0）
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.title("Observed (Noisy) Wear")
    plt.imshow(d_noisy[0], origin='lower', cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Fitted Wear")
    plt.imshow(d_fit[0], origin='lower', cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Residual (Fitted - Observed)")
    plt.imshow(residual[0], origin='lower', cmap='bwr')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    # 三维可视化观测、拟合和残差数据
    plot_3d_wear(d_noisy, title="Observed (Noisy) Wear 3D", step_index=0, cmap='viridis')
    plot_3d_wear(d_fit, title="Fitted Wear 3D", step_index=0, cmap='viridis')
    plot_3d_wear(residual, title="Residual (Fitted - Observed) 3D", step_index=0, cmap='bwr')
    print("代码结束。")