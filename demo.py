import json
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from initial_states_p import generate_for_two_particles, PARTICLE_TYPES

Qe = 1.602176634e-19
RB87_MASS = PARTICLE_TYPES['Rb87']['mass']
RB85_MASS = PARTICLE_TYPES['Rb85']['mass']


def load_magnetic_field_data(ft_path=None):
    """加载磁场表 ft.txt 并按 y 方向建立插值函数。"""
    if ft_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.environ.get("FENLI_FT_PATH")
        ft_path = env_path if env_path else os.path.join(base_dir, "data", "ft.txt")

    data = np.loadtxt(ft_path)
    t_values = data[:, 0]
    ft_values = data[:, 1]

    return interp1d(t_values, ft_values, kind='cubic', bounds_error=False,
                    fill_value=(ft_values[0], ft_values[-1]))


def calculate_magnetic_field(position, interpolator):
    """按当前位置 y 取插值，转换为 Bz 分量；B=(0,0,bzy)。"""
    y = position[1]
    ft_y = interpolator(y)
    bzy = ft_y * 1e-4 * 0.9
    return np.array([0.0, 0.0, bzy])


def lorentz_force(t, state, mass, charge, interpolator):
    """洛伦兹力 F=q(v×B) → a=F/m；返回状态导数。"""
    position = state[:3]
    velocity = state[3:]
    B_field = calculate_magnetic_field(position, interpolator)
    force = charge * np.cross(velocity, B_field)
    acceleration = force / mass
    return np.concatenate([velocity, acceleration])


def calculate_trajectory(initial_position, initial_velocity, mass, charge,
                         interpolator, t_max=2e-5, num_points=100):
    """不设几何边界，积分得到三维轨迹点阵。"""
    initial_state = np.concatenate([initial_position, initial_velocity])
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, num_points)

    solution = solve_ivp(
        lambda t, y: lorentz_force(t, y, mass, charge, interpolator),
        t_span,
        initial_state,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-8,
        max_step=t_max / 100,
        dense_output=True
    )
    return solution.y[:3, :].T


def find_narrowest_region(trajectories, start_percentage=50):
    """在所有轨迹的后半段上搜索过所有轨迹的水平线，找出聚焦区域。"""
    narrowest_width = float('inf')
    narrowest_y = None
    narrowest_x_min = None
    narrowest_x_max = None

    total_trajectories = len(trajectories)
    start_indices = [int(len(traj) * start_percentage / 100) for traj in trajectories]

    # 收集所有 y 值用于采样
    all_y_values = []
    for traj_idx, trajectory in enumerate(trajectories):
        for i in range(start_indices[traj_idx], len(trajectory)):
            all_y_values.append(trajectory[i][1])

    if not all_y_values:
        return None, None, None, None

    y_min = min(all_y_values)
    y_max = max(all_y_values)
    y_samples = np.linspace(y_min, y_max, 2000)

    for y in y_samples:
        x_positions = []
        trajectory_count = 0

        for traj_idx, trajectory in enumerate(trajectories):
            trajectory_intersects = False
            for i in range(start_indices[traj_idx], len(trajectory) - 1):
                p1 = trajectory[i]
                p2 = trajectory[i + 1]

                if (p1[1] <= y <= p2[1]) or (p2[1] <= y <= p1[1]):
                    if abs(p2[1] - p1[1]) > 1e-10:
                        t = (y - p1[1]) / (p2[1] - p1[1])
                        x = p1[0] + t * (p2[0] - p1[0])
                        x_positions.append(x)

                        if not trajectory_intersects:
                            trajectory_count += 1
                            trajectory_intersects = True

        # 检查是否所有轨迹都相交且至少有两个交点
        if len(x_positions) >= 2 and trajectory_count == total_trajectories:
            x_min = min(x_positions)
            x_max = max(x_positions)
            width = x_max - x_min

            if width < narrowest_width:
                narrowest_width = width
                narrowest_y = y
                narrowest_x_min = x_min
                narrowest_x_max = x_max

    return narrowest_x_min, narrowest_x_max, narrowest_y, narrowest_width


def analyze(arr, mass, label, interpolator):
    """对单一粒子类型：积分轨迹，搜索最窄区域，并输出JSON轨迹文件路径。"""
    trajectories = []
    for i in range(arr.shape[0]):
        pos = arr[i, :3]
        vel = arr[i, 3:]
        traj = calculate_trajectory(pos, vel, mass, Qe, interpolator)
        trajectories.append(traj)

    x_min, x_max, y, width = find_narrowest_region(trajectories)

    # 保存轨迹到 JSON 文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    traj_json = [{"id": i + 1, "trajectory": trajectories[i].tolist()}
                 for i in range(len(trajectories))]
    traj_path = os.path.join(base_dir, f"trajectories_{label}.json")

    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(traj_json, f, ensure_ascii=False)

    if x_min is None:
        return {"result": "未找到最窄区域", "trajectories_path": traj_path}

    s = f"聚焦位置: [{x_min:.6f}, {y:.6f}], [{x_max:.6f}, {y:.6f}]\n   宽度: {width:.6f} 米"
    return {
        "result": s,
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y": float(y),
        "width": float(width),
        "trajectories_path": traj_path
    }


def focus_analysis(U=35000.0, theta1=None, count=100, ft_path=None):
    """生成初始状态，积分轨迹，查找聚焦区域并输出结果。"""
    try:
        interpolator = load_magnetic_field_data(ft_path)
    except Exception as e:
        return {"error": "磁场文件加载失败", "message": str(e)}

    # 生成初始状态
    theta1_range = theta1 if theta1 is not None else (-15, 15)
    arr_rb87_list, arr_rb85_list = generate_for_two_particles(U, theta1_range, count)

    arr_rb87 = np.array(arr_rb87_list, dtype=float)
    arr_rb85 = np.array(arr_rb85_list, dtype=float)

    # 保存初始状态文件
    np.savetxt("initial_states_rb87.txt", arr_rb87, fmt="%.6f", delimiter="\t")
    np.savetxt("initial_states_rb85.txt", arr_rb85, fmt="%.6f", delimiter="\t")

    # 分析两种粒子
    rb87_result = analyze(arr_rb87, RB87_MASS, "Rb87", interpolator)
    rb85_result = analyze(arr_rb85, RB85_MASS, "Rb85", interpolator)

    return {
        "Rb87": rb87_result,
        "Rb85": rb85_result,
    }


if __name__ == "__main__":
    result = focus_analysis(U=35e3, theta1=(-12, 12), count=20, ft_path=None)
    print(json.dumps(result, ensure_ascii=False, indent=2))