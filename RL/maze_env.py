import numpy as np          # 引入数值计算库
import time               # 引入时间库，用于暂停
import sys                # 引入系统库，用于判断Python版本

# 根据Python版本选择Tkinter库
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # 每个格子的像素大小
MAZE_H = 4  # 迷宫高度（格子数）
MAZE_W = 4  # 迷宫宽度（格子数）


class Maze(tk.Tk, object):
    """
    迷宫环境类，继承自tk.Tk，用于可视化迷宫及智能体交互
    """
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']  # 动作空间：上下左右
        self.n_actions = len(self.action_space)   # 动作数量
        self.n_features = 2                       # 状态特征维度（相对坐标差）
        self.title('maze')                        # 窗口标题
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))  # 窗口大小
        self._build_maze()                        # 构建迷宫界面

    def _build_maze(self):
        """
        构建迷宫界面，包括网格、障碍物、目标点和智能体
        """
        # 创建画布，白色背景
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # 绘制纵向网格线
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        # 绘制横向网格线
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 定义原点坐标（左上角格子中心）
        origin = np.array([20, 20])

        # 创建第一个陷阱（黑色方块）
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # 第二个陷阱（已注释）
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 15, hell2_center[1] - 15,
        #     hell2_center[0] + 15, hell2_center[1] + 15,
        #     fill='black')

        # 创建目标点（黄色圆形）
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # 创建智能体（红色方块）
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # 将画布添加到窗口
        self.canvas.pack()

    def reset(self):
        """
        重置环境，将智能体放回起点，返回初始状态
        """
        # 刷新界面
        self.update()
        # 短暂暂停，便于观察
        time.sleep(0.1)
        # 删除旧的红色方块
        self.canvas.delete(self.rect)
        # 重新设置起始位置
        origin = np.array([20, 20])
        # 重新绘制红色方块
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # 返回归一化后的状态（智能体与目标之间的相对坐标差）
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)

    def step(self, action):
        """
        执行一步动作，返回下一个状态、奖励和是否结束
        """
        s = self.canvas.coords(self.rect)  # 当前智能体坐标
        base_action = np.array([0, 0])     # 基础移动量
        # 根据动作更新移动量
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        # 移动智能体
        self.canvas.move(self.rect, base_action[0], base_action[1])

        # 获取下一个状态坐标
        next_coords = self.canvas.coords(self.rect)

        # 奖励函数
        if next_coords == self.canvas.coords(self.oval):
            reward = 1   # 到达目标，奖励1
            done = True  # 结束
        elif next_coords in [self.canvas.coords(self.hell1)]:
            reward = -1  # 掉入陷阱，奖励-1
            done = True  # 结束
        else:
            reward = 0   # 普通移动，奖励0
            done = False # 未结束
        # 计算归一化状态（相对目标坐标差）
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    def render(self):
        """
        渲染界面（更新窗口）
        """
        # time.sleep(0.01)
        self.update()