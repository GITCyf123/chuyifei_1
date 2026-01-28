"""
Deep Q Network off-policy
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 固定随机种子，保证实验可复现
np.random.seed(42)
torch.manual_seed(2)


class Network(nn.Module):
    """
    Network Structure
    """
    def __init__(self,
                 n_features,
                 n_actions,
                 n_neuron=10
                 ):
        super(Network, self).__init__()
        # 构建两层全连接网络：输入层->隐藏层->输出层，并在最后添加ReLU激活
        self.net = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_neuron, bias=True),
            nn.Linear(in_features=n_neuron, out_features=n_actions, bias=True),
            nn.ReLU()
        )

    def forward(self, s):
        """
        前向传播，输入状态s，返回各动作的Q值
        :param s: 状态向量
        :return: q值向量
        """
        q = self.net(s)
        return q


class DeepQNetwork(nn.Module):
    """
    Q Learning Algorithm
    """
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None):
        super(DeepQNetwork, self).__init__()

        self.n_actions = n_actions          # 动作空间维度
        self.n_features = n_features        # 状态空间维度
        self.lr = learning_rate             # 学习率
        self.gamma = reward_decay           # 折扣因子
        self.epsilon_max = e_greedy         # epsilon最大值
        self.replace_target_iter = replace_target_iter  # 目标网络更新频率
        self.memory_size = memory_size      # 经验回放缓冲区大小
        self.batch_size = batch_size        # 每次训练采样的批量大小
        self.epsilon_increment = e_greedy_increment  # epsilon增量
        # 若提供增量，则初始epsilon为0，否则直接使用最大值
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # 总学习步数计数器
        self.learn_step_counter = 0

        # 初始化经验回放缓冲区，用DataFrame存储[s, a, r, s_]格式的transition
        # 表格行数为memory_size，列数为2*n_features + 2（状态*2 + 动作 + 奖励）
        self.memory = pd.DataFrame(np.zeros((self.memory_size, self.n_features*2+2)))

        # 构建两个网络：eval_net用于当前策略，target_net用于计算目标Q值
        self.eval_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        self.target_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        # 定义损失函数与优化器
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        # 记录每一步的误差，用于后续绘图
        self.cost_his = []


    def store_transition(self, s, a, r, s_):
        """
        存储一条transition到经验回放缓冲区
        :param s: 当前状态
        :param a: 采取的动作
        :param r: 获得的奖励
        :param s_: 下一状态
        """
        # 若首次调用，初始化memory_counter
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 将状态、动作、奖励、下一状态拼接成一条transition
        transition = np.hstack((s, [a,r], s_))

        # 使用循环队列方式覆盖旧数据
        index = self.memory_counter % self.memory_size
        self.memory.iloc[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        """
        根据epsilon-greedy策略选择动作
        :param observation: 当前观测（状态）
        :return: 选择的动作索引
        """
        # 将观测升维成二维，适配网络输入
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # 利用当前eval_net计算各动作Q值，并选择最大值对应的动作
            s = torch.FloatTensor(observation)
            actions_value = self.eval_net(s)
            action = [np.argmax(actions_value.detach().numpy())][0]
        else:
            # 以1-epsilon概率随机探索
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        """
        将eval_net的参数复制给target_net，实现软/硬更新
        """
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        """
        执行一次DQN学习步骤：采样、计算目标Q值、更新eval_net
        """
        # 每隔replace_target_iter步，同步target网络参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget params replaced\n')

        # 从经验池中随机采样batch_size条transition
        # 若memory未填满，则允许重复采样
        batch_memory = self.memory.sample(self.batch_size) \
            if self.memory_counter > self.memory_size \
            else self.memory.iloc[:self.memory_counter].sample(self.batch_size, replace=True)

        # 提取状态、下一状态
        s = torch.FloatTensor(batch_memory.iloc[:, :self.n_features].values)
        s_ = torch.FloatTensor(batch_memory.iloc[:, -self.n_features:].values)
        # 分别用eval_net与target_net计算Q值
        q_eval = self.eval_net(s)
        q_next = self.target_net(s_)

        # 复制eval_net输出作为目标，后续仅更新对应动作的Q值
        q_target = q_eval.clone()

        # 提取动作索引与奖励
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory.iloc[:, self.n_features].values.astype(int)
        reward = batch_memory.iloc[:, self.n_features + 1].values

        # 使用贝尔曼方程更新目标Q值
        q_target[batch_index, eval_act_index] = torch.FloatTensor(reward) + self.gamma * q_next.max(dim=1).values

        # 计算损失并反向传播更新eval_net
        loss = self.loss_function(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 记录损失值
        self.cost_his.append(loss.detach().numpy())

        # 逐步增加epsilon，直至最大值
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        """
        绘制训练过程中损失变化曲线
        """
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.show()