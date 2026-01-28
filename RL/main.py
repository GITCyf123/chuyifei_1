from maze_env import Maze
from RL_brain import DeepQNetwork

def run_maze():
    step = 0  # 为了记录走到第几步，记忆录中积累经验（也就是积累一些transition）之后再开始学习
    for episode in range(200):
        # 初始化观测
        observation = env.reset()

        while True:
            # 刷新环境
            env.render()

            # RL 基于观测选择动作
            action = RL.choose_action(observation)

            # RL 执行动作并获取下一观测与奖励
            observation_, reward, done = env.step(action)

            # !! 存储 transition
            RL.store_transition(observation, action, reward, observation_)

            # 超过200条transition之后每隔5步学习一次
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 更新观测
            observation = observation_

            # 本回合结束则跳出循环
            if done:
                break
            step += 1

    # 游戏结束
    print("game over")
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000)
    # 100毫秒后调用run_maze函数，开始运行迷宫游戏
    env.after(100, run_maze)
    # 启动GUI主循环，等待并响应用户交互
    env.mainloop()
    # 训练结束后绘制损失曲线，观察学习过程中的cost变化
    RL.plot_cost()