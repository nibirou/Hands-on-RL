# 我们仍然在悬崖漫步环境下尝试 Sarsa 算法。首先来看一下悬崖漫步环境的代码，
# 这份环境代码和第 4 章中的不一样，因为此时环境不需要提供奖励函数和状态转移函数，
# 而需要提供一个和智能体进行交互的函数step()，
# 该函数将智能体的动作作为输入，输出奖励和下一个状态给智能体。

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库

class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标
    
    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上（纵坐标-1）, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 原点（0，0）定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done
    
    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x

# 然后我们来实现 Sarsa 算法，主要维护一个表格Q_table()，用来储存当前策略下所有状态动作对的价值，在用 Sarsa 算法和环境交互时，用-贪婪策略进行采样，在更新 Sarsa 算法时，使用时序差分的公式。
# 我们默认终止状态时所有动作的价值都是 0，这些价值在初始化为 0 后就不会进行更新。
class Sarsa:
    """ Sarsa算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

    def take_action(self, state):  # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def print_agent(self, agent, env, action_meaning, disaster=[], end=[]):
        for i in range(env.nrow):
            for j in range(env.ncol):
                if (i * env.ncol + j) in disaster:
                    print('****', end=' ')
                elif (i * env.ncol + j) in end:
                    print('EEEE', end=' ')
                else:
                    a = self.best_action(i * env.ncol + j)
                    pi_str = ''
                    for k in range(len(action_meaning)):
                        pi_str += action_meaning[k] if a[k] > 0 else 'o'
                    print(pi_str, end=' ')
            print()

if __name__ == "__main__":
    # 在悬崖漫步环境中运行 Sarsa 算法
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 500  # 智能体在环境中运行的序列的数量

    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False
                while not done:
                    next_state, reward, done = env.step(action)
                    next_action = agent.take_action(next_state)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state, next_action)
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa on {}'.format('Cliff Walking'))
plt.savefig('./Chapter5/Sarsa_on_Cliff_Walking.png')
# plt.show()

# 我们发现，随着训练的进行，Sarsa 算法获得的回报越来越高。在进行 500 条序列的学习后，
# 可以获得 −20 左右的回报，此时已经非常接近最优策略了。
# 然后我们看一下 Sarsa 算法得到的策略在各个状态下会使智能体采取什么样的动作。
action_meaning = ['^', 'v', '<', '>']
print('Sarsa算法最终收敛得到的策略为：')
agent.print_agent(agent, env, action_meaning, list(range(37, 47)), [47])