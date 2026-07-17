# 价值迭代算法

import copy

class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P
    
class ValueIteration:
    """ 价值迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值为0
        self.theta = theta  # 价值收敛阈值
        self.gamma = gamma
        # 价值迭代结束后得到的策略
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]
    
    def value_iteration(self):
        cnt = 0
        # 相当于直接找每个状态s下最大的动作价值
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += r + self.gamma * self.v[next_state] * (1 - done)
                        # qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)  # 这一行和下一行代码是价值迭代和策略迭代的主要区别
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break  # 满足收敛条件,退出评估迭代
            cnt += 1
        print("价值迭代一共进行%d轮" % cnt)
        self.get_policy()
    
    def get_policy(self):  # 根据价值函数导出一个贪婪策略
        # 根据贝尔曼最优方程，此时最优状态价值是选择此时使最优动作价值最大的那一个动作时的状态价值
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa) 
            maxq = max(qsa_list) # 找到使最优动作价值最大的那一个动作
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
    
    def print_agent(self, agent, action_meaning, disaster=[], end=[]):
        print("状态价值：")
        for i in range(agent.env.nrow):
            for j in range(agent.env.ncol):
                # 为了输出美观,保持输出6个字符
                print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
            print()
        # 经过 5 次策略评估和策略提升的循环迭代，策略收敛了，此时将获得的策略打印出来。
        # 用贝尔曼最优方程去检验其中每一个状态的价值，可以发现最终输出的策略的确是最优策略。
        print("策略：")
        for i in range(agent.env.nrow):
            for j in range(agent.env.ncol):
                # 一些特殊的状态,例如悬崖漫步中的悬崖
                if (i * agent.env.ncol + j) in disaster:
                    print('****', end=' ')
                elif (i * agent.env.ncol + j) in end:  # 目标状态
                    print('EEEE', end=' ')
                else:
                    a = agent.pi[i * agent.env.ncol + j]
                    pi_str = ''
                    for k in range(len(action_meaning)):
                        pi_str += action_meaning[k] if a[k] > 0 else 'o'
                    print(pi_str, end=' ')
            print()

if __name__ == "__main__":
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = ValueIteration(env, theta, gamma)
    agent.value_iteration()
    agent.print_agent(agent, action_meaning, list(range(37, 47)), [47])