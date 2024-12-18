import random
import matplotlib.pyplot as plt
import numpy as np


def generate_er_network(n, p):
    """
    生成一个 ER（Erdős-Rényi）随机图。

    参数：
        n (int): 网络的节点总数。
        p (float): 边生成的概率。

    返回：
        nodes (list): 节点列表。
        adjacency_list (dict): 邻接表表示的图。一个字典，表示图的邻接表，其中的键是节点，值是与该节点相邻的节点列表。
    """
    nodes = list(range(n))  # 所有节点的编号
    adjacency_list = {i: [] for i in range(n)}  # 使用邻接表存储图结构

    # 按概率 p 为每对节点添加边
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    return nodes, adjacency_list


def initialize_sis_state(nodes, initial_infected_ratio):
    """
    初始化SIS（易感者-感染者-易感者）模型中每个节点的状态。

    参数：
        nodes (list): 节点列表。
        initial_infected_ratio (float): 初始感染节点比例。

    返回：
        states (list): 节点状态列表，'S' 表示易感，'I' 表示感染。
    """
    states = ['S'] * len(nodes)  # 所有节点初始状态为易感
    initial_infected_count = int(len(nodes) * initial_infected_ratio)  # 计算初始感染节点数量
    infected_nodes = random.sample(nodes, initial_infected_count)  # 随机选择感染节点（random.sample函数确保选择的节点是随机的，不会重复）

    for node in infected_nodes:   #遍历所有被随机选择为初始感染的节点
        states[node] = 'I'  # 设置它们在states列表中的状态从s变为i的感染状态

    return states


def simulate_sis(nodes, adjacency_list, states, beta, gamma, steps):
    """
    模拟 SIS 模型。

    参数：
        nodes (list): 节点列表。
        adjacency_list (dict): 邻接表表示的图。
        states (list): 节点状态列表。
        beta (float): 感染率。
        gamma (float): 恢复率。
        steps (int): 模拟步数。

    返回：
        states (list): 最终的节点状态列表。
    """
    for _ in range(steps):    #循环重复steps次
        new_states = states[:]   #通过复制当前的状态列表来创建新的列表（为了在当前迭代中更新状态，而不改变之前迭代的结果）

        for node in nodes:    #遍历每个节点，检查状态
            if states[node] == 'I':  # 如果节点处于感染状态
                # 找到所有与当前节点连接的邻居
                neighbors = adjacency_list[node]

                # 感染邻居
                for neighbor in neighbors:
                    if states[neighbor] == 'S' and random.random() < beta:
                        new_states[neighbor] = 'I'

                # 恢复为易感状态
                if random.random() < gamma:
                    new_states[node] = 'S'

        states = new_states

    return states


def plot_phase_transition(betas, gamma, steps, initial_infected_ratio,runs):
    """
    绘制 SIS 模型的相变图。

    参数：
        betas (list): 一组不同的感染率值。
        gamma (float): 恢复率。
        steps (int): 模拟步数。
        initial_infected_ratio (float): 初始感染节点比例。
    """
    prevalences = []  # 用于存储每个感染率下的感染比例

    for beta in betas:    #遍历所有感染率

        avg_prevalence = 0
        for _ in range(runs):   #对于每个感染率值，重复实验runs次，这样做可以减少随机波动对结果的影响，更稳定

            nodes, adjacency_list = generate_er_network(1000, 0.01)  # 生成一个 ER 网络
            states = initialize_sis_state(nodes, initial_infected_ratio)  # 初始化节点状态
            final_states = simulate_sis(nodes, adjacency_list, states, beta, gamma, steps)  # 模拟 SIS 模型
            prevalence = final_states.count('I') / len(nodes)  # 计算感染比例

            avg_prevalence += prevalence

        avg_prevalence /= runs  # 对多次运行取平均

        prevalences.append(prevalence)

    # 绘制结果
    plt.plot(betas, prevalences)
    plt.xlabel('Infection Rate (β)')
    plt.ylabel('Prevalence')
    # plt.title('SIS Model Phase Transition')
    plt.title('ER_SIS Model Phase Transition (Averaged over {} runs)'.format(runs))
    plt.show()


# 设置参数
betas = np.linspace(0.001, 0.1, 100)  # 不同的感染率值（列表，包含多个不同的感染率值）
# (当前的 betas 是从 0.001 到 0.1 之间取了 100 个点，分辨率可能稍微偏低。你可以通过增加 betas 的分辨率，获得更细腻的曲线

gamma = 0.1  # 恢复率
steps = 1000  # 模拟步数
initial_infected_ratio = 0.01  # 初始感染节点比例

# 绘制相变图
plot_phase_transition(betas, gamma, steps, initial_infected_ratio,runs=100)
