import random
import matplotlib.pyplot as plt
import numpy as np


def generate_scale_free_network(n, m):
    """
    生成一个无标度网络。

    参数：
        n (int): 网络的节点总数。
        m (int): 每个新加入节点连接的已有节点数量。

    返回：
        nodes (list): 节点列表。
        adjacency_list (dict): 邻接表表示的图。
    """
    nodes = list(range(n))  # 所有节点的编号
    adjacency_list = {i: [] for i in range(n)}  # 使用邻接表存储图结构

    # 初始化网络：第一个节点连接到前 m 个节点
    for i in range(1, m + 1):
        adjacency_list[0].append(i)
        adjacency_list[i].append(0)

    # 度列表，用于快速选择连接的节点
    degrees = [0] * n
    for i in range(m + 1):
        degrees[i] = len(adjacency_list[i])

    # 构建网络：逐步添加节点
    for new_node in range(m + 1, n):
        total_degree = sum(degrees[:new_node])
        connected_nodes = random.choices(
            population=list(range(new_node)), weights=degrees[:new_node], k=m
        )

        for node in connected_nodes:
            adjacency_list[new_node].append(node)
            adjacency_list[node].append(new_node)
            degrees[new_node] += 1
            degrees[node] += 1

    return nodes, adjacency_list


def initialize_sis_state(nodes, initial_infected_ratio):
    """
    初始化 SIS 模型的节点状态。

    参数：
        nodes (list): 节点列表。
        initial_infected_ratio (float): 初始感染节点比例。

    返回：
        states (list): 节点状态列表，'S' 表示易感，'I' 表示感染。
    """
    states = ['S'] * len(nodes)  # 所有节点初始状态为易感
    initial_infected_count = int(len(nodes) * initial_infected_ratio)  # 初始感染节点数量
    infected_nodes = random.sample(nodes, initial_infected_count)  # 随机选择感染节点

    for node in infected_nodes:
        states[node] = 'I'  # 设置感染状态

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
    for _ in range(steps):
        new_states = states[:]

        for node in nodes:
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


def plot_phase_transition(betas, gamma, steps, initial_infected_ratio):
    """
    绘制 SIS 模型的相变图。

    参数：
        betas (list): 一组不同的感染率值。
        gamma (float): 恢复率。
        steps (int): 模拟步数。
        initial_infected_ratio (float): 初始感染节点比例。
    """
    prevalences = []  # 用于存储每个感染率下的感染比例

    for beta in betas:
        nodes, adjacency_list = generate_scale_free_network(1000, 3)  # 生成一个无标度网络
        states = initialize_sis_state(nodes, initial_infected_ratio)  # 初始化节点状态
        final_states = simulate_sis(nodes, adjacency_list, states, beta, gamma, steps)  # 模拟 SIS 模型
        prevalence = final_states.count('I') / len(nodes)  # 计算感染比例
        prevalences.append(prevalence)

    # 绘制结果
    plt.plot(betas, prevalences)
    plt.xlabel('Infection Rate (β)')
    plt.ylabel('Prevalence')
    plt.title('SIS Model Phase Transition')
    plt.show()


# 设置参数
betas = np.linspace(0.001, 0.1, 100)  # 不同的感染率值
gamma = 0.1  # 恢复率
steps = 1000  # 模拟步数
initial_infected_ratio = 0.01  # 初始感染节点比例

# 绘制相变图
plot_phase_transition(betas, gamma, steps, initial_infected_ratio)
