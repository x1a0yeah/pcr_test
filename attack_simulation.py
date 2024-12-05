import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# 设置字体以支持中文和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 确保中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_adjacency_matrix(filename):
    return nx.from_numpy_array(np.load(filename, allow_pickle=True))

def calculate_controllability(G):
    # 这里是计算控制性的代码,计算控制性是一个复杂的过程
    N = len(G.nodes)
    A = nx.adjacency_matrix(G).todense()
    # print(np.linalg.matrix_rank(A))
    t = max(N - np.linalg.matrix_rank(A), 1) / N
    return t

def simulate_random_attacks(G, num_nodes):
    controllability_curve = []
    # 模拟移除i个节点
    attacked_G = G.copy()
    for i in range(1, num_nodes):

        nodes_to_remove = np.random.choice(list(attacked_G.nodes), 1)
        attacked_G.remove_nodes_from(nodes_to_remove)
        # print(len(attacked_G.nodes))
        # 计算攻击后的控制性
        controllability = calculate_controllability(attacked_G)
        controllability_curve.append(controllability)

    return np.array(controllability_curve)

def plot_controllability_curve(controllability_curve, title):
    plt.figure(figsize=(10, 6))
    plt.plot(controllability_curve, label='Controllability')
    plt.title(title)
    plt.xlabel('删除的节点数')
    plt.ylabel('可控性')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def main():
    # 加载邻接矩阵
    networks = {
        'ER': 'er_adjacency_matrix.npy',
        'SF': 'sf_adjacency_matrix.npy',
        # 'QSN': 'qsn_adjacency_matrix.npy',
        'SW': 'sw_adjacency_matrix.npy'
    }

    for name, filename in networks.items():
        G = load_adjacency_matrix(filename)
        num_nodes = G.number_of_nodes()
        controllability_curve = simulate_random_attacks(G, num_nodes)

        # 保存控制性曲线
        np.save(f'{name}_controllability_curve.npy', controllability_curve)

        # 可视化控制性曲线
        plot_controllability_curve(controllability_curve, f'{name} 可控性曲线')

if __name__ == "__main__":
    main()