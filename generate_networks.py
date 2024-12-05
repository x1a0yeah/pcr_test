import networkx as nx
import numpy as np
import pandas as pd

# 定义生成ER随机图的函数
def generate_erdos_renyi(n, p):
    return nx.erdos_renyi_graph(n, p)

# 定义生成BA无标度网络的函数
def generate_scale_free(m, k):
    return nx.barabasi_albert_graph(m, k)

# 定义生成WS小世界网络的函数
def generate_small_world(k, n, p):
    if k >= n:
        # 根据 NetworkX 的实现规则：k必须小于等于 n-1，否则无法生成一个有效的小世界网络。
        raise ValueError("k must be less than n to create a valid small-world network")
    return nx.watts_strogatz_graph(n, k, p)

# # 定义生成q-snapback网络的函数
# def generate_q_snapback(n, q, r):
#     # 初始化邻接矩阵
#     adj_matrix = np.zeros((n, n), dtype=int)
#     # 创建主链
#     for i in range(1, n):
#         adj_matrix[i][i - 1] = 1  # 指向前一个节点
#
#     # 添加回溯链接
#     for i in range(r + 1, n):
#         for _ in range(r):
#             # 随机选择一个回溯链接的目标节点
#             # 确保下限小于上限
#             low = max(0, i - q) if i - q > 0 else 0
#             high = i - 1
#             target = np.random.randint(low, high)
#             # 添加回溯链接
#             adj_matrix[i][target] = 1
#
#     return adj_matrix

# 定义保存图的邻接矩阵为.npy文件的函数
def save_adjacency_matrix(adj_matrix, filename):
    # 将一个图对象G转换为邻接矩阵
    # adj_matrix = nx.to_numpy_array(G)
    np.save(filename, adj_matrix)

def main():
    n = 1000  # 节点数量
    p = 0.05  # 边的连接概率
    k = 3  # Barabasi-Albert模型的k值
    # q = 0.3  # q-snapback模型中回溯链接的概率阈值
    # r = 1    # q-snapback模型中每个节点生成的回溯链接数

    # 生成网络
    G_ER = generate_erdos_renyi(n, p)
    G_SF = generate_scale_free(n, k)
    G_SW = generate_small_world(4, n, 0.1)  # k值设置为4
    # adj_matrix_QSN = generate_q_snapback(n, q, r)

    # 保存邻接矩阵
    save_adjacency_matrix(nx.to_numpy_array(G_ER), 'er_adjacency_matrix.npy')
    save_adjacency_matrix(nx.to_numpy_array(G_SF), 'sf_adjacency_matrix.npy')
    save_adjacency_matrix(nx.to_numpy_array(G_SW), 'sw_adjacency_matrix.npy')
    # save_adjacency_matrix(adj_matrix_QSN, 'qsn_adjacency_matrix.csv')

if __name__ == "__main__":
    main()
