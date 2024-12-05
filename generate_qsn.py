import numpy as np


def generate_q_snapback(n, q, r):
    """
    生成一个 q-snapback 网络。

    参数:
    n -- 网络中的节点总数
    q -- 回溯链接的最大索引，即一个节点至多可以连接到编号为 n - q 的节点
    r -- 每个节点生成的回溯链接数

    返回:
    adj_matrix -- 网络的邻接矩阵
    edges -- 网络中的边及其方向
    """
    # 初始化邻接矩阵
    adj_matrix = np.zeros((n, n), dtype=int)

    # 初始化边的列表
    edges = []

    # 创建主链
    for i in range(1, n):
        adj_matrix[i][i - 1] = 1  # 指向前一个节点
        edges.append((i, i - 1))  # 记录边的方向

    # 添加回溯链接
    for i in range(r + 1, n):
        for _ in range(r):
            # 随机选择一个回溯链接的目标节点
            target = np.random.randint(max(0, i - q), i - 1)
            adj_matrix[i][target] = 1
            edges.append((i, target))  # 记录边的方向

    return adj_matrix, edges


# 参数
n = 10  # 节点数
q = 3  # 回溯链接的最大索引
r = 1  # 每个节点生成的回溯链接数

# 生成 q-snapback 网络
adj_matrix, edges = generate_q_snapback(n, q, r)

# 打印邻接矩阵
print("邻接矩阵:")
print(adj_matrix)

# 打印每组连边的方向
print("\n每组连边的方向:")

for edge in edges:
    print(edge)