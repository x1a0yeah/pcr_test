import numpy as np

# 使用np.load()读取.npy文件中的数据
file_path = 'ER_controllability_curve.npy'

# 加载文件中的数据
data = np.load(file_path)

# 验证数组是否在预期范围内（比如检查值是否在 0 到 1 之间）
if np.all((data >= 0) & (data <= 998)):
    print("数据在预期范围内。")
else:
    print("数据超出预期范围！")

# 打印数据
print(data)

# 打印形状和大小
# shape = (999,) 表示一个包含 999 个元素的 一维数组。
print("数据形状:", data.shape)
print("数据元素数量:", data.size)