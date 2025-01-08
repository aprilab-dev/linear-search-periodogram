import numpy as np

import numpy as np

# 行向量和列向量
A = np.array([1, 2, 3])  # 行向量
B = np.array([[4], [5], [6]])  # 列向量
C = np.array([1, 2, 3])  # 行向量

# 行向量和列向量的点积
result1 = A * B 
result2 = B * A 
print(result1)
print(result2)
print(A @ B)
