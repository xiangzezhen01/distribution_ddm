import numpy as np

# 列堆叠，适用于一维数组，效果类似于hstack
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])


# 垂直堆叠，效果类似于axis=0的concatenate
result = np.vstack((array1, array2))
print("垂直堆叠结果：\n", result)

