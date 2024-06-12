import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

# 使用pandas读取CSV文件
"""df1 = pd.read_csv('data/storm-obj1_feature6.csv')
df2 = pd.read_csv('data/storm-obj1_feature7.csv')
df3 = pd.read_csv('data/storm-obj1_feature8.csv')
df4 = pd.read_csv('data/storm-obj1_feature9.csv')"""

"""df1 = pd.read_csv('data/data1/sac_2.csv')
df2 = pd.read_csv('data/data1/sac_5.csv')
df3 = pd.read_csv('data/data1/sac_7.csv')
df4 = pd.read_csv('data/data1/sac_8.csv')"""

df1 = pd.read_csv('data/data2/x264_0.csv')
df2 = pd.read_csv('data/data2/x264_1.csv')
df3 = pd.read_csv('data/data2/x264_2.csv')
df4 = pd.read_csv('data/data2/x264_3.csv')

count = 400

# 从每个DataFrame中抽样count个样本
df11 = df1.sample(count)
df22 = df2.sample(count)
df33 = df3.sample(count)
df44 = df4.sample(count)

# 将DataFrame转换为numpy ndarray
data_array1 = df11.values
data_array2 = df22.values
data_array3 = df33.values
data_array4 = df44.values

# 将两个数组垂直堆叠
result = np.vstack((data_array1, data_array2, data_array3, data_array4))
(N, n) = result.shape  # 获取结果的形状


zeros = np.zeros(count, dtype=int)

# 创建一个包含200个1的数组
ones = np.ones(count, dtype=int)

# 创建一个包含200个2的数组
twos = np.full(count, 2, dtype=int)

# 创建一个包含200个3的数组
threes = np.full(count, 3, dtype=int)

# 将这些数组连接起来，形成一个最终的数组


# 分离特征和标签
X = result  # 特征
y = np.concatenate((zeros, ones, twos, threes))   # 标签

"""max_X = np.amax(X, axis=0)
min_X = np.amin(X, axis=0)
X = (X - min_X) / (max_X - min_X)"""


# 初始化TSNE对象，并将高维数据降维到二维
tsne = TSNE(n_components=2, perplexity=300)
X_embedded = tsne.fit_transform(X)

# 可视化降维后的数据
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(scatter, ticks=[0, 1, 2], label='Species')
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()
