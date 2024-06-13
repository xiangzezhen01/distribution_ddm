import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



base_data = "data/data2/x264_0.csv"
data1 = "data/data2/x264_1.csv"
data2 = "data/data2/x264_2.csv"
data3 = "data/data2/x264_3.csv"
data4 = "data/data2/x264_4.csv"
data5 = "data/data2/x264_5.csv"
data6 = "data/data2/x264_6.csv"

d1 = pd.read_csv(base_data)
d2 = pd.read_csv(data1)
d3 = pd.read_csv(data2)
d4 = pd.read_csv(data3)


whole_data = np.vstack([d1.values, d2.values, d3.values, d4.values])
(N, n) = whole_data.shape

feature_weights = mutual_info_regression(whole_data[:, :n-1], whole_data[:, -1], random_state=0)
max = 0
maxi = 0
for i,v in enumerate(feature_weights):
    if v > max:
        max = v
        maxi = i






# 生成示例数据

x = d1.values[:, maxi]
y = whole_data[:, -1]
z = np.random.standard_normal(100)

# 创建一个新的图形
fig = plt.figure()

# 添加一个3D子图
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图
ax.scatter(x, y, z, c='r', marker='o')

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图形
plt.show()
