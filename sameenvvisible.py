import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns

data = "data/storm-obj1_feature6.csv"


def eq_on_indexs(l1, l2, indexs):
    for i in indexs:
        if l1[i] != l2[i]:
            return False
    return True


whole_data = pd.read_csv(data).values
(N, n) = whole_data.shape
xs = whole_data[:, :-1]
ys = whole_data[:, -1]
feature_weights = mutual_info_regression(whole_data[:, 0:n - 1], whole_data[:, n - 1], random_state=0)
valid_features = list()
for i, w in enumerate(feature_weights):
    if w >= 0.2:
        valid_features.append(i)
print(feature_weights)
print(valid_features)
xs = xs.tolist()
ys = ys.tolist()
clusters = list()
for i, x in enumerate(xs):
    exist = False
    for c in clusters:
        if eq_on_indexs(x,xs[c[0]],valid_features):
            exist = True
            c.append(i)
    if exist:
        continue
    newc = list()
    newc.append(i)
    clusters.append(newc)
print(clusters)
print(len(clusters))

X2I = dict()
for i, c in enumerate(clusters):
    for x in c:
        X2I[x] = i
gX = list(range(len(xs)))
for i, x in enumerate(gX):
    gX[i] = X2I[x]
gY = ys
gL = np.zeros(len(xs)).tolist()

data = {
    "X": gX,
    "Y": gY,
    "L": gL
}
df = pd.DataFrame(data)
# 创建颜色映射
palette = sns.color_palette("hsv", len(df['L'].unique()))

# 绘制点图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='X', y='Y', hue='L', palette=palette, s=100)

# 显示图例
plt.legend(title='L')

# 显示图形
plt.show()
