import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 示例数据
data = {
    'X': [1, 2, 3, 4, 5],
    'Y': [5, 4, 3, 2, 1],
    'Label': [1, 2, 1, 2, 1]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 创建颜色映射
palette = sns.color_palette("hsv", len(df['Label'].unique()))

# 绘制点图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='X', y='Y', hue='Label', palette=palette, s=100)

# 显示图例
plt.legend(title='Label')

# 显示图形
plt.show()