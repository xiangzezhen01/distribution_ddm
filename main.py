import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 生成两个正态分布的样本
np.random.seed(0)
sample1 = np.random.normal(loc=5, scale=1, size=100)
sample2 = np.random.normal(loc=0.5, scale=1, size=100)

# 计算K-S检验
ks_statistic, p_value = stats.ks_2samp(sample1, sample2)

print(f"K-S Statistic: {ks_statistic}")
print(f"P-value: {p_value}")

# 可视化ECDF
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return x, y

x1, y1 = ecdf(sample1)
x2, y2 = ecdf(sample2)

plt.step(x1, y1, label='Sample 1')
plt.step(x2, y2, label='Sample 2')

plt.xlabel('Value')
plt.ylabel('ECDF')
plt.legend()
plt.show()
