import matplotlib.pyplot as plt
import numpy as np
# for pre
# 生成数据
np.random.seed(0)
# 两个不同区域的x值
x1 = 2 * np.random.rand(100, 1) + 1  # x 在 (1,3) 区域
x2 = 2 * np.random.rand(100, 1) + 4  # x 在 (4,6) 区域

# 相同的y值关系
y1 = 4 + 3 * x1 + np.random.randn(100, 1)
y2 = 4 + 3 * x2 + np.random.randn(100, 1)

# 计算相同的回归线
# 我们在整个范围上生成一组线性回归模型
x_all = np.vstack((x1, x2))
y_all = np.vstack((y1, y2))
regression_line = np.linspace(0, 7, 100).reshape(-1, 1)
y_regression = 4 + 3 * regression_line

# 创建图形
plt.figure(figsize=(14, 6))

# 第一个图：x 在 (1,3) 区域
plt.subplot(1, 2, 1)
plt.scatter(x1, y1, color='blue', label='Data Points (1,3)')
plt.plot(regression_line, y_regression, color='red', linewidth=2, label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression (x in [1, 3])')
plt.legend()

# 第二个图：x 在 (4,6) 区域
plt.subplot(1, 2, 2)
plt.scatter(x2, y2, color='blue', label='Data Points (4,6)')
plt.plot(regression_line, y_regression, color='red', linewidth=2, label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression (x in [4, 6])')
plt.legend()

plt.show()
