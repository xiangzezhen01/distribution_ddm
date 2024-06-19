import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from river import drift
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import configs


def re_to_bool(val, threshold = configs.RESIDUAL_ERRORS_THRESHOLD):
    return 0 if val <= threshold else 1

"""not good 534"""
base_data = "data/data1/sac_2.csv"
data1 = "data/data1/sac_4.csv"
data2 = "data/data1/sac_5.csv"
data3 = "data/data1/sac_6.csv"
data4 = "data/data1/sac_7.csv"
data5 = "data/data1/sac_8.csv"
data6 = "data/data1/sac_9.csv"

"""534"""
"""base_data = "data/data2/x264_0.csv"
data1 = "data/data2/x264_1.csv"
data2 = "data/data2/x264_2.csv"
data3 = "data/data2/x264_3.csv"
data4 = "data/data2/x264_4.csv"
data5 = "data/data2/x264_5.csv"
data6 = "data/data2/x264_6.csv"""

"""base_data = "data/storm-obj1_feature6.csv"
data1 = "data/storm-obj1_feature7.csv"
data2 = "data/storm-obj1_feature8.csv"
data3 = "data/storm-obj1_feature9.csv"""


others = [data2, data3, data4]

ddm = drift.binary.EDDM()

# Generate train and test data
train = pd.read_csv(base_data)
(N, n) = train.shape
x_train = train.iloc[:, :n-1]
y_train = train.iloc[:, -1]
rfr = RandomForestRegressor()
rfr.fit(x_train.values, y_train.values)

base_sample = train.sample(800)
x_base = base_sample.iloc[:, :n-1]
y_base = base_sample.iloc[:, -1]
y_base_pre = rfr.predict(x_base.values)
aes = abs(y_base - y_base_pre)

test1 = pd.read_csv(data6)
test2 = pd.read_csv(data2)
test3 = pd.read_csv(data3)

(N1, n1) = test1.shape
(N2, n2) = test2.shape
(N3, n3) = test3.shape
test1 = test1.sample(N1)
test2 = test2.sample(N2)
test3 = test3.sample(N3)

test = np.vstack((base_sample.values, test1.values, test2.values, test3.values))
x_test = test[:, :n-1]
y_test = test[:, -1]
print(f"in test data, drift appear in index 16, {N1/50+16}, {(N1+N2)/50+16}")
start = N1+N2+N3

windows = []
models = []

models.append(rfr)
x = range(len(x_test))
y = list()
wait_data = False
enough_data = False
warning = False
for i, sample in enumerate(x_test):
    y_pre = rfr.predict([sample])
    mae = abs(y_pre-y_test[i])
    y.append(mae)


    if wait_data:
        if i - start >= 100:
            enough_data = True
        else:
            continue

    ddm.update(re_to_bool(mae[0]))


    if ddm.warning_detected:
        if not warning:
            start = i
            warning = True
        print(f"warning detected in index {i/50}")



    if ddm.drift_detected or (wait_data and enough_data):
        if (warning and i - start < 100) or not warning:
            print(f"drift detected but data in index {i/50} is not enough")
            wait_data = True
            enough_data = False
            start = i
            warning = True
            continue
        print(f"drift detected in index {i/50}, new train data count : "+str(i - start))
        print("the old model mae:")
        wait_data = False
        enough_data = False
        warning = False
        new_train = test[range(start, i+1)]
        new_x = new_train[:, :n-1]
        new_y = new_train[:, -1]
        old_y = rfr.predict(new_x)

        maes = abs(old_y - new_y)
        mmae = np.mean(maes)
        print(mmae)
        windows.clear()
        rfr.fit(new_x, new_y)
        new_y_pre = rfr.predict(new_x)
        maes = abs(new_y_pre - new_y)
        mmae = np.mean(maes)
        print("the new model mae:")
        print(mmae)
        start = N1+N2+N3
        continue


def average_of_chunks(data, chunk_size=50):
    # 初始化一个空列表，用于存储每组的平均值
    averages = []

    # 计算列表的长度
    length = len(data)

    # 使用range函数按chunk_size步长进行迭代
    for i in range(0, length, chunk_size):
        # 获取当前chunk
        chunk = data[i:i + chunk_size]

        # 计算当前chunk的平均值
        chunk_average = sum(chunk) / len(chunk)

        # 将平均值添加到averages列表中
        averages.append(chunk_average)

    return averages


# 示例使用 # 示例浮点数列表
y = average_of_chunks(y)
x = range(len(y))

plt.figure()

# 绘制点图
plt.scatter(x, y, color='blue', label='Scatter Plot')

# 绘制线图
plt.plot(x, y, color='red', label='Line Plot')
plt.plot(x, y)

# 添加标题和标签
plt.title('absolute error')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图例
plt.legend()

# 显示图形
plt.show()


