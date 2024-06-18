import numpy as np
import pandas as pd

data0 = "data/data1/sac_2.csv"
data1 = "data/data1/sac_4.csv"
data2 = "data/data1/sac_5.csv"
data3 = "data/data1/sac_6.csv"
data4 = "data/data1/sac_7.csv"
data5 = "data/data1/sac_8.csv"
data6 = "data/data1/sac_9.csv"

data0 = "data/data2/x264_0.csv"
data1 = "data/data2/x264_1.csv"
data2 = "data/data2/x264_2.csv"
data3 = "data/data2/x264_3.csv"
data4 = "data/data2/x264_4.csv"
data5 = "data/data2/x264_5.csv"
data6 = "data/data2/x264_6.csv"


data_files = [data0, data1, data2, data3, data4]
test_data = []

for data_file in data_files:
    test_data.append(pd.read_csv(data_file).values)

X = None
Y = None
L = np.array(list())
i = 0

for data in test_data:
    tL = np.zeros(len(data))
    tx = data[:, :-1]
    ty = data[:, -1]
    if i == 0:
        X = tx
        Y = ty
        L = tL
        i += 1
        continue
    X = np.vstack((X, tx))
    Y = np.hstack((Y, ty))
    tL.fill(i)
    L = np.hstack((L, tL))
    i += 1

X = X.tolist()
Y = Y.tolist()

cluster = list()
for i, x in enumerate(X):
    exist = False
    for c in cluster:
        if x.__eq__(X[c[0]]):
            exist = True
            c.append(i)
    if exist:
        continue
    newc = list()
    newc.append(i)
    cluster.append(newc)

print(cluster)
X2I = dict()
for i, c in enumerate(cluster):
    for x in c:
        X2I[x] = i
gX = list(range(len(X)))
for i,x in enumerate(gX):
    gX[i] = X2I[x]
gY = Y
gL = L





