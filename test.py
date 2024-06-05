import numpy as np
import pandas as pd

df = pd.read_csv('data/storm-obj1_feature6.csv')

# 将数据框转换为ndarray
whole_data = df.values
X = whole_data[:, :-1]
Y = whole_data[:, -1]

print("gh")
