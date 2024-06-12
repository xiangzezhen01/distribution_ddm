import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

data = sns.load_dataset('iris') #总数据集
data_train = data.iloc[np.r_[0:30, 50:80, 100:130]] #训练集
data_test = data.iloc[np.r_[30:50, 80:100, 130:150]] #测试集
train_x = data_train.iloc[:,:4] #训练集数据
train_y = data_train.iloc[:,4] #训练集分类
test_x = data_test.iloc[:,:4] #测试集数据
test_y = data_test.iloc[:,4] #测试集分类
clf = LDA(n_components=2) #分类器
clf.fit(train_x, train_y) #用数据进行训练

pre_y = clf.predict(test_x) #对测试数据进行预测