import os
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from imblearn.over_sampling import SMOTE
import configs
from utils.general import build_model, recursive_dividing
from numpy import recfromcsv

from utils.mlp_sparse_model_tf2 import MLPSparseModel


# todo:  2. clustering_indexs and clustering_index_all 3. total index and training index(no zero)  4. if N_train to big?

def get_whole_data(path):
    df = pd.read_csv(path)
    ndarray = df.values
    return ndarray


class DaL_Regressor:

    def __init__(self, data_path):
        self.test_mode = True  # if tune hyper-parameter? False->tune True->default configration
        self.whole_data = None
        self.other_env_data = None
        self.data_path = data_path
        self.whole_data = get_whole_data(data_path)
        (self.N, self.n) = self.whole_data.shape
        (self.sample_count, self.feather_count) = (self.N, self.n - 1)
        # todo: remove 0 performance samples
        self.train_count = int(len(self.whole_data) * configs.TRAIN_TEST_SPLIT)
        self.test_count = len(self.whole_data) - self.train_count
        temp_total_index = range(len(list(self.whole_data)))
        random.seed = 0
        self.test_index = random.sample(list(temp_total_index), self.test_count)
        self.train_index = np.setdiff1d(temp_total_index, self.test_index)  # test: divide successfully
        self.weights = list()
        feature_weights = mutual_info_regression(self.whole_data[self.train_index, 0:self.feather_count],
                                                 self.whole_data[self.train_index, self.n - 1], random_state=0)
        for i in range(self.feather_count):
            weight = feature_weights[i]
            self.weights.append(weight)
        self.train_set = self.whole_data[self.train_index]
        self.clusters_train = self.divide_cluster(self.whole_data[self.train_index])
        self.clusters_train_label = list()
        self.clusters_test_label = list()
        self.generate_cluster_train_label()
        self.ordered_train_list = list()
        for l in self.clusters_train:
            self.ordered_train_list += l
        self.x_smo = None
        self.y_smo = None
        self.data_smo()
        self.RFC = None
        self.train_RFC()
        self.generate_cluster_test_label()
        self.x_train = []
        self.y_train = []
        self.models = list()


    # todo: need to test the function
    def divide_cluster(self, whole_train_data):
        """
        using DecisionTreeRegressor divide train data into clusters
        :param whole_train_data: all train data
        :return: clusters list
        """
        DT = DecisionTreeRegressor(random_state=3, criterion='squared_error', splitter='best')
        X = whole_train_data[:, :-1]
        Y = whole_train_data[:, -1]
        DT.fit(X, Y)
        tree_ = DT.tree_
        cluster_indexes_all = []  # for recursive algorithms
        cluster_indexes_all = recursive_dividing(0, 1, tree_, X, self.train_index, configs.MAX_DEPTH,
                                                 configs.MIN_SAMPLES, cluster_indexes_all)

        return cluster_indexes_all

    # build a random forest classifier to classify testing samples

    def generate_cluster_train_label(self):
        clusters_list = list()
        for i in range(len(self.clusters_train)):
            clusters_list.append(np.full(len(self.clusters_train[i]), i))
        self.clusters_train_label = np.concatenate(clusters_list)
        return None


    def data_smo(self):
        x_smo = self.whole_data[self.train_index, 0:self.feather_count]
        x_smo = x_smo * self.weights[:self.feather_count]
        y_smo = self.clusters_train_label
        smo = SMOTE(random_state=1, k_neighbors=3)
        self.x_smo, self.y_smo = smo.fit_resample(x_smo, y_smo)

    def train_RFC(self):
        self.RFC = RandomForestClassifier(criterion='gini', random_state=3)
        if self.test_mode:
            self.RFC.fit(self.x_smo, self.y_smo)
            return
        # else, tune hyper-parameters
        param = {'n_estimators': np.arange(10, 100, 10)}
        gridS = GridSearchCV(self.RFC, param)
        gridS.fit(self.x_smo, self.y_smo)
        self.RFC = RandomForestClassifier(**gridS.best_params_, random_state=3, criterion='gini')
        self.RFC.fit(self.x_smo, self.y_smo)

    def generate_cluster_test_label(self):
        x_test = self.whole_data[self.test_index, 0:self.feather_count]
        x_test = x_test * self.weights[:self.feather_count]
        self.clusters_test_label = self.RFC.predict(x_test)

    def DaL_Train_data(self):
        max_X = np.amax(self.whole_data[self.train_index, 0:self.feather_count], axis=0)  # scale X to 0-1
        if 0 in max_X:
            max_X[max_X == 0] = 1
        max_Y = np.max(self.whole_data[self.train_index, -1]) / 100  # scale Y to 0-100
        if max_Y == 0:
            max_Y = 1
        for i in range(len(self.clusters_train)):  # for each cluster
            temp_X = self.whole_data[self.clusters_train[i], 0:self.feather_count]
            temp_Y = self.whole_data[self.clusters_train[i], -1][:, np.newaxis]
            # Scale X and Y
            self.x_train.append(np.divide(temp_X, max_X))
            self.y_train.append(np.divide(temp_Y, max_Y))
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)

    def train_DaL_DNNs(self):
        config = []  # model config for each cluster
        lr_opt = [] # model learn rate for each cluster
        if self.test_mode: # default hyperparameters
            for i in range(self.clusters_train):
                temp_lr_opt = 0.123  # 临时学习率
                n_layer_opt = 3  # 临时层数
                lambda_f = 0.123  # 正则化参数
                # 初始化配置字典
                temp_config = {
                    'num_neuron': 128,  # 每层的神经元数量
                    'num_input': self.feather_count,  # 输入特征的数量
                    'num_layer': n_layer_opt,  # 神经网络的层数
                    'lambda': lambda_f,  # 正则化参数
                    'verbose': 0  # 不输出详细信息
                }
                # 将配置添加到 config 列表中
                config.append(temp_config)
                # 将学习率添加到 lr_opt 列表中
                lr_opt.append(temp_lr_opt)
        else: # hyper tune
            # todo: tune the hyper parameter
            pass

        for i in range(len(self.clusters_train)):
            self.x_train.append(self.whole_data[self.clusters_train[i], 0:self.feather_count])
            self.y_train.append(self.whole_data[self.clusters_train[i], -1])

        for i in range(len(self.clusters_train)):
            print('Training DNN for division {}... ({} samples)'.format(i + 1, len(self.clusters_train[i])))
            model = MLPSparseModel(config[i])
            model.build_train()
            model.train(self.x_train,self.y_train,lr_opt[i])
            self.models.append(model)




dal = DaL_Regressor("data/Apache_AllNumeric.csv")
