import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.tree import DecisionTreeRegressor

import configs
from utils.general import build_model, recursive_dividing
from numpy import recfromcsv


# todo:  2. clustering_indexs and clustering_index_all 3. total index and training index(no zero)  4. if N_train to big?

def get_whole_data(path):
    df = pd.read_csv(path)
    ndarray = df.values
    return ndarray


class DaL_Regressor:

    def __init__(self, data_path):
        self.RFC = None
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
        self.clusters = self.get_cluster(self.whole_data[self.train_index])  # todo: right？
        self.train_RFC()
        pass

    # todo: need to test the function
    def get_cluster(self, whole_train_data):
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

    def generate_cluster_label(self):

        return None

    def data_smo(self):
        x_som = self.whole_data[self.train_index, 0:self.feather_count]
        y_som = None

    def train_RFC(self):
        self.RFC = RandomForestClassifier(criterion='gini', random_state=0)
        pass


dal = DaL_Regressor("data/Apache_AllNumeric.csv")
