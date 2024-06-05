import random

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import configs
from ddm import DataProcessor
from utils.general import build_model, recursive_dividing
from numpy import recfromcsv


def get_whole_data(path):
    df = pd.read_csv(path)
    ndarray = df.values
    return ndarray

class DaL_Regressor:

    def __init__(self, data_path):
        self.whole_data = None
        self.other_env_data = None
        self.data_path = data_path
        self.whole_data = get_whole_data(data_path)
        # todo: remove 0 performance samples
        self.train_count = int(len(self.whole_data)*configs.TRAIN_TEST_SPLIT)
        self.test_count = len(self.whole_data) - self.train_count
        temp_total_index = range(len(list(self.whole_data)))
        self.test_index = random.sample(list(temp_total_index), self.test_count)
        self.train_index = np.setdiff1d(temp_total_index, self.test_index)  # test: divide successfully
        pass


    def feature_weight(self):

        pass

    # todo: need to test the function
    def get_cluster(self, whole_train_data):
        DT = DecisionTreeRegressor(random_state=3, criterion='squared_error', splitter='best')
        X = whole_train_data[:, :-1]
        Y = whole_train_data[:, -1]
        DT.fit(X, Y)
        tree_ = DT.tree_
        cluster_indexes_all = [] # for recursive algorithms
        cluster_indexes_all = recursive_dividing(0, 1, tree_, X, self.train_index, configs.MAX_DEPTH,
                                                 configs.MIN_SAMPLES, cluster_indexes_all)

        return cluster_indexes_all



dal = DaL_Regressor("data/storm-obj1_feature6.csv")