import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from imblearn.over_sampling import SMOTE
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
        self.test_mode = True  # if tune hyper-parameter? False->tune True->no
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
        self.x_smo = None
        self.y_smo = None
        self.data_smo()
        self.RFC = None
        self.train_RFC()
        self.generate_cluster_test_label()

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



dal = DaL_Regressor("data/Apache_AllNumeric.csv")
