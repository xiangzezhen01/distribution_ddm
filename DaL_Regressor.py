from sklearn.tree import DecisionTreeRegressor
from ddm import DataProcessor
from utils.general import build_model, recursive_dividing

class DaL_Regressor:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        pass

    def feature_weight(self):
        pass
    def devide_set(self, X, Y):
        DT = DecisionTreeRegressor(random_state=3, criterion='squared_error', splitter='best')
        DT.fit(X, Y)
        tree_ = DT.tree_
        #cluster_indexes_all = recursive_dividing(0, 1, tree_, X, non_zero_indexes, max_depth, min_samples,
        #                                         cluster_indexes_all)



