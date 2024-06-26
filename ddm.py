import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from river import drift
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import configs

def re_to_bool(val, threshold = configs.RESIDUAL_ERRORS_THRESHOLD):
    return 0 if val <= threshold else 1


def mean_absolute_percentage_error(y_true, y_pred):  # todo: how to choice a suitable one
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    y_true = y_true[non_zero_indices]
    y_pred = y_pred[non_zero_indices]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class DataProcessor:

    def __init__(self, data_path):
        self.train_count = 0
        self.test_count = 0
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path, encoding="utf-8")
        self.data.sample(200)
        (self.sample_count, self.feather_count) = self.data.shape
        self.feather_count -= 1

    def divide_dataset(self, rate=configs.TRAIN_TEST_SPLIT, seed=42):
        self.train_count = int(self.sample_count * rate)
        self.test_count = self.sample_count - self.train_count
        X = self.data.iloc[:, :self.feather_count]
        Y = self.data.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=rate, random_state=seed)
        return x_train, x_test, y_train, y_test

    def data_formulate(self):
        X = self.data.iloc[:, :12]
        Y = self.data.iloc[:, -1]
        return X, Y

    def get_whole_data(self):
        df = pd.read_csv('data/storm-obj1_feature6.csv')
        ndarray = df.values
        return ndarray

def detect_drift_in_same_environment(data_path):
    dp = DataProcessor(data_path)
    x_train, x_test, y_train, y_test = dp.divide_dataset()
    rfr = RandomForestRegressor()
    rfr.fit(x_train, y_train)
    y_pre = rfr.predict(x_test)
    #mmre = abs(y_pre - y_test) / y_test
    mmse = (abs(y_pre - y_test)) ^ 2
    PredPDF = pd.DataFrame({"实际值": y_test,
                            "预测值": y_pre,
                            "误差": mmse})
    print("[Message] Prediction Results on The Test Data Set for RandomForestRegressor:")
    print(PredPDF)
    # using PH test:
    print("PH TEST WITH ABSOLUTE ERROR...")
    phtest = drift.PageHinkley()
    for i, val in enumerate(mmse):
        phtest.update(val)
        if phtest.drift_detected:
            print(f"Change detected at index {i}, input value: {val}")

    print("ADWIN WITH ABSOLUTE ERROR...")
    adwin = drift.ADWIN()
    for i, val in enumerate(mmse):
        adwin.update(val)
        if adwin.drift_detected:
            print(f"Change detected at index {i}, input value: {val}")

    print("KSWIN WITH ABSOLUTE ERROR...")
    kswin = drift.KSWIN()
    for i, val in enumerate(mmse):
        kswin.update(val)
        if kswin.drift_detected:
            print(f"Change detected at index {i}, input value: {val}")

    print("DDM WITH ABSOLUTE ERROR...")
    ddm = drift.binary.DDM()
    for i, val in enumerate(mmse):
        if val > 0.5:
            ddm.update(False)
        if val <= 0.5:
            ddm.update(True)
        if ddm.drift_detected:
            print(f"Change detected at index {i}, input value: {val}")

    print("EDDM WITH ABSOLUTE ERROR...")
    eddm = drift.binary.EDDM()
    for i, val in enumerate(mmse):
        if val > 0.5:
            eddm.update(False)
        if val <= 0.5:
            eddm.update(True)
        if eddm.drift_detected:
            print(f"Change detected at index {i}, input value: {val}")


def detect_drift_in_different_environment(basedata, others: list):
    dp = DataProcessor(basedata)
    x_train, x_test, y_train, y_test = dp.divide_dataset()
    rfr = RandomForestRegressor()
    rfr.fit(x_train.values, y_train.values)
    print("NO drift sample count: " + str(dp.test_count))
    y_pre0 = rfr.predict(x_test.values)
    mae0 = abs(y_pre0 - y_test)
    ae0 = mean_absolute_error(
        y_test, y_pre0
    )
    mmre0 = abs(y_pre0 - y_test) / y_test
    #mre0 = mean_absolute_percentage_error(y_test, y_pre0)
    mmse0 = (y_pre0 - y_test) ** 2
    mse0 = mean_squared_error(y_test, y_pre0)
    print("in base dataset(without drift), the mre value is" + str(ae0))
    phtest = drift.PageHinkley(delta=0.002, threshold=10)
    adwin = drift.ADWIN()

    kswin = drift.KSWIN()
    ddm = drift.binary.DDM()
    eddm = drift.binary.EDDM()
    hddma = drift.binary.HDDM_A()  # DDM based on Hoeffding's bounds with moving average-test
    hddmw = drift.binary.HDDM_W()  # DDM based on Hoeffding's bounds with moving weighted average-test

    print("------base test set------")
    PredPDF = pd.DataFrame({"实际值": y_test,
                            "预测值": y_pre0,
                            "误差": mae0})
    print("[Message] Prediction Results on The Test Data Set for RandomForestRegressor:")
    print(PredPDF)
    for i, val in enumerate(mae0):
        phtest.update(val)
        if phtest.drift_detected:
            print(f"phChange detected at index {i}, input value: {val}")


        adwin.update(val)
        if adwin.drift_detected:
            print(f"adwinChange detected at index {i}, input value: {val}")


        kswin.update(val)
        if kswin.drift_detected:
            print(f"kswinChange detected at index {i}, input value: {val}")

        ddm.update(re_to_bool(val))
        if ddm.drift_detected:
            print(f"ddmChange detected at index {i}, input value: {val}")


        eddm.update(re_to_bool(val))
        if eddm.drift_detected:
            print(f"eddmChange detected at index {i}, input value: {val}")

        hddma.update(re_to_bool(val))
        if hddma.drift_detected:
            print(f"hddmaChange detected at index {i}, input value: {val}")

        hddmw.update(re_to_bool(val))
        if hddmw.drift_detected:
            print(f"hddmaChange detected at index {i}, input value: {val}")

    for dataset in others:
        # dp = DataProcessor(dataset)
        print("------in env:" + dataset.split('/')[-1]+"------")
        # X, Y = dp.data_formulate()
        df = pd.read_csv(dataset)
        (N, n) = df.shape
        dff = df.sample(int(N * 0.35))
        X = dff.values[:, :-1]
        Y = dff.values[:, -1]
        y_pre = rfr.predict(X)
        mae = abs(y_pre - Y)
        #mmre = abs(y_pre - Y) / Y
        mmse = (abs(y_pre - Y)) ** 2
        #mre = mean_absolute_percentage_error(Y, y_pre)
        mse = mean_squared_error(Y, y_pre)
        print("mre value is" + str(mse))
        PredPDF = pd.DataFrame({"实际值": Y,
                                "预测值": y_pre,
                                "误差": mae})
        print("[Message] Prediction Results on The Test Data Set for RandomForestRegressor:")
        print(PredPDF)
        print("PH TEST WITH ABSOLUTE ERROR...")

        for i, val in enumerate(mae):
            phtest.update(val)
            if phtest.drift_detected:
                print(f"Change detected at index {i}, input value: {val}")

        print("ADWIN WITH ABSOLUTE ERROR...")

        for i, val in enumerate(mae):
            adwin.update(val)
            if adwin.drift_detected:
                print(f"Change detected at index {i}, input value: {val}")

        print("KSWIN WITH ABSOLUTE ERROR...")

        for i, val in enumerate(mae):
            kswin.update(val)
            if kswin.drift_detected:
                print(f"Change detected at index {i}, input value: {val}")

        print("DDM WITH ABSOLUTE ERROR...")
        for i, val in enumerate(mae):
            result = re_to_bool(val)
            ddm.update(result)
            if ddm.drift_detected:
                print(f"Change detected at index {i}, input value: {val}")

        print("EDDM WITH ABSOLUTE ERROR...")
        for i, val in enumerate(mae):
            result = re_to_bool(val)
            eddm.update(result)
            if eddm.drift_detected:
                print(f"Change detected at index {i}, input value: {val}")

        print("HDDMA WITH ABSOLUTE ERROR...")
        for i, val in enumerate(mae):
            result = re_to_bool(val)
            hddma.update(result)
            if hddma.drift_detected:
                print(f"Change detected at index {i}, input value: {val}")

        print("HDDMW WITH ABSOLUTE ERROR...")
        for i, val in enumerate(mae):
            result = re_to_bool(val)
            hddmw.update(result)
            if hddmw.drift_detected:
                print(f"Change detected at index {i}, input value: {val}")


data_path = "data/storm-obj1_feature6.csv"
other_data1 = "data/storm-obj1_feature7.csv"
other_data2 = "data/storm-obj1_feature8.csv"
other_data3 = "data/storm-obj1_feature9.csv"
other_data4 = "data/storm-obj2_feature6.csv"
other_data5 = "data/storm-obj2_feature7.csv"
other_data6 = "data/storm-obj2_feature8.csv"
other_data7 = "data/storm-obj2_feature9.csv"

# detect_drift_in_different_environment(data_path, [other_data1, other_data2, other_data3, other_data4, other_data5, other_data6, other_data7])

# detect_drift_in_different_environment(data_path, [other_data7, other_data2, other_data3, other_data4, other_data5, other_data6, other_data1])


#  目前没有添加适应算法
#  训练新模型的时候需要添加到detector中
#  detect_drift_in_same_environment(data_path)
"""envlist = [other_data1, other_data2, other_data3, other_data4, other_data5, other_data6, other_data7]
for env in envlist:
    detect_drift_in_different_environment(basedata=data_path, others=[env])"""

data_path = "data/data1/sac_2.csv"
other_data1 = "data/data1/sac_4.csv"
other_data2 = "data/data1/sac_5.csv"
other_data3 = "data/data1/sac_6.csv"
other_data4 = "data/data1/sac_7.csv"
other_data5 = "data/data1/sac_8.csv"
other_data6 = "data/data1/sac_9.csv"
"""envlist = [data_path, other_data1, other_data2, other_data3, other_data4, other_data5, other_data6]

for env in envlist:
    detect_drift_in_different_environment(basedata=data_path, others=[env])"""

data_path = "data/data2/x264_0.csv"
other_data1 = "data/data2/x264_1.csv"
other_data2 = "data/data2/x264_2.csv"
other_data3 = "data/data2/x264_3.csv"
other_data4 = "data/data2/x264_4.csv"
other_data5 = "data/data2/x264_5.csv"
other_data6 = "data/data2/x264_6.csv"

envlist = [data_path, other_data1, other_data2, other_data3, other_data4, other_data5, other_data6]

for env in envlist:
    detect_drift_in_different_environment(basedata=data_path, others=[env])