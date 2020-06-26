import pandas as pd
import numpy as np
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN


def all_1():
    df = pd.read_csv("D:/data/abalone_benchmark.csv")
    X1 = df['original.label'].values.reshape(-1, 1)
    X2 = df['diff.score'].values.reshape(-1, 1)
    X3 = df['V1'].values.reshape(-1, 1)
    X4 = df['V2'].values.reshape(-1, 1)
    X5 = df['V3'].values.reshape(-1, 1)
    X6 = df['V4'].values.reshape(-1, 1)
    X7 = df['V5'].values.reshape(-1, 1)
    X8 = df['V6'].values.reshape(-1, 1)
    X9 = df['V7'].values.reshape(-1, 1)
    X = np.concatenate((X1, X2, X3, X4, X5, X6, X7, X8, X9), axis=1)
    random_state = np.random.RandomState(33)
    outliers_fraction = 0.15

    classifiers = {
        "Cluster-based Local Outiler Factor (CBLOF)": CBLOF(contamination=outliers_fraction, check_estimator=False,
                                                            random_state=random_state),
        "Histogram-base Outlier Detection(HBOS)": HBOS(contamination=outliers_fraction),
        # "Isolation Forest": IForest(contamination=outliers_fraction, random_state=random_state),
        "KNN": KNN(contamination=outliers_fraction),
        "Average KNN": KNN(method='mean', contamination=outliers_fraction)
    }

    # 逐一比较模型
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        clf.fit(X)
        n_inliers = 0
        n_outliers = 0
        y_pred = clf.predict(X)
        outliers = []
        for x, y in zip(X, y_pred):
            x = np.array(x).reshape(1, -1)
            if int(y) == 0:
                n_inliers += 1
            else:
                n_outliers += 1
                outliers.append(x)

        print("模型 %s 检测到的" % clf_name, "离群点有 ", n_outliers, "非离群点有", n_inliers)
        print("离群点为：")
        i = 0
        for x in outliers[0:50]:
            print(x[0].tolist(), end="  ")
            i += 1
            if i % 5 == 0:
                print()
        print()
        print()


def all_2():
    df = pd.read_csv("D:/data/wine_benchmark.csv")
    X1 = df['original.label'].values.reshape(-1, 1)
    X2 = df['diff.score'].values.reshape(-1, 1)
    X3 = df['fixed.acidity'].values.reshape(-1, 1)
    X4 = df['volatile.acidity'].values.reshape(-1, 1)
    X5 = df['citric.acid'].values.reshape(-1, 1)
    X6 = df['residual.sugar'].values.reshape(-1, 1)
    X7 = df['chlorides'].values.reshape(-1, 1)
    X8 = df['free.sulfur.dioxide'].values.reshape(-1, 1)
    X9 = df['total.sulfur.dioxide'].values.reshape(-1, 1)
    X10 = df['density'].values.reshape(-1, 1)
    X11 = df['pH'].values.reshape(-1, 1)
    X12 = df['sulphates'].values.reshape(-1, 1)
    X13 = df['alcohol'].values.reshape(-1, 1)

    X = np.concatenate((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13), axis=1)
    random_state = np.random.RandomState(42)
    outliers_fraction = 0.15

    classifiers = {
        "Cluster-based Local Outiler Factor (CBLOF)": CBLOF(contamination=outliers_fraction, check_estimator=False,
                                                            random_state=random_state),
        "Histogram-base Outlier Detection(HBOS)": HBOS(contamination=outliers_fraction),
        # "Isolation Forest": IForest(contamination=outliers_fraction, random_state=random_state),
        "KNN": KNN(contamination=outliers_fraction),
        "Average KNN": KNN(method='mean', contamination=outliers_fraction)
    }

    # 逐一比较模型
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        clf.fit(X)
        n_inliers = 0
        n_outliers = 0
        y_pred = clf.predict(X)
        outliers = []
        for x, y in zip(X, y_pred):
            x = np.array(x).reshape(1, -1)
            if int(y) == 0:
                n_inliers += 1
            else:
                n_outliers += 1
                outliers.append(x)

        print("模型 %s 检测到的" % clf_name, "离群点有 ", n_outliers, "非离群点有", n_inliers)
        print("离群点为：")
        i = 0
        for x in outliers[0:50]:
            print(x[0].tolist(), end="  ")
            i += 1
            if i % 5 == 0:
                print()
        print()
        print()


ss = 1
if ss == 0:
    all_1()
else:
    all_2()
