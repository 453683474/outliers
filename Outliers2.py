import pandas as pd
import numpy as np
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.lof import LOF
from scipy import stats
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm


def all_1():
    df = pd.read_csv("D:/data/abalone_benchmark.csv")
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[['original.label', 'diff.score', ]] = scaler.fit_transform(df[['original.label', 'diff.score']])
    df[['original.label', 'diff.score']].head()
    X1 = df['original.label'].values.reshape(-1, 1)
    X2 = df['diff.score'].values.reshape(-1, 1)
    X = np.concatenate((X1, X2), axis=1)
    random_state = np.random.RandomState(42)
    outliers_fraction = 0.15

    classifiers = {
        "Cluster-based Local Outiler Factor (CBLOF)": CBLOF(contamination=outliers_fraction, check_estimator=False,
                                                            random_state=random_state),
        "Histogram-base Outlier Detection(HBOS)": HBOS(contamination=outliers_fraction),
        "Isolation Forest": IForest(contamination=outliers_fraction, random_state=random_state),
        "KNN": KNN(contamination=outliers_fraction),
        "Average KNN": KNN(method='mean', contamination=outliers_fraction)
    }

    xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))

    for i, (clf_name, clf) in enumerate(classifiers.items()):
        clf.fit(X)
        scores_pred = clf.decision_function(X) * -1
        # 预测数据点是否为离群点
        y_pred = clf.predict(X)
        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)
        plt.figure(figsize=(10, 10))

        dfx = df
        dfx['outlier'] = y_pred.tolist()
        IX1 = np.array(dfx['original.label'][dfx['outlier'] == 0]).reshape(-1, 1)
        IX2 = np.array(dfx['diff.score'][dfx['outlier'] == 0]).reshape(-1, 1)

        # OX1 离群点的特征1，OX2离群点特征2
        OX1 = np.array(dfx['original.label'][dfx['outlier'] == 1]).reshape(-1, 1)
        OX2 = np.array(dfx['diff.score'][dfx['outlier'] == 1]).reshape(-1, 1)

        print("模型 %s 检测到的" % clf_name, "离群点有 ", n_outliers, "非离群点有", n_inliers)

        threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        z = z.reshape(xx.shape)
        # 最小离群得分和阈值之间的点 使用绿色填充
        plt.contourf(xx, yy, z, levels=np.linspace(z.min(), threshold, 7), cmap=plt.cm.Greens_r)
        # 离群得分等于阈值的数据点 使用红色填充
        a = plt.contour(xx, yy, z, levels=[threshold], linewidths=2, colors='red')
        # 离群得分在阈值和最大离群得分之间的数据 使用橘色填充
        plt.contourf(xx, yy, z, levels=[threshold, z.max()], colors='orange')
        b = plt.scatter(IX1, IX2, c='white', s=20, edgecolor='k')
        c = plt.scatter(OX1, OX2, c='black', s=20, edgecolor='k')
        plt.axis('tight')
        plt.legend(
            [a.collections[0], b, c],
            ['learned decision function', 'inliers', 'outliers'],
            prop=mfm.FontProperties(size=20),
            loc=2
        )
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.title(clf_name)
        plt.show()


def all_2():
    df = pd.read_csv("D:/data/wine_benchmark.csv")
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[['original.label', 'diff.score', ]] = scaler.fit_transform(df[['original.label', 'diff.score']])
    df[['original.label', 'diff.score']].head()
    X1 = df['original.label'].values.reshape(-1, 1)
    X2 = df['diff.score'].values.reshape(-1, 1)
    X = np.concatenate((X1, X2), axis=1)
    random_state = np.random.RandomState(42)
    outliers_fraction = 0.15

    classifiers = {
        'Feature Bagging': FeatureBagging(LOF(n_neighbors=35), contamination=outliers_fraction, check_estimator=False,
                                          random_state=random_state),
        "Cluster-based Local Outiler Factor (CBLOF)": CBLOF(contamination=outliers_fraction, check_estimator=False,
                                                            random_state=random_state),
        "Histogram-base Outlier Detection(HBOS)": HBOS(contamination=outliers_fraction),
        "Isolation Forest": IForest(contamination=outliers_fraction, random_state=random_state),
    }

    xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))

    for i, (clf_name, clf) in enumerate(classifiers.items()):
        clf.fit(X)
        scores_pred = clf.decision_function(X) * -1
        # 预测数据点是否为离群点
        y_pred = clf.predict(X)
        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)
        plt.figure(figsize=(10, 10))

        dfx = df
        dfx['outlier'] = y_pred.tolist()
        IX1 = np.array(dfx['original.label'][dfx['outlier'] == 0]).reshape(-1, 1)
        IX2 = np.array(dfx['diff.score'][dfx['outlier'] == 0]).reshape(-1, 1)

        # OX1 离群点的特征1，OX2离群点特征2
        OX1 = np.array(dfx['original.label'][dfx['outlier'] == 1]).reshape(-1, 1)
        OX2 = np.array(dfx['diff.score'][dfx['outlier'] == 1]).reshape(-1, 1)

        print("模型 %s 检测到的" % clf_name, "离群点有 ", n_outliers, "非离群点有", n_inliers)

        threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        z = z.reshape(xx.shape)
        # 最小离群得分和阈值之间的点 使用绿色填充
        plt.contourf(xx, yy, z, levels=np.linspace(z.min(), threshold, 7), cmap=plt.cm.Greens_r)
        # 离群得分等于阈值的数据点 使用红色填充
        a = plt.contour(xx, yy, z, levels=[threshold], linewidths=2, colors='red')
        # 离群得分在阈值和最大离群得分之间的数据 使用橘色填充
        plt.contourf(xx, yy, z, levels=[threshold, z.max()], colors='orange')
        b = plt.scatter(IX1, IX2, c='white', s=20, edgecolor='k')
        c = plt.scatter(OX1, OX2, c='black', s=20, edgecolor='k')
        plt.axis('tight')
        plt.legend(
            [a.collections[0], b, c],
            ['learned decision function', 'inliers', 'outliers'],
            prop=mfm.FontProperties(size=20),
            loc=2
        )
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.title(clf_name)
        plt.show()


ss = 1
if ss == 0:
    all_1()
else:
    all_2()