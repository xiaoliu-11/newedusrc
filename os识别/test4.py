# @author lsg
# @date 2023/4/22
# @file test4.py
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
from sklearn.preprocessing import LabelEncoder

def loadcsv1(path):
    data = pd.read_csv(path,header=None)
    X = data.iloc[:, 1:-1]  #
    X = X.to_numpy()
    y = data.iloc[:, -1]  #
    y = y.to_numpy()
    # X = pd.DataFrame(X)
    # print(X.shape)
    return  X,y




def avtice(X,y):
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 初始化训练集
    initial_idx = np.random.choice(range(len(X_train)), size=10, replace=False)
    X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]

    # 创建一个随机森林分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 训练分类器
    clf.fit(X_initial, y_initial)

    # 选择需要标记的样本
    n_queries = 100
    for i in range(n_queries):
        # 预测未标记样本
        X_unlabeled = np.delete(X_train, initial_idx, axis=0)
        y_pred = clf.predict(X_unlabeled)
        # 计算每个未标记样本的不确定性度量
        uncertainty = clf.predict_proba(X_unlabeled)
        uncertainty = np.max(uncertainty, axis=1)
        # 选择不确定性最高的样本进行标记
        query_idx = np.argmin(uncertainty)
        # 将新标记的样本添加到训练集中
        initial_idx = np.append(initial_idx, query_idx)
        X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
        # 重新训练分类器
        clf.fit(X_initial, y_initial)

    # 在测试集上评估分类器的性能
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)


if __name__ == '__main__':
    csvfile = r"C:\Users\Administrator\Desktop\os识别\dataos1.csv"
    X, y = loadcsv1(csvfile)
    avtice(X, y)