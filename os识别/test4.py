# @author lsg
# @date 2023/4/22
# @file test4.py
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")
font = {'family':'MicroSoft YaHei'}
plt.rc("font",**font)

def loadcsv1(path):
    data = pd.read_csv(path,header=None)
    X = data.iloc[:, 1:-1]  #
    X = X.to_numpy()
    y = data.iloc[:, -1]  #
    y = y.to_numpy()
    # X = pd.DataFrame(X)
    # print(X.shape)
    return  X,y




def avtice(clf,X,y):
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 初始化训练集 从初始训练集中随机抽取5000分数据
    initial_idx = np.random.choice(range(len(X_train)), size=2000, replace=False)
    # print("initial_idx")
    # print(initial_idx)
    X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
    # print("X_initial")
    # print(X_initial)
    # print("y_initial")
    # print(y_initial)
    # 训练分类器
    clf.fit(X_initial, y_initial)

    # 选择需要标记的样本
    n_queries = 50
    for i in range(10):
        # 预测未标记样本
        X_unlabeled = np.delete(X_train, initial_idx, axis=0)
        y_pred = clf.predict(X_unlabeled)
        # 计算每个未标记样本的不确定性度量
        # 预测值为，获得所有结果的概率。（有多少个分类结果，每行就有多少个概率，以至于它对每个结果都有一个可能，如0、1就有两个概率）
        uncertainty = clf.predict_proba(X_unlabeled)
        uncertainty = np.max(uncertainty, axis=1)
        # 选择不确定性最高的样本进行标记
        # 给出axis方向最小值的下标
        query_idx = np.argmin(uncertainty)
        # 将新标记的样本添加到训练集中
        initial_idx = np.append(initial_idx, query_idx)
        X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
        # 重新训练分类器
        clf.fit(X_initial, y_initial)

    # 在测试集上评估分类器的性能
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precison = precision_score(y_test, y_pred,average='weighted')
    rescore = recall_score(y_test, y_pred,average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print('Accuracy:', accuracy)
    print('precison:', precison)
    print('rescore:', rescore)
    print('f1:', f1)
    print("-----------------------------")



if __name__ == '__main__':
    csvfile = r"C:\Users\Administrator\Desktop\os识别\dataos1.csv"
    X, y = loadcsv1(csvfile)
    # 创建一个随机森林分类器
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    # knn
    knn =KNeighborsClassifier(n_neighbors=7)
    # dt
    dt = DecisionTreeClassifier(criterion="gini",max_depth=10)
    # svm
    svm = SVC(C=130, gamma=13, decision_function_shape="ovo",probability=True)
    #逻辑回归
    lr_l1 = LogisticRegression(penalty="l1", C=1.0, solver="liblinear")

    avtice(clf, X, y)
    avtice(knn, X, y)
    avtice(dt, X, y)
    avtice(svm, X, y)
    avtice(lr_l1, X, y)