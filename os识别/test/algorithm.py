# @author lsg
# @date 2023/3/14
# @file algorithm.py
import re
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling


import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import sklearn.svm as svm
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.naive_bayes import GaussianNB
import joblib


# 1.打开url文件，读取
def loadUrl(fileName):
    f = open(fileName, encoding='utf-8')
    urlList = []
    for line in f:
        urlList.append(line.strip())
    f.close()
    return urlList


def loadcsv1(path):
    data = pd.read_csv(path,header=None)
    X = data.iloc[:, 1:-1]  #
    X = X.to_numpy()
    y = data.iloc[:, -1]  #
    y = y.to_numpy()
    # X = pd.DataFrame(X)
    # print(X.shape)
    return  X,y



def load_data(train,target):
    x_train, x_test, y_train, y_test \
        = train_test_split(train, target, test_size=0.3)
    return x_train, x_test, y_train, y_test

# 2.1决策树
def dt(data):
    x_train, x_test, y_train, y_test = data
    estimator = DecisionTreeClassifier(criterion="gini", max_depth=16)
    estimator.fit(x_train, y_train)
    t_score = estimator.score(x_test, y_test)
    print(f"dt test score:{t_score}")

# 2.3 使用knn训练模型
def knn(data):
    x_train, x_test, y_train, y_test = data
    estimator = KNeighborsClassifier(n_neighbors=11)
    estimator.fit(x_train, y_train)
    t_score = estimator.score(x_test, y_test)
    print(f"knn test score:{t_score}")


# 2.7 使用bayes
def bayes(data):
    x_train, x_test, y_train, y_test = data
    estimator = GaussianNB()
    estimator.fit(x_train, y_train)
    t_score = estimator.score(x_test, y_test)
    print(f"bayes test score:{t_score}")



if __name__ == '__main__':
    csvfile = r"C:\Users\Administrator\Desktop\os识别\dataos1.csv"
    X, y = loadcsv1(csvfile)
    data = load_data(X,y)
    dt(data)
    knn(data)
    bayes(data)
