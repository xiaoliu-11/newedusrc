# @author lsg
# @date 2023/3/22
# @file 备份.py
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

def load_data(train,target):
    x_train, x_test, y_train, y_test \
        = train_test_split(train, target, test_size=0.3)
    print("x_train")
    print(x_train)
    print("x_test")
    print(x_test)
    print("y_train")
    print(y_train)
    print("y_test")
    print(y_test)
    return x_train, x_test, y_train, y_test

def loadcsv(path):
    data = pd.read_csv(path, header=1,)
    x = data.iloc[:, 1:-1]  #
    y = data.iloc[:, -1]  #

    le = LabelEncoder()
    newx = pd.DataFrame(x)
    X = pd.get_dummies(newx)
    #X = np.asarray(X.columns.astype(str), dtype=object)
    Y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)
    print("X_train")
    print(X_train)
    print("X_test")
    print(X_test)
    print("y_train")
    print(y_train)
    print("y_test")
    print(y_test)
    return X_train, X_test, y_train, y_test


# 2.1决策树
def dt(data):
    x_train, x_test, y_train, y_test = data
    estimator = DecisionTreeClassifier(criterion="gini", )
    estimator.fit(x_train, y_train)
    t_score = estimator.score(x_test, y_test)
    print(f"dt test score:{t_score}")

# 2.加载数据
X_training = "标注数据"
y_training = "标注数据的标签"
X_pool = "未标注数据"


def activelearn(X_training,y_training,X_pool):
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=entropy_sampling,
        X_training=X_training,
        y_training=y_training
    )
    query_idx, query_inst = learner.query(X_pool)

    learner.teach(X_pool[query_idx], y_new)


if __name__ == '__main__':
    filename = r"C:\Users\Administrator\Desktop\os识别\321.txt"
    csvfile = r"C:\Users\Administrator\Desktop\os识别\dataos1.csv"
    urlList = loadUrl(filename)
    data1 = loadcsv(csvfile)
    print("data1")
    print(data1)
    dt(data1)




    # listos = []
    # iplist = []
    # saw_sig = []
    # for i in urlList:
    #     if "syn" in i :
    #         #print(i)
    #         temp = i.split("|")
    #         for j in temp:
    #             if "os=" in j:
    #                 listos.append(j[3:])
    #                 #print(j)
    #             if "srv=" in j:
    #                 if len(j) < 20:
    #                     tm = j.split("/")[0]
    #                     iplist.append(tm)
    #             if "raw_sig=" in j :
    #                 saw_sig.append(re.split(':|,', j[8:]))
    #
    # # 将标签编码为数字
    # le = LabelEncoder()
    # listos = le.fit_transform(listos)
    # print("listos")
    # print(listos)
    # print(len(listos))
    # saw_sig = pd.DataFrame(saw_sig)
    # print("saw_sig")
    # print(saw_sig)
    #
    # new_saw_sig = pd.get_dummies(saw_sig)
    # print("new_saw_sig")
    # print(new_saw_sig)
    # data = load_data(new_saw_sig,listos)

