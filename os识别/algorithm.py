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
        = train_test_split(train, target, test_size=0.7)
    print("x_train")
    print(x_train)
    print("x_test")
    print(x_test)
    print("y_train")
    print(y_train)
    print("y_test")
    print(y_test)
    return x_train, x_test, y_train, y_test

# 2.1决策树
def dt(data):
    x_train, x_test, y_train, y_test = data
    estimator = DecisionTreeClassifier(criterion="gini", max_depth=2)
    estimator.fit(x_train, y_train)
    t_score = estimator.score(x_test, y_test)
    print(f"dt test score:{t_score}")

# 2.3 使用knn训练模型
def knn(data):
    x_train, x_test, y_train, y_test = data
    estimator = KNeighborsClassifier(n_neighbors=1)
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
    filename = r"C:\Users\Administrator\Desktop\os识别\321.txt"
    #fileexcel = r"C:\Users\Administrator\Desktop\dataos.xlsx"
    urlList = loadUrl(filename)
    listos = []
    iplist = []
    saw_sig = []
    for i in urlList:
        if "syn" in i:
            #print(i)
            temp = i.split("|")
            for j in temp:
                if "os=" in j:
                    listos.append(j[3:])
                    #print(j)
                if "srv=" in j:
                    if len(j) < 20:
                        tm = j.split("/")[0]
                        iplist.append(tm)
                if "raw_sig=" in j :
                    saw_sig.append(re.split(':|,', j[8:]))

    # 将标签编码为数字
    le = LabelEncoder()
    listos = le.fit_transform(listos)
    print("listos")
    print(listos)
    print(len(listos))
    saw_sig = pd.DataFrame(saw_sig)
    print("saw_sig")
    print(saw_sig)

    new_saw_sig = pd.get_dummies(saw_sig)
    print("new_saw_sig")
    print(new_saw_sig)

    data = load_data(new_saw_sig,listos)
    dt(data)
    knn(data)
    bayes(data)
