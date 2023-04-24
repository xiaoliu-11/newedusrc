# @author lsg
# @date 2023/3/14
# @file algorithm.py
import re
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, uncertainty_sampling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

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


def loadcsv(path):
    data = pd.read_csv(path, header=1,)
    x = data.iloc[:40000, 1:-1]  #
    y = data.iloc[:40000, -1]  #
    le = LabelEncoder()
    newx = pd.DataFrame(x)
    #X = pd.get_dummies(newx)
    #X = np.asarray(X.columns.astype(str), dtype=object)
    Y = le.fit_transform(y)
    # 分割数据集
    X_train, X_pool, y_train, y_pool = train_test_split(newx, Y, test_size=0.9, random_state=42)
    print("X_train")
    print(X_train)
    print("X_pool")
    print(X_pool)
    print("y_train")
    print(y_train)
    print("y_pool")
    print(y_pool)
    return X_train, X_pool, y_train, y_pool


# 读测试集
def loadcsv1(path):
    data = pd.read_csv(path, header=1,)
    x = data.iloc[40000:, 1:-1]  #
    y = data.iloc[40000:, -1]  #
    le = LabelEncoder()
    newx = pd.DataFrame(x)
    X = pd.get_dummies(newx)
    #X = np.asarray(X.columns.astype(str), dtype=object)
    Y = le.fit_transform(y)
    return  X,Y



def load_data(train,target):
    X_train,X_pool,y_train,y_pool \
        = train_test_split(train, target, test_size=0.7)
    print("X_train")
    print(X_train)
    print("X_pool")
    print(X_pool)
    print("y_train")
    print(y_train)
    print("y_pool")
    print(y_pool)
    return X_train,X_pool,y_train,y_pool


def activelearn(X_train,X_pool,y_train,y_pool):
    # 初始化分类器和向量化器
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    vectorizer = CountVectorizer()
    print(X_train.shape)
    print(y_train.shape)
    # 向量化训练集
    X_train_vec = vectorizer.fit_transform(X_train)
    print(X_train_vec.shape)
    learner = ActiveLearner(
        estimator=clf,
        query_strategy=uncertainty_sampling,
        X_training=X_train_vec, y_training=y_train
    )
    # 设置迭代次数
    n_queries = 20
    # 进行主动学习迭代
    for idx in range(n_queries):
        # 向量化池集
        X_pool_vec = vectorizer.transform(X_pool)
        # 使用主动学习器选择新的样本
        query_idx, query_instance = learner.query(X_pool_vec)
        # 根据标注请求
        y_label = y_pool[query_idx]
        learner.teach(query_instance.reshape(1, -1), y_label.reshape(1, ))
        # 从池中移除已选择的样本
        X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx)

    # 使用测试集进行评估
    X_test,y_test = loadcsv1(csvfile)
    X_test_vec = vectorizer.transform(X_test)
    y_pred = learner.predict(X_test_vec)
    print('Accuracy:', accuracy_score(y_test, y_pred))



if __name__ == '__main__':
    # filename = r"C:\Users\Administrator\Desktop\os识别\321.txt"
    csvfile = r"C:\Users\Administrator\Desktop\os识别\dataos1.csv"
    # #urlList = loadUrl(filename)
    # X_train,X_pool,y_train,y_pool = loadcsv(csvfile)
    # activelearn(X_train,X_pool,y_train,y_pool)

    filename = r"C:\Users\Administrator\Desktop\os识别\321.txt"
    # fileexcel = r"C:\Users\Administrator\Desktop\dataos.xlsx"
    urlList = loadUrl(filename)
    listos = []
    iplist = []
    saw_sig = []
    for i in urlList:
        if "syn" in i:
            # print(i)
            temp = i.split("|")
            for j in temp:
                if "os=" in j:
                    listos.append(j[3:])
                    # print(j)
                if "srv=" in j:
                    if len(j) < 20:
                        tm = j.split("/")[0]
                        iplist.append(tm)
                if "raw_sig=" in j:
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

    X_train,X_pool,y_train,y_pool = load_data(new_saw_sig, listos)
    activelearn(X_train, X_pool, y_train, y_pool)





