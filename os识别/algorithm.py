# @author lsg
# @date 2023/3/14
# @file algorithm.py
import re
import warnings

from matplotlib import pyplot as plt
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
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
warnings.filterwarnings("ignore")
font = {'family':'MicroSoft YaHei'}
plt.rc("font",**font)




accuracyList = []
precisionList = []
recallList = []
f1List = []
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


# 2.1决策树
def dt(data):
    x_train, x_test, y_train, y_test = data
    estimator = DecisionTreeClassifier(criterion="gini", max_depth=8)
    estimator.fit(x_train, y_train)
    t_score = estimator.score(x_test, y_test)
    p_score = precision_score(y_test, estimator.predict(x_test), average="weighted")
    r_score = recall_score(y_test, estimator.predict(x_test), average="weighted")
    f_score = f1_score(y_test, estimator.predict(x_test), average="weighted")
    print(f"dt t_score score:{t_score}")
    print(f"dt p_score score:{p_score}")
    print(f"dt r_score score:{r_score}")
    print(f"dt f_score score:{f_score}")
    accuracyList.append(t_score)
    precisionList.append(p_score)
    recallList.append(r_score)
    f1List.append(f_score)


def rf(data):
    x_train, x_test, y_train, y_test = data
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    estimator.fit(x_train, y_train)
    t_score = estimator.score(x_test, y_test)
    p_score = precision_score(y_test, estimator.predict(x_test), average="weighted")
    r_score = recall_score(y_test, estimator.predict(x_test), average="weighted")
    f_score = f1_score(y_test, estimator.predict(x_test), average="weighted")
    print(f"rf t_score score:{t_score}")
    print(f"rf p_score score:{p_score}")
    print(f"rf r_score score:{r_score}")
    print(f"rf f_score score:{f_score}")
    accuracyList.append(t_score)
    precisionList.append(p_score)
    recallList.append(r_score)
    f1List.append(f_score)


# 2.3 使用knn训练模型
def knn(data):
    x_train, x_test, y_train, y_test = data
    estimator = KNeighborsClassifier(n_neighbors=5)
    estimator.fit(x_train, y_train)
    t_score = estimator.score(x_test, y_test)
    p_score = precision_score(y_test, estimator.predict(x_test), average="weighted")
    r_score = recall_score(y_test, estimator.predict(x_test), average="weighted")
    f_score = f1_score(y_test, estimator.predict(x_test), average="weighted")
    print(f"knn t_score score:{t_score}")
    print(f"knn p_score score:{p_score}")
    print(f"knn r_score score:{r_score}")
    print(f"knn f_score score:{f_score}")
    accuracyList.append(t_score)
    precisionList.append(p_score)
    recallList.append(r_score)
    f1List.append(f_score)


def svm(data):
    x_train, x_test, y_train, y_test = data
    # svm
    estimator = SVC(C=130, gamma=13, decision_function_shape="ovo", probability=True)
    estimator.fit(x_train, y_train)
    t_score = estimator.score(x_test, y_test)
    p_score = precision_score(y_test, estimator.predict(x_test), average="weighted")
    r_score = recall_score(y_test, estimator.predict(x_test), average="weighted")
    f_score = f1_score(y_test, estimator.predict(x_test), average="weighted")
    print(f"svm t_score score:{t_score}")
    print(f"svm p_score score:{p_score}")
    print(f"svm r_score score:{r_score}")
    print(f"svm f_score score:{f_score}")
    accuracyList.append(t_score)
    precisionList.append(p_score)
    recallList.append(r_score)
    f1List.append(f_score)


# 2.7 使用bayes
def lr(data):
    x_train, x_test, y_train, y_test = data
    estimator = LogisticRegression(penalty="l1", C=1.0, solver="liblinear")
    estimator.fit(x_train, y_train)
    t_score = estimator.score(x_test, y_test)
    p_score = precision_score(y_test, estimator.predict(x_test), average="weighted")
    r_score = recall_score(y_test, estimator.predict(x_test), average="weighted")
    f_score = f1_score(y_test, estimator.predict(x_test), average="weighted")
    print(f"lr t_score score:{t_score}")
    print(f"lr p_score score:{p_score}")
    print(f"lr r_score score:{r_score}")
    print(f"lr f_score score:{f_score}")
    accuracyList.append(t_score)
    precisionList.append(p_score)
    recallList.append(r_score)
    f1List.append(f_score)


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
    print(set(listos))
    print(len(set(listos)))
    print(len(listos))
    saw_sig = pd.DataFrame(saw_sig)
    print("saw_sig")
    print(saw_sig)

    new_saw_sig = pd.get_dummies(saw_sig)
    print("new_saw_sig")
    print(new_saw_sig)

    data = load_data(new_saw_sig,listos)
    timelist = []
    start = time.time()
    dt(data)
    a = time.time()
    timelist.append(a - start)
    rf(data)
    b = time.time()
    timelist.append(b - a)
    knn(data)
    c = time.time()
    timelist.append(c - b)
    svm(data)
    d = time.time()
    timelist.append(d - c)
    lr(data)
    e = time.time()
    timelist.append(e - d)

    # 3.2 柱状图
    # size = 5
    # x = np.arange(size)
    # total_width, n = 0.8, 4
    # width = total_width / n
    # x = x - (total_width - width) / 2
    # plt.figure(figsize=(20, 10), dpi=80)
    # plt.bar(x, accuracyList, width=width, label='准确率')
    # plt.bar(x + width,precisionList, width=width, label='精确率')
    # plt.bar(x + 2 * width, recallList, width=width, label='召回率')
    # plt.bar(x + 3 * width, f1List, width=width, label='F1值')
    # plt.xticks(x, ['Decision Tree', 'Random Forest', 'KNN', 'SVM', 'Logistic Regression'],fontsize=20)
    # y = np.arange(0, 1.1, 0.1)
    # xx = np.arange(5)
    # plt.xticks( xx,['Decision Tree', 'Random Forest', 'KNN', 'SVM', 'Logistic Regression'],fontsize=24)
    # plt.yticks( y,fontsize=24)
    # plt.ylabel("准确率",fontsize=24)
    # plt.legend(bbox_to_anchor=(1, 0.9), prop={'size': 20})
    # plt.show()

    # # 时间图
    print("timlist",timelist)
    # 时间柱状图
    x = ['Decision Tree', 'Random Forest', 'KNN', 'SVM', 'Logistic Regression']
    plt.bar(x,timelist)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.ylabel("单位：s")
    plt.show()

