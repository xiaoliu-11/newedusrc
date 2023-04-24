# @author lsg
# @date 2023/4/22
# @file test2.py
import numpy as np
import re
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

import pandas as pd
from sklearn.preprocessing import LabelEncoder

path = r"C:\Users\Administrator\Desktop\os识别\dataos1.csv"


def loadUrl(fileName):
    f = open(fileName, encoding='utf-8')
    urlList = []
    for line in f:
        urlList.append(line.strip())
    f.close()
    return urlList



def loadcsv(path):
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
    print(new_saw_sig.reset_index(drop=True).values)
    return  new_saw_sig.reset_index(drop=True).values, listos


# 读取数据集并进行预处理
X, y = loadcsv(path) # 加载数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42) # 划分训练集和测试集
X_pool, y_pool = X_train, y_train # 初始化未标记数据集为训练集

# 初始化分类器和主动学习器
classifier = RandomForestClassifier() # 选择分类器
learner = ActiveLearner(
    estimator=classifier,
    query_strategy=uncertainty_sampling,
    X_training=X_train, y_training=y_train
)

# 训练主动学习器
while len(X_pool) > 0: # 当未标记数据集非空
    query_idx, query_instance = learner.query(X_pool) # 获取具有最高不确定性的未标记实例
    learner.teach(X_pool[query_idx], y_pool[query_idx]) # 标记实例并训练模型
    X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx) # 从未标记数据集中删除已标记实例

# 在测试集上评估模型性能
accuracy = learner.score(X_test, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100))

