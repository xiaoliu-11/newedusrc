# @author lsg
# @date 2023/3/22
# @file chatgpt.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"C:\Users\Administrator\Desktop\os识别\dataos1.csv", header=1, )
x = data.iloc[:40000, 1:-1]  #
y = data.iloc[:40000, -1]  #
le = LabelEncoder()
newx = pd.DataFrame(x)
X = pd.get_dummies(newx)
# X = np.asarray(X.columns.astype(str), dtype=object)
Y = le.fit_transform(y)
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 初始化分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 初始化选择的数据点数量
n_queries = 20

# 初始化选择的数据点的索引
query_idx = np.zeros(n_queries, dtype=int)

# 训练分类器
clf.fit(X_train, y_train)

# 在测试集上评估分类器的准确率
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"初始分类器准确率: {accuracy:.2f}")

# 主动学习
for i in range(n_queries):
    # 用分类器预测测试集中每个样本的概率
    proba = clf.predict_proba(X_test)

    # 计算测试集中每个样本的不确定度
    uncertainty = np.zeros(len(X_test))
    for j in range(len(X_test)):
        uncertainty[j] = -np.max(proba[j])  # 使用负最大概率作为不确定度

    # 选择不确定度最大的样本
    query_idx[i] = np.argmax(uncertainty)

    # 将选择的样本添加到训练集中
    X_train = np.vstack((X_train, X_test[query_idx[i]]))
    y_train = np.append(y_train, y_test[query_idx[i]])

    # 在扩充后的训练集上重新训练分类器
    clf.fit(X_train, y_train)

    # 在测试集上评估分类器的准确率
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"第{i+1}次选择后的分类器准确率: {accuracy:.2f}")
