# @author lsg
# @date 2023/5/2
# @file zhudongxuexi2.py
import re
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")
font = {'family':'MicroSoft YaHei'}
plt.rc("font",**font)

def load_data(train,target):
    x_train, x_test, y_train, y_test \
        = train_test_split(train, target, test_size=0.7)
    # print("x_train")
    # print(x_train)
    # print("x_test")
    # print(x_test)
    # print("y_train")
    # print(y_train)
    # print("y_test")
    # print(y_test)
    return x_train, x_test, y_train, y_test


def loadUrl(fileName):
    f = open(fileName, encoding='utf-8')
    urlList = []
    for line in f:
        urlList.append(line.strip())
    f.close()
    return urlList




accuracyList = []
precisionList = []
recallList = []
f1List = []
itlist = []
def avtice(clf,X,y):
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # X_tarin 数量 17046

    # 初始化训练集 从初始训练集中随机抽取500分数据
    # 17，170，852，1704，3409
    initial_idx = np.random.choice(range(len(X_train)), size=17, replace=False)
    # print("initial_idx")
    # print(initial_idx)
    X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
    # print("X_initial")
    # print(X_initial)
    # print("y_initial")
    # print(y_initial)
    # 训练分类器
    clf.fit(X_initial, y_initial)
    for i in range(20):
        # 预测未标记样本
        X_unlabeled = np.delete(X_train, initial_idx, axis=0)
        print(X_unlabeled)
        print(len(X_unlabeled))
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
        y_pred = clf.predict(X_test)
        tmp = accuracy_score(y_test, y_pred)
    itlist.append(tmp)
    print(itlist)

    # # 在测试集上评估分类器的性能
    # y_pred = clf.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # precison = precision_score(y_test, y_pred,average='weighted')
    # rescore = recall_score(y_test, y_pred,average='weighted')
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print('Accuracy:', accuracy)
    # print('precison:', precison)
    # print('rescore:', rescore)
    # print('f1:', f1)
    # print("-----------------------------")
    # accuracyList.append(accuracy)
    # precisionList.append(precison)
    # recallList.append(rescore)
    # f1List.append(f1)




if __name__ == '__main__':
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
    print(len(listos))
    # 将标签编码为数字
    le = LabelEncoder()
    y = le.fit_transform(listos)
    print("Linux 3.x", y.tolist().count(0))
    print("Windows 7 or 8", y.tolist().count(1))
    print("Linux 2.2.x", y.tolist().count(2))
    print("Windows NT kernel", y.tolist().count(3))
    print("Windows Server 2003", y.tolist().count(4))
    print("Windows XP", y.tolist().count(5))
    print("Windows Server 2008", y.tolist().count(6))
    print("Windows NT kernel 5.x", y.tolist().count(7))
    # print("y")
    # print(y)
    # print(set(y))
    # print(len(set(y)))
    # print(len(y))
    saw_sig = pd.DataFrame(saw_sig)
    # print("saw_sig")
    # print(saw_sig)

    X = pd.get_dummies(saw_sig)
    # print("X")
    # print(X)

    data = load_data(X, y)
    X = X.to_numpy()
    #X, y = loadcsv1(csvfile)
    # dt
    dt = DecisionTreeClassifier(criterion="gini", max_depth=16)
    # 创建一个随机森林分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # knn
    knn = KNeighborsClassifier(n_neighbors=7)
    # svm
    svm = SVC(C=130, gamma=13, decision_function_shape="ovo", probability=True)
    # 逻辑回归
    lr_l1 = LogisticRegression(penalty="l1", C=1.0, solver="liblinear")
    timelist = []
    start = time.time()
    avtice(clf, X, y)
    a = time.time()
    timelist.append(a - start)
    avtice(knn, X, y)
    b = time.time()
    timelist.append(b-a)
    avtice(dt, X, y)
    c = time.time()
    timelist.append(c-b)
    avtice(svm, X, y)
    d = time.time()
    timelist.append(d-c)
    avtice(lr_l1, X, y)
    e = time.time()
    timelist.append(e-d)





    # # 3.2 柱状图
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
    #
    # # # 时间图
    print("timlist",timelist)
    # 时间柱状图
    x = ['Decision Tree', 'Random Forest', 'KNN', 'SVM', 'Logistic Regression']
    plt.bar(x,timelist)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.ylabel("单位：s")
    plt.show()

    # one = [0.8947440459895976, 0.8714754995893786, 0.9329071995620039, 0.9622228305502327, 0.9627703257596496]
    # five = [0.912263892690939, 0.877634820695319, 0.9420859567478784, 0.9650971803996715, 0.9705967697782645]
    # ten = [0.9631809471667123, 0.8918696961401588, 0.9401697235149192, 0.9638653161784835, 0.9650971803996715]
    # fifteen = [0.9167807281686285, 0.9047358335614564, 0.9195182042157131, 0.9644128113879004, 0.9616753353408157]
    # tween = [0.9604434711196277, 0.9295510539282781, 0.9137695045168355, 0.9596222283055024, 0.9753791404325213]
    # teenfive = [0.9578428688748973, 0.9544785108130304, 0.910580344921982, 0.96181220914317, 0.9725458527237886]
    # thrity = [0.9646865589926088, 0.9524254037777169, 0.9637284423761292, 0.9612647139337531, 0.9572953736654805]
    #
    #
    # a = [one[0],five[0],ten[0],fifteen[0],tween[0],teenfive[0],thrity[0]]
    # b = [one[1],five[1],ten[1],fifteen[1],tween[1],teenfive[1],thrity[1]]
    # c = [one[2],five[2],ten[2],fifteen[2],tween[2],teenfive[2],thrity[2]]
    # d = [one[3],five[3],ten[3],fifteen[3],tween[3],teenfive[3],thrity[3]]
    # e = [one[4],five[4],ten[4],fifteen[4],tween[4],teenfive[4],thrity[4]]
    #
    # x = [0,5,10,15,20,25,30]
    # plt.figure(figsize=(15, 8), dpi=80)
    # plt.plot(x, a, label="Decision Tree", linewidth=1, marker='p', markersize=10)
    # # plt.plot(x, sc.kscores[1], label="rfc", linewidth=1, marker='8', markersize=10)
    # plt.plot(x, b, label="Random Forest", linewidth=1, marker='D', markersize=10)
    # plt.plot(x, c, label="KNN", color='black', linewidth=1, marker='X', markersize=10)
    # plt.plot(x, d, label="SVM", linewidth=1, marker='v', markersize=10)
    # #plt.plot(x, sc.kscores[5], label="svmBagging", linewidth=1, marker='o', markersize=10)
    # plt.plot(x, e, label="Logistic Regression", linewidth=1, marker='^', markersize=10)
    #
    #
    # plt.xlabel("迭代次数", fontsize=20)
    # plt.ylabel("准确率", fontsize=20)
    # plt.grid(alpha=0.1)
    # # 设置坐标轴刻度
    # #my_x_ticks = np.arange(1, 11, 1)
    # # 对比范围和名称的区别
    # # my_x_ticks = np.arange(-5, 2, 0.5)
    # plt.xticks( fontsize=20)
    # plt.yticks( fontsize=20)
    # plt.ylabel("准确率",fontsize=20)
    # # 显示图例
    # plt.legend(loc='lower right',prop={'size':12})
    # plt.show()
