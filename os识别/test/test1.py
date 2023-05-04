# @author lsg
# @date 2023/4/25
# @file test1.py
import pandas as pd


def loadcsv1(path):
    data = pd.read_csv(path,header=None)
    X = data.iloc[:, 1:-1]  #
    X = X.to_numpy()
    y = data.iloc[:, -1]  #
    print(set(y))
    y = y.to_numpy()
    print("Windows NT kernel",y.tolist().count(0))
    print("Windows 7 or 8",y.tolist().count(1))
    print("Linux 2.2.x",y.tolist().count(2))
    print("Linux 3.x",y.tolist().count(3))
    print("Windows Server 2003",y.tolist().count(4))
    print("Windows XP",y.tolist().count(5))
    print("Windows Server 2008",y.tolist().count(6))
    print("Windows NT kernel 5.x",y.tolist().count(7))

    return  X,y

if __name__ == '__main__':
    csvfile1 = r"C:\Users\Administrator\Desktop\os识别\dataos1.csv"
    #csvfile2 = r"C:\Users\Administrator\Desktop\os识别\dataos2.csv"
    loadcsv1(csvfile1)

