# @author lsg
# @date 2023/2/9
# @file domain.py
import pandas as pd
import csv


# 从execl读取域名。然后去重，然后存到txt。然后用rad爬，然后用xray扫。

# 从execl读取 并去重
# 从execl读取 并去重
def loadExecl(path):
    with open(path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        column = [row[1] for row in reader]
        ids = list(set(column))
        print("一共{}个域名".format(len(ids)))
        return ids

# 重复的消掉插入txt
def inserturl(urlList):
    f = open(r"C:\Users\Administrator\Desktop\360.txt", "a")
    a=0
    for line in urlList:
        f.write(line+"\n")
        a +=1
    print("插入了{}行".format(a))
    f.close()
    print("insert txt success!")




if __name__ == '__main__':
    list1 = ["360.com.csv",
                "360.cn.csv"]
    for i in list1:
        ids = loadExecl(i)
        print(ids)
        inserturl(ids)
