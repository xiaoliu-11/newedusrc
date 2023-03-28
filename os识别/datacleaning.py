# @author lsg
# @date 2023/3/12
# @file datacleaning.py
import pandas as pd
import re
import random
import openpyxl as op
# 1.打开url文件，读取
def loadUrl(fileName):
    f = open(fileName, encoding='utf-8')
    urlList = []
    for line in f:
        urlList.append(line.strip())
    f.close()
    return urlList



# 将数组存到execl
def pd_tocsv(data): # pandas库储存数据到excel
    city = pd.DataFrame([i for i in data])
    city.to_csv('dataos3.csv',mode="a")

def load_csv(path):
    df = pd.read_csv(path)
    return df.iloc[:, 16]


if __name__ == '__main__':
    filename = r"C:\Users\Administrator\Desktop\os识别\321.txt"
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
                    #print(j)
    #print(list(set(listos)))
    # print(list(set(iplist)))
    # print(len(list(set(iplist))))
    # print(listos)
    #print(saw_sig)
    print(len(listos))
    print(len(saw_sig))
    #pd_tocsv(saw_sig)
    #pd_tocsv(listos)
    #os = load_csv(r"C:\Users\Administrator\Desktop\os识别\dataos1.csv")
    #print(os)
    osls = ['Linux 3.x', 'Windows XP', 'Windows NT kernel', 'Windows 7 or 8', 'Linux 2.4.x',
            'Windows NT kernel 5.x','Windows Server 2008',"Windows Server 2003","Linux 2.2.x"]









