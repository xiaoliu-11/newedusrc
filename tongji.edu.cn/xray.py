# @author lsg
# @date 2022/12/10
# @file autoscript.py
import os
import threading
import time

# 1.打开url文件读取
def loadUrl(fileName):
    f = open(fileName, encoding='utf-8')
    urlList = []
    for line in f:
        urlList.append(line.strip())
    f.close()
    ids = list(set(urlList))
    return ids


def insertxrayexe(urlList):
    f = open(r"C:\Users\Administrator\Desktop\oppo1.txt", "a")
    a=0
    for line in urlList:
        # d: & cd D:\chromedownload\xray & .\xray_windows_amd64.exe webscan --url http://ishnc.sjtu.edu.cn/  --html-output D:\edusrc\xray\c.html
        f.write(r"cd D:\chromedownload\xray & .\xray_windows_amd64.exe webscan --basic-crawler "+ line +r" --html-output D:\edusrc\xray\{}.html".format("417_"+str(a))+"\n")
        a +=1
        print(a)
    f.close()
    print("insertxrayexe success!")





if __name__ == '__main__':
    exexray = r"C:\Users\Administrator\Desktop\oppo.txt"
    urlList = loadUrl(exexray)
    insertxrayexe(urlList)




