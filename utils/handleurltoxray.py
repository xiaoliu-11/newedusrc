# @author lsg
# @date 2023/2/9
# @file handleurltoxray.py
import os

import os.path
import re


# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir = os.listdir(filepath)  # 这一步就已经实现拿到文件夹下的文件名了
    for allDir in pathDir:
        child = os.path.join('%s\%s' % (filepath, allDir))
        print(child)
        if os.path.isfile(child):
            readFile(child)
            continue


# 遍历出结果 返回文件的名字
def readFile(filenames):
    f = open(filenames, encoding='utf-8')
    for line in f:
        line = line.replace("GET","")
        line = line.replace("POST","")
        arr.append(line.strip())
    f.close()


# 添加xray信息。准备爬虫。
def insertxrayurl(urlList):
    f = open(r"D:\edusrc\zhengshuzhan\tongji\xrayurl.txt", "a")
    a=0
    for line in urlList:
        f.write(r"cd C:\chromedownload\xray & .\xray_windows_amd64.exe webscan --basic-crawler "+ line +r" --html-output C:\edusrc\xray\{}.html".format(str(a))+"\n")
        a +=1
        print(a)
    f.close()
    print("insertxray success!")


if __name__ == "__main__":
    filenames = r'D:\edusrc\zhengshuzhan\tongji\output'  # refer root dir
    arr = []
    eachFile(filenames)
    print(len(arr))
    insertxrayurl(arr)

