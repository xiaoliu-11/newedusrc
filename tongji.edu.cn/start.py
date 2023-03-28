# @author lsg
# @date 2023/2/9
# @file start.py
import os


# 1.打开url文件，读取
def loadUrl(fileName):
    f = open(fileName, encoding='utf-8')
    urlList = []
    for line in f:
        urlList.append(line.strip())
    f.close()
    return urlList
# 开始自动扫描脚本。
def radcrawler(urlList):
    os.system('chcp 65001')
    for i in urlList:
        try:
            print("================{}开始扫描！==============\n".format(i))
            os.system(i)
        except(Exception):
            print("rad出现未知异常！")
            continue


if __name__ == '__main__':
    urlpath = r"D:\edusrc\zhengshuzhan\tongji\radurl.txt"
    urlList = loadUrl(urlpath)
    radcrawler(urlList)