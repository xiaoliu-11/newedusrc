# @author lsg
# @date 2023/2/7
# @file removal.py
import os


# 1.打开url文件，读取
def loadUrl(fileName):
    f = open(fileName, encoding='utf-8')
    urlList = []
    for line in f:
        urlList.append(line.strip())
    f.close()
    return urlList

# 重复的消掉
def inserturl(urlList):
    f = open(r"D:\edusrc\fourexp\removalurl.txt", "a")
    a=0
    for line in urlList:
        f.write(line+"\n")
        a +=1
        print(a)
    f.close()
    print("removal success!")

# 添加rad信息。准备爬虫。
def insertradurl(urlList):
    f = open(r"D:\edusrc\fourexp\radurl.txt", "a")
    a=0
    for line in urlList:
        # d: & cd D:\chromedownload\xray & .\xray_windows_amd64.exe webscan --url http://ishnc.sjtu.edu.cn/  --html-output D:\edusrc\xray\c.html
        f.write(r"cd D:\edusrc\fourexp\rad & D: & .\rad.exe -t http://"+ line +r" --text D:\edusrc\fourexp\output\test.txt --index"+"\n")
        a +=1
        print(a)
    f.close()
    print("insertrad success!")

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
     # urlpath = r"D:\edusrc\fourexp\url.txt"
    # urlList = loadUrl(urlpath)
    # ids = list(set(urlList))
    # inserturl(ids)

     # urlpath = r"D:\edusrc\fourexp\removalurl.txt"
     # urlList = loadUrl(urlpath)
     # insertradurl(urlList)

     urlpath = r"D:\edusrc\fourexp\radurl.txt"
     urlList = loadUrl(urlpath)
     radcrawler(urlList)


