# @author lsg
# @date 2022/12/10
# @file autoscript.py
import os
import threading
import time



# 1.打开url文件，读取
def loadUrl(fileName):
    f = open(fileName, encoding='utf-8')
    urlList = []
    for line in f:
        urlList.append(line.strip())
    f.close()
    return urlList

# os.system('chcp 65001')
# os.system(r'd: & cd D:\chromedownload\rad & .\rad_windows_amd64.exe -t http://electsys.sjtu.edu.cn --text-output D:\edusrc\rad\electsys.sjtu.txt  ')
#
# 2.通过rad爬取每个url的网站资源
def radcrawler(urlList):
        os.system('chcp 65001')
        for i in urlList:
            try:
                os.system(r"d: & cd D:\chromedownload\rad & .\rad_windows_amd64.exe -t " + i +r" --text-output D:\edusrc\rad\\"+i[7:]+".txt")
            except(Exception):
                print("rad出现未知异常！")
                continue


def xraycrawler(exexraylist):
    os.system('chcp 65001')
    for i in exexraylist:
        try:
            print("================{}开始扫描！==============\n".format(i))
            os.system(i)
        except(Exception):
            print("xray出现未知异常！")
            continue


def insertxrayexe(urlList):
    f = open(r"D:\edusrc\new_exexray.txt", "a")
    a=0
    for line in urlList:
        # d: & cd D:\chromedownload\xray & .\xray_windows_amd64.exe webscan --url http://ishnc.sjtu.edu.cn/  --html-output D:\edusrc\xray\c.html
        f.write(r"cd C:\chromedownload\xray & .\xray_windows_amd64.exe webscan --basic-crawler "+"http://"+ line +r" --html-output C:\edusrc\xray\{}.html".format(line+str(a))+"\n")
        a +=1
        print(a)
    f.close()
    print("insertxrayexe success!")

    # 删除前N行
def delete_first_lines(filename, count):
    fin = open(filename, 'r')
    a = fin.readlines()
    fout = open(filename, 'w')
    b = ''.join(a[count:])
    fout.write(b)


if __name__ == '__main__':
    filename = r'D:\edusrc\tmp.txt'
    filename1 = r"D:\pycharmprojects\edusrc\1.collection\subdomain.txt"
    exexray = r'D:\edusrc\exexray.txt'
    fiveurl = r'D:\edusrc\url.txt'
    new_exexray = r'D:\edusrc\new_exexray.txt'
    urlList = loadUrl(exexray)
    #insertxrayexe(urlList)
    # delete_first_lines(new_exexray,250000)
    xraycrawler(urlList)


