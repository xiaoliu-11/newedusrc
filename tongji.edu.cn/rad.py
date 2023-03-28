# @author lsg
# @date 2023/2/9
# @file rad.py


# 1.打开url文件，读取
def loadUrl(fileName):
    f = open(fileName, encoding='utf-8')
    urlList = []
    for line in f:
        urlList.append(line.strip())
    f.close()
    return urlList

# 添加rad信息。准备爬虫。
def insertradurl(urlList):
    f = open(r"D:\edusrc\zhengshuzhan\tongji\radurl.txt", "a")
    a=0
    for line in urlList:
        f.write(r"cd D:\edusrc\fourexp\rad & D: & .\rad.exe -t "+ line +r" --text D:\edusrc\zhengshuzhan\tongji\output\test.txt --index"+"\n")
        a +=1
        print(a)
    f.close()
    print("insertrad success!")


if __name__ == '__main__':
    path = r"D:\edusrc\zhengshuzhan\tongji\tongji.edu.cn.txt"
    urlist = loadUrl(path)
    insertradurl(urlist)

