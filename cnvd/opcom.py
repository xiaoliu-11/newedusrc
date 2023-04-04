# @author lsg
# @date 2023/3/29
# @file opcom.py


def loadUrl(fileName):
    f = open(fileName, encoding='utf-8')
    urlList = []
    for line in f:
        urlList.append(line.strip())
    f.close()
    return urlList


def inserturl(urlList):
    f = open(r"E:\PycharmProjects\edusrc\cnvd\educom.txt", "a")
    a=0
    for line in urlList:
        line = line.replace("有限公司","")
        f.write("\"技术支持："+line+"\""+"\n")
        a +=1
    print("插入了{}行".format(a))
    f.close()
    print("insert txt success!")

if __name__ == '__main__':
    com = loadUrl("edu.txt")
    inserturl(com)