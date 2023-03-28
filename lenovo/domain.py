# @author lsg
# @date 2023/2/9
# @file domain.py
import pandas as pd
import csv


# 从execl读取域名。然后去重，然后存到txt。然后用rad爬，然后用xray扫。

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
    f = open(r"D:\edusrc\zhengshuzhan\src.txt", "a")
    a=0
    for line in urlList:
        f.write("http://"+line+"\n")
        a +=1
    print("插入了{}行".format(a))
    f.close()
    print("insert txt success!")




if __name__ == '__main__':
    path = "motorola.com.csv"
    list1 = ["360.cn.csv",
            "alipay.com.csv",
            "baiying.cn.csv",
            "ctrip.com.csv",
            "epet.com.csv",
            "iflytek.com.csv",
            "immomo.com.csv",
            "kuaishou.com.csv",
            "liepin.com.csv",
            "kugou.com.csv",
            "ly.com.csv",
            "mafengwo.cn.csv",
            "meizu.com.csv",
            "oneplus.com.csv",
            "oppo.com.csv",
            "pingan.com.csv",
            "qianmi.com.csv",
            "rong360.com.csv",
            "sangfor.com.cn.csv",
            "saicmobility.com.csv",
            "sangfor.com.csv",
            "sf-express.com.csv",
            "shein.com.csv",
            "shuidihuzhu.com.csv",
            "t3go.cn.csv",
            "soulapp.cn.csv",
            "tcl.com.csv",
            "tongji.edu.cn.csv",
            "tuhu.cn.csv",
            "unionpay.com.csv",
            "vip.com.csv",
            "wacai.com.csv",
            "webank.com.csv",
            "weibo.com.csv",
            "wulintang.net.csv",
            "xiaomi.com.csv",
            "xiaoying.com.csv",
            "ximalaya.com.csv",
            "ys7.com.csv"]
    for i in list1:
        ids = loadExecl(i)
        inserturl(ids)
