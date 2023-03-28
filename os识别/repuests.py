# @author lsg
# @date 2023/3/13
# @file repuests.py


# @author lsg
# @date 2023/2/14
# @file urlcollection_v1.py
# @author lsg
# @date 2023/1/27
# @file urlcollecion.py
import time

import requests
from lxml import etree


# 1.打开url文件，读取
def loadUrl(fileName):
    f = open(fileName, encoding='utf-8')
    urlList = []
    for line in f:
        urlList.append(line.strip())
    f.close()
    return urlList



def getGoogleUrl(url,headers):
        resp = requests.get(url, headers=headers)
        html_content = resp.content.decode("utf-8")
        print(url +"-->"+ str(len(html_content)))
        resp.close()





if __name__ == '__main__':
    filename = r"D:\edusrc\sqlurl\url.txt"
    urlList = loadUrl(filename)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
    }
    ans = 0
    for url in urlList:
        try:
            getGoogleUrl(url, headers)
            ans += 1
            print(ans)
        except(Exception):
            print("未知异常！")
            continue




