# @author lsg
# @date 2023/1/27
# @file urlcollecion.py
import time

import requests
from lxml import etree



def getGoogleUrl(url,headers):
        resp = requests.get(url, headers=headers)
        html_content = resp.content.decode("utf-8")
        html = etree.HTML(html_content)
        url_list = html.xpath("//*/@href")
        f = open(r"D:\edusrc\sqlurl\bytedance.txt", "a")
        try:
            a = 0
            for url in url_list:
                if "/search?" in url:
                    continue
                if "google.com" in url:
                    continue
                if ".jp" in url:
                    continue
                if "#" in url:
                    continue
                if url == "":
                    break
                f.write(url + "\n")
                print(url+"成功存入！")
                a = a + 1
            f.close()
        except(Exception):
            print("爬取失败！")
        return a






if __name__ == '__main__':
    # 地区inurl:"search?kw="
    # id uid typeid u name page productid BigClassName

    search = ""
    str1 = '\"'
    type = ["php", "jsp", "asp", "aspx"]
    parameter = ["id", "uid", "typeid", "u", "name", "page", "productid", "BigClassName", "searchkeywords", "keywords","type_id","orderby"]
    content = ["今日头条", "抖音短视频", "抖音极速版", "抖音火山版", "西瓜视频", "飞书", "火山引擎", "抖音电商",
               "番茄小说", "番茄畅听", "幸福里", "住小帮", "小荷医疗", "醒图", "头条搜索", "皮皮虾", "懂车帝",
               "Faceu激萌", "轻颜相机", "剪映", "头条百科", "图虫", "大力智能"]

    # url = "https://www.google.com.hk/search?q=%22%E5%AD%97%E8%8A%82%E8%B7%B3%E5%8A%A8inurl%3Aphp+id%22&hl=zh-cn&source=hp&ei=2-DsY5esLrHj2roPsvqIiAg&iflsig=AK50M_UAAAAAY-zu6wqUGqe5gGw0GFsAFysHlxlD_m2V&ved=0ahUKEwjXhcXQ05f9AhWxsVYBHTI9AoEQ4dUDCAg&uact=5&sclient=gws-wiz"
    url = "https://www.google.com.hk/search?q="+ search +"22&hl=zh-cn&ei=OJDrY_boAa3n2roP7-OT6Ao&ved=0ahUKEwj25OvKkpX9AhWts1YBHe_xBK0Q4dUDCA8&uact=5&oq=%22%E5%90%8C%E6%B5%8E%E5%A4%A7%E5%AD%A6inurl%3Aphp+id%22&sclient=gws-wiz-serp&start="
    # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
    }

    for i in content:
        for j in type:
            for k in parameter:
                search = i + "inurl:" + j + " " + k
                search = str1 + search + str1
                print("此时的search",search)
                for page in range(0,200,10):
                    # print(url+str(i))
                    a = getGoogleUrl(url+str(page),headers)
                    print("第{}頁".format(page // 10))
                    print("本页共有{}个数据url".format(a))
                    time.sleep(3)
                    if a == 0 :
                        break

