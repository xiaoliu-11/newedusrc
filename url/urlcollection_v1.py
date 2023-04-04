# @author lsg
# @date 2023/2/14
# @file urlcollection_v1.py
# @author lsg
# @date 2023/1/27
# @file urlcollecion.py
import time

import requests
from lxml import etree
from requests.exceptions import ProxyError


def getGoogleUrl(url,headers):
       try:
           resp = requests.get(url, headers=headers)
           html_content = resp.content.decode("utf-8")
           html = etree.HTML(html_content)
           url_list = html.xpath("//*/@href")
           f = open(r"C:\Users\Administrator\Desktop\dida.txt", "a")
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
                   if "tw" in url:
                       continue
                   if "hk" in url:
                       continue
                   if "facebook" in url:
                       continue
                   if url == "":
                       break
                   f.write(url + "\n")
                   print(url + "成功存入！")
                   a = a + 1
               f.close()
           except(Exception):
               print("爬取失败！")
           return a
       except ProxyError:
           print("网络问题，连接失败，准备重新连接！")
           x = 0
           while x < 3:
               getGoogleUrl(url, headers)
               x += 1









if __name__ == '__main__':
    #id q  type_id  bid pid a c d e m
    #uid cid classid  sid tid  typeid u name page productid BigClassName
    #  search  kw orderby start year month
    #  sortid sort date pageid catid itemid num no cat_id
    #  siteid articleid eventid g eid fid kid mid lid
    search = 'intitle:"技术学院" "招生" "青海"'
    #url = "https://www.google.com.hk/search?q="+ search +"&hl=zh-cn&source=hp&ei=gS_zY6vqN63j2roP2K6PsAU&iflsig=AK50M_UAAAAAY_M9kdOR5JPu1J17Uy_BD6_HuS1dj_q5&sclient=gws-wiz&start="
    #url ="https://www.google.com.hk/search?q="+ search +"&hl=zh-cn&gs_lcp=Cgdnd3Mtd2l6EANQ1jRY1jRgnzloAXAAeACAAZMBiAGTAZIBAzAuMZgBAKABAqABAbABAA&sclient=gws-wiz&start="
    url = "https://www.google.com.hk/search?q="+ search +"&hl=zh-cn&ei=ft8BZOjdJ8GfseMP59C8oA0&ved=0ahUKEwjolImk2b_9AhXBT2wGHWcoD9QQ4dUDCBA&uact=5&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQA0oECEEYAFAAWPH6TmDm_E5oBHAAeAGAAf8EiAGwJ5IBCjItMTIuMS4xLjKYAQCgAQHAAQE&sclient=gws-wiz-serp&start="
    # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
    }


    for page in range(0,360,10):
        # print(url+str(i))
        a = getGoogleUrl(url+str(page),headers)
        print("第{}頁".format(page // 10))
        print("本页共有{}个数据url".format(a))
        time.sleep(3)


