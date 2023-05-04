# @author lsg
# @date 2023/5/3
# @file reusets.py

import requests

a = ['tar', 'tar.gz', 'zip', 'rar']
b = ['web', 'website', 'backup', 'back', 'www', 'wwwroot', 'temp']

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}




res = requests.get("http://challenge-b6ce4d3cc193e1a8.sandbox.ctfhub.com:10800/website.zip", headers=headers)
print("http://challenge-b6ce4d3cc193e1a8.sandbox.ctfhub.com:10800/")
res.close()#记得要关闭
# if res.status_code == 200:
#     print("泄露文件位置：",tmp)
# else:
#     print("该路径不存在:",tmp)
