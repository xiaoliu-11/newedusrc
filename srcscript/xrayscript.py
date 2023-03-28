# @author lsg
# @date 2023/3/17
# @file xrayscript.py

import os
import hashlib
import re

# 扫描
def get_url():
    f = open("xray_url.txt")
    lines = f.readlines()
    # 匹配http | https请求头
    pattern = re.compile(r'^(https|http)://')
    for line in lines:
        try:
            if not pattern.match(line.strip()):
                targeturl="http://"+line.strip()
            else:
                targeturl=line.strip()
            # print(targeturl.strip())
            outputfilename=hashlib.md5(targeturl.encode("utf-8"))
            do_scan(targeturl.strip(), outputfilename.hexdigest())
        except Exception as e:
            print(e)
            pass
    f.close()
    print("Xray Scan End~")
    return

# 报告
def do_scan(targeturl,outputfilename="test"):
    scan_command="C:/Users/Administrator/Desktop/xray_windows_amd64.exe.lnk webscan --basic-crawler {} --html-output {}.html".format(targeturl,outputfilename)
    # scan_command = "ping 943ogg.dnslog.cn"
    # print(scan_command)
    os.system(scan_command)
    return

if __name__ == '__main__':
    get_url()
