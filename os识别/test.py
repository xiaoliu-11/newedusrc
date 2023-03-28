# @author lsg
# @date 2023/3/12
# @file test.py

ss = "4:125+3:0:1460:8192,8:mss,nop,ws,nop,nop,sok:df,id+:0"
import re
re.split(':|,',ss)
print(re.split(':|,',ss))