
import os
import shutil

list1 = []
def get_path(file_path):
    for root, dirs, files in os.walk(file_path):
        s = root+"\log"
        if os.path.getsize(s) == 0:
            shutil.rmtree(root)

if __name__ == "__main__":
    file_path = r"C:\Users\Administrator\Desktop\补天\云主机扫描的注入\4.18"
    get_path(file_path)
