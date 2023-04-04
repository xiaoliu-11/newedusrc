import matplotlib.pyplot as plt
import warnings

import numpy as np

warnings.filterwarnings("ignore")
font = {'family':'MicroSoft YaHei'}
plt.rc("font",**font)

# 1.打开url文件读取
def loadUrl(fileName):
    f = open(fileName, encoding='utf-8')
    urlList = []
    for line in f:
        urlList.append(float(line.strip()))
    f.close()
    return urlList


def paintplot(datalist):
    print(datalist)
    print(len(datalist))
    print(max(datalist))
    plt.figure(figsize=(20, 14), dpi=80,)
    x = [x for x in range(0,1201)]
    plt.plot(x,datalist[0:1201],linewidth = 2, )
    plt.fill(x,datalist[0:1201], 'b', alpha=0.9)
    plt.ylim(0,6000)
    plt.xlim(0,1201)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.xlabel('道址', fontsize=35, loc="center")
    plt.ylabel('计数率', labelpad=40,  # 调整y轴标签与y轴的距离
                  y=0.5,  # 调整y轴标签的上下位置
                  rotation=90, fontsize=35)
    plt.legend(frameon=False)
    plt.show()


if __name__ == '__main__':
    path = r"C:\Users\Administrator\Desktop\射线图\paint4.txt"
    datalist = loadUrl(path)
    paintplot(datalist)