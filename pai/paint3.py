# @author lsg
# @date 2023/3/30
# @file paint3.py.py
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
font = {'family':'MicroSoft YaHei'}
plt.rc("font",**font)
import csv

def getdata(path):
    alldata = []
    try:
        with open(path, 'r', encoding='utf-8') as datalist:
            next(datalist)
            reader = csv.reader(datalist)
            for row in reader:
              alldata.append(row[1:])
    except csv.Error as e:
        print("Error at line %s :%s", reader.line_num, e)
    return alldata


def paint(alldata):
    fig, axs = plt.subplots(figsize=(25, 10))

    # sum = salary1 + salary2 + salary3
    # percentage1 = salary1 / sum
    # percentage2 = salary2 / sum
    # percentage3 = salary3 / sum

    # width = 0.2
    # index = np.arange(52)
    # axs[1].bar(index, percentage1, width=width)
    # axs[1].bar(index, percentage2, width=width, bottom=percentage1)
    # axs[1].bar(index, percentage3, width=width,
    #            bottom=percentage1 + percentage2)
    # axs[1].set_ylim(0, 1)
    # axs[1].set_xticks(index)
    # axs[1].set_xticklabels(index, rotation=90)
    #
    # plt.savefig('9.tiff', dpi=300)
    # plt.show()




if __name__ == '__main__':
    path = r'C:\Users\Administrator\Desktop\p4.csv'
    alldata = getdata(path)
    paint(alldata)
