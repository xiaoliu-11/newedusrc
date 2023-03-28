# @author lsg
# @date 2023/3/26
# @file paint2.py

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
    fig = plt.figure(figsize=(20, 14), dpi=80,)
    ka_x = ["17.67497324","145.4993412","162.064294","179.7134982","323.8071105","352.9614274","383.3807044","448.0743746","482.4016004","1053.496531"]
    ka_y = ["1.10202E-05",	"0.000316629","0.000228979","0.000102377","0.061555344","0.036878964","0.380041814","0.013460475","0.000654996","0.000268651"]
    kb_x = ["151.0410708", "169.3528732", "189.1103442", "356.9370161", "388.6212531", "422.7751739", "495.4802577","534.0314205", "1181.739383"]
    kb_y = ["4.22172E-05", "3.05305E-05", "1.36503E-05", "0.008207379", "0.004917195", "0.050672242", "0.00179473","8.73328E-05", "3.58202E-05"]
    # for i in range(len(ka_y)):
    #     ka_y[i] = float(ka_y[i]) * 90000
    # for i in range(len(kb_y)):
    #     kb_y[i] = float(kb_y[i]) * 90000

    ka_y[0] = float(ka_y[0]) * 130000000
    ka_y[1] = float(ka_y[1]) * 5000000
    ka_y[2] = float(ka_y[2]) * 7000000
    ka_y[3] = float(ka_y[3]) * 15000000
    ka_y[4] = float(ka_y[4]) * 150000
    ka_y[5] = float(ka_y[5]) * 170000
    ka_y[6] = float(ka_y[6]) * 90000
    ka_y[7] = float(ka_y[7]) * 390000
    ka_y[8] = float(ka_y[8]) * 2800000
    ka_y[9] = float(ka_y[9]) * 5300000

    kb_y[0] = float(kb_y[0]) * 40000000
    kb_y[1] = float(kb_y[1]) * 55000000
    kb_y[2] = float(kb_y[2]) * 115000000
    kb_y[3] = float(kb_y[3]) * 330000
    kb_y[4] = float(kb_y[4]) * 500000
    kb_y[5] = float(kb_y[5]) * 160000
    kb_y[6] = float(kb_y[6]) * 1000000
    kb_y[7] = float(kb_y[7]) * 20000000
    kb_y[8] = float(kb_y[8]) * 40000000


    x = [x for x in range(0, 1200)]
    plt.plot(x,datalist[:1200],linewidth = 2, )
    plt.fill(x,datalist[:1200], 'b', alpha=0.5)

    plt.scatter(float(ka_x[0]), ka_y[0], marker='s', color='black', s=160)


    plt.scatter(float(ka_x[1]), ka_y[1], marker='p', color='black', s=160)
    plt.scatter(float(kb_x[0]), kb_y[0], marker='p', color='r', s=160)

    plt.scatter(float(ka_x[2]), ka_y[2], marker='3', color='black', s=160)
    plt.scatter(float(kb_x[1]), kb_y[1], marker='3', color='r', s=160)

    plt.scatter(float(ka_x[3]), ka_y[3], marker='v', color='black', s=160)
    plt.scatter(float(kb_x[2]), kb_y[2], marker='v', color='r', s=160)

    plt.scatter(float(ka_x[4]), ka_y[4], marker='+', color='black', s=160)
    plt.scatter(float(kb_x[3]), kb_y[3], marker='+', color='r', s=160)

    plt.scatter(float(ka_x[5]), ka_y[5], marker='^', color='black', s=160)
    plt.scatter(float(kb_x[4]), kb_y[4], marker='^', color='r', s=160)

    plt.scatter(float(ka_x[6]), ka_y[6], marker='X', color='black', s=160)
    plt.scatter(float(kb_x[5]), kb_y[5], marker='X', color='r', s=160)

    plt.scatter(float(ka_x[7]), ka_y[7], marker='*', color='black', s=160)
    plt.scatter(float(kb_x[6]), kb_y[6], marker='*', color='r', s=160)

    plt.scatter(float(ka_x[8]), ka_y[8], marker='d', color='black', s=160)
    plt.scatter(float(kb_x[7]), kb_y[7], marker='d', color='r', s=160)

    plt.scatter(float(ka_x[9]), ka_y[9], marker='8', color='black', s=160)
    plt.scatter(float(kb_x[8]), kb_y[8], marker='8', color='r', s=160)


    # kb_y kb_x
    # plt.scatter(float(kb_x[0]), kb_y[0], marker='p', color='r', s=160)
    # plt.scatter(float(kb_x[1]), kb_y[1], marker='3', color='r', s=160)
    # plt.scatter(float(kb_x[2]), kb_y[2], marker='v', color='r', s=160)
    # plt.scatter(float(kb_x[3]), kb_y[3], marker='+', color='r', s=160)
    # plt.scatter(float(kb_x[4]), kb_y[4], marker='^', color='r', s=160)
    # plt.scatter(float(kb_x[5]), kb_y[5], marker='X', color='r', s=160)
    # plt.scatter(float(kb_x[6]), kb_y[6], marker='*', color='r', s=160)
    # plt.scatter(float(kb_x[7]), kb_y[7], marker='d', color='r', s=160)
    # plt.scatter(float(kb_x[8]), kb_y[8], marker='8', color='r', s=160)



    plt.axvline(x=float(ka_x[0]), c="black",ymax=0.04, ls="-", lw=1)
    plt.axvline(x=float(ka_x[1]), c="black", ymax=0.04, ls="-", lw=1)
    plt.axvline(x=float(ka_x[2]), c="black", ymax=0.04, ls="-", lw=1)
    plt.axvline(x=float(ka_x[3]), c="black", ymax=0.04, ls="-", lw=1)
    plt.axvline(x=float(ka_x[4]), c="black", ymax=0.26, ls="-", lw=1)
    plt.axvline(x=float(ka_x[5]), c="black", ymax=0.18, ls="-", lw=1)
    plt.axvline(x=float(ka_x[6]), c="black", ymax=0.941, ls="-", lw=1)
    plt.axvline(x=float(ka_x[7]), c="black", ymax=0.14, ls="-", lw=1)
    plt.axvline(x=float(ka_x[8]), c="black", ymax=0.05, ls="-", lw=1)
    plt.axvline(x=float(ka_x[9]), c="black", ymax=0.05, ls="-", lw=1)


    # kb_y kb_x
    plt.axvline(x=float(kb_x[0]), c="r",ymax=0.04, ls="--", lw=1)
    plt.axvline(x=float(kb_x[1]), c="r", ymax=0.04, ls="--", lw=1)
    plt.axvline(x=float(kb_x[2]), c="r", ymax=0.04, ls="--", lw=1)
    plt.axvline(x=float(kb_x[3]), c="r", ymax=0.08, ls="--", lw=1)
    plt.axvline(x=float(kb_x[4]), c="r", ymax=0.07, ls="--", lw=1)
    plt.axvline(x=float(kb_x[5]), c="r", ymax=0.23, ls="--", lw=1)
    plt.axvline(x=float(kb_x[6]), c="r", ymax=0.05, ls="--", lw=1)
    plt.axvline(x=float(kb_x[7]), c="r", ymax=0.05, ls="--", lw=1)
    plt.axvline(x=float(kb_x[8]), c="r", ymax=0.05, ls="--", lw=1)

    # ax = plt.gca()
    # # 把上面和右面的轴设置为消失
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # # 设置下面的线为x轴，左边的线为y轴
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    # # x轴的position放在y轴的0点上
    # ax.spines['bottom'].set_position(('data', 0))  # outward,axes
    # ax.spines['left'].set_position(('data', 0))
    plt.ylim(0,36000)
    plt.xlim(0,1250)


    plt.yticks([])  # 去y坐标刻度
    plt.xticks(fontsize = 25)
    plt.xlabel('道址', fontsize=25, loc="center")
    plt.ylabel('能量 (keV)', labelpad=70,  # 调整y轴标签与y轴的距离
                  y=0.9,  # 调整y轴标签的上下位置
                  rotation=0, fontsize=25)
    # 显示图例
    # l2 = plt.legend(
    #     labels=["","",'C(Kα)', 'Si(Kα)','Si(Kβ)', 'P(Kα)','P(Kβ)', 'S(Kα)','S(Kβ)', 'Cr(Kα)','Cr(Kβ)', 'Mn(Kα)','Mn(Kβ)', 'Fe(Kα)','Fe(Kβ)', 'Ni(Kα)','Ni(Kβ)', 'Cu(Kα)','Cu(Kβ)','Mo(Kα)','Mo(Kβ)'], loc=5, fontsize=18)
    # plt.gca().add_artist(l2)

    # 添加图例
    l1 = plt.legend(labels=['Kα能量', 'Kβ能量'], loc=1, fontsize=20)
    l2 = plt.legend(
        labels=["","",'C(Kα)', 'Si(Kα)', 'Si(Kβ)', 'P(Kα)', 'P(Kβ)', 'S(Kα)', 'S(Kβ)', 'Cr(Kα)', 'Cr(Kβ)', 'Mn(Kα)', 'Mn(Kβ)',
                'Fe(Kα)', 'Fe(Kβ)', 'Ni(Kα)', 'Ni(Kβ)', 'Cu(Kα)', 'Cu(Kβ)', 'Mo(Kα)', 'Mo(Kβ)'], loc=5, fontsize=18)

    fig.gca().add_artist(l1)
    fig.gca().add_artist(l2)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # BX-01-1
    path = r"C:\Users\Administrator\Desktop\射线图\BX-01-1.txt"
    datalist = loadUrl(path)
    paintplot(datalist)