import matplotlib.pyplot as plt
import warnings

import numpy as np

warnings.filterwarnings("ignore")
font = {'family':'MicroSoft YaHei'}
plt.rc("font",**font)

# 设置纵坐标
y_labels = ['C', 'Si', 'P', 'S', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Mo']

# 设置两组横坐标数据
# ka_x = [0.282, 1.74, 2.015, 2.308, 4.51, 5.414, 5.898, 6.403, 6.93, 7.477, 8.047, 8.638, 17.478]
# ka_y = [3.23E-06,	0.000443281,	4.28498E-05,	3.35663E-06,	7.26232E-06,	0.079472862,	0.005795199,	0.365054105,	0.001189986,0.048384633,	0.00116247,	0,	0.000296205]
#
# kb_x = [ 1.832, 2.136, 2.464, 4.931, 5.964, 6.49, 7.057, 7.649, 8.264, 8.904, 9.571, 19.607]
# kb_y = [	5.91041E-05,	5.71331E-06,	4.4755E-07,	9.68309E-07,	0.010596382,	0.000772693,	0.048673881,	0.000158665,	0.006451284,	0.000154996,	0,	3.9494E-05]


# 29号
ka_y = [1.10202E-05,0.000316629,0.000228979,0.000102377,0.061555344,0.036878964,0.380041814,0.013460475,0.000654996,0.000268651]
ka_x = [0.282,1.74,2.015,2.308,5.414,5.898,6.403,7.477,8.047,17.478]
kb_y = [4.22172E-05,3.05305E-05,1.36503E-05,0.008207379,0.004917195,0.050672242,0.00179473,8.73328E-05,3.58202E-05]
kb_x = [1.832,2.136,2.464,5.964,6.49,7.057,8.264,8.904,19.607]



# 创建画布和子图对象
fig, ax = plt.subplots(figsize=(20,8))

# 绘制散点图
line1 = ax.scatter(ka_x, [x*100000 for x in ka_y], marker='s', color='black', s=60)
line2 = ax.scatter(kb_x, [x*100000 for x in kb_y], marker='^', color='r', s=60)


tmpa = 1
for i in ka_x:
    ax.axvline(x=i, c="black",ymax=0.05, ls="-", lw=1)
    tmpa += 1
ax.axvline(x=5.414, c="black",ymax=0.2, ls="-", lw=1)
ax.axvline(x=5.898, c="black",ymax=0.14, ls="-", lw=1)
ax.axvline(x=6.403, c="black",ymax=0.945, ls="-", lw=1)
ax.axvline(x=7.477, c="black",ymax=0.09, ls="-", lw=1)


tmpb = 1
for i in kb_x:
    ax.axvline(x=i, c="r",ymax=0.05, ls="--", lw=1)
    tmpb += 1

ax.axvline(x=7.057, c="r",ymax=0.18, ls="--", lw=1)

# 设置图形样式
# 设置图表标题和轴标签
ax.set_title('')
ax.set_xlabel('能量 (keV)',fontsize=25,loc="center")
ax.set_ylabel('强度(I)',labelpad=40,  #调整y轴标签与y轴的距离
           y=0.9,  #调整y轴标签的上下位置
           rotation=0,fontsize=25)
# 设置坐标轴刻度
my_x_ticks = np.arange(1, 20, 1)
# 对比范围和名称的区别
ax.set_xticks(my_x_ticks,)
ax.set_xticklabels(my_x_ticks, fontsize=25)
# 添加图例
ax.legend( labels=['Kα能量', 'Kβ能量'],loc=1,fontsize=25)
ax.yaxis.set_ticks([]) # 隐藏y坐标轴
# ax.annotate('WW', xy=(0.282,3.23), xytext=(0.282,-1))

# labelsKa = ["C(Kα)","Si(Kα)","P(Kα)","S(Kα)","Ti(Kα)","Cr(Kα)","Mn(Kα)","Fe(Kα)","Co(Kα)","Ni(Kα)","Cu(Kα)","Zn(Kα)","Mo(Kα)"]
# for i, label in enumerate(labelsKa ):
#     plt.annotate(label, (ka_x[i], ka_y[i]), textcoords="offset points", xytext=(-15,-60), fontsize=10,)
#
# labelsKb = ["Si(Kβ)","P(Kβ)","S(Kβ)","Ti(Kβ)","Cr(Kβ)","Mn(Kβ)","Fe(Kβ)","Co(Kβ)","Ni(Kβ)","Cu(Kβ)","Zn(Kβ)","Mo(Kβ)"]
# for i, label in enumerate(labelsKb):
#     plt.annotate(label, (kb_x[i], kb_y[i]), textcoords="offset points", xytext=(-15,-80), fontsize=10,color='red', )


# 显示图形
plt.show()
