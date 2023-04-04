# @author lsg
# @date 2023/3/30
# @file test.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

import numpy as np
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore")
font = {'family':'MicroSoft YaHei'}
plt.rc("font",**font)


table = pd.read_table(r"C:\Users\Administrator\Desktop\p5.xls", sep="\t", index_col=0)  # 读取文件，并将第一列设为index
table2 = table.sum(axis=1, skipna=True)  # dataframe 每行求和
df = table.div(table2, axis='rows')  # 计算占比并存为新的dataframe

category_names = df.columns.values.tolist()  # 提取列名 存为list

results = df.T.to_dict('list')  # 行名为key  值为value 存为dic

labels = list(results.keys())  # 确定x轴
print(labels)
data = np.array(list(results.values()))  # 确定数据
data_cum = data.cumsum(axis=1)  # 确定柱子起始位置

category_colors = plt.get_cmap('hsv')(np.linspace(0, 1, data.shape[1]))  # 提取颜色   'hsv' 该值需根据实际情况选择

fig, ax = plt.subplots(figsize=(40, 20))  # 绘图
plt.xticks(fontsize=30,rotation=30)  # x轴坐标刻度字体大小
plt.yticks(fontsize=30)  # y轴坐标刻度字体大小
ax.set_ylim(0, np.sum(data, axis=1).max())  # 设置y轴范围
x_num = np.arange(len(category_names))
#ax.set_xlim(min(x_num) - 1, max(x_num) + 1)  # 设置x轴范围
ax.set_xlim(0.6,59)
x = MultipleLocator(2)    # x轴每1一个刻度
ax = plt.gca()
ax.xaxis.set_major_locator(x)


for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    heights = data[:, i]
    starts = (data_cum[:, i] - heights)
    ax.bar(labels, heights, bottom=starts, width=0.7, label=colname, color=color)

plt.xlabel('合金钢编号', fontsize=50, loc="center")
plt.ylabel('各元素含量推荐值成分占比（%）', labelpad=70,  # 调整y轴标签与y轴的距离
           y=0.5,  # 调整y轴标签的上下位置
           rotation=90, fontsize=50)

ax.legend(loc='right',prop={'size':40})  # legend  位置


plt.show()
