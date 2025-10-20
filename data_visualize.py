# -*- coding: utf-8 -*-
# 读取 Excel 并画 SOC vs 幅度图
# 功能：STEP抽样 + 原始SOC顺序 + 三色频率分类

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

STEP = 5       # 每隔多少个频率点画一条线
INVERT_X = True  # 是否反转横轴方向

def main(xlsx_path: str, step: int = STEP, invert_x: bool = INVERT_X):
    # 读取原始数据
    df_raw = pd.read_excel(xlsx_path, header=None)

    # 第三行（索引2）SOC 标签，第一列是频率，第四行起是幅度
    soc_labels = df_raw.iloc[2, 1:].values       # SOC 原始顺序
    freqs = df_raw.iloc[3:, 0].values            # 频率点
    amp_data = df_raw.iloc[3:, 1:].values        # 幅度矩阵

    # 按步长抽取频率点索引
    idx = np.arange(0, len(freqs), step)

    # 定义三种颜色
    colors = ['blue', 'green', 'red']

    # 按分位数划分频率范围
    low_th = np.percentile(freqs, 33)   # 低频阈值
    high_th = np.percentile(freqs, 66)  # 高频阈值

    # 用索引做横轴，再映射SOC标签
    x_index = np.arange(len(soc_labels))

    plt.figure(figsize=(10, 6))
    for i in idx:
        f = freqs[i]
        if f <= low_th:
            c = colors[0]   # 低频
        elif f <= high_th:
            c = colors[1]   # 中频
        else:
            c = colors[2]   # 高频
        plt.plot(x_index, amp_data[i, :], color=c)

    plt.xlabel('SOC')
    plt.ylabel('phase')
    plt.title(f'Amplitude vs SOC (every {step}th freq, 3 groups)')
    tick_step = max(1, len(soc_labels)//10)
    plt.xticks(x_index[::tick_step], soc_labels[::tick_step])
    if invert_x:
        plt.gca().invert_xaxis()  # 反转横轴方向
    
    plt.tight_layout()

    out_dir = Path("vis_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"Amplitude_vs_SOC_step{step}_3groups_2.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("图已保存到：", out_path.resolve())
    print(f"低频 <= {low_th:.2f}, 中频 <= {high_th:.2f}, 高频 > {high_th:.2f}")

if __name__ == "__main__":
    # 没有命令行参数就用默认路径
    xlsx = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\86182\Desktop\SOC_DATA\0.3c_Cycle1-Cycle6\phase\phase_c1.xlsx"
    main(xlsx, step=STEP, invert_x=INVERT_X)
