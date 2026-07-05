#!/usr/bin/python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    base_dir = os.path.abspath('')
    # jammed_freq = 2442
    # jammer_dist = 20
    # jamming_power = 10

    # --- 仅修改以下参数以匹配 5GHz 描述，读取方式与逻辑保持不变 ---
    jammed_freq = 5180  # 修改为 5GHz 频段示例频率
    jammer_dist = 20
    jamming_power = 10

    raw_dir = f'{base_dir}/raw/spectral_scans_QC9880_ht20_background'
    scenario = f'samples_chamber_{jammed_freq}MHz_{jammer_dist}cm_{jamming_power}dBm'
    # scenario = 'samples_chamber_None'
    df = pd.read_csv(f'{raw_dir}/{scenario}_5.csv')
    # Getting channel information
    # band = '2.4GHz'
    # all_channels = None
    # if band == '2.4GHz':
    #     all_channels = list(range(2412, 2467, 5))
    # elif band == '5GHz':
    #     all_channels = [list(range(5180, 5340, 20)), list(range(5745, 5865, 20))]
    #     all_channels = [freq for band in range(len(all_channels)) for freq in all_channels[band]]
    # else:
    #     print('Invalid band')
    #
    # x = []
    # y = []
    # --- 修改 band 为 5GHz 并精确匹配你要求的 8 个信道 ---
    band = '5GHz'
    all_channels = None
    if band == '2.4GHz':
        all_channels = list(range(2412, 2467, 5))
    elif band == '5GHz':
        # 修改此处：5180MHz 到 5320MHz，步长 20MHz，正好 8 个候选信道
        all_channels = list(range(5180, 5321, 20))
    else:
        print('Invalid band')

    x = []
    y = []
    for channel in all_channels:
        df_channel = df.where(df['freq1'] == channel).dropna()
        df_channel[df_channel['snr'] < -100] = np.NaN
        df_channel[df_channel['snr'] > 100] = np.NaN
        df_channel.fillna(df_channel['snr'].mean(), inplace=True)
        x.append(channel)
        y.append(df_channel['snr'].mean())

    plotName = f'{scenario}.png'

    # 核心修改点：
    # 1. 设置 y 轴的起始点为 -10
    y_start = -10

    # 2. bottom=y_start 确保所有柱子都从 -10 这个基准线开始向上“生长”
    # height 参数需要传入 (实际值 - 基准值)，否则柱子高度会多出一段
    plt.bar(x, [val - y_start for val in y], width=12, bottom=y_start, color='royalblue', edgecolor='black')

    # 3. 强制设置坐标轴范围，确保下方从 -10 开始，上方留出足够空间（比如到 40）
    plt.ylim(y_start, 40)

    # 4. 优化横坐标刻度，确保 8 个频率点都能对齐显示
    plt.xticks(all_channels)

    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Average received power (dBm)')

    # 添加一个网格线方便审稿人对比数值
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(plotName, bbox_inches='tight', dpi=300)
    plt.show()
