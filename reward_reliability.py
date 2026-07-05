import pickle
import numpy as np
import pandas as pd

def analyze_simple_comparison(sweep_path, random_path):
    files = {
        'Sweeping (扫频)': sweep_path,
        'Random (随机)': random_path
    }

    all_results = []

    for mode_name, path in files.items():
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            continue

        for agent_name, runs in data.items():
            all_scores = []
            for run in runs:
                if len(run) >= 2:
                    # run[1] 通常是训练过程中的 rolling_scores
                    all_scores.extend(run[1])

            if all_scores:
                all_scores = np.array(all_scores)
                # 1. 计算全过程平均奖励
                mean_reward = np.mean(all_scores)
                # 2. 计算可靠性 (得分 > 90 的比例)
                reliability = np.mean(all_scores > 90) * 100

                all_results.append({
                    'Mode': mode_name,
                    'Algorithm': agent_name,
                    'Mean Reward': round(mean_reward, 2),
                    'Reliability': f"{reliability:.2f}%"
                })

    # 转化为 DataFrame 并格式化展示
    df = pd.DataFrame(all_results)
    if not df.empty:
        df = df.set_index(['Mode', 'Algorithm'])
    return df

# --- 设置路径并执行 ---
sweep_file = "/content/Anti_Jam_training_0.1_sweeping.pkl"
random_file = "/content/Anti_Jam_training_0.1_random.pkl"

comparison_df = analyze_simple_comparison(sweep_file, random_file)

# 打印最终结果
print("\n" + "="*25 + " 算法性能对比简表 " + "="*25)
if not comparison_df.empty:
    print(comparison_df)
else:
    print("未发现有效数据，请检查文件路径或 .pkl 内部结构。")
print("="*68)