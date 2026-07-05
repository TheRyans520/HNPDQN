import pickle
import numpy as np
import pandas as pd


def analyze_loss_only(files_dict, target_steps=100):
    loss_results = []

    for mode_name, path in files_dict.items():
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f"--- 正在分析模式: {mode_name} ---")
        except Exception as e:
            print(f"读取文件失败: {path}, 错误: {e}")
            continue

        for agent_name, runs in data.items():
            run_metrics = []
            valid_runs_count = 0

            for run in runs:
                # 检查索引 6 是否存在 (对应 loss_history)
                if len(run) > 6:
                    raw_loss = np.array(run[6])
                    # 确保 loss 序列不为空
                    if raw_loss.size > 0:
                        # 截取目标步数
                        loss_segment = raw_loss[:target_steps]

                        # 指标 1: 初始 Loss 水平 (第 1 步)
                        initial_val = loss_segment[0]
                        # 指标 2: 前 100 步的平均波动 (标准差)
                        fluctuation = np.std(loss_segment)
                        # 指标 3: 100 步结束时的收敛深度 (最后 10 步均值)
                        final_depth = np.mean(loss_segment[-10:]) if len(loss_segment) >= 10 else loss_segment[-1]

                        run_metrics.append([initial_val, fluctuation, final_depth])
                        valid_runs_count += 1

            if run_metrics:
                metrics_mean = np.mean(run_metrics, axis=0)
                loss_results.append({
                    'Mode': mode_name,
                    'Algorithm': agent_name,
                    'Initial Loss': metrics_mean[0],
                    'Loss Fluctuation': metrics_mean[1],
                    'Conv Depth (100th)': metrics_mean[2]
                })
                print(f"  [成功] Agent: {agent_name} ({valid_runs_count} 次实验有效)")
            else:
                print(f"  [跳过] Agent: {agent_name} (未发现有效的 loss_history 数据)")

    # 核心修复：检查结果集是否为空
    if not loss_results:
        print("\n错误：所有文件均未提取到有效的 Loss 数据！请检查 Agent 类是否正确保存了索引为 6 的 loss_history。")
        return pd.DataFrame()  # 返回空表不报错

    df = pd.DataFrame(loss_results)
    # 设置索引
    df = df.set_index(['Mode', 'Algorithm'])

    # 格式化输出：科学计数法转为更易读的 4 位小数
    pd.options.display.float_format = '{:.4f}'.format
    return df


# ==========================================
# 自定义读取路径
# ==========================================
custom_loss_files = {
    'Random Mode Training': "/content/Anti_Jam_training1_0.1_random.pkl",
    'Sweeping Mode Training': "/content/Anti_Jam_training1_0.1_sweeping.pkl"
}

# 执行
loss_df = analyze_loss_only(custom_loss_files, target_steps=100)

if not loss_df.empty:
    print("\n" + "=" * 20 + " 初始收敛机理分析表 (Top 100 Steps) " + "=" * 20)
    print(loss_df)
    print("=" * 76)