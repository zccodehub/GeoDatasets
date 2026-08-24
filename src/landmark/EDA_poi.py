import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

# 设置绘图风格 (Windows 环境下中文支持)
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Windows 系统下的路径设置
DATA_PATH = r"E:\Projects\PycharmProjects\Geocoding_dataset\data\Beijing_POI_composite_3.csv"
OUTPUT_DIR = r"E:\Projects\PycharmProjects\Geocoding_dataset\output"

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    return pd.read_csv(path)

def analyze_poi_scores(df):
    """
    执行探索性数据分析 (EDA) - 已修复描述性统计逻辑
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 描述性统计
    print("--- 第一步：描述性统计 ---")
    # 修复：使用字符串名称调用 agg
    stats_df = df[['spatial_score', 'cognition_score']].agg(['mean', 'median', 'std', 'min', 'max', 'skew', 'kurt'])
    print(stats_df)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats.csv"))

    # 2. 可视化：分布分析 (直方图/密度图 + 箱线图)
    print("\n--- 第一步：生成分布图 ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.histplot(df['spatial_score'], kde=True, ax=axes[0], color='blue')
    axes[0].set_title('SIS (spatial_score) 分布图')
    
    sns.histplot(df['cognition_score'], kde=True, ax=axes[1], color='green')
    axes[1].set_title('CIS (cognition_score) 分布图')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distribution_analysis.png"))
    print(f"分布图已保存至: {os.path.join(OUTPUT_DIR, 'distribution_analysis.png')}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(x=df['spatial_score'], ax=axes[0], color='blue')
    axes[0].set_title('SIS (spatial_score) 箱线图')

    sns.boxplot(x=df['cognition_score'], ax=axes[1], color='green')
    axes[1].set_title('CIS (cognition_score) 箱线图')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_analysis.png"))
    print(f"箱线图已保存至: {os.path.join(OUTPUT_DIR, 'boxplot_analysis.png')}")

    # 3. 双变量分析
    print("\n--- 第二步：相关性分析 ---")
    correlation = df['spatial_score'].corr(df['cognition_score'], method='pearson')
    print(f"SIS 与 CIS 的 Pearson 相关系数: {correlation:.4f}")
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='spatial_score', y='cognition_score', data=df, alpha=0.3)
    plt.title(f'SIS vs CIS 散点图 (相关系数: {correlation:.4f})')
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_scatter.png"))
    print(f"相关性散点图已保存至: {os.path.join(OUTPUT_DIR, 'correlation_scatter.png')}")

    # 4. 分布形态检验 (Q-Q图)
    print("\n--- 第三步：Q-Q图检验 ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stats.probplot(df['spatial_score'], dist="norm", plot=axes[0])
    axes[0].set_title('SIS Q-Q Plot')
    stats.probplot(df['cognition_score'], dist="norm", plot=axes[1])
    axes[1].set_title('CIS Q-Q Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qq_plot.png"))
    print(f"Q-Q图已保存至: {os.path.join(OUTPUT_DIR, 'qq_plot.png')}")

if __name__ == "__main__":
    try:
        data = load_data(DATA_PATH)
        analyze_poi_scores(data)
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
