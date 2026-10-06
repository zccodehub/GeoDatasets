# -*- coding: utf-8 -*-
"""
Composite Salience Fusion Evaluation Script for IJGIS Section 5.2
综合显著性融合效果评估脚本（对应学术论文 5.2 章节）

PostGIS 数据库 landmarks 表字段映射：
  - cognition_score AS cis
  - spatial_score   AS sis
  - composite_score AS lis
  - priority (0: 背景 POI; 1~4: 入选地标)
"""

import os
import sys
import psycopg2
import pandas as pd
import numpy as np
import sqlalchemy
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# ---------- 配置 ----------
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')


def fetch_landmark_data(conn_str=DATABASE_URL):
    """
    从 PostGIS 数据库提取 landmarks 表的核心量化字段
    """
    print("[INFO] 正在连接 PostgreSQL/PostGIS 数据库提取数据...")
    engine = sqlalchemy.create_engine(conn_str)
    query = """
        SELECT 
            poi_id,
            
            type_1,
            type_2,
            type_3,
            cognition_score AS cis,
            spatial_score AS sis,
            composite_score AS lis,
            priority
        FROM landmarks;
    """
    df = pd.read_sql(query, engine)
    engine.dispose()
    print(f"[INFO] 成功读取 {len(df):,} 条 POI 数据。")
    return df


def analyze_parameter_sensitivity(df, output_dir):
    """
    Sigmoid 参数 x0 与 k 的敏感性网格分析及热力图绘制
    """
    print("\n================ 5.2.1 Sigmoid 参数敏感性分析 (Parameter Sensitivity) ================")

    x0_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    k_list = [4.0, 6.0, 8.0, 10.0, 12.0]

    # 全局 P75
    p75_sis = df['sis'].quantile(0.75)
    p75_cis = df['cis'].quantile(0.75)

    q1_mask = (df['sis'] >= p75_sis) & (df['cis'] >= p75_cis)
    q4_mask = (df['sis'] < p75_sis) & (df['cis'] < p75_cis)

    results = []
    suppression_grid = np.zeros((len(x0_list), len(k_list)))
    ratio_grid = np.zeros((len(x0_list), len(k_list)))

    w_s = 0.5
    s_blend = w_s * df['sis'] + (1 - w_s) * df['cis']

    for i, x0 in enumerate(x0_list):
        for j, k in enumerate(k_list):
            lis_sim = 1.0 / (1.0 + np.exp(-k * (s_blend - x0)))

            mean_all = lis_sim.mean()
            mean_q1 = lis_sim[q1_mask].mean()
            mean_q4 = lis_sim[q4_mask].mean()

            suppression_rate = (lis_sim < 0.10).mean() * 100.0
            distinction_ratio = mean_q1 / mean_q4 if mean_q4 > 0 else 0

            suppression_grid[i, j] = suppression_rate
            ratio_grid[i, j] = distinction_ratio

            results.append({
                'x0': x0,
                'k': k,
                'Mean_LIS': mean_all,
                'Q1_Mean': mean_q1,
                'Q4_Mean': mean_q4,
                'Suppression_Rate (%)': suppression_rate,
                'Distinction_Ratio': distinction_ratio
            })

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, 'sensitivity_analysis_results.csv'), index=False, encoding='utf-8-sig')
    print(res_df.to_string(index=False))

    # 绘制双热力图 Fig 5.4
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='white', font_scale=1.0, font='Times New Roman')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(suppression_grid, annot=True, fmt=".1f", xticklabels=k_list, yticklabels=x0_list,
                cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'LSR(%)'})
    axes[0].set_title('(a) Low-score Suppression Ratio (LIS < 0.10)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Steepness Slope Coefficient (k)', fontsize=10)
    axes[0].set_ylabel('Offset Threshold (μ0)', fontsize=10)

    sns.heatmap(ratio_grid, annot=True, fmt=".2f", xticklabels=k_list, yticklabels=x0_list,
                cmap='Blues', ax=axes[1], cbar_kws={'label': 'LDR(%)'})
    axes[1].set_title('(b) Landmark Distinction Ratio (Q1 Mean / Q4 Mean)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Steepness Coefficient (k)', fontsize=10)
    axes[1].set_ylabel('Offset Threshold (μ0)', fontsize=10)

    plt.tight_layout()
    fig54_path = os.path.join(output_dir, 'fig_5_4_sensitivity_analysis.png')
    plt.savefig(fig54_path, dpi=300)
    plt.close()
    print(f"[INFO] 成功生成图 5.2.1 参数敏感性分析热力图: {fig54_path}")
    return res_df

def evaluate_different_model(df):
    """
    5.2.2 非线性 Sigmoid 融合模型与基线模型对比分析
    比较：
      - Sigmoid 融合模型 (LIS)
      - 加权线性模型 (0.5 * SIS + 0.5 * CIS)
      - 乘积模型 sqrt(SIS * CIS)
    """
    print("\n================ 5.2.2 融合模型对比与区分度评价 (Fusion Models Comparison) ================")
    
    # 构建对比模型
    df['linear_fusion'] = 0.5 * df['sis'] + 0.5 * df['cis']
    df['mult_fusion'] = np.sqrt(np.maximum(df['sis'] * df['cis'], 0.0))
    
    models = {
        'Sigmoid 非线性激活模型 (LIS)': df['lis'],
        '加权线性模型 (0.5*SIS + 0.5*CIS)': df['linear_fusion'],
        '乘积模型 (sqrt(SIS*CIS))': df['mult_fusion']
    }
    
    summary_dict = {}
    for name, s in models.items():
        summary_dict[name] = {
            '均值 (Mean)': float(s.mean()),
            '标准差 (SD)': float(s.std()),
            '中位数 (Median)': float(s.median()),
            'P75 分位数': float(s.quantile(0.75)),
            '偏度 (Skewness)': float(stats.skew(s)),
            '峰度 (Kurtosis)': float(stats.kurtosis(s)),
            'IQR (四分位距)': float(s.quantile(0.75) - s.quantile(0.25))
        }
    
    summary_df = pd.DataFrame(summary_dict).T
    summary_df.to_csv(os.path.join(output_dir, 'fusion_models_comparison_results.csv'), index=False, encoding='utf-8-sig')
    print(summary_df.round(4).to_string())
    return summary_df


def evaluate_quadrant_responses(df):
    """
    5.2.3 四类典型实体的 Sigmoid 响应与解耦特征
    """
    print("\n================ 5.2.3 典型实体类型响应特征 (Quadrant Group Responses) ================")
    
    p75_sis = df['sis'].quantile(0.75)
    p75_cis = df['cis'].quantile(0.75)
    
    # 划分四类典型实体
    q1 = df[(df['sis'] >= p75_sis) & (df['cis'] >= p75_cis)]
    q2 = df[(df['sis'] < p75_sis) & (df['cis'] >= p75_cis)]
    q3 = df[(df['sis'] >= p75_sis) & (df['cis'] < p75_cis)]
    q4 = df[(df['sis'] < p75_sis) & (df['cis'] < p75_cis)]
    
    groups = {
        '双高核心实体 (Q1: High-SIS & High-CIS)': q1,
        '高认知微观实体 (Q2: Low-SIS & High-CIS)': q2,
        '高空间拓扑实体 (Q3: High-SIS & Low-CIS)': q3,
        '双低背景实体 (Q4: Low-SIS & Low-CIS)': q4
    }
    
    group_stats = {}
    for name, g_df in groups.items():
        group_stats[name] = {
            '样本数 (N)': len(g_df),
            '占比 (%)': round(len(g_df) / len(df) * 100, 2),
            'SIS 均值': float(g_df['sis'].mean()),
            'CIS 均值': float(g_df['cis'].mean()),
            'LIS 均值': float(g_df['lis'].mean()),
            'LIS 中位数': float(g_df['lis'].median()),
            'LIS 标准差': float(g_df['lis'].std())
        }
    
    group_df = pd.DataFrame(group_stats).T
    group_df.to_csv(os.path.join(output_dir, 'quadrant_responses_results.csv'), index=False,
                      encoding='utf-8-sig')
    print(group_df.round(4).to_string())
    return group_df


def plot_section_5_2_figures(df, output_dir):
    """
    绘制并导出 5.2 章节高清学术图表 (IJGIS 标准):
      - 图 5.3 (fig_5_3_fusion_comparison.png): 融合模型 (Sigmoid vs Linear vs Multiplicative) 密度分布对比
      - 图 5.4 (fig_5_4_landmark_vs_background.png): 入选地标集 vs 背景实体的 LIS 箱线图/小提琴图与累积分布曲线 (CDF)
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.1, font='Times New Roman')
    
    # ---------------- 绘制 图 5.3: 三种融合模型密度分布对比 ----------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    df['linear_fusion'] = 0.5 * df['sis'] + 0.5 * df['cis']
    df['mult_fusion'] = np.sqrt(np.maximum(df['sis'] * df['cis'], 0.0))
    
    sns.kdeplot(data=df['lis'], ax=ax, color='#B22222', linewidth=2.2, label='SNLM')
    sns.kdeplot(data=df['linear_fusion'], ax=ax, color='#104E8B', linewidth=1.8, linestyle='--', label='WLM')
    sns.kdeplot(data=df['mult_fusion'], ax=ax, color='#2ca02c', linewidth=1.8, linestyle=':', label='PM')
    
    # ax.set_title('Density Distributions of Different Salience Fusion Models', fontsize=12, fontweight='bold')
    ax.set_xlabel('LIS', fontsize=10, fontweight='bold')
    ax.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    fig53_path = os.path.join(output_dir, 'fig_5_3_fusion_comparison.png')
    plt.savefig(fig53_path, dpi=300)
    plt.close()
    print(f"[INFO] 成功导出图 5.3 融合模型对比图: {fig53_path}")
    
    # ---------------- 绘制 图 5.4: 入选地标集 vs 背景实体分布对比 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    df['group'] = df['priority'].apply(lambda x: 'Selected Landmarks (N=26,800)' if x > 0 else 'Background POIs (N=258,613)')
    palette = {'Selected Landmarks (N=26,800)': '#d62728', 'Background POIs (N=258,613)': '#7f7f7f'}
    
    # (a) 小提琴图与箱线图结合
    sns.violinplot(data=df, x='group', y='lis', palette=palette, ax=axes[0], inner='box', cut=0)
    axes[0].set_title('(a) LIS Distribution: Selected Landmarks vs. Background', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Landmark Index of Salience (LIS)', fontsize=10, fontweight='bold')
    
    # (b) 累积概率分布曲线 (CDF)
    sns.ecdfplot(data=df, x='lis', hue='group', palette=palette, ax=axes[1], linewidth=2.0)
    axes[1].set_title('(b) Empirical Cumulative Distribution Function (ECDF)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Landmark Index of Salience (LIS)', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Cumulative Probability', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig54_path = os.path.join(output_dir, 'fig_5_4_landmark_vs_background.png')
    plt.savefig(fig54_path, dpi=300)
    plt.close()
    print(f"[INFO] 成功导出图 5.4 地标与背景实体对比图: {fig54_path}")


if __name__ == '__main__':
    try:
        data_df = fetch_landmark_data()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'output'))

        # 1. 评估 5.2.1
        analyze_parameter_sensitivity(data_df,output_dir)

        # 2. 评估 5.2.2
        evaluate_different_model(data_df)
        
        # 3. 评估 5.2.3
        evaluate_quadrant_responses(data_df)

        # 5. 绘图
        plot_section_5_2_figures(data_df, output_dir)
        
        print("\n[SUCCESS] 5.2 章节综合显著性融合效果评估计算与图表生成顺利完成！")
    except Exception as e:
        print(f"[ERROR] 运行分析失败: {e}", file=sys.stderr)
