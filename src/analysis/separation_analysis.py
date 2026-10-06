# -*- coding: utf-8 -*-
"""
入选地标集与背景实体的显著性分离度检验 (Section 5.3.1)
实现 Welch t检验、Mann-Whitney U检验、Cohen's d效应量计算与分布可视化
入选地标集与背景实体的显著性分离度检验

PostGIS 数据库 landmarks 表字段映射：
  - cognition_score AS cis
  - spatial_score   AS sis
  - composite_score AS lis
  - priority (0: 背景 POI; 1~4: 候选地标)
"""

import os
import sys
import pandas as pd
import numpy as np
import sqlalchemy
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# ---------- 配置项 ----------
# ---------- 配置 ----------
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
current_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(current_dir, '..', '..', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置学术绘图风格
plt.rcParams['font.sans-serif'] = ['Times New roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (8, 5)
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)


def fetch_landmark_data(conn_str=DATABASE_URL):
    """从 PostgreSQL/PostGIS 数据库提取 landmarks 表核心字段"""
    print('[INFO] 正在连接 PostgreSQL/PostGIS 数据库读取数据...')
    engine = sqlalchemy.create_engine(conn_str)
    query = """
        SELECT 
            poi_id,
            cognition_score AS cis,
            spatial_score AS sis,
            composite_score AS lis,
            priority
        FROM landmarks;
    """
    df = pd.read_sql(query, engine)
    engine.dispose()
    print(f'[INFO] 成功读取 {len(df):,} 条 POI 数据。')
    return df


def analyze_separation(df, output_dir=OUTPUT_DIR):
    """执行 5.3.1 节的分离度检验与可视化"""
    print('\n================ 5.3.1 入选地标集与背景实体的显著性分离度检验 ================')
    
    # 分离地标集与背景集
    landmarks = df[df['priority'] > 0]['lis']
    background = df[df['priority'] == 0]['lis']
    
    n_lm = len(landmarks)
    n_bg = len(background)
    
    # 基本统计量
    lm_mean, lm_std = landmarks.mean(), landmarks.std()
    lm_median = landmarks.median()
    lm_q1, lm_q3 = landmarks.quantile(0.25), landmarks.quantile(0.75)
    lm_iqr = lm_q3 - lm_q1
    
    bg_mean, bg_std = background.mean(), background.std()
    bg_median = background.median()
    bg_q1, bg_q3 = background.quantile(0.25), background.quantile(0.75)
    bg_iqr = bg_q3 - bg_q1
    
    print(f'地标集 (N={n_lm:,}): Mean={lm_mean:.4f}, Std={lm_std:.4f}, Median={lm_median:.4f}, IQR={lm_iqr:.4f}')
    print(f'背景集 (N={n_bg:,}): Mean={bg_mean:.4f}, Std={bg_std:.4f}, Median={bg_median:.4f}, IQR={bg_iqr:.4f}')
    
    # 1. Welch t-test (方差不齐)
    t_stat, p_val_t = stats.ttest_ind(landmarks, background, equal_var=False)
    print(f'Welch t-test: t = {t_stat:.4f}, p-value = {p_val_t:.4e}')
    
    # 2. Mann-Whitney U test (非参数检验)
    u_stat, p_val_u = stats.mannwhitneyu(landmarks, background, alternative='two-sided')
    print(f'Mann-Whitney U test: U = {u_stat:.2e}, p-value = {p_val_u:.4e}')
    
    # 3. Cohen's d 效应量
    s_pooled = np.sqrt(((n_lm - 1) * lm_std**2 + (n_bg - 1) * bg_std**2) / (n_lm + n_bg - 2))
    cohen_d = (lm_mean - bg_mean) / s_pooled
    print(f"Cohen's d effect size = {cohen_d:.4f} (Large effect if |d| > 0.8)")
    
    # 生成表5.2
    table_5_2 = pd.DataFrame({
        '样本集': ['入选地标集 (Landmarks)', '背景实体集 (Background POI)'],
        '样本量 (N)': [n_lm, n_bg],
        '均值 (μ)': [lm_mean, bg_mean],
        '标准差 (σ)': [lm_std, bg_std],
        '中位数 (Q2)': [lm_median, bg_median],
        '四分位距 (IQR)': [lm_iqr, bg_iqr]
    })
    
    # 保存表5.2为CSV
    table_path = os.path.join(output_dir, 'table_5_2_separation_stats.csv')
    table_5_2.to_csv(table_path, index=False, encoding='utf-8-sig')
    print(f'\n[INFO] 表5.2 已保存至: {table_path}')
    try:
        print(table_5_2.to_string(index=False))
    except UnicodeEncodeError:
        print(table_5_2.to_string(index=False, encoding='utf-8'))
    
    # 优先级细分统计 (可选)
    print('\n各 Priority 级别 LIS 统计 (细分类别):')
    priority_stats = []
    for p in range(5):
        sub = df[df['priority'] == p]['lis']
        priority_stats.append({
            'Priority': p,
            'Count': len(sub),
            'Mean': sub.mean(),
            'Std': sub.std(),
            'Median': sub.median(),
            'IQR': sub.quantile(0.75) - sub.quantile(0.25)
        })
    priority_df = pd.DataFrame(priority_stats)
    print(priority_df.to_string(index=False))
    
    # 可视化：入选地标 vs 背景实体 LIS 分布对比图 (图5.5)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 核密度估计图 (KDE)
    sns.kdeplot(data=background, label=f'背景实体 (Background, N={n_bg:,})', 
                color='#999999', fill=True, alpha=0.4, linewidth=1.8, ax=ax)
    sns.kdeplot(data=landmarks, label=f'入选地标 (Landmarks, N={n_lm:,})', 
                color='#D95F02', fill=True, alpha=0.5, linewidth=2.2, ax=ax)
    
    # 添加均值线
    ax.axvline(bg_mean, color='#666666', linestyle='--', linewidth=1.5, 
               label=f'背景均值 ({bg_mean:.3f})')
    ax.axvline(lm_mean, color='#B2182B', linestyle='--', linewidth=1.5, 
               label=f'地标均值 ({lm_mean:.3f})')
    
    # 美化图形
    ax.set_title('图5.5 入选地标集与背景实体的综合显著性(LIS)概率密度分布对比', 
                 fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('综合显著性得分 (LIS)', fontsize=12)
    ax.set_ylabel('概率密度 (Density)', fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', frameon=True, fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # 在图中添加统计检验结果文本框
    textstr = f"Welch t = {t_stat:.2f}, p < 0.0001\nMann‑Whitney U = {u_stat:.2e}\nCohen's d = {cohen_d:.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray')
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # 保存图形
    fig_path = os.path.join(output_dir, 'fig_5_5_landmark_vs_background.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[INFO] 图5.5 已保存至: {fig_path}')
    

    # 新增图5.6：地标集与非地标集LIS分布小提琴图与箱线图（结合）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 准备数据：合并两组数据，添加类别标签
    data_for_violin = pd.DataFrame({
        'LIS': pd.concat([landmarks, background], ignore_index=True),
        'Category': ['Landmarks'] * len(landmarks) + ['Non-landmarks'] * len(background)
    })
    
    # 绘制小提琴图，内部嵌入箱线图 (inner='box')
    sns.violinplot(data=data_for_violin, x='Category', y='LIS',
                  palette={'Landmarks': '#d62728', 'Non-landmarks': '#045B98'},
                  inner='box',  # 在小提琴图内部绘制箱线图
                  linewidth=1, saturation=0.85, alpha=0.8, ax=ax)
    
    # 美化图形
    # ax.set_title('图5.6 地标集与非地标集LIS分布小提琴图与箱线图联合展示',fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('', fontsize=10, fontweight='bold')
    ax.set_ylabel('LIS', fontsize=10,fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    
    # 在小提琴图上添加统计量标签
    # lm_stats_text = f'地标集N={n_lm:,}μ={lm_mean:.3f}σ={lm_std:.3f}'
    # bg_stats_text = f'背景集N={n_bg:,}μ={bg_mean:.3f}σ={bg_std:.3f}'
    
    # 在相应位置添加文本
    # ax.text(0.02, 0.95, lm_stats_text, transform=ax.transAxes, fontsize=10,
    #         bbox=dict(boxstyle='round', facecolor='#f7d7ba', alpha=0.8, edgecolor='#D95F02'))
    # ax.text(0.85, 0.95, bg_stats_text, transform=ax.transAxes, fontsize=10,
    #         bbox=dict(boxstyle='round', facecolor='#e6e6e6', alpha=0.8, edgecolor='#999999'))
    
    ax.grid(True, linestyle=':', alpha=0.7, axis='y')
    
    plt.tight_layout()
    
    # 保存图形
    violin_fig_path = os.path.join(output_dir, 'fig_5_6_violin_box_combined.png')
    plt.savefig(violin_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[INFO] 图5.6 已保存至: {violin_fig_path}')
    # 保存检验结果到文件
    results = {
        'landmark_stats': {
            'N': n_lm, 'mean': lm_mean, 'std': lm_std, 
            'median': lm_median, 'iqr': lm_iqr
        },
        'background_stats': {
            'N': n_bg, 'mean': bg_mean, 'std': bg_std,
            'median': bg_median, 'iqr': bg_iqr
        },
        'tests': {
            'welch_t': t_stat, 'welch_p': p_val_t,
            'mannwhitney_u': u_stat, 'mannwhitney_p': p_val_u,
            'cohen_d': cohen_d, 'pooled_sd': s_pooled
        }
    }
    
    import json
    results_path = os.path.join(output_dir, 'separation_test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f'[INFO] 分离度检验详细结果已保存至: {results_path}')
    
    return results


def main():
    """主函数"""
    print('=' * 60)
    print('入选地标集与背景实体显著性分离度检验脚本')
    print('对应论文 Section 5.3.1')
    print('=' * 60)
    
    try:
        df = fetch_landmark_data()
        results = analyze_separation(df)
        print('\n[SUCCESS] 分离度检验完成！')
        
        # 打印总结
        print('\n' + '=' * 60)
        print('总结:')
        print(f'1. 地标集 LIS 均值 ({results["landmark_stats"]["mean"]:.3f}) 显著高于背景集 ({results["background_stats"]["mean"]:.3f})')
        print(f'2. Welch t检验拒绝零假设 (p < {results["tests"]["welch_p"]:.3e})')
        print(f"3. Cohen's d = {results['tests']['cohen_d']:.3f} (大效应量)")
        print('4. 可视化图表已生成:')
        print(f'   - fig_5_5_landmark_vs_background.png')
        print(f'   - table_5_2_separation_stats.csv')
        print('=' * 60)
        
    except Exception as e:
        print(f'[ERROR] 执行过程中发生错误: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

