# -*- coding: utf-8 -*-
"""
Landmark Salience Dual-Dimensional Analysis Script for IJGIS Section 5.1
地标显著性双维分布特征分析脚本（对应学术论文 5.1.1 与 5.1.2 章节）

PostGIS 数据库 landmarks 表字段映射对应关系：
  - cognition_score -> CIS (Cognitive Index of Salience)
  - spatial_score   -> SIS (Spatial Index of Salience)
  - composite_score -> LIS (Landmark Index of Salience)
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
# 数据库连接配置 (可从环境变量 DATABASE_URL 获取)
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:123@localhost:5432/poi_db')


def fetch_landmark_data(conn_str=DATABASE_URL):
    """
    从 PostGIS 数据库连接并提取 landmarks 表的核心分析字段
    """
    print(f"[INFO] 正在连接 PostgreSQL/PostGIS 数据库提取 landmarks 数据...")
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
            priority,
            ST_X(geom) AS x,
            ST_Y(geom) AS y
        FROM landmarks;
    """
    df = pd.read_sql(query, engine)
    engine.dispose()
    print(f"[INFO] 成功从 PostGIS 读取 {len(df):,} 条 POI 数据。")
    return df


def analyze_section_5_1_1(df):
    """
    对应论文 5.1.1 SIS 与 CIS 的单变量描述性统计分析
    计算容量、均值、中位数、标准差、极值、偏度、峰度及关键分位数
    """
    print("\n================ 5.1.1 单变量统计描述 (Univariate Profiles) ================")
    results = {}
    for col, name in [('sis', 'SIS (空间结构显著性)'), ('cis', 'CIS (认知感知显著性)'), ('lis', 'LIS (综合显著性)')]:
        s = df[col]
        results[name] = {
            '样本容量(N)': len(s),
            '均值(Mean)': float(s.mean()),
            '中位数(Median)': float(s.median()),
            '标准差(SD)': float(s.std()),
            '最小值(Min)': float(s.min()),
            '最大值(Max)': float(s.max()),
            '偏度(Skewness)': float(stats.skew(s)),
            '峰度(Kurtosis)': float(stats.kurtosis(s)),
            'P25': float(s.quantile(0.25)),
            'P75': float(s.quantile(0.75)),
            'P90': float(s.quantile(0.90)),
            'P95': float(s.quantile(0.95))
        }
    
    stats_df = pd.DataFrame(results)
    print(stats_df.round(4).to_string())
    return stats_df


def analyze_section_5_1_2(df):
    """
    对应论文 5.1.2 双变量相关性分析与四象限聚类模式解耦
    """
    print("\n================ 5.1.2 双变量相关性与四象限解耦分析 (Bivariate & Quadrants) ================")
    sis = df['sis']
    cis = df['cis']
    
    # 1. 维度相关性检验
    pearson_r, pearson_p = stats.pearsonr(sis, cis)
    spearman_rho, spearman_p = stats.spearmanr(sis, cis)
    
    print("[1. 维度相关性检验结果]")
    print(f"  * Pearson 线性相关系数 r   : {pearson_r:.4f} (p-value: {pearson_p:.4e})")
    print(f"  * Spearman 秩相关系数 rho : {spearman_rho:.4f} (p-value: {spearman_p:.4e})")
    
    # 2. 划分四象限实体类型
    p75_sis = sis.quantile(0.75)
    p75_cis = cis.quantile(0.75)
    p80_sis = sis.quantile(0.80)
    p80_cis = cis.quantile(0.80)
    p60_sis = sis.quantile(0.60)
    p60_cis = cis.quantile(0.60)
    
    q1 = df[(df['sis'] >= p80_sis) & (df['cis'] >= p80_cis)]
    q2 = df[(df['cis'] >= p75_cis) & (df['sis'] <= p60_sis)]
    q3 = df[(df['sis'] >= p75_sis) & (df['cis'] <= p60_cis)]
    q4 = df[(df['sis'] < p75_sis) & (df['cis'] < p75_cis)]
    
    total_n = len(df)
    quadrant_stats = {
        '象限 I (地标群 Landmark Clusters P1)': {'数量': len(q1), '占比(%)': round(len(q1)/total_n*100, 2)},
        '象限 II (潜在地标 Potential Landmarks P2)': {'数量': len(q2), '占比(%)': round(len(q2)/total_n*100, 2)},
        '象限 III (空间锚点 Spatial Anchors P3)': {'数量': len(q3), '占比(%)': round(len(q3)/total_n*100, 2)},
        '象限 IV (背景实体 Background Entities P4)': {'数量': len(q4), '占比(%)': round(len(q4)/total_n*100, 2)}
    }
    
    quadrant_df = pd.DataFrame(quadrant_stats).T
    print("\n[2. 四象限实体解耦分布]")
    print(quadrant_df.to_string())
    
    return quadrant_df


def analyze_section_5_1_2_75(df):
    """对应论文 5.1.2 双变量相关性分析与基于 P75 正交四象限解耦分析"""
    print("\n================ 5.1.2 双变量相关性与四象限解耦分析 (Bivariate & Quadrants) ================")
    sis = df['sis']
    cis = df['cis']

    # 1. 维度相关性检验
    pearson_r, pearson_p = stats.pearsonr(sis, cis)
    spearman_rho, spearman_p = stats.spearmanr(sis, cis)

    print("[1. 维度相关性检验结果]")
    print(f"  * Pearson 线性相关系数 r   : {pearson_r:.4f} (p-value: {pearson_p:.4e})")
    print(f"  * Spearman 秩相关系数 rho : {spearman_rho:.4f} (p-value: {spearman_p:.4e})")

    # 2. 统一以 P75 为分界线划分严格正交四象限
    p75_sis = sis.quantile(0.75)
    p75_cis = cis.quantile(0.75)

    q1 = df[(df['sis'] >= p75_sis) & (df['cis'] >= p75_cis)]
    q2 = df[(df['sis'] < p75_sis) & (df['cis'] >= p75_cis)]
    q3 = df[(df['sis'] >= p75_sis) & (df['cis'] < p75_cis)]
    q4 = df[(df['sis'] < p75_sis) & (df['cis'] < p75_cis)]

    total_n = len(df)
    quadrant_stats = {
        '象限 I (核心地标群 Landmark Clusters P1)': {
            '阈值条件': f'SIS >= P75 ({p75_sis:.4f}) & CIS >= P75 ({p75_cis:.4f})', '数量': len(q1),
            '占比(%)': round(len(q1) / total_n * 100, 2)},
        '象限 II (潜在认知地标 Potential Landmarks P2)': {
            '阈值条件': f'SIS < P75 ({p75_sis:.4f}) & CIS >= P75 ({p75_cis:.4f})', '数量': len(q2),
            '占比(%)': round(len(q2) / total_n * 100, 2)},
        '象限 III (空间拓扑锚点 Spatial Anchors P3)': {
            '阈值条件': f'SIS >= P75 ({p75_sis:.4f}) & CIS < P75 ({p75_cis:.4f})', '数量': len(q3),
            '占比(%)': round(len(q3) / total_n * 100, 2)},
        '象限 IV (背景普通实体 Background Entities P4)': {
            '阈值条件': f'SIS < P75 ({p75_sis:.4f}) & CIS < P75 ({p75_cis:.4f})', '数量': len(q4),
            '占比(%)': round(len(q4) / total_n * 100, 2)}
    }

    quadrant_df = pd.DataFrame(quadrant_stats).T
    print("\n[2. P75 正交四象限实体解耦分布]")
    print(quadrant_df.to_string())

    return quadrant_df

def plot_academic_figures(df, output_dir):
    """
    绘制符合 IJGIS 期刊标准的图 5.1（单变量分布）与图 5.2（四象限散点图）
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.1,font='Times New Roman')

    # 绘制 图 5.1 单变量直方图与 KDE
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.histplot(df['sis'], kde=True, ax=axes[0], color='#B22222', bins=100, stat='density')
    axes[0].set_title('(a) Spatial Index of Salience (SIS) Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('SIS', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=10, fontweight='bold')
    
    sns.histplot(df['cis'], kde=True, ax=axes[1], color='#104E8B', bins=100, stat='density')
    axes[1].set_title('(b) Cognitive Index of Salience (CIS) Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('CIS', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Density', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig51_path = os.path.join(output_dir, 'fig_5_1_univariate_distribution.png')
    plt.savefig(fig51_path, dpi=300)
    plt.close()
    print(f"[INFO] 成功导出图 5.1 单变量分布图: {fig51_path}")
    
    # 绘制 图 5.2 四象限散点图
    fig, ax = plt.subplots(figsize=(8, 7))
    sample_size = min(20000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    sns.scatterplot(data=sample_df, x='sis', y='cis', alpha=0.25, s=12, color='#F37254', ax=ax)
    
    p75_sis = df['sis'].quantile(0.75)
    p75_cis = df['cis'].quantile(0.75)
    ax.axvline(x=p75_sis, color='crimson', linestyle='--', linewidth=1.5, label=f'SIS P75 ({p75_sis:.2f})')
    ax.axhline(y=p75_cis, color='navy', linestyle='--', linewidth=1.5, label=f'CIS P75 ({p75_cis:.2f})')
    
    ax.set_title('Bivariate Scatter & Four-Quadrant Decoupling (SIS vs CIS)', fontsize=12, fontweight='bold')
    ax.set_xlabel('SIS', fontsize=10, fontweight='bold')
    ax.set_ylabel('CIS', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    fig52_path = os.path.join(output_dir, 'fig_5_2_bivariate_quadrants.png')
    plt.savefig(fig52_path, dpi=300)
    plt.close()
    print(f"[INFO] 成功导出图 5.2 四象限散点图: {fig52_path}")

    # 3. 绘制 图 5.2 正交四象限散点解耦图 (P75 切线与区域标注)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sample_size = min(20000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    sns.scatterplot(data=sample_df, x='sis', y='cis', alpha=0.85, s=8, color='#2ca02c', ax=ax)

    p75_sis = df['sis'].quantile(0.78)
    p75_cis = df['cis'].quantile(0.75)

    ax.axvline(x=p75_sis, color='#B22222', linestyle='--', linewidth=1.8, label=f'SIS P75 Boundary')
    ax.axhline(y=p75_cis, color='#104E8B', linestyle='--', linewidth=1.8, label=f'CIS P75 Boundary')

    # 添加象限标签
    ax.text(0.5, 0.65, 'Q1: Core Landmarks', transform=ax.transAxes,
            fontsize=10,fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='crimson'))
    ax.text(0.01, 0.65, 'Q2: Potential Landmarks)', transform=ax.transAxes,
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='#1f77b4'))
    ax.text(0.5, 0.15, 'Q3: Spatial Anchors)', transform=ax.transAxes,
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='#ff7f0e'))
    ax.text(0.03, 0.15, 'Q4: Backup Entities)', transform=ax.transAxes,
            fontsize=10,fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

    # ax.set_title('Bivariate Scatter & Four-Quadrant Decoupling (SIS vs CIS)', fontsize=12, fontweight='bold')
    ax.set_xlabel('SIS', fontsize=10, fontweight='bold')
    ax.set_ylabel('CIS', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    fig52_path = os.path.join(output_dir, 'fig_5_2_bivariate_quadrants_75.png')
    plt.savefig(fig52_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    try:
        data_df = fetch_landmark_data()
        stats_df = analyze_section_5_1_1(data_df)
        quadrant_df = analyze_section_5_1_2_75(data_df)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'output'))
        plot_academic_figures(data_df, output_dir)
        print("\n[SUCCESS] 5.1 章节数据统计与图表绘制完整执行完成！")
    except Exception as e:
        print(f"[ERROR] 运行分析失败: {e}", file=sys.stderr)
