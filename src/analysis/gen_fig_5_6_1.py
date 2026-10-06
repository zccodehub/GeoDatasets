# -*- coding: utf-8 -*-
"""
图 5.6.1 生成脚本（matplotlib + seaborn 版）
统一分类体系下七项数据集的七维能力对比

数据源：IJGIS_GeoDatasets.docx 中的 Table 28（代表性基准数据集属性对比，
规模列取文档最新数值）。本脚本从 Table 28 的属性出发，计算七维归一化向量、
综合得分 S(D) 与相对优势指数 A(D)，并绘制"左：归一化热力矩阵 / 右：综合得分"图。

布局：左 = 归一化维度热力矩阵；右 = 综合数据集能力得分 S(D) 条形图。

依赖：matplotlib、seaborn、numpy（运行前请确保已安装，例如：
    pip install matplotlib seaborn numpy）

运行：
    python gen_fig_5_6_1.py
输出：
    E:/Agent_workspace/GeoDatasets/5.6_figure.svg   （矢量图，可直接嵌入论文）
    E:/Agent_workspace/GeoDatasets/5.6_figure.png   （位图，便于快速预览）
"""

import matplotlib
matplotlib.use('Agg')  # 无界面后端，适合服务器/脚本环境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
from pathlib import Path

# ---------- 0. 中文字体配置 ----------
# matplotlib 默认不含 CJK 字形，需显式指定系统中文字体，否则中文会显示为方块。
CJK_NAMES = [
    'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC',
]
available = {f.name for f in fm.fontManager.ttflist}
_chosen = None
for _c in CJK_NAMES:
    if _c in available:
        _chosen = _c
        break
else:
    # 若字体名未注册，则尝试直接加载常见字体文件
    for _p in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf',
              r'C:\Windows\Fonts\simsun.ttc']:
        if Path(_p).exists():
            fm.fontManager.addfont(_p)
            _chosen = fm.FontProperties(fname=_p).get_name()
            break
if _chosen:
    plt.rcParams['font.sans-serif'] = [_chosen]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# ---------- 1. 数据定义（来自 docx Table 28 最新数值） ----------
# 数据集顺序（行）：与 Table 28 完全一致（7 项）
DATASETS = [
    'Nominatim', 'GeoBenchmark', 'GeoCorpora', 'Touchdown',
    'GeoText-1652', 'GeoGLUE', 'AIGeoRel',
]
# 七维维度（列）
DIMS = ['规模D1', '双语D2', '关系D3', '几何D4', '标注D5', '模糊D6', '开源D7']
# 中文短标签（用于热力矩阵坐标轴）
DIMS_CN = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']

# --- 1.1 规模维度 D1 的输入：Table 28 的"规模"列（最新数值） ---
# 计量单元：各数据集"核心地理定位样本单元"的已发布数量。
N = {
    'Nominatim': 1.01e10,   # 101亿对象（服务级全量数据库，不参与 D1 归一化，赋参考上限 0.90）
    'GeoBenchmark': 39000,  # 39,000 合成问答条目（26K 二值 + 13K 多选）
    'GeoCorpora': 6711,     # 6,711 含地名标注推文
    'Touchdown': 9326,      # 9,326 导航示例
    'GeoText-1652': 316335, # 316,335 地理定位图像
    'GeoGLUE': 579042,      # 579,042 任务实例（6 任务合计）
    'AIGeoRel': 532285,      # 532,285 描述对（cn+en）
}

# --- 1.2 其余六维（D2–D7）依据 Table 28 属性直接映射（0/0.33/0.5/0.66/1.0） ---
#   映射规则（见论文 5.6.2）：
#   D2 双语：单语=0.5，中英双语=1.0
#   D3 关系覆盖：无=0，方向/参照=0.67，方向/距离/拓扑 或 四者全=1.0
#   D4 几何真值：有=1.0，部分=0.5，无=0
#   D5 标注范式：规则/人工=0.33，半自动=0.66，模板+LLM生成=1.0
#   D6 模糊性：有=1.0，部分=0.5，无=0
#   D7 开源：是=1.0
V = {
    'Nominatim':    [None, 0.50, 0.00, 1.00, 0.66, 0.50, 1.00],
    'GeoBenchmark': [None, 0.50, 1.00, 1.00, 0.66, 0.00, 1.00],
    'GeoCorpora':   [None, 0.50, 0.00, 1.00, 0.66, 0.50, 1.00],
    'Touchdown':    [None, 0.50, 0.67, 0.50, 0.33, 0.50, 1.00],
    'GeoText-1652': [None, 0.50, 0.67, 0.50, 0.33, 0.50, 1.00],
    'GeoGLUE':      [None, 0.50, 0.00, 1.00, 0.33, 0.50, 1.00],
    'AIGeoRel':     [None, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
}

# --- 1.3 D1 计算：对数最小—最大归一化（排除服务级 Nominatim） ---
#   D1(D)=max(0.05,(log10 N - log10 Nmin)/(log10 Nmax - log10 Nmin))
_bench = [d for d in DATASETS if d != 'Nominatim']
_nmin = min(np.log10(N[d]) for d in _bench)   # GeoCorpora: log10(6711)
_nmax = max(np.log10(N[d]) for d in _bench)   # GeoGLUE:   log10(579042)
D1_DETAIL = {}  # 记录逐项明细，便于核对
for d in _bench:
    raw = (np.log10(N[d]) - _nmin) / (_nmax - _nmin)
    d1 = max(0.05, raw)
    V[d][0] = round(float(d1), 2)
    D1_DETAIL[d] = (N[d], round(np.log10(N[d]), 3), round(float(d1), 2))
# Nominatim 作为参考上限，赋 0.90（不参与归一化）
V['Nominatim'][0] = 0.90
NOMINATIM_D1 = 0.90

# 综合数据集能力得分：七维归一化得分的算术平均
S = {d: round(sum(V[d]) / len(DIMS), 3) for d in DATASETS}
# 维度矩阵 (7 行 × 7 列)
M = np.array([V[d] for d in DATASETS])

# ---------- 2. 绘图：左右两个子图 ----------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2))

# ===== 左：归一化维度热力矩阵 =====
sns.heatmap(
    M, annot=True, fmt='.2f', cmap='Blues', cbar=True,
    xticklabels=DIMS_CN, yticklabels=DATASETS,
    linewidths=1, linecolor='white', ax=axL,
    vmin=0, vmax=1, annot_kws={'size': 11},
)
axL.set_title('(a) Normalized Dimension Heatmap Matrix', fontsize=12, fontweight='bold', pad=10)
axL.set_xlabel('')
axL.set_ylabel('')
plt.setp(axL.get_xticklabels(), rotation=0, ha='center', fontsize=11)
plt.setp(axL.get_yticklabels(), rotation=0, fontsize=11)

# ===== 右：综合数据集能力得分 S(D) 条形图 =====
order = DATASETS
scores = [S[d] for d in order]
colors = ['#E3454A' if d == 'AIGeoRel' else '#08306B' for d in order]
y = np.arange(len(order))
bars = axR.barh(y, scores, color=colors, height=0.62, edgecolor='white')
axR.set_yticks(y)
axR.set_yticklabels(order, fontsize=11)
axR.invert_yaxis()  # 使第一个数据集位于顶部
axR.set_ylim(len(order) - 0.5, -1.05)  # 顶部留白，供基线均值标签使用
axR.set_xlim(0, 1.12)
axR.set_title('(b) Overall Capability Scores of Datasets', fontsize=12, fontweight='bold', pad=10)
# 数值标注
for b, s in zip(bars, scores):
    axR.text(b.get_width() + 0.012, b.get_y() + b.get_height() / 2, f'{s:.3f}',
             va='center', fontsize=10, fontweight='bold', color='#222')
# 基线均值参考线（以除本数据集外的六项为基线）
baseline = [S[d] for d in DATASETS if d != 'AIGeoRel']
mean = sum(baseline) / len(baseline)
axR.axvline(mean, color='red', linestyle='--', linewidth=2)
axR.text(mean + 0.03, -0.72, f'Baseline Mean {mean:.3f}', color='#333333',fontweight='bold',
         fontsize=10, ha='left', va='center')

# 总标题
# fig.suptitle('图 5.6.1　统一分类体系下七项数据集的七维能力对比（左：热力矩阵　右：综合得分）',
#              fontsize=15, fontweight='bold', color='#08306b')

plt.tight_layout(rect=[0, 0, 1, 0.94])

# ---------- 3. 输出 ----------
OUT_DIR = Path(__file__).resolve().parents[2]
out_svg = OUT_DIR / 'output/5.6_figure.svg'
out_png = OUT_DIR / 'output/5.6_figure.png'
fig.savefig(out_svg, format='svg', dpi=300, bbox_inches='tight')
fig.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------- 4. 控制台核对输出 ----------
A = S['AIGeoRel'] / mean
print('使用字体:', _chosen)
print('--- D1 计算明细（Table 28 规模列 → 对数归一化）---')
print(f'{"数据集":<12}{"N":>12}{"log10N":>10}{"D1":>8}')
for d, (nn, lg, d1) in D1_DETAIL.items():
    print(f'{d:<12}{nn:>12}{lg:>10}{d1:>8}')
print(f'{"Nominatim(参考上限)":<12}{int(N["Nominatim"]):>12}{round(np.log10(N["Nominatim"]),3):>10}{NOMINATIM_D1:>8}')
print('--- S(D) ---')
for d in DATASETS:
    print(f'{d:<12}{S[d]:>8}')
print('baseline mean =', round(mean, 4))
print('A(本数据集) =', round(A, 3))
print('SVG ->', out_svg)
print('PNG ->', out_png)
