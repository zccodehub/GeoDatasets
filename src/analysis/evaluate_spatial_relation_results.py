# -*- coding: utf-8 -*-
"""
空间关系计算结果与质量校验 —— 评估数据获取脚本（对应学术论文 5.2 节）

本脚本连接 PostGIS 数据库中的 spatial_relations 表，提取并复算 5.2 节
"空间关系计算结果与质量校验" 所需的全部统计量与校验指标，包括：

  5.2.1 关系记录规模与候选空间分布
        - 关系记录总数、去重地标覆盖度、平均每地标关联目标数
        - 方向 (direction_8) 分布、距离等级 (dist_level) 分布、模糊标记 (is_ambiguous) 分布
        - 总体精确距离统计：均值 (mean) 与中位数 (median)（数据集整体尺度特征）
  5.2.2 方向—距离计算的几何与逻辑一致性
        - 由 azimuth_deg / exact_distance_m 服务端复算 direction_8 / dist_level，
          与已存储值比对，输出不一致率；并校验方位角/距离范围与坐标合法性
  5.2.3 多维分层抽样的均衡性达成
        - 输出抽样后距离等级占比，与离线自然候选分布对比，计算归一化熵均衡度
  5.2.4 拓扑关系判定的可靠性验证
        - 六类拓扑关系类型分布统计（计数 / 占比 / 归一化熵均衡度）
        - 分层抽样导出专家复标样本 (CSV)；
          提供 Cohen's κ 与一致率计算函数供复标后调用
  5.2.5 计算效率与可扩展性
        - 记录规模、每地标目标数极值、关键校验查询耗时

输出：
  - Geocoding_dataset/output/evaluation_results_5_2.json  (结构化结果)
  - Geocoding_dataset/output/expert_review_sample.csv     (专家复标分层样本)
  - Geocoding_dataset/output/distribution_stats_5_2_1.xlsx  (8 方向/模糊等级/距离分级多维分布 + 方向×距离交叉表)
  - Geocoding_dataset/output/figures/fig_5_2_1a_direction.(png|svg)  图 5.2.1(a) 8 方位自然候选分布
  - Geocoding_dataset/output/figures/fig_5_2_1b_distance.(png|svg)   图 5.2.1(b) 距离分级自然候选分布
  - Geocoding_dataset/output/figures/fig_5_2_1_rose_direction.(png|svg)  图 5.2.1(c) 方向关系分布玫瑰图

纯函数 (recompute_direction_8 / recompute_dist_level / cohens_kappa /
normalized_entropy) 已在模块顶层定义，可独立单元测试，不依赖数据库连接。
"""

import os
import sys
import json
import math
import time
import argparse
import csv
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import sqlalchemy
from sqlalchemy import text

# openpyxl 用于导出多维分布统计 Excel；若运行环境未安装则 Excel 导出会给出明确提示。
try:
    import openpyxl  # noqa: F401
    _HAS_OPENPYXL = True
except Exception:  # pragma: no cover - 环境依赖差异
    _HAS_OPENPYXL = False

# Matplotlib 用于绘制 5.2.1 节分布图；若运行环境未安装则自动回退为纯 SVG 生成。
try:
    import matplotlib
    matplotlib.use("Agg")  # 无界面后端，适配服务器 / 批处理环境
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    _HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover - 不同环境依赖差异
    plt = None
    fm = None
    _HAS_MATPLOTLIB = False

# ======================== 配置 ========================
load_dotenv()
# 数据库连接配置（与环境变量 DATABASE_URL 保持一致，缺省回退到项目链接）
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://postgres:123@localhost:5432/poi_db"
)

# 5.2 章节所用有序分类（与 Spatial_relation_calculation.py 中写入值严格对应）
DIRECTION_8: List[str] = [
    "North", "Northeast", "East", "Southeast",
    "South", "Southwest", "West", "Northwest",
]
DIRECTION_8_CN: Dict[str, str] = {
    "North": "北", "Northeast": "东北", "East": "东", "Southeast": "东南",
    "South": "南", "Southwest": "西南", "West": "西", "Northwest": "西北",
}
DIRECTION_8_ABBR: Dict[str, str] = {
    "North": "N", "Northeast": "NE", "East": "E", "Southeast": "SE",
    "South": "S", "Southwest": "SW", "West": "W", "Northwest": "NW",
}
DIST_LEVEL: List[str] = ["VeryClose", "Close", "Medium", "Far"]
DIST_LEVEL_CN: Dict[str, str] = {
    "VeryClose": "极近 (0–50 m)", "Close": "近 (50–200 m)",
    "Medium": "中 (200–500 m)", "Far": "远 (500–1000 m)",
}
# 模糊等级（由 azimuth_deg 与其 direction_8 主中心偏差 δ 划分，详见 4.3.1.1 偏差角标定）
AMBIG_LEVEL: List[str] = ["EXACT", "SLIGHT", "MODERATE"]
AMBIG_LEVEL_CN: Dict[str, str] = {
    "EXACT": "精确 (δ≤3°)", "SLIGHT": "轻微 (3°<δ≤11.25°)",
    "MODERATE": "明显 (δ>11.25°)",
}
# 8 方位主中心角（度，北为 0、顺时针），用于由 azimuth_deg 反算偏差 δ
DIRECTION_8_CENTER: Dict[str, float] = {
    "North": 0.0, "Northeast": 45.0, "East": 90.0, "Southeast": 135.0,
    "South": 180.0, "Southwest": 225.0, "West": 270.0, "Northwest": 315.0,
}
TOPOLOGY_TYPES: List[str] = [
    "Coincident", "Abutting", "Opposite", "SameSide", "Intersection", "Isolated",
]
TOPOLOGY_CN: Dict[str, str] = {
    "Coincident": "重合 (T1)", "Abutting": "贴边近邻 (T2)",
    "Opposite": "隔路正对 (T3)", "SameSide": "临街同侧 (T5)",
    "Intersection": "路口把角 (T4)", "Isolated": "空间孤立 (T6)",
}
# 北京市行政边界（经度 115.7°~117.4°，纬度 39.4°~41.6°）
BEIJING_BOUNDS: Tuple[float, float, float, float] = (115.7, 39.4, 117.4, 41.6)
# 离线自然候选分布（由真实坐标全量复算得到，详见 paper_section_4_3/_relation_dist_real.json）
OFFLINE_CANDIDATE_JSON: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "paper_section_4_3", "_relation_dist_real.json",
)
EXPERT_SAMPLE_SIZE: int = 2000          # 专家复标分层样本总量
EXPERT_MIN_PER_CLASS: int = 30          # 每拓扑类最少抽取量
CANDIDATE_RADIUS_M: int = 1000          # 候选配对邻域半径（米），与 Spatial_relation_calculation 一致


# ======================== 纯函数（可独立测试） ========================
def recompute_direction_8(az: Optional[float]) -> Optional[str]:
    """
    由方位角(度) 复算 8 方位标签，逻辑与 Spatial_relation_calculation.py 写入规则严格一致。

    参数:
        az: 方位角（0–360，北为 0，顺时针），可为 None
    返回:
        8 方位英文标签；方位角非法时返回 None
    """
    if az is None:
        return None
    a: float = az % 360.0
    if (337.5 <= a <= 360.0) or (0.0 <= a <= 22.5):
        return "North"
    if 22.5 <= a <= 67.5:
        return "Northeast"
    if 67.5 <= a <= 112.5:
        return "East"
    if 112.5 <= a <= 157.5:
        return "Southeast"
    if 157.5 <= a <= 202.5:
        return "South"
    if 202.5 <= a <= 247.5:
        return "Southwest"
    if 247.5 <= a <= 292.5:
        return "West"
    if 292.5 <= a <= 337.5:
        return "Northwest"
    return None


def recompute_dist_level(d: Optional[float]) -> Optional[str]:
    """
    由直线距离(米) 复算距离等级，逻辑与写入规则严格一致。

    参数:
        d: 直线距离（米），可为 None
    返回:
        VeryClose/Close/Medium/Far；超出 (0, 1000] 返回 None
    """
    if d is None:
        return None
    if d < 50:
        return "VeryClose"
    if d < 200:
        return "Close"
    if d < 500:
        return "Medium"
    if d <= 1000:
        return "Far"
    return None


def normalized_entropy(counts: List[float]) -> float:
    """
    计算计数分布的相对均衡度（归一化香农熵，取值 0–1，越接近 1 越均衡）。

    参数:
        counts: 各类别频数列表
    返回:
        归一化熵；仅一类时返回 1.0
    """
    total: float = float(sum(c for c in counts if c > 0))
    if total <= 0:
        return 0.0
    probs: List[float] = [c / total for c in counts if c > 0]
    if len(probs) <= 1:
        # 仅单一类别存在，无法构成均衡分布
        return 0.0
    h: float = -sum(p * math.log(p) for p in probs)
    h_max: float = math.log(len(probs))
    return h / h_max if h_max > 0 else 0.0


def cohens_kappa(labels_a: List[Any], labels_b: List[Any]) -> float:
    """
    计算 Cohen's κ 一致性系数（用于专家复标可靠性评估）。

    参数:
        labels_a, labels_b: 两位标注者的标签序列（等长）
    返回:
        κ 值；完全一致返回 1.0，期望一致时返回 0.0
    """
    n: int = len(labels_a)
    if n == 0 or n != len(labels_b):
        return 0.0
    po: float = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / n
    cats: List[Any] = sorted(set(labels_a) | set(labels_b))
    ca: Counter = Counter(labels_a)
    cb: Counter = Counter(labels_b)
    pe: float = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return (po - pe) / (1.0 - pe) if (1.0 - pe) != 0 else 1.0


# ======================== 数据库连接 ========================
def get_engine() -> sqlalchemy.engine.Engine:
    """创建并返回 SQLAlchemy 引擎（只读查询使用）。"""
    print(f"[INFO] 正在连接 PostgreSQL/PostGIS 数据库: {DATABASE_URL}")
    return sqlalchemy.create_engine(DATABASE_URL)


# ======================== 5.2.1 规模与分布 ========================
def fetch_scale_and_coverage(engine: sqlalchemy.engine.Engine) -> Dict[str, Any]:
    """提取关系记录规模与地标覆盖度指标。"""
    print("\n[5.2.1] 提取关系记录规模与地标覆盖度 ...")
    sql = text("""
        SELECT
            (SELECT COUNT(*) FROM spatial_relations) AS total_records,
            COUNT(DISTINCT landmark_id)              AS distinct_landmarks,
            (SELECT COUNT(*) FROM landmarks WHERE priority > 0) AS total_landmarks
        FROM spatial_relations;
    """)
    row = engine.connect().execute(sql).mappings().first()
    total_records: int = int(row["total_records"])
    distinct_landmarks: int = int(row["distinct_landmarks"])
    total_landmarks: int = int(row["total_landmarks"])
    coverage: float = (distinct_landmarks / total_landmarks * 100.0) if total_landmarks else 0.0

    # 每地标关联目标数的统计
    per = text("""
        SELECT AVG(c) AS avg_t, MAX(c) AS max_t, MIN(c) AS min_t,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY c) AS median_t
        FROM (SELECT landmark_id, COUNT(*) AS c FROM spatial_relations
              GROUP BY landmark_id) AS t;
    """)
    r2 = engine.connect().execute(per).mappings().first()
    res = {
        "total_records": total_records,
        "distinct_landmarks": distinct_landmarks,
        "total_landmarks": total_landmarks,
        "landmark_coverage_pct": round(coverage, 2),
        "avg_targets_per_landmark": round(float(r2["avg_t"]), 3) if r2["avg_t"] else None,
        "median_targets_per_landmark": round(float(r2["median_t"]), 2) if r2["median_t"] else None,
        "max_targets_per_landmark": int(r2["max_t"]) if r2["max_t"] else None,
        "min_targets_per_landmark": int(r2["min_t"]) if r2["min_t"] else None,
    }
    print(f"        关系记录总数={res['total_records']:,}  去重地标={res['distinct_landmarks']:,}"
          f"/{res['total_landmarks']:,} (覆盖 {res['landmark_coverage_pct']}%)")
    return res


def fetch_natural_candidate_count(engine: sqlalchemy.engine.Engine) -> Dict[str, Any]:
    """
    计算自然候选配对记录数（写入 spatial_relations 之前的完整候选空间）。

    以地标(priority>0)为参照点，对其 CANDIDATE_RADIUS_M 米邻域内全部非地标
    目标(priority=0)执行空间自连接配对，统计候选配对总规模及其每地标分布。
    配对口径与 Spatial_relation_calculation.py 中 temp_candidates 的插入逻辑严格一致：
        FROM landmarks l JOIN landmarks t ON ST_DWithin(l.geom, t.geom, 半径)
        WHERE l.priority > 0 AND t.priority = 0 AND l.poi_id != t.poi_id

    注意：该查询为全表空间自连接，依赖 landmarks.geom 上的 GiST 空间索引；
          首次运行可能耗时数分钟，仅建议在离线批处理或论文统计时执行一次。

    返回:
        含 total_candidates（自然候选配对总数）、有效地标数、每地标候选数极值与中位数等。
    """
    print(f"\n[5.2.1] 计算自然候选配对记录数（邻域半径={CANDIDATE_RADIUS_M} m）...")
    t0 = time.time()
    sql = text("""
        WITH per_lm AS (
            SELECT l.poi_id AS lm, COUNT(*) AS cnt
            FROM landmarks l
            JOIN landmarks t
              ON ST_DWithin(l.geom, t.geom, :radius)
            WHERE l.priority > 0
              AND t.priority = 0
              AND l.poi_id != t.poi_id
            GROUP BY l.poi_id
        )
        SELECT
            (SELECT COUNT(*) FROM landmarks WHERE priority > 0) AS total_landmarks,
            COUNT(*)                                           AS landmarks_with_candidates,
            COALESCE(SUM(cnt), 0)                              AS total_candidates,
            ROUND(AVG(cnt), 3)                                 AS avg_candidates,
            MAX(cnt)                                           AS max_candidates,
            MIN(cnt)                                           AS min_candidates,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cnt)   AS median_candidates
        FROM per_lm;
    """)
    r = engine.connect().execute(sql, {"radius": CANDIDATE_RADIUS_M}).mappings().first()
    dt = time.time() - t0
    res = {
        "candidate_radius_m": CANDIDATE_RADIUS_M,
        "total_landmarks": int(r["total_landmarks"]),
        "landmarks_with_candidates": int(r["landmarks_with_candidates"]),
        "total_candidates": int(r["total_candidates"]),
        "avg_candidates_per_landmark": round(float(r["avg_candidates"]), 3) if r["avg_candidates"] else 0.0,
        "median_candidates_per_landmark": round(float(r["median_candidates"]), 2) if r["median_candidates"] else 0.0,
        "max_candidates_per_landmark": int(r["max_candidates"]) if r["max_candidates"] else 0,
        "min_candidates_per_landmark": int(r["min_candidates"]) if r["min_candidates"] else 0,
        "query_seconds": round(dt, 3),
    }
    print(f"        自然候选配对总数={res['total_candidates']:,}"
          f"  有效地标={res['landmarks_with_candidates']:,}/{res['total_landmarks']:,}"
          f"  每地标均值={res['avg_candidates_per_landmark']}  耗时={res['query_seconds']}s")
    return res


def fetch_distribution(engine: sqlalchemy.engine.Engine,
                       column: str,
                       order: List[str]) -> Dict[str, Dict[str, Any]]:
    """通用分布提取：按预定义顺序返回各类别计数与占比。"""
    sql = text(f"SELECT {column} AS k, COUNT(*) AS c FROM spatial_relations "
               f"WHERE {column} IS NOT NULL GROUP BY {column};")
    rows = engine.connect().execute(sql).mappings().all()
    cnt: Dict[str, int] = {str(r["k"]): int(r["c"]) for r in rows}
    total = sum(cnt.values())
    out: Dict[str, Dict[str, Any]] = {}
    for k in order:
        c = cnt.get(k, 0)
        out[k] = {"count": c, "pct": round(c / total * 100.0, 2) if total else 0.0}
    return out


def fetch_ambiguity_distribution(engine: sqlalchemy.engine.Engine) -> Dict[str, Dict[str, Any]]:
    """模糊标记分布（is_ambiguous）。"""
    print("[5.2.1] 提取模糊标记分布 ...")
    sql = text("SELECT COALESCE(is_ambiguous, FALSE) AS k, COUNT(*) AS c "
               "FROM spatial_relations GROUP BY k;")
    rows = engine.connect().execute(sql).mappings().all()
    cnt: Dict[str, int] = {("ambiguous" if r["k"] else "clear"): int(r["c"]) for r in rows}
    total = sum(cnt.values())
    out = {k: {"count": v, "pct": round(v / total * 100.0, 2)} for k, v in cnt.items()}
    print(f"        模糊(ambiguous)={out.get('ambiguous', {}).get('count', 0):,}"
          f" / 清晰(clear)={out.get('clear', {}).get('count', 0):,}")
    return out


def fetch_ambiguity_level_distribution(engine: sqlalchemy.engine.Engine) -> Dict[str, Dict[str, Any]]:
    """
    由 azimuth_deg 与其 direction_8 主中心偏差 δ 划分模糊等级（EXACT/SLIGHT/MODERATE），
    对应论文 4.3.1.1 的偏差角标定机制。返回三级模糊等级计数与占比。

    偏差 δ = min(|azimuth_deg - center|, 360 - |azimuth_deg - center|)；
    分级阈值：EXACT(δ≤3°)、SLIGHT(δ≤11.25°)、MODERATE(δ>11.25°)。
    """
    print("[5.2.1] 提取模糊等级（偏差角 δ）分布 ...")
    case_sql = " ".join(
        f"WHEN '{k}' THEN {v}" for k, v in DIRECTION_8_CENTER.items())
    sql = text(f"""
        SELECT amb, COUNT(*) AS c FROM (
            SELECT
                CASE
                    WHEN d <= 3        THEN 'EXACT'
                    WHEN d <= 11.25    THEN 'SLIGHT'
                    ELSE 'MODERATE'
                END AS amb
            FROM (
                SELECT LEAST(ABS(azimuth_deg - center),
                             360 - ABS(azimuth_deg - center)) AS d
                FROM (
                    SELECT azimuth_deg,
                        CASE direction_8 {case_sql} ELSE NULL END AS center
                    FROM spatial_relations
                    WHERE direction_8 IS NOT NULL AND azimuth_deg IS NOT NULL
                ) t1
            ) t2
        ) grouped
        GROUP BY amb;
    """)
    rows = engine.connect().execute(sql).mappings().all()
    total = sum(int(r["c"]) for r in rows)
    by_amb = {str(r["amb"]): int(r["c"]) for r in rows}
    out = {k: {"count": by_amb.get(k, 0),
               "pct": round(by_amb.get(k, 0) / total * 100.0, 2) if total else 0.0}
           for k in AMBIG_LEVEL}
    print("        模糊等级: " + ", ".join(
        f"{AMBIG_LEVEL_CN[k]}={out[k]['pct']}%" for k in AMBIG_LEVEL))
    return out


def fetch_direction_distance_crosstab(engine: sqlalchemy.engine.Engine) -> pd.DataFrame:
    """
    提取方向 × 距离分级的二维交叉分布（计数），作为"多维分布"的核心交叉表。
    行索引为 8 方位（有序），列索引为 4 个距离等级（有序）。
    """
    print("[5.2.1] 提取方向 × 距离分级交叉分布 ...")
    sql = text("""
        SELECT direction_8 AS d, dist_level AS l, COUNT(*) AS c
        FROM spatial_relations
        WHERE direction_8 IS NOT NULL AND dist_level IS NOT NULL
        GROUP BY direction_8, dist_level;
    """)
    rows = engine.connect().execute(sql).mappings().all()
    df = pd.DataFrame([{"d": str(r["d"]), "l": str(r["l"]), "c": int(r["c"])}
                       for r in rows])
    pivot = (df.pivot(index="d", columns="l", values="c")
                .reindex(index=DIRECTION_8, columns=DIST_LEVEL)
                .fillna(0).astype(int))
    pivot.index.name = "direction_8"
    pivot.columns.name = "dist_level"
    return pivot


def export_multidim_distribution_excel(engine: sqlalchemy.engine.Engine,
                                       output_path: str) -> str:
    """
    从 spatial_relations 表提取 8 方向、模糊等级、距离分级的多维分布统计，
    连同方向×距离交叉表一并写入 Excel 工作簿。

    工作表：
        - 概览 (overview)       ：总记录数、各维度类别数、各维度归一化熵均衡度
        - 方向分布 (direction)  ：direction_8 计数与占比
        - 距离分级分布 (distance)：dist_level 计数与占比
        - 模糊等级分布 (ambiguity)：EXACT/SLIGHT/MODERATE 计数与占比
        - 方向×距离交叉表 (crosstab)：各方向×距离等级计数
        - 方向×距离交叉表(行%)：按方向行归一化的占比

    参数:
        engine: 数据库引擎（基于 spatial_relations 表，必须连库）
        output_path: 输出 .xlsx 路径
    返回:
        实际写出的文件路径
    """
    if not _HAS_OPENPYXL:
        raise RuntimeError(
            "未安装 openpyxl，无法导出 Excel。请运行 `pip install openpyxl` 后重试。")
    print(f"\n[5.2.1] 导出多维分布统计 Excel -> {output_path}")

    dir_dist = fetch_distribution(engine, "direction_8", DIRECTION_8)
    dist_dist = fetch_distribution(engine, "dist_level", DIST_LEVEL)
    amb_dist = fetch_ambiguity_level_distribution(engine)
    cross = fetch_direction_distance_crosstab(engine)

    total_dir = sum(dir_dist[k]["count"] for k in DIRECTION_8)
    total_dist = sum(dist_dist[k]["count"] for k in DIST_LEVEL)
    total_amb = sum(amb_dist[k]["count"] for k in AMBIG_LEVEL)

    df_dir = pd.DataFrame([{
        "方向(direction_8)": k, "中文": DIRECTION_8_CN[k],
        "记录数": dir_dist[k]["count"], "占比(%)": dir_dist[k]["pct"],
    } for k in DIRECTION_8])

    df_dist = pd.DataFrame([{
        "距离分级(dist_level)": k, "中文": DIST_LEVEL_CN[k],
        "记录数": dist_dist[k]["count"], "占比(%)": dist_dist[k]["pct"],
    } for k in DIST_LEVEL])

    df_amb = pd.DataFrame([{
        "模糊等级(ambiguity)": k, "中文": AMBIG_LEVEL_CN[k],
        "记录数": amb_dist[k]["count"], "占比(%)": amb_dist[k]["pct"],
    } for k in AMBIG_LEVEL])

    # 交叉表行归一化占比（按方向）
    cross_pct = (cross.div(cross.sum(axis=1), axis=0)
                      .mul(100).round(2).reset_index())
    cross_out = cross.reset_index()

    df_overview = pd.DataFrame({
        "指标": [
            "关系记录总数(有效)",
            "方向维度类别数", "距离维度类别数", "模糊等级类别数",
            "方向分布归一化熵(均衡度)", "距离分布归一化熵(均衡度)",
            "模糊等级归一化熵(均衡度)",
            "方向×距离交叉表总样本数",
        ],
        "取值": [
            total_dir,
            len(DIRECTION_8), len(DIST_LEVEL), len(AMBIG_LEVEL),
            round(normalized_entropy([dir_dist[k]["count"] for k in DIRECTION_8]), 4),
            round(normalized_entropy([dist_dist[k]["count"] for k in DIST_LEVEL]), 4),
            round(normalized_entropy([amb_dist[k]["count"] for k in AMBIG_LEVEL]), 4),
            int(cross.values.sum()),
        ],
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as xw:
        df_overview.to_excel(xw, sheet_name="概览", index=False)
        df_dir.to_excel(xw, sheet_name="方向分布", index=False)
        df_dist.to_excel(xw, sheet_name="距离分级分布", index=False)
        df_amb.to_excel(xw, sheet_name="模糊等级分布", index=False)
        cross_out.to_excel(xw, sheet_name="方向×距离交叉表", index=False)
        cross_pct.to_excel(xw, sheet_name="方向×距离交叉表(行%)", index=False)

    print(f"        工作表: 概览 / 方向分布 / 距离分级分布 / 模糊等级分布 / "
          f"方向×距离交叉表 / 方向×距离交叉表(行%)")
    print(f"        方向分布归一化熵={df_overview['取值'][4]}  "
          f"距离分布归一化熵={df_overview['取值'][5]}  "
          f"模糊等级归一化熵={df_overview['取值'][6]}")
    return output_path


# ======================== 5.2.2 几何与逻辑一致性 ========================
def validate_geometric_consistency(engine: sqlalchemy.engine.Engine) -> Dict[str, Any]:
    """
    服务端复算 direction_8 与 dist_level，与已存储值比对，
    并校验方位角/距离范围与坐标合法性。返回不一致率与各类异常计数。
    """
    print("[5.2.2] 服务端复算方向/距离并校验几何与逻辑一致性 ...")
    t0 = time.time()
    sql = text("""
        WITH recomputed AS (
            SELECT id,
                CASE
                    WHEN azimuth_deg BETWEEN 337.5 AND 360 OR azimuth_deg BETWEEN 0 AND 22.5 THEN 'North'
                    WHEN azimuth_deg BETWEEN 22.5 AND 67.5 THEN 'Northeast'
                    WHEN azimuth_deg BETWEEN 67.5 AND 112.5 THEN 'East'
                    WHEN azimuth_deg BETWEEN 112.5 AND 157.5 THEN 'Southeast'
                    WHEN azimuth_deg BETWEEN 157.5 AND 202.5 THEN 'South'
                    WHEN azimuth_deg BETWEEN 202.5 AND 247.5 THEN 'Southwest'
                    WHEN azimuth_deg BETWEEN 247.5 AND 292.5 THEN 'West'
                    WHEN azimuth_deg BETWEEN 292.5 AND 337.5 THEN 'Northwest'
                    ELSE NULL
                END AS calc_dir8,
                CASE
                    WHEN exact_distance_m < 50 THEN 'VeryClose'
                    WHEN exact_distance_m < 200 THEN 'Close'
                    WHEN exact_distance_m < 500 THEN 'Medium'
                    WHEN exact_distance_m <= 1000 THEN 'Far'
                    ELSE NULL
                END AS calc_dist
            FROM spatial_relations
        )
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE recomputed.calc_dir8 IS DISTINCT FROM sr.direction_8) AS dir8_mismatch,
            COUNT(*) FILTER (WHERE recomputed.calc_dist IS DISTINCT FROM sr.dist_level)   AS dist_mismatch
        FROM recomputed
        JOIN spatial_relations sr ON sr.id = recomputed.id;
    """)
    r1 = engine.connect().execute(sql).mappings().first()
    total = int(r1["total"])
    dir8_mm = int(r1["dir8_mismatch"])
    dist_mm = int(r1["dist_mismatch"])

    # 范围与坐标合法性校验
    sql2 = text("""
        SELECT
            COUNT(*) FILTER (WHERE exact_distance_m < 0 OR exact_distance_m > 1000
                             OR exact_distance_m IS NULL) AS distance_out_of_range,
            COUNT(*) FILTER (WHERE azimuth_deg IS NULL OR azimuth_deg < 0 OR azimuth_deg > 360) AS azimuth_out_of_range,
            COUNT(*) FILTER (
                WHERE landmark_x::numeric < 115.7 OR landmark_x::numeric > 117.4
                   OR landmark_y::numeric < 39.4  OR landmark_y::numeric > 41.6
                   OR target_x::numeric  < 115.7 OR target_x::numeric  > 117.4
                   OR target_y::numeric  < 39.4  OR target_y::numeric  > 41.6
            ) AS coord_out_of_bounds,
            COUNT(*) FILTER (WHERE direction_8 IS NULL OR dist_level IS NULL
                             OR topology_type IS NULL) AS null_field
        FROM spatial_relations;
    """)
    r2 = engine.connect().execute(sql2).mappings().first()
    dt = time.time() - t0

    res = {
        "total": total,
        "direction_8_mismatch": dir8_mm,
        "direction_8_mismatch_pct": round(dir8_mm / total * 100.0, 4) if total else 0.0,
        "dist_level_mismatch": dist_mm,
        "dist_level_mismatch_pct": round(dist_mm / total * 100.0, 4) if total else 0.0,
        "distance_out_of_range": int(r2["distance_out_of_range"]),
        "azimuth_out_of_range": int(r2["azimuth_out_of_range"]),
        "coord_out_of_bounds": int(r2["coord_out_of_bounds"]),
        "null_field_records": int(r2["null_field"]),
        "consistency_pass_rate_pct": round(
            (1 - (dir8_mm + dist_mm) / (2 * total)) * 100.0, 4) if total else 100.0,
        "validation_query_seconds": round(dt, 3),
    }
    print(f"        方向不一致={res['direction_8_mismatch']:,} ({res['direction_8_mismatch_pct']}%)"
          f"  距离不一致={res['dist_level_mismatch']:,} ({res['dist_level_mismatch_pct']}%)")
    print(f"        坐标越界={res['coord_out_of_bounds']:,}  空字段={res['null_field_records']:,}"
          f"  复算耗时={res['validation_query_seconds']}s")
    return res


# ======================== 5.2.3 抽样均衡性 ========================
def analyze_sampling_balance(engine: sqlalchemy.engine.Engine) -> Dict[str, Any]:
    """输出抽样后距离等级占比，并对照离线自然候选分布计算均衡度。"""
    print("[5.2.3] 分析多维分层抽样均衡性 ...")
    dist = fetch_distribution(engine, "dist_level", DIST_LEVEL)
    final_pct = [dist[k]["pct"] for k in DIST_LEVEL]
    balance = normalized_entropy([dist[k]["count"] for k in DIST_LEVEL])

    # 对照离线自然候选分布（若存在）
    offline_pct: Optional[List[float]] = None
    if os.path.exists(OFFLINE_CANDIDATE_JSON):
        with open(OFFLINE_CANDIDATE_JSON, "r", encoding="utf-8") as f:
            offline = json.load(f)
        offline_pct = [offline["grade_pct"][DIST_LEVEL.index(k)] for k in DIST_LEVEL]

    res = {
        "final_dist_level_pct": dict(zip(DIST_LEVEL, final_pct)),
        "balance_index_normalized_entropy": round(balance, 4),
        "offline_candidate_dist_level_pct": offline_pct,
    }
    print(f"        抽样后距离等级占比: " + ", ".join(
        f"{DIST_LEVEL_CN[k]}={dist[k]['pct']}%" for k in DIST_LEVEL))
    print(f"        距离维度均衡度(归一化熵)={balance:.4f}")
    if offline_pct:
        print(f"        自然候选距离占比(对照): " + ", ".join(
            f"{DIST_LEVEL[k]}={offline_pct[k]}%" for k in range(len(DIST_LEVEL))))
    return res


# ======================== 5.2.4 拓扑可靠性 ========================
def fetch_topology_distribution(engine: sqlalchemy.engine.Engine) -> Dict[str, Dict[str, Any]]:
    """拓扑类型分布。"""
    print("[5.2.4] 提取拓扑关系类型分布 ...")
    dist = fetch_distribution(engine, "topology_type", TOPOLOGY_TYPES)
    for k in TOPOLOGY_TYPES:
        print(f"        {TOPOLOGY_CN[k]:<14}: {dist[k]['count']:,} ({dist[k]['pct']}%)")
    return dist


def fetch_topology_distribution_stats(engine: sqlalchemy.engine.Engine) -> Dict[str, Any]:
    """
    六类拓扑关系类型分布统计（Results 章节要求）。

    基于 spatial_relations.topology_type 统计六类拓扑关系
    (Coincident / Abutting / Opposite / SameSide / Intersection / Isolated)
    的计数与占比，并给出拓扑类型的归一化熵均衡度（衡量六类是否均衡分布）。

    返回:
        {
          "distribution": {拓扑类型: {"count", "pct"}, ...},   # 按 TOPOLOGY_TYPES 有序
          "total_topology_records": int,                       # 已标注拓扑的记录数
          "balance_index_normalized_entropy": float,           # 0–1，越接近 1 越均衡
        }
    """
    print("\n[统计] 提取六类拓扑关系类型分布 ...")
    dist = fetch_distribution(engine, "topology_type", TOPOLOGY_TYPES)
    total = sum(dist[k]["count"] for k in TOPOLOGY_TYPES)
    balance = normalized_entropy([dist[k]["count"] for k in TOPOLOGY_TYPES])
    res = {
        "distribution": dist,
        "total_topology_records": total,
        "balance_index_normalized_entropy": round(balance, 4),
    }
    for k in TOPOLOGY_TYPES:
        print(f"        {TOPOLOGY_CN[k]:<14}: {dist[k]['count']:,} ({dist[k]['pct']}%)")
    print(f"        拓扑类型总数(已标注)={total:,}  归一化熵均衡度={balance:.4f}")
    return res


def export_expert_sample(engine: sqlalchemy.engine.Engine,
                         output_path: str,
                         n_total: int = EXPERT_SAMPLE_SIZE) -> Dict[str, Any]:
    """
    按 topology_type 比例分层抽样，导出专家复标样本 CSV（含空白标注列）。

    参数:
        engine: 数据库引擎
        output_path: 输出 CSV 路径
        n_total: 样本总量
    返回:
        各类别实际抽取数量字典
    """
    print(f"[5.2.4] 导出专家复标分层样本 (n={n_total}) -> {output_path}")
    counts = engine.connect().execute(
        text("SELECT topology_type AS k, COUNT(*) AS c FROM spatial_relations "
             "WHERE topology_type IS NOT NULL GROUP BY topology_type;")
    ).mappings().all()
    cnt_map = {str(r["k"]): int(r["c"]) for r in counts}
    total = sum(cnt_map.values())

    # 比例分配 + 最小样本保底
    alloc: Dict[str, int] = {}
    for k in TOPOLOGY_TYPES:
        c = cnt_map.get(k, 0)
        nk = max(EXPERT_MIN_PER_CLASS, round(n_total * c / total)) if total else 0
        alloc[k] = min(nk, c)

    cols = ["id", "landmark_name", "target_name", "exact_distance_m",
            "direction_8", "direction_16", "topology_type", "dist_level",
            "road_name", "landmark_x", "landmark_y", "target_x", "target_y",
            "expert1_label", "expert2_label"]
    written: Dict[str, int] = {}
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in TOPOLOGY_TYPES:
            nk = alloc.get(k, 0)
            if nk <= 0:
                continue
            q = text(
                "SELECT id, landmark_name, target_name, exact_distance_m, direction_8, "
                "direction_16, topology_type, dist_level, road_name, landmark_x, "
                "landmark_y, target_x, target_y "
                "FROM spatial_relations WHERE topology_type = :topo "
                "ORDER BY random() LIMIT :lim;"
            )
            rows = engine.connect().execute(
                q, {"topo": k, "lim": nk}).fetchall()
            for r in rows:
                w.writerow(list(r) + ["", ""])
            written[k] = len(rows)
    print(f"        实际导出: " + ", ".join(
        f"{TOPOLOGY_CN[k]}={written.get(k, 0)}" for k in TOPOLOGY_TYPES
        if written.get(k, 0) > 0))
    return written


def evaluate_expert_sample(csv_path: str) -> Dict[str, Any]:
    """
    读取已标注样本 CSV，计算专家一致率与 Cohen's κ。
    标注列: expert1_label, expert2_label（与 topology_type 取值一致）。
    """
    print(f"[5.2.4] 评估专家复标样本: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    a = df["expert1_label"].dropna().astype(str).tolist()
    b = df["expert2_label"].dropna().astype(str).tolist()
    # 仅取两位均标注的行
    pairs = [(x, y) for x, y in zip(a, b) if x and y and x != "nan" and y != "nan"]
    if not pairs:
        print("        [WARN] 未找到双专家标注行，无法计算 κ。")
        return {"agreement_rate": None, "cohens_kappa": None, "n": 0}
    la = [p[0] for p in pairs]
    lb = [p[1] for p in pairs]
    agree = sum(1 for x, y in pairs if x == y) / len(pairs)
    kappa = cohens_kappa(la, lb)
    print(f"        一致率={agree:.4f}  Cohen's κ={kappa:.4f}  (n={len(pairs)})")
    return {"agreement_rate": round(agree, 4),
            "cohens_kappa": round(kappa, 4), "n": len(pairs)}


# ======================== 5.2.5 效率与可扩展性 ========================
def analyze_efficiency(engine: sqlalchemy.engine.Engine,
                       consistency_sec: float) -> Dict[str, Any]:
    """计算效率指标：记录规模、分批粒度、关键校验查询耗时。"""
    print("[5.2.5] 分析计算效率与可扩展性 ...")
    # 表行数（以 spatial_relations 为准）
    total = int(engine.connect().execute(
        text("SELECT COUNT(*) FROM spatial_relations;")).scalar())
    res = {
        "total_records": total,
        "key_validation_query_seconds": round(consistency_sec, 3),
        "approx_throughput_records_per_sec": round(
            total / consistency_sec, 1) if consistency_sec > 0 else None,
    }
    print(f"        记录规模={res['total_records']:,}  "
          f"校验吞吐≈{res['approx_throughput_records_per_sec']} 条/秒")
    return res


# ======================== 方向关系分布玫瑰图（极坐标） ========================
_ROSE_PALETTE: List[str] = [
    # "#F37252", "#6F78B9","#F7935A","#779FC6",
    # "#FCB481","#7FC6D3", "#FED297","#ABE0E4"
    # "#5F89B1", "#E5B5B5", "#7FC6D3", "#F37252",
    # "#104E8B", "#B22222", "#376B9E", "#D89090"

    "#DD161D", "#C1D9ED", "#FC512B", "#9DCAE1",
    "#FECE6A", "#5CA4D0", "#FFFFCC", "#1562A9"
]
# 模糊等级配色：随模糊程度递增由冷转暖（精确=青绿 / 轻微=琥珀 / 明显=珊瑚红），
# 与方向玫瑰图同属 5.2.1 多维分布可视化，配色语义化、彼此区分度高。
_AMBIG_PALETTE: List[str] = ["#9DCAE1", "#5CA4D0", "#1562A9"]


def _save_rose_matplotlib(out_path: str, title: str,
                          values: List[float], labels: List[str],
                          counts: Optional[List[Optional[int]]] = None) -> None:
    """使用 matplotlib 极坐标投影绘制方向关系分布玫瑰图（北朝上、顺时针），
    样式与 _save_rose_svg 保持一致：浅灰同心圆网格、半透明扇区、扇区内标注占比。"""
    f0 = _cjk_font()
    if f0:
        plt.rcParams["font.family"] = "Times New roman"
    plt.rcParams["axes.unicode_minus"] = False
    n = len(values)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n
    vmax = max(values) if max(values) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=300, edgecolor='#333', linewidth=2,
                            subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#ffffff")
    ax.set_theta_zero_location("N")   # 方位角 0（北）指向正上方
    ax.set_theta_direction(-1)        # 顺时针（北→东→南→西）

    # 同心圆网格（25/50/75/100%），浅灰细线 + 刻度标注，风格同 _save_rose_svg
    ax.set_ylim(0, vmax * 1.2)
    rticks = [vmax * f for f in (0.25, 0.5, 0.75, 1.0)]
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{vmax * f:.1f}%" for f in (0.25, 0.5, 0.75, 1.0)],
                       fontsize=8, color="#FFF")
    ax.set_rlabel_position(90)        # 刻度标签置于正上方
    ax.grid(True, linestyle="-", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.spines["polar"].set_color("#e3e3e3")
    ax.spines["polar"].set_linewidth(0.8)

    # 扇区：半透明填充 + 细深边，与 SVG 一致
    def _hex_alpha(hexc: str, a: float):
        h = hexc.lstrip("#")
        return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0,
                int(h[4:6], 16) / 255.0, a)
    colors = [_hex_alpha(_ROSE_PALETTE[i % len(_ROSE_PALETTE)], 1)
              for i in range(n)]
    bars = ax.bar(angles, values, width=width, color=colors,
                  edgecolor="#1f2d3d", linewidth=0.5, zorder=3, alpha=0.8)

    # 占比标注：扇区中点半径处（约 0.62·value），与 SVG 一致
    for i, b in enumerate(bars):
        rad = b.get_height()
        if rad <= 0:
            continue
        ax.text(angles[i], rad * 0.62, f"{values[i]:.2f}%",
                ha="center", va="center", fontsize=9, color="#111", zorder=5)
        if counts and counts[i] is not None:
            # n= 标注置于外圈（与 SVG 一致的固定半径 vmax·1.12）
            ax.text(angles[i], vmax * 1.12, f"n={counts[i]:,}",
                    ha="center", va="center", fontsize=8, color="#666", zorder=5)

    ax.set_xticks(angles)
    ax.set_xticklabels([labels[i] for i in range(n)], fontsize=11, color="#111",fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#222", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def _save_rose_svg(out_path: str, title: str,
                   values: List[float], labels: List[str],
                   counts: Optional[List[Optional[int]]] = None) -> None:
    """纯 Python 回退：将方向关系分布玫瑰图导出为 SVG（极坐标柱状/扇区）。"""
    W = H = 640
    cx = cy = W / 2
    R = 232                       # 100% 值对应的最大半径
    vmax = max(values) if max(values) > 0 else 1.0
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="\'Microsoft YaHei\',\'SimHei\',\'Noto Sans CJK SC\',sans-serif">']
    p.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    p.append(f'<text x="{cx:.0f}" y="28" text-anchor="middle" font-size="15" '
             f'font-weight="bold" fill="#222">{_xml(title)}</text>')

    def _pt(r: float, deg: float) -> Tuple[float, float]:
        a = math.radians(deg)
        return (cx + r * math.cos(a), cy - r * math.sin(a))

    # 同心圆网格（25/50/75/100%）
    for frac in (0.25, 0.5, 0.75, 1.0):
        rr = R * frac
        p.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{rr:.1f}" '
                 f'fill="none" stroke="#dddddd" stroke-width="1"/>')
        p.append(f'<text x="{cx + 4:.0f}" y="{cy - rr + 12:.0f}" font-size="9" '
                 f'fill="#888">{vmax * frac:.1f}%</text>')
    # 轴线
    for i in range(len(values)):
        deg = 90.0 - i * 45.0     # 北朝上、顺时针
        x, y = _pt(R * 1.04, deg)
        p.append(f'<line x1="{cx:.0f}" y1="{cy:.0f}" x2="{x:.1f}" y2="{y:.1f}" '
                 f'stroke="#e3e3e3" stroke-width="1"/>')

    n = len(values)
    half = 22.5                   # 半扇区角（45°/2）
    for i in range(n):
        deg = 90.0 - i * 45.0
        rv = (values[i] / vmax) * R
        if rv <= 0:
            continue
        s0, e0 = deg - half, deg + half
        x0, y0 = _pt(rv, s0)
        x1, y1 = _pt(rv, e0)
        col = _ROSE_PALETTE[i % len(_ROSE_PALETTE)]
        p.append(f'<path d="M{cx:.1f},{cy:.1f} L{x0:.1f},{y0:.1f} '
                 f'A{rv:.1f},{rv:.1f} 0 0 1 {x1:.1f},{y1:.1f} Z" '
                 f'fill="{col}" fill-opacity="0.85" stroke="#1f2d3d" stroke-width="0.8"/>')
        # 占比标注（扇区中点半径处）
        mx, my = _pt(rv * 0.62, deg)
        p.append(f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" '
                 f'font-size="9" fill="#111">{values[i]:.2f}%</text>')
        if counts and counts[i] is not None:
            lx, ly = _pt(R * 1.12, deg)
            p.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
                     f'font-size="8.5" fill="#666">n={counts[i]:,}</text>')
        # 方向标签（最外圈）
        tx, ty = _pt(R * 1.2, deg)
        p.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
                 f'font-size="11" fill="#333">{_xml(labels[i])}</text>')
    p.append('</svg>')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(p))


def generate_direction_rose_chart(engine: Optional[sqlalchemy.engine.Engine],
                                  out_dir: str,
                                  use_offline: bool = False) -> str:
    """
    绘制方向关系分布玫瑰图（图 5.2.1(c)）。

    参数:
        engine: 数据库引擎（use_offline=True 且离线 JSON 存在时可传 None）
        out_dir: 输出根目录，图写入其子目录 figures/
        use_offline: 优先读取离线自然候选分布 JSON 绘制（跳过数据库重算）
    返回:
        实际写出文件的路径（PNG 或 SVG）
    """
    print("\n[5.2.1] 生成方向关系分布玫瑰图（图 5.2.1(c)）...")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if use_offline and os.path.exists(OFFLINE_CANDIDATE_JSON):
        with open(OFFLINE_CANDIDATE_JSON, "r", encoding="utf-8") as f:
            offline = json.load(f)
        dir_dist = {k: {"count": int(offline["dir8"].get(k, 0)),
                        "pct": offline["dir8_pct"][DIRECTION_8.index(k)]}
                    for k in DIRECTION_8}
        print("        使用离线自然候选分布 JSON 绘制（跳过数据库重算）。")
    else:
        if engine is None:
            raise ValueError("缺少数据库引擎且离线 JSON 不可用，无法绘制玫瑰图。")
        dir_dist = fetch_distribution(engine, "direction_8", DIRECTION_8)

    values = [dir_dist[k]["pct"] for k in DIRECTION_8]
    counts = [dir_dist[k]["count"] for k in DIRECTION_8]
    labels = [DIRECTION_8_ABBR[k] for k in DIRECTION_8]

    out = os.path.join(fig_dir, "fig_5_2_1_rose_direction.png")
    if _HAS_MATPLOTLIB:
        _save_rose_matplotlib(out, "",
                              values, labels)
        print(f"        玫瑰图 -> {out}")
        return out
    svg = os.path.splitext(out)[0] + ".svg"
    _save_rose_svg(svg, "图 5.2.1(c) 方向关系分布玫瑰图",
                   values, labels, counts)
    print(f"        玫瑰图(SVG 回退) -> {svg}")
    return svg


# ======================== 方向模糊等级分布柱形图 ========================
def _save_ambiguity_bar_matplotlib(out_path: str, title: str,
                                   levels: List[str],
                                   values: List[float],
                                   counts: List[int]) -> None:
    """
    使用 matplotlib 绘制方向模糊等级分布柱形图。

    设计要点（与方向玫瑰图 5.2.1(c) 保持一致）：
        - 同尺寸 figsize=(6.4, 6.0)、dpi=300、白底；
        - 每个等级一根柱，按 _AMBIG_PALETTE 多色填充（冷→暖语义化）；
        - 浅灰横向网格 + 隐藏上/右边框，风格同玫瑰图。
    """
    f0 = _cjk_font()
    if f0:
        plt.rcParams["font.family"] = f0
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=300,edgecolor='#333', linewidth=2)
    fig.patch.set_facecolor("#ffffff")
    n = len(levels)
    bars = ax.bar(range(n), values,
                  color=[_AMBIG_PALETTE[i % len(_AMBIG_PALETTE)] for i in range(n)],
                  edgecolor="#4B4B4B", linewidth=0.5, width=0.5, zorder=3, alpha=0.9)
    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [levels[i] for i in range(n)],
        fontsize=11, color="#111",fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=11, color="#111",fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", color="#222", pad=14)
    vmax = max(values) if max(values) > 0 else 1.0
    ax.set_ylim(0, vmax * 1.22)
    ax.grid(axis="y", linestyle="-", color="#dddddd", linewidth=0.8, alpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#e3e3e3")
    ax.spines["bottom"].set_color("#e3e3e3")
    for i, b in enumerate(bars):
        txt = f"{values[i]:.2f}%"
        if i < len(counts):
            txt += f"\n(n={counts[i]:,})"
        ax.annotate(txt, (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=10, color="#000", zorder=5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def _save_ambiguity_bar_svg(out_path: str, title: str,
                            levels: List[str],
                            values: List[float],
                            counts: List[int]) -> None:
    """纯 Python 回退：方向模糊等级分布柱形图导出为 SVG（无第三方库依赖）。"""
    W, H = 640, 600
    ml, mr, mt, mb = 96, 32, 64, 150
    pw = W - ml - mr
    ph = H - mt - mb
    vmax = max(values) if max(values) > 0 else 1.0
    ytop = vmax * 1.2
    n = len(levels)
    slot = pw / n
    bw = slot * 0.5
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H} " '
         f'font-family="\'Microsoft YaHei\',\'SimHei\',\'Noto Sans CJK SC\',sans-serif">']
    p.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    p.append(f'<text x="{W/2:.0f}" y="34" text-anchor="middle" font-size="16" '
             f'font-weight="bold" fill="#222">{_xml(title)}</text>')
    p.append(f'<text x="22" y="{mt+ph/2:.0f}" text-anchor="middle" font-size="13" '
             f'fill="#333" transform="rotate(-90 22 {mt+ph/2:.0f})">占比 (%)</text>')
    ticks = 5
    for t in range(ticks + 1):
        val = ytop * t / ticks
        y = mt + ph - (val / ytop) * ph
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                 f'stroke="#dddddd" stroke-width="1"/>')
        p.append(f'<text x="{ml-10}" y="{y+4:.1f}" text-anchor="end" font-size="10" '
                 f'fill="#555">{val:.1f}</text>')
    p.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#e3e3e3" stroke-width="1.2"/>')
    p.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#e3e3e3" stroke-width="1.2"/>')
    for i in range(n):
        cx = ml + slot * (i + 0.5)
        x = cx - bw / 2
        h = (values[i] / ytop) * ph
        y = mt + ph - h
        col = _AMBIG_PALETTE[i % len(_AMBIG_PALETTE)]
        p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                 f'fill="{col}" stroke="#1f2d3d" stroke-width="0.9"/>')
        ann = f"{values[i]:.2f}%"
        if i < len(counts):
            ann += f" (n={counts[i]:,})"
        p.append(f'<text x="{cx:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="11" '
                 f'fill="#111">{_xml(ann)}</text>')
        p.append(f'<text x="{cx:.1f}" y="{mt+ph+30:.0f}" text-anchor="middle" font-size="12" '
                 f'fill="#333">{_xml(AMBIG_LEVEL_CN[levels[i]])}</text>')
        p.append(f'<text x="{cx:.1f}" y="{mt+ph+48:.0f}" text-anchor="middle" font-size="10" '
                 f'fill="#777">{_xml(levels[i])}</text>')
    p.append('</svg>')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(p))


def generate_ambiguity_level_bar_chart(engine: Optional[sqlalchemy.engine.Engine],
                                       out_dir: str,
                                       use_offline: bool = False) -> str:
    """
    绘制方向模糊等级（EXACT/SLIGHT/MODERATE）分布柱形图（图 5.2.1(d)）。

    模糊等级由 azimuth_deg 与其 direction_8 主中心偏差 δ 划分（详见 4.3.1.1 偏差角标定），
    反映方向描述的精确程度，是 5.2.1 多维分布特征的关键维度之一。

    参数:
        engine: 数据库引擎（模糊等级需由 spatial_relations 实算，离线 JSON 不含该维度）
        out_dir: 输出根目录，图写入其子目录 figures/
        use_offline: 为兼容 rose 图调用约定保留，模糊等级始终由数据库实算（忽略该参数）
    返回:
        实际写出文件的路径（PNG 或 SVG 回退）
    """
    print("\n[5.2.1] 生成方向模糊等级分布柱形图（图 5.2.1(d)）...")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    if engine is None:
        raise ValueError("模糊等级分布需由 spatial_relations 实算，缺少数据库引擎时无法绘制。")
    amb_dist = fetch_ambiguity_level_distribution(engine)
    levels = AMBIG_LEVEL
    values = [amb_dist[k]["pct"] for k in levels]
    counts = [amb_dist[k]["count"] for k in levels]

    out = os.path.join(fig_dir, "fig_5_2_1d_ambiguity_bar.png")
    if _HAS_MATPLOTLIB:
        _save_ambiguity_bar_matplotlib(out, "",
                                       levels, values, counts)
        print(f"        模糊等级柱形图 -> {out}")
        return out
    svg = os.path.splitext(out)[0] + ".svg"
    _save_ambiguity_bar_svg(svg, "图 5.2.1(d) 方向模糊等级分布",
                            levels, values, counts)
    print(f"        模糊等级柱形图(SVG 回退) -> {svg}")
    return svg


# ======================== 距离分级对比哑铃图（自然候选 vs 最终数据集） ========================
_DIST_NAT_COLOR: str = "#045B98"    # 自然候选空间（基线，蓝色）
_DIST_FINAL_COLOR: str = "#A91F1F"  # 最终数据集（目标，红色高亮）


def _save_dumbbell_matplotlib(out_path: str, title: str, cats: List[str],
                              nat: List[float], final: List[float],
                              nat_cnt: List[int], final_cnt: List[int],
                              labels: List[str]) -> None:
    """
    使用 matplotlib 绘制距离分级对比哑铃图（横向）。

    设计要点：
        - 每个距离分级一条水平杆，两端分别为自然候选（灰）与最终数据集（蓝）占比圆点；
        - 杆长 = 占比变化幅度，直观呈现分层抽样对长尾自然分布的再平衡效果；
        - 中点标注 Δpp（最终−自然），强化"校正量"叙事；
        - 横向布局、尺寸 7.6×4.6（与玫瑰图解耦，更适合类别数少的对比）。
    """
    f0 = _cjk_font()
    if f0:
        plt.rcParams["font.family"] = f0
    plt.rcParams["axes.unicode_minus"] = False
    n = len(cats)
    y = list(range(n))
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    xmax = max(max(nat), max(final)) * 1.12
    xmin = -6.0
    # 连接线（哑铃杆）
    for i in range(n):
        ax.plot([nat[i], final[i]], [y[i], y[i]],
                color="#c9d1da", linewidth=3.2, zorder=1, solid_capstyle="round")
    # 圆点：自然候选（蓝）/ 最终数据集（红）
    ax.scatter(nat, y, s=70, color=_DIST_NAT_COLOR, edgecolor="#203A5E",
               linewidth=1.1, zorder=3, label="Natural Candidate Pool")
    ax.scatter(final, y, s=90, color=_DIST_FINAL_COLOR, edgecolor="#811216",
               linewidth=1.1, zorder=4, label="Final Dataset")
    # 端点标注：自然(左) / 最终(右，加粗)；中点标注 Δpp
    for i in range(n):
        ax.annotate(f"{nat[i]:.2f}%", (nat[i], y[i]),
                    xytext=(15, 10), textcoords="offset points",
                    ha="right", va="center", fontsize=9.5, color="#1f4e79", fontweight="bold")
        ax.annotate(f"{final[i]:.2f}%", (final[i], y[i]),
                    xytext=(-10, 10), textcoords="offset points",
                    ha="left", va="center", fontsize=9.5, color="#A91F1F",
                    fontweight="bold")
        d = final[i] - nat[i]
        ax.annotate(f"Δ{d:+.1f}%", ((nat[i] + final[i]) / 2, y[i]),
                    xytext=(0, -12), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{labels[i]}" for i in range(n)], fontsize=10.5)
    ax.set_ylim(n - 0.5, -0.5)           # VeryClose 在顶部，Far 在底部
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Percentage(%)", fontsize=11, color="#111")
    ax.set_title(title, fontsize=13, fontweight="bold", color="#222", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4, color="#dddddd")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def _save_dumbbell_svg(out_path: str, title: str, cats: List[str],
                       nat: List[float], final: List[float],
                       nat_cnt: List[int], final_cnt: List[int],
                       labels: List[str]) -> None:
    """纯 Python 回退：距离分级对比哑铃图导出为 SVG（无第三方库依赖）。"""
    W, H = 820, 470
    ml, mr, mt, mb = 122, 120, 60, 84
    pw = W - ml - mr
    ph = H - mt - mb
    xmax = max(max(nat), max(final)) * 1.12
    xmin = -6.0
    n = len(cats)
    rowh = ph / n

    def X(v: float) -> float:
        return ml + (v - xmin) / (xmax - xmin) * pw

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="\'Microsoft YaHei\',\'SimHei\',\'Noto Sans CJK SC\',sans-serif">']
    p.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    p.append(f'<text x="{W/2:.0f}" y="32" text-anchor="middle" font-size="16" '
             f'font-weight="bold" fill="#222">{_xml(title)}</text>')
    # x 网格 + 刻度
    ticks = 5
    for t in range(ticks + 1):
        val = xmax * t / ticks
        x = X(val)
        p.append(f'<line x1="{x:.1f}" y1="{mt:.0f}" x2="{x:.1f}" y2="{mt+ph:.0f}" '
                 f'stroke="#eef0f2" stroke-width="1"/>')
        p.append(f'<text x="{x:.1f}" y="{mt+ph+18:.0f}" text-anchor="middle" font-size="9.5" '
                 f'fill="#888">{val:.0f}</text>')
    p.append(f'<text x="{W/2:.0f}" y="{mt+ph+38:.0f}" text-anchor="middle" font-size="11" '
             f'fill="#333">占比 (%)</text>')
    for i in range(n):
        yc = mt + rowh * (i + 0.5)
        xn, xf = X(nat[i]), X(final[i])
        p.append(f'<line x1="{xn:.1f}" y1="{yc:.1f}" x2="{xf:.1f}" y2="{yc:.1f}" '
                 f'stroke="#c9d1da" stroke-width="4" stroke-linecap="round"/>')
        p.append(f'<circle cx="{xn:.1f}" cy="{yc:.1f}" r="6" fill="{_DIST_NAT_COLOR}" '
                 f'stroke="#5b6675" stroke-width="1.1"/>')
        p.append(f'<circle cx="{xf:.1f}" cy="{yc:.1f}" r="7" fill="{_DIST_FINAL_COLOR}" '
                 f'stroke="#1f4e79" stroke-width="1.1"/>')
        p.append(f'<text x="{xn-9:.1f}" y="{yc+4:.1f}" text-anchor="end" font-size="11" '
                 f'fill="#5b6675">{nat[i]:.2f}%</text>')
        p.append(f'<text x="{xf+9:.1f}" y="{yc+4:.1f}" text-anchor="start" font-size="11" '
                 f'font-weight="bold" fill="#1f4e79">{final[i]:.2f}%</text>')
        d = final[i] - nat[i]
        p.append(f'<text x="{(xn+xf)/2:.1f}" y="{yc+22:.1f}" text-anchor="middle" '
                 f'font-size="9.5" fill="#9aa0a6">Δ{d:+.1f}pp</text>')
        # 左侧类别标签（两行：中文 + 英文）
        p.append(f'<text x="{ml-12:.0f}" y="{yc-3:.0f}" text-anchor="end" font-size="12" '
                 f'fill="#333">{_xml(labels[i])}</text>')
        p.append(f'<text x="{ml-12:.0f}" y="{yc+14:.0f}" text-anchor="end" font-size="9.5" '
                 f'fill="#777">{_xml(cats[i])}</text>')
    # 图例
    ly = H - 30
    p.append(f'<circle cx="300" cy="{ly-4:.0f}" r="6" fill="{_DIST_NAT_COLOR}" '
             f'stroke="#5b6675" stroke-width="1.1"/>')
    p.append(f'<text x="312" y="{ly:.0f}" font-size="11" fill="#333">自然候选空间</text>')
    p.append(f'<circle cx="440" cy="{ly-4:.0f}" r="7" fill="{_DIST_FINAL_COLOR}" '
             f'stroke="#1f4e79" stroke-width="1.1"/>')
    p.append(f'<text x="452" y="{ly:.0f}" font-size="11" fill="#333">最终数据集</text>')
    p.append('</svg>')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(p))


def generate_distance_grade_compare_dumbbell(engine: Optional[sqlalchemy.engine.Engine],
                                             out_dir: str,
                                             use_offline: bool = False) -> str:
    """
    绘制距离分级对比哑铃图（自然候选空间 vs 最终数据集，图 5.2.1(e)）。

    自然候选距离分布优先来自离线 JSON（--use-offline）；否则由
    fetch_natural_distance_distribution 实算。最终数据集距离分布来自
    spatial_relations.dist_level（始终需数据库实算）。哑铃图的杆长即两分布占比之差，
    直观呈现分层抽样对长尾自然分布的再平衡幅度。

    参数:
        engine: 数据库引擎（最终数据集分布需实算；自然候选可走离线 JSON）
        out_dir: 输出根目录，图写入其子目录 figures/
        use_offline: 优先读取离线自然候选分布 JSON
    返回:
        实际写出文件路径（PNG 或 SVG 回退）
    """
    print("\n[5.2.1] 生成距离分级对比哑铃图（图 5.2.1(e)）...")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    if engine is None:
        raise ValueError("最终数据集距离分级需由 spatial_relations 实算，缺少数据库引擎。")
    final_dist = fetch_distribution(engine, "dist_level", DIST_LEVEL)
    if use_offline and os.path.exists(OFFLINE_CANDIDATE_JSON):
        with open(OFFLINE_CANDIDATE_JSON, "r", encoding="utf-8") as f:
            offline = json.load(f)
        offline_grade_keys = ["VeryClose(0-50)", "Close(50-200)",
                              "Medium(200-500)", "Far(500-1000)"]
        nat_dist = {k: {"count": int(offline["grade"].get(offline_grade_keys[i], 0)),
                        "pct": offline["grade_pct"][i]}
                    for i, k in enumerate(DIST_LEVEL)}
        print("        自然候选距离分布使用离线 JSON（跳过数据库重算）。")
    else:
        nat_dist = fetch_natural_distance_distribution(engine)

    cats = DIST_LEVEL
    labels = DIST_LEVEL
    nat_pct = [nat_dist[k]["pct"] for k in cats]
    final_pct = [final_dist[k]["pct"] for k in cats]
    nat_cnt = [nat_dist[k]["count"] for k in cats]
    final_cnt = [final_dist[k]["count"] for k in cats]

    out = os.path.join(fig_dir, "fig_5_2_1e_distance_dumbbell.png")
    if _HAS_MATPLOTLIB:
        _save_dumbbell_matplotlib(
            out, "",
            cats, nat_pct, final_pct, nat_cnt, final_cnt, labels)
        print(f"        距离分级对比哑铃图 -> {out}")
        return out
    svg = os.path.splitext(out)[0] + ".svg"
    _save_dumbbell_svg(
        svg, "图 5.2.1(e) 距离分级：自然候选空间 vs 最终数据集",
        cats, nat_pct, final_pct, nat_cnt, final_cnt, labels)
    print(f"        距离分级对比哑铃图(SVG 回退) -> {svg}")
    return svg


# ======================== 图 5.2.1 自然候选空间分布可视化 ========================
def fetch_natural_direction_distribution(engine: sqlalchemy.engine.Engine) -> Dict[str, Dict[str, Any]]:
    """
    在自然候选配对空间（地标 × 其 1 km 邻域内全部非地标目标）上统计 8 方位分布。
    配对口径与 fetch_natural_candidate_count 严格一致；方位角与方向判定采用与
    Spatial_relation_calculation.py 完全相同的平面几何计算
    （ST_Azimuth(l.geom, t.geom) 取弧度、北为 0、顺时针，再 degrees() 转角度），
    以保证自然候选分布与已写入 spatial_relations 的方向/距离口径严格对齐。
    注意：landmarks.geom 为度量投影坐标系（非 lon/lat），不可强转 geography，
          故一律使用原始 geom 列，与源计算流水线保持一致。

    注意：该查询为全表空间自连接，依赖 landmarks.geom 上的 GiST 索引；
          首次运行可能耗时数分钟，仅建议在离线批处理或论文统计时执行一次。
    """
    print("[5.2.1] 统计自然候选空间的 8 方位分布 ...")
    t0 = time.time()
    sql = text("""
        SELECT dir8, COUNT(*) AS c FROM (
            SELECT
                CASE
                    WHEN az BETWEEN 337.5 AND 360 OR az BETWEEN 0 AND 22.5 THEN 'North'
                    WHEN az BETWEEN 22.5 AND 67.5 THEN 'Northeast'
                    WHEN az BETWEEN 67.5 AND 112.5 THEN 'East'
                    WHEN az BETWEEN 112.5 AND 157.5 THEN 'Southeast'
                    WHEN az BETWEEN 157.5 AND 202.5 THEN 'South'
                    WHEN az BETWEEN 202.5 AND 247.5 THEN 'Southwest'
                    WHEN az BETWEEN 247.5 AND 292.5 THEN 'West'
                    WHEN az BETWEEN 292.5 AND 337.5 THEN 'Northwest'
                    ELSE NULL
                END AS dir8
            FROM (
                SELECT degrees(ST_Azimuth(l.geom, t.geom)) AS az
                FROM landmarks l
                JOIN landmarks t ON ST_DWithin(l.geom, t.geom, :radius)
                WHERE l.priority > 0 AND t.priority = 0 AND l.poi_id != t.poi_id
            ) sub
        ) grouped
        WHERE dir8 IS NOT NULL
        GROUP BY dir8;
    """)
    rows = engine.connect().execute(sql, {"radius": CANDIDATE_RADIUS_M}).mappings().all()
    total = sum(int(r["c"]) for r in rows)
    by_dir = {str(r["dir8"]): int(r["c"]) for r in rows}
    out = {k: {"count": by_dir.get(k, 0),
               "pct": round(by_dir.get(k, 0) / total * 100.0, 2) if total else 0.0}
           for k in DIRECTION_8}
    print(f"        8 方位自然候选分布: " + ", ".join(f"{k}={out[k]['pct']}%" for k in DIRECTION_8)
          + f"  (耗时 {round(time.time()-t0, 1)}s)")
    return out


def fetch_natural_distance_distribution(engine: sqlalchemy.engine.Engine) -> Dict[str, Dict[str, Any]]:
    """
    在自然候选配对空间上统计距离分级分布（米）。配对口径与
    fetch_natural_candidate_count 一致；距离采用与 Spatial_relation_calculation.py
    完全一致的平面几何 ST_Distance(l.geom, t.geom)（geom 为度量投影坐标系，
    返回单位即米），确保与已写入 spatial_relations.exact_distance_m 口径一致。
    """
    print("[5.2.1] 统计自然候选空间的距离分级分布 ...")
    t0 = time.time()
    sql = text("""
        SELECT dl, COUNT(*) AS c FROM (
            SELECT
                CASE
                    WHEN d < 50 THEN 'VeryClose'
                    WHEN d < 200 THEN 'Close'
                    WHEN d < 500 THEN 'Medium'
                    WHEN d <= 1000 THEN 'Far'
                    ELSE NULL
                END AS dl
            FROM (
                SELECT ST_Distance(l.geom, t.geom) AS d
                FROM landmarks l
                JOIN landmarks t ON ST_DWithin(l.geom, t.geom, :radius)
                WHERE l.priority > 0 AND t.priority = 0 AND l.poi_id != t.poi_id
            ) sub
        ) grouped
        WHERE dl IS NOT NULL
        GROUP BY dl;
    """)
    rows = engine.connect().execute(sql, {"radius": CANDIDATE_RADIUS_M}).mappings().all()
    total = sum(int(r["c"]) for r in rows)
    by_lv = {str(r["dl"]): int(r["c"]) for r in rows}
    out = {k: {"count": by_lv.get(k, 0),
               "pct": round(by_lv.get(k, 0) / total * 100.0, 2) if total else 0.0}
           for k in DIST_LEVEL}
    print(f"        距离分级自然候选分布: " + ", ".join(f"{k}={out[k]['pct']}%" for k in DIST_LEVEL)
          + f"  (耗时 {round(time.time()-t0, 1)}s)")
    return out


def compute_overall_distance_statistics(engine: sqlalchemy.engine.Engine) -> Dict[str, Any]:
    """
    空间关系数据集总体记录的精确距离统计（Results 章节要求）。

    基于 spatial_relations.exact_distance_m 计算全数据集的精确距离
    均值 (mean) 与中位数 (median)，并附有效样本数、极值与标准差，
    用于描述最终数据集的整体空间尺度特征。

    返回:
        {
          "n_valid": int,                       # 有效距离样本数
          "mean_exact_distance_m": float,       # 精确距离均值 (米)
          "median_exact_distance_m": float,     # 精确距离中位数 (米)
          "min_exact_distance_m": float,
          "max_exact_distance_m": float,
          "std_exact_distance_m": float,
        }
    """
    print("\n[统计] 计算空间关系数据集总体精确距离均值与中位数 ...")
    sql = text("""
        SELECT
            COUNT(exact_distance_m)                                                  AS n_valid,
            ROUND(AVG(exact_distance_m)::numeric, 2)                                AS mean_m,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY exact_distance_m)::numeric, 2)
                                                                                    AS median_m,
            ROUND(MIN(exact_distance_m)::numeric, 2)                                AS min_m,
            ROUND(MAX(exact_distance_m)::numeric, 2)                                AS max_m,
            ROUND(STDDEV(exact_distance_m)::numeric, 2)                             AS std_m
        FROM spatial_relations
        WHERE exact_distance_m IS NOT NULL;
    """)
    r = engine.connect().execute(sql).mappings().first()
    res = {
        "n_valid": int(r["n_valid"]),
        "mean_exact_distance_m": float(r["mean_m"]),
        "median_exact_distance_m": float(r["median_m"]),
        "min_exact_distance_m": float(r["min_m"]),
        "max_exact_distance_m": float(r["max_m"]),
        "std_exact_distance_m": float(r["std_m"]),
    }
    print(f"        有效样本={res['n_valid']:,}  均值={res['mean_exact_distance_m']} m  "
          f"中位数={res['median_exact_distance_m']} m  "
          f"(min={res['min_exact_distance_m']}, max={res['max_exact_distance_m']}, "
          f"std={res['std_exact_distance_m']})")
    return res


def _xml(s: Any) -> str:
    """转义字符串以满足 SVG/XML 文本节点要求。"""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _cjk_font() -> Optional[str]:
    """在 matplotlib 字体库中查找一款支持中文的字体；未找到则返回 None。"""
    if not fm:
        return None
    return "Times New roman"
    names = {f.name for f in fm.fontManager.ttflist}
    for c in cands:
        if c in names:
            return c
    return None


def _save_bar_matplotlib(out_path: str, title: str, categories: List[str],
                         values: List[float], labels: List[str], xlabel: str,
                         ylabel: str, value_suffix: str = "%",
                         annotate_counts: Optional[List[Optional[int]]] = None) -> None:
    """使用 matplotlib 绘制纵向柱状图（Agg 后端，适配论文出版）。"""
    f0 = _cjk_font()
    if f0:
        plt.rcParams["font.family"] = f0
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=150)
    bars = ax.bar(categories, values, color="#4C72B0",
                  edgecolor="#2c3e50", linewidth=0.7, width=0.62)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#222")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_xticklabels(
        [f"{labels[i]}\n({categories[i]})" for i in range(len(categories))],
        fontsize=10)
    for i, b in enumerate(bars):
        txt = f"{values[i]:.2f}{value_suffix}"
        if annotate_counts and annotate_counts[i] is not None:
            txt += f"\n(n={annotate_counts[i]:,})"
        ax.annotate(txt, (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8.5, color="#222")
    ax.set_ylim(0, max(values) * 1.18 if max(values) > 0 else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_bar_svg(out_path: str, title: str, categories: List[str],
                  values: List[float], labels: List[str], xlabel: str,
                  ylabel: str, value_suffix: str = "%",
                  annotate_counts: Optional[List[Optional[int]]] = None) -> None:
    """纯 Python 回退：将纵向柱状图导出为 SVG（无需第三方库）。"""
    W, H = 780, 470
    ml, mr, mt, mb = 72, 24, 56, 100
    pw = W - ml - mr
    ph = H - mt - mb
    vmax = max(values) if max(values) > 0 else 1.0
    ytop = vmax * 1.15  # 顶部留白，便于标注
    n = len(categories)
    slot = pw / n
    bw = slot * 0.6
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="\'Microsoft YaHei\',\'SimHei\',\'Noto Sans CJK SC\',sans-serif">']
    p.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    p.append(f'<text x="{W/2:.0f}" y="30" text-anchor="middle" font-size="15" '
             f'font-weight="bold" fill="#222">{_xml(title)}</text>')
    p.append(f'<text x="18" y="{mt+ph/2:.0f}" text-anchor="middle" font-size="12" '
             f'fill="#333" transform="rotate(-90 18 {mt+ph/2:.0f})">{_xml(ylabel)}</text>')
    p.append(f'<text x="{ml+pw/2:.0f}" y="{H-16}" text-anchor="middle" font-size="12" '
             f'fill="#333">{_xml(xlabel)}</text>')
    ticks = 5
    for t in range(ticks + 1):
        val = ytop * t / ticks
        y = mt + ph - (val / ytop) * ph
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                 f'stroke="#dddddd" stroke-width="1"/>')
        p.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-size="10" '
                 f'fill="#555">{val:.1f}</text>')
    p.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333" stroke-width="1.2"/>')
    p.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333" stroke-width="1.2"/>')
    for i in range(n):
        cx = ml + slot * (i + 0.5)
        x = cx - bw / 2
        h = (values[i] / ytop) * ph
        y = mt + ph - h
        p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                 f'fill="#4C72B0" stroke="#2c3e50" stroke-width="0.8"/>')
        ann = f"{values[i]:.2f}{value_suffix}"
        if annotate_counts and annotate_counts[i] is not None:
            ann += f" (n={annotate_counts[i]:,})"
        p.append(f'<text x="{cx:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="9.5" '
                 f'fill="#222">{_xml(ann)}</text>')
        p.append(f'<text x="{cx:.1f}" y="{mt+ph+22:.0f}" text-anchor="middle" font-size="11" '
                 f'fill="#333">{_xml(labels[i])}</text>')
        p.append(f'<text x="{cx:.1f}" y="{mt+ph+38:.0f}" text-anchor="middle" font-size="9.5" '
                 f'fill="#777">{_xml(categories[i])}</text>')
    p.append('</svg>')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(p))


def _save_bar_figure(out_path: str, title: str, categories: List[str],
                     values: List[float], labels: List[str], xlabel: str,
                     ylabel: str, value_suffix: str = "%",
                     annotate_counts: Optional[List[Optional[int]]] = None) -> str:
    """
    绘制纵向柱状图：优先使用 matplotlib（PNG），不可用时回退为 SVG。
    返回实际写出文件的路径。
    """
    if _HAS_MATPLOTLIB:
        _save_bar_matplotlib(out_path, title, categories, values, labels,
                             xlabel, ylabel, value_suffix, annotate_counts)
        return out_path
    svg_path = os.path.splitext(out_path)[0] + ".svg"
    _save_bar_svg(svg_path, title, categories, values, labels,
                  xlabel, ylabel, value_suffix, annotate_counts)
    return svg_path


def generate_figures_5_2_1(engine: Optional[sqlalchemy.engine.Engine],
                           out_dir: str,
                           use_offline: bool = False) -> Dict[str, str]:
    """
    绘制 5.2.1 节两幅自然候选空间分布图：
        图 5.2.1(a) 8 方位在自然候选空间中的分布
        图 5.2.1(b) 距离分级在自然候选空间中的分布

    参数:
        engine: 数据库引擎（use_offline=True 且离线 JSON 存在时可传 None）
        out_dir: 输出根目录（默认 Geocoding_dataset/output），图写入其子目录 figures/
        use_offline: 优先读取离线自然候选分布 JSON
                     （paper_section_4_3/_relation_dist_real.json）绘制，跳过数据库重算
    返回:
        含两幅图路径的字典 {"fig_5_2_1a": ..., "fig_5_2_1b": ...}
    """
    print("\n[5.2.1] 生成自然候选空间分布图（图 5.2.1(a)/(b)）...")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- 数据来源：离线 JSON 或数据库实算 ----
    if use_offline and os.path.exists(OFFLINE_CANDIDATE_JSON):
        with open(OFFLINE_CANDIDATE_JSON, "r", encoding="utf-8") as f:
            offline = json.load(f)
        dir_dist = {k: {"count": int(offline["dir8"].get(k, 0)),
                        "pct": offline["dir8_pct"][DIRECTION_8.index(k)]}
                    for k in DIRECTION_8}
        offline_grade_keys = ["VeryClose(0-50)", "Close(50-200)",
                              "Medium(200-500)", "Far(500-1000)"]
        dist_dist = {k: {"count": int(offline["grade"].get(offline_grade_keys[i], 0)),
                         "pct": offline["grade_pct"][i]}
                     for i, k in enumerate(DIST_LEVEL)}
        print("        使用离线自然候选分布 JSON 绘制（跳过数据库重算）。")
    else:
        if engine is None:
            raise ValueError("缺少数据库引擎且离线 JSON 不可用，无法绘制 5.2.1 图。")
        dir_dist = fetch_natural_direction_distribution(engine)
        dist_dist = fetch_natural_distance_distribution(engine)

    # ---- 图 5.2.1(a)：8 方位分布 ----
    pa = os.path.join(fig_dir, "fig_5_2_1a_direction.png")
    pa_out = _save_bar_figure(
        pa, "图 5.2.1(a) 8 方位在自然候选空间中的分布",
        DIRECTION_8, [dir_dist[k]["pct"] for k in DIRECTION_8],
        [DIRECTION_8_CN[k] for k in DIRECTION_8],
        "方位 (direction_8)", "占比 (%)",
        annotate_counts=[dir_dist[k]["count"] for k in DIRECTION_8])

    # ---- 图 5.2.1(b)：距离分级分布 ----
    pb = os.path.join(fig_dir, "fig_5_2_1b_distance.png")
    pb_out = _save_bar_figure(
        pb, "图 5.2.1(b) 距离分级在自然候选空间中的分布",
        DIST_LEVEL, [dist_dist[k]["pct"] for k in DIST_LEVEL],
        [DIST_LEVEL_CN[k] for k in DIST_LEVEL],
        "距离分级 (dist_level)", "占比 (%)",
        annotate_counts=[dist_dist[k]["count"] for k in DIST_LEVEL])

    print(f"        图 5.2.1(a) -> {pa_out}")
    print(f"        图 5.2.1(b) -> {pb_out}")
    return {"fig_5_2_1a": pa_out, "fig_5_2_1b": pb_out}


# ======================== 主流程 ========================
def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="空间关系计算结果与质量校验数据获取")
    parser.add_argument("--eval-sample", type=str, default=None,
                        help="若提供已标注样本 CSV 路径，则仅执行专家复标评估并退出")
    parser.add_argument("--use-offline", action="store_true",
                        help="优先使用离线自然候选分布 JSON 绘制 5.2.1 图，跳过数据库重算"
                             "（需 paper_section_4_3/_relation_dist_real.json 存在）")
    args = parser.parse_args()

    if args.eval_sample:
        evaluate_expert_sample(args.eval_sample)
        return {}

    engine = get_engine()
    report: Dict[str, Any] = {"meta": {"database_url": DATABASE_URL,
                                       "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}}
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "output"))
    os.makedirs(out_dir, exist_ok=True)

    # 5.2.1 规模与分布
    # report["5.2.1_scale"] = fetch_scale_and_coverage(engine)
    # report["5.2.1_direction_8"] = fetch_distribution(engine, "direction_8", DIRECTION_8)
    # report["5.2.1_dist_level"] = fetch_distribution(engine, "dist_level", DIST_LEVEL)
    # report["5.2.1_ambiguity"] = fetch_ambiguity_distribution(engine)
    # # 5.2.1 总体精确距离均值与中位数（数据集整体尺度特征）
    # report["5.2.1_overall_distance"] = compute_overall_distance_statistics(engine)
    #
    # # 5.2.1 多维分布统计导出 Excel（8 方向 / 模糊等级 / 距离分级 + 方向×距离交叉表）

    # excel_path = os.path.join(out_dir, "distribution_stats_5_2_1.xlsx")
    # report["5.2.1_excel"] = export_multidim_distribution_excel(engine, excel_path)
    #
    # # 5.2.1 自然候选配对规模（写入 spatial_relations 前的完整候选空间）与保留率
    # nat = fetch_natural_candidate_count(engine)
    # report["5.2.1_natural_candidates"] = nat
    # final_n = report["5.2.1_scale"]["total_records"]
    # nat_n = nat["total_candidates"]
    # if nat_n:
    #     report["5.2.1_retention_pct"] = round(final_n / nat_n * 100.0, 3)
    #
    # # 5.2.2 一致性（记录耗时供 5.2.5 使用）
    # consistency = validate_geometric_consistency(engine)
    #
    # # 5.2.3 均衡性
    # report["5.2.3_sampling_balance"] = analyze_sampling_balance(engine)
    #
    # 5.2.4 拓扑可靠性 + 导出样本
    report["5.2.4_topology"] = fetch_topology_distribution_stats(engine)
    sample_csv = os.path.join(out_dir, "expert_review_sample.csv")
    report["5.2.4_expert_sample_alloc"] = export_expert_sample(engine, sample_csv)
    #
    # # 5.2.5 效率
    # report["5.2.5_efficiency"] = analyze_efficiency(
    #     engine, consistency["validation_query_seconds"])
    # # 将一致性结果并入报告
    # report["5.2.2_consistency"] = consistency
    #
    # # 写出 JSON
    # json_path = os.path.join(out_dir, "evaluation_results_5_2.json")
    # with open(json_path, "w", encoding="utf-8") as f:
    #     json.dump(report, f, ensure_ascii=False, indent=2)
    # print(f"\n[SUCCESS] 评估结果已写出: {json_path}")
    # print(f"[SUCCESS] 专家复标样本已写出: {sample_csv}")
    #
    # # 5.2.1 自然候选空间分布可视化（图 5.2.1(a) / (b)）+ 方向关系分布玫瑰图（5.2.1(c)）
    # report["5.2.1_figures"] = generate_figures_5_2_1(
    #     engine, out_dir, use_offline=args.use_offline)
    # 5.2.1 方向关系分布玫瑰图（5.2.1(c)）+ 方向模糊等级分布柱形图（5.2.1(d)）
    report["5.2.1_figures"] = {}
    report["5.2.1_figures"]["fig_5_2_1_rose"] = generate_direction_rose_chart(
        engine, out_dir, use_offline=True)
    report["5.2.1_figures"]["fig_5_2_1d_ambiguity_bar"] = \
        generate_ambiguity_level_bar_chart(engine, out_dir, use_offline=args.use_offline)
    report["5.2.1_figures"]["fig_5_2_1e_distance_dumbbell"] = \
        generate_distance_grade_compare_dumbbell(engine, out_dir, use_offline=args.use_offline)
    print(f"[SUCCESS] 分布图已写出: {report['5.2.1_figures']}")
    return report


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] 运行评估失败: {e}", file=sys.stderr)
        sys.exit(1)
