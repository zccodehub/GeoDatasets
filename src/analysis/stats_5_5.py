# -*- coding: utf-8 -*-
"""
5.5 数据集统计特征与质量评估 —— 统计结果计算实现（PostGIS 数据源）

对应 IJGIS_GeoDatasets.docx 第 5.5 节（5.5.1 ~ 5.5.5）：
  5.5.1 数据集总体规模与构成            -> 表 22
  5.5.2 文本描述的语言统计特征          -> 表 23（句长 / 词汇丰富度 / 表达多样性；含去地名掩码对照）
  5.5.3 自然语言生成质量评估
        5.5.3.1 自动化指标评估          -> 表 24（BLEU-4 / ROUGE-L 相对改进 + Self-BLEU / Distinct）
        5.5.3.2 人工抽样评估            -> 表 25（Likert 聚合 + Cohen's kappa）
  5.5.4 几何一致性与空间逻辑保真度校验  -> 表 26（文本-坐标误差 / 方向·距离·拓扑一致性 / 收敛半径）
  5.5.5 数据集完整性与合规性校验        -> 异常率报告

数据来源
========
  geo_desc(id, description 中文, description_en 英文, description_raw 模板基线,
           ref_name, target_x, target_y)
  spatial_relations(id, scene_code, ambiguity_level, target_lng, target_lat,
                   direction, distance, topology, ref_landmark_id, ref_landmark_name,
                   landmark_x, landmark_y  -> 参考地标经纬度)
  二者通过 id 关联。5.5.4 反向坐标推理所需的"参考地标坐标"直接来自
  spatial_relations.landmark_x / landmark_y（按 ref_landmark_id 或 ref_name 索引）；
  5.5.3.1 的模板基线直接来自 geo_desc.description_raw。

实现约定
========
  - 中文分词：优先复用 linguistic_stats_5_5_2（已装 jieba 则词级，否则字符级）。
  - 英文：正则切词（小写、去标点）。
  - 连接管理：上下文管理器 + 流式读取（chunksize=10000）防 OOM，遵循项目规范。
  - 所有指标均提供"内置示例数据"(--demo) 以便无数据库时校验算法。

依赖：pandas, numpy；（PostGIS）sqlalchemy, psycopg2, python-dotenv；（可选）jieba
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 北京行政边界（项目规范 7.3）：经度 115.7~117.4，纬度 39.4~41.6
BJ_BBOX = dict(lng_min=115.7, lng_max=117.4, lat_min=39.4, lat_max=41.6)
CHUNK_SIZE = 10_000
地球半径_M = 6_371_000.0

# ----------------------------------------------------------------------------
# 复用 5.5.2 已实现的底层指标（TTR / 信息熵 / Distinct-n / Self-BLEU / 分词）
# ----------------------------------------------------------------------------
try:
    import importlib.util as _ilu

    _ling_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linguistic_stats_5_5_2.py")
    _spec = _ilu.spec_from_file_location("ling_552", _ling_path)
    _ling = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_ling)
    HAVE_LING = True
    logger.debug("已复用 linguistic_stats_5_5_2 的底层指标函数")
except Exception as _e:  # pragma: no cover - 依赖可选
    HAVE_LING = False
    logger.warning("未能导入 linguistic_stats_5_5_2（%s）；将使用内置最小实现", _e)


if HAVE_LING:
    计算类符形符比 = _ling.计算类符形符比
    计算词汇熵 = _ling.计算词汇熵
    计算_distinct_n = _ling.计算_distinct_n
    计算_self_bleu = _ling.计算_self_bleu
    选择分词函数 = _ling.选择分词函数
    分词_英文 = _ling.分词_英文
else:  # 内置最小实现（保证独立可运行）
    _EN_RE = re.compile(r"[a-zA-Z0-9']+")

    def 分词_英文(text: str) -> List[str]:
        text = (text or "").lower()
        return _EN_RE.findall(text) if text.strip() else []

    def 选择分词函数(语言: str, zh_mode: str = "auto"):
        if 语言 == "en":
            return 分词_英文
        # 中文退回字符级（无 jieba 时不保证词级语义，仅保运行）
        return lambda t: [c for c in (t or "") if not c.isspace()]

    def 计算类符形符比(tokens: Sequence[str]) -> float:
        return len(set(tokens)) / len(tokens) if tokens else 0.0

    def 计算词汇熵(tokens: Sequence[str]) -> Tuple[float, float, float]:
        if not tokens:
            return 0.0, 0.0, 1.0
        counts = Counter(tokens)
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()], dtype=float)
        H = float(-np.sum(probs * np.log2(probs)))
        V = len(counts)
        return H, float(H / np.log2(V) if V > 1 else 0.0), float(2.0 ** H)

    def 计算_distinct_n(texts: Sequence[str], n: int, 分词函数) -> float:
        grams: List[tuple] = []
        for t in texts:
            toks = 分词函数(t)
            grams.extend(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
        return len(set(grams)) / len(grams) if grams else 0.0

    def 计算_self_bleu(texts, 分词函数, max_n=4, 假设样本数=500, 参考池大小=200, random_state=42):
        # 简化抽样近似（与 5.5.2 一致思路）
        N = len(texts)
        if N < 2:
            return 0.0
        rng = np.random.default_rng(random_state)
        total = min(N, 假设样本数 + 参考池大小)
        idx = rng.choice(N, size=total, replace=False)
        hyp = list(idx[: min(假设样本数, total)])
        ref = list(idx[min(假设样本数, total):])

        def ngrams(toks, n):
            return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))

        ref_data = [(len(分词函数(texts[i])), ngrams(分词函数(texts[i]), max_n)) for i in ref]
        scores = []
        for i in hyp:
            ht = 分词函数(texts[i])
            c = len(ht)
            r = min((rd[0] for rd in ref_data), key=lambda L: (abs(L - c), L)) if ref_data else 0
            bp = 1.0 if c >= r else math.exp(1.0 - r / c)
            logs = []
            for n in range(1, max_n + 1):
                hg = ngrams(ht, n)
                if not hg:
                    logs.append(float("-inf"))
                    continue
                clipped = 0
                for g, cnt in hg.items():
                    mx = max((rd[1].get(n, {}).get(g, 0) for rd in ref_data), default=0)
                    clipped += min(cnt, mx)
                p = clipped / sum(hg.values())
                logs.append(math.log(p) if p > 0 else float("-inf"))
            if any(v == float("-inf") for v in logs):
                scores.append(0.0)
            else:
                scores.append(bp * math.exp(np.mean(logs)))
        return float(np.mean(scores)) if scores else 0.0


# ----------------------------------------------------------------------------
# 数据加载：PostGIS（主） / CSV（测试） / 内置示例（demo）
# ----------------------------------------------------------------------------
def 加载数据(
    *,
    数据库_URL: Optional[str] = None,
    csv路径: Optional[str] = None,
    landmark_csv: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
) -> Tuple[pd.DataFrame, Dict]:
    """加载 5.5 所需的全部字段。

    返回 (描述数据框, 地标坐标字典)。地标坐标字典键为 ref_landmark_id（或 ref_name），
    值为 (lng, lat)。

    地标坐标来源优先级：
      1. 若显式传入 landmark_csv，则以其为准；
      2. 否则优先使用 joined 字段 spatial_relations.landmark_x / landmark_y
         （按 ref_landmark_id 或 ref_name 索引）；
      3. 若两者皆无，则为空字典（5.5.4 的反向坐标推理将跳过）。
    """
    if csv路径:
        logger.info("从 CSV 读取：%s", csv路径)
        df = pd.read_csv(csv路径)
    else:
        from dotenv import load_dotenv
        from sqlalchemy import create_engine, text

        _env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        load_dotenv(dotenv_path=os.path.abspath(_env_path))
        if 数据库_URL is None:
            数据库_URL = os.getenv("DATABASE_URL")
        if not 数据库_URL:
            raise ValueError("未找到 DATABASE_URL，请检查 .env 或传入 --database-url")

        engine = create_engine(数据库_URL, pool_pre_ping=True)
        # geo_desc 与 spatial_relations 通过 id 关联；
        # 参考地标经纬度取自 spatial_relations.landmark_x / landmark_y；
        # 模板基线文本取自 geo_desc.description_raw
        sql = text("""
            SELECT g.id,
                   g.description,
                   g.description_en,
                   g.description_raw,
                   g.ref_name,
                   s.scene_code,
                   s.ambiguity_level,
                   s.target_x,
                   s.target_y,
                   s.direction_8,
                   s.exact_distance_m,
                   s.topology_type,
                   s.landmark_id,
                   s.landmark_name,
                   s.landmark_x,
                   s.landmark_y
            FROM geo_desc g
            JOIN spatial_relations s ON g.id = s.id
            ORDER BY g.id
        """)
        logger.info("从 PostGIS 读取 geo_desc JOIN spatial_relations ...")
        with engine.connect() as conn:
            reader = pd.read_sql(sql, conn, chunksize=chunk_size)
            df = pd.concat([chunk for chunk in reader], ignore_index=True)
        logger.info("读取完成，样本数 = %d", len(df))

    # 地标坐标：landmark_csv 优先；否则从 joined 的 landmark_x/landmark_y 构建
    if landmark_csv:
        logger.info("使用外部地标坐标 CSV 覆盖：%s", landmark_csv)
        地标坐标 = _加载地标坐标(landmark_csv)
    else:
        地标坐标 = _构建地标坐标_from_df(df)
    logger.info("构建地标坐标 %d 条（来自 landmark_x/landmark_y 或 CSV）", len(地标坐标))
    return df, 地标坐标


def _加载地标坐标(path: str) -> Dict:
    """从 CSV 加载地标坐标，支持 landmark_id/lng/lat 或 name/lng/lat 两列风格。"""
    ld = pd.read_csv(path)
    out: Dict = {}
    if {"landmark_id", "lng", "lat"}.issubset(ld.columns):
        for _, r in ld.iterrows():
            out[str(r["landmark_id"])] = (float(r["lng"]), float(r["lat"]))
    elif {"name", "lng", "lat"}.issubset(ld.columns):
        for _, r in ld.iterrows():
            out[str(r["name"])] = (float(r["lng"]), float(r["lat"]))
    logger.info("加载地标坐标 %d 条", len(out))
    return out


def _构建地标坐标_from_df(df: pd.DataFrame) -> Dict:
    """从 joined 数据框的 landmark_x / landmark_y 列构建地标坐标字典。

    键优先取 ref_landmark_id，缺失时回退 ref_name；值为 (lng, lat)。
    """
    out: Dict = {}
    if "landmark_x" not in df or "landmark_y" not in df:
        return out
    lx = pd.to_numeric(df["landmark_x"], errors="coerce")
    ly = pd.to_numeric(df["landmark_y"], errors="coerce")
    rid = df.get("ref_landmark_id")
    rname = df.get("ref_name")
    for i in range(len(df)):
        key = None
        if rid is not None and pd.notna(rid.iloc[i]):
            key = str(rid.iloc[i])
        elif rname is not None and pd.notna(rname.iloc[i]):
            key = str(rname.iloc[i])
        x, y = lx.iloc[i], ly.iloc[i]
        if key is not None and pd.notna(x) and pd.notna(y):
            out[key] = (float(x), float(y))
    return out


# ----------------------------------------------------------------------------
# 5.5.1 数据集总体规模与构成（表 22）
# ----------------------------------------------------------------------------
def 计算_规模与构成(df: pd.DataFrame) -> pd.DataFrame:
    """从样本规模、参照锚点、空间覆盖三个维度刻画数据集整体画像。"""
    n = len(df)
    # 参照锚点（去重地标）
    n_landmark = df["ref_name"].nunique(dropna=True) if "ref_name" in df else 0
    n_landmark_id = df["ref_landmark_id"].nunique(dropna=True) if "ref_landmark_id" in df else n_landmark
    # 空间覆盖（基于 target 坐标；优先 target_lng/lat，回退 target_x/y）
    lng = df.get("target_lng") if "target_lng" in df else df.get("target_x")
    lat = df.get("target_lat") if "target_lat" in df else df.get("target_y")
    if lng is not None and lat is not None:
        lng = pd.to_numeric(lng, errors="coerce")
        lat = pd.to_numeric(lat, errors="coerce")
        inside = ((lng >= BJ_BBOX["lng_min"]) & (lng <= BJ_BBOX["lng_max"]) &
                  (lat >= BJ_BBOX["lat_min"]) & (lat <= BJ_BBOX["lat_max"])).mean()
        bbox = (f"[{lng.min():.3f},{lng.max():.3f}] × [{lat.min():.3f},{lat.max():.3f}]")
    else:
        inside = float("nan")
        bbox = "N/A"

    rows = [
        ("最终描述总数", f"{n:,}"),
        ("去重参照地标数(ref_name)", f"{n_landmark:,}"),
        ("去重参照地标数(ref_landmark_id)", f"{n_landmark_id:,}"),
        ("空间覆盖边界框(经度×纬度)", bbox),
        ("北京行政边界内占比", f"{inside * 100:.2f}%" if not math.isnan(inside) else "N/A"),
    ]
    # 按场景 / 模糊等级的分布（若有）
    if "scene_code" in df:
        for sc, cnt in df["scene_code"].value_counts().items():
            rows.append((f"场景分布: {sc}", f"{cnt:,} ({cnt / n * 100:.2f}%)"))
    if "ambiguity_level" in df:
        for al, cnt in df["ambiguity_level"].value_counts().items():
            rows.append((f"模糊等级分布: {al}", f"{cnt:,} ({cnt / n * 100:.2f}%)"))
    return pd.DataFrame(rows, columns=["指标", "数值"])


# ----------------------------------------------------------------------------
# 5.5.2 文本描述的语言统计特征（表 23）
# ----------------------------------------------------------------------------
def 掩码_地名(text: str, name: Optional[str]) -> str:
    """将文本中的参照地标专名替换为统一占位符『某』，以消除专有名词长尾对
    词汇多样性度量的干扰——避免评审误判数据集"同质化"。

    占位符『某』在字符级与 jieba 词级分词下均计为单一词元，保证中英文口径一致；
    同一地标名被替换为同一占位符后，掩码文本的多样性即反映"功能词/句式模板"
    的真实丰富程度，而非由海量唯一地名堆砌出的伪多样性。
    """
    text = text or ""
    name = (str(name) or "").strip()
    if name:
        text = text.replace(name, "某")
    return text


def 计算_语言统计(df: pd.DataFrame, zh_mode: str = "auto") -> pd.DataFrame:
    """句长（词数 均值/中位/标准差）、词汇丰富度（TTR/信息熵/归一化熵）、
    表达多样性（Distinct-1/2），并额外给出『去地名掩码』对照分组。

    返回表 23：列 = 类型（原始 / 去地名掩码）、指标、中文描述、英文描述，
    可直接用于论文 5.5.2 节，直观展示"去除专有地名后数据仍具表达多样性"。
    """
    if "description" not in df.columns:
        raise ValueError("数据需包含 description（中文描述）列")

    zh_texts = df["description"].astype(str).tolist()
    en_texts = df["description_en"].fillna("").astype(str).tolist()
    ref_names = df["ref_name"].astype(str).tolist() if "ref_name" in df else [""] * len(df)
    # 去地名掩码：将每条描述中的参照地标专名统一替换为占位符『某』
    zh_masked = [掩码_地名(t, n) for t, n in zip(zh_texts, ref_names)]
    en_masked = [掩码_地名(t, n) for t, n in zip(en_texts, ref_names)]

    zh_tok = 选择分词函数("zh", zh_mode)
    en_tok = 选择分词函数("en")

    def 单行(texts: Sequence[str], tok) -> Dict[str, float]:
        """针对一套文本计算句长 / 词汇丰富度 / 表达多样性。"""
        词数 = np.array([len(tok(t)) for t in texts], dtype=float)
        语料_token = [t for s in texts for t in tok(s)]
        H, H_norm, _ = 计算词汇熵(语料_token)
        return {
            "平均词数": float(词数.mean()),
            "中位词数": float(np.median(词数)),
            "词数标准差": float(词数.std()),
            "TTR": 计算类符形符比(语料_token),
            "信息熵(bit)": H,
            "归一化熵H_norm": H_norm,
            "Distinct-1": 计算_distinct_n(texts, 1, tok),
            "Distinct-2": 计算_distinct_n(texts, 2, tok),
        }

    原始组 = 单行(zh_texts, zh_tok), 单行(en_texts, en_tok)
    掩码组 = 单行(zh_masked, zh_tok), 单行(en_masked, en_tok)

    指标序 = ["平均词数", "中位词数", "词数标准差", "TTR", "信息熵(bit)",
              "归一化熵H_norm", "Distinct-1", "Distinct-2"]
    rows = []
    for 类型, (zg, eg) in [("原始", 原始组), ("去地名掩码", 掩码组)]:
        for k in 指标序:
            rows.append({
                "类型": 类型,
                "指标": k,
                "中文描述": f"{zg[k]:.4f}",
                "英文描述": f"{eg[k]:.4f}",
            })
    return pd.DataFrame(rows, columns=["类型", "指标", "中文描述", "英文描述"])


# ----------------------------------------------------------------------------
# 5.5.3.1 自动化指标评估（表 24）
# ----------------------------------------------------------------------------
def _ngram计数(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def 计算_单句_bleu(hyp: List[str], ref: List[str], max_n: int = 4) -> float:
    """单句 BLEU（n=1..4，closest 长度惩罚，多参考以最大计数计）。"""
    if not hyp:
        return 0.0
    c = len(hyp)
    r = len(ref)
    bp = 1.0 if c >= r else math.exp(1.0 - r / c)
    logs = []
    for n in range(1, max_n + 1):
        hg = _ngram计数(hyp, n)
        rg = _ngram计数(ref, n)
        if not hg:
            logs.append(float("-inf"))
            continue
        clipped = sum(min(cnt, rg.get(g, 0)) for g, cnt in hg.items())
        p = clipped / sum(hg.values())
        logs.append(math.log(p) if p > 0 else float("-inf"))
    if any(v == float("-inf") for v in logs):
        return 0.0
    return bp * math.exp(np.mean(logs))


def 计算_语料_bleu(hyps: Sequence[str], refs: Sequence[str], 分词函数, max_n: int = 4) -> float:
    """语料级 BLEU（逐句计算后平均，closest 参考长度）。"""
    return float(np.mean([
        计算_单句_bleu(分词函数(h), 分词函数(r), max_n)
        for h, r in zip(hyps, refs)
    ])) if len(hyps) else 0.0


def _lcs_length(a: List[str], b: List[str]) -> int:
    """最长公共子序列长度（动态规划）。"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i, j] = (dp[i - 1, j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1, j], dp[i, j - 1]))
    return int(dp[m, n])


def 计算_rouge_l(hyp: str, ref: str, 分词函数) -> float:
    """ROUGE-L（基于 LCS 的 F1）。"""
    a, b = 分词函数(hyp), 分词函数(ref)
    if not a or not b:
        return 0.0
    lcs = _lcs_length(a, b)
    p = lcs / len(a)
    r = lcs / len(b)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def 计算_语料_rouge_l(hyps: Sequence[str], refs: Sequence[str], 分词函数) -> float:
    return float(np.mean([计算_rouge_l(h, r, 分词函数) for h, r in zip(hyps, refs)])) if len(hyps) else 0.0


def 计算_自动指标(df: pd.DataFrame, template_col: str = "description_raw", zh_mode: str = "auto") -> pd.DataFrame:
    """计算 5.5.3.1 自动化指标（表 24），仅基于中文文本。

    表 24 列 = 模板基线（geo_desc.description_raw）/ LLM精修（geo_desc.description）；
    行 = 各项指标，分别针对两套中文文本给出数值：
      【内在多样性 / 词汇丰富度】（各自独立计算）
        - Self-BLEU（越低越好，衡量样本间重复度）
        - Distinct-1 / Distinct-2（n-gram 去重率，衡量表达多样性）
        - 词汇熵 H(bit) / 归一化熵 H_norm（衡量词表均衡度）
      【跨体系保真度】（模板基线 ⇄ LLM精修 互为参考，双向交叉）
        - BLEU-4：LLM精修列 = 精修文本以模板为参考的得分（即论文所指
          "LLM 精修文本相对模板基线的 BLEU-4"）；模板基线列 = 反向得分。
        - ROUGE-L：同上，基于 LCS 的 F1。
    若未提供 template_col 或该列不存在，则"模板基线"列及跨体系两行均显示为 N/A。
    """
    if "description" not in df.columns:
        raise ValueError("数据需包含 description（中文 LLM 精修文本）列")

    精修文本 = df["description"].astype(str).tolist()
    基线文本 = df[template_col].astype(str).tolist() if (template_col and template_col in df.columns) else None

    zh_tok = 选择分词函数("zh", zh_mode)

    def 侧指标(texts: Sequence[str]) -> Dict[str, float]:
        """针对一套中文文本计算各内在指标。"""
        # 词汇熵在语料层级统计，需先拼接全部 token
        语料_token = [tok for s in texts for tok in zh_tok(s)]
        H, H_norm, _ = 计算词汇熵(语料_token)
        return {
            "Self-BLEU(越低越好)": 计算_self_bleu(texts, zh_tok, max_n=4),
            "Distinct-1": 计算_distinct_n(texts, 1, zh_tok),
            "Distinct-2": 计算_distinct_n(texts, 2, zh_tok),
            "词汇熵 H(bit)": H,
            "归一化熵 H_norm": H_norm,
        }

    精修饰标 = 侧指标(精修文本)
    基线指标 = 侧指标(基线文本) if 基线文本 is not None else None

    # 跨体系保真度：两套中文文本互为参考，双向交叉（需模板基线列存在）
    if 基线文本 is not None:
        bleu_基线 = 计算_语料_bleu(基线文本, 精修文本, zh_tok)   # 模板为假设、精修为参考
        bleu_精修 = 计算_语料_bleu(精修文本, 基线文本, zh_tok)   # 精修为假设、模板为参考
        rouge_基线 = 计算_语料_rouge_l(基线文本, 精修文本, zh_tok)
        rouge_精修 = 计算_语料_rouge_l(精修文本, 基线文本, zh_tok)
    else:
        bleu_基线 = bleu_精修 = rouge_基线 = rouge_精修 = None

    def _fmt(v: Optional[float]) -> str:
        return f"{v:.4f}" if v is not None else "N/A"

    # 行：指标；列：模板基线 / LLM精修
    rows = []
    for k in 精修饰标.keys():
        if 基线指标 is not None:
            rows.append({"指标": k, "模板基线": f"{基线指标[k]:.4f}", "LLM精修": f"{精修饰标[k]:.4f}"})
        else:
            rows.append({"指标": k, "模板基线": "N/A", "LLM精修": f"{精修饰标[k]:.4f}"})
    # 追加跨体系比较指标（模板基线 ⇄ LLM精修）
    rows.append({"指标": "BLEU-4(精修⇄模板)", "模板基线": _fmt(bleu_基线), "LLM精修": _fmt(bleu_精修)})
    rows.append({"指标": "ROUGE-L(精修⇄模板)", "模板基线": _fmt(rouge_基线), "LLM精修": _fmt(rouge_精修)})
    return pd.DataFrame(rows, columns=["指标", "模板基线", "LLM精修"])


# ----------------------------------------------------------------------------
# 5.5.3.2 人工抽样评估：Likert 聚合 + Cohen's kappa（表 25）
# ----------------------------------------------------------------------------
def 计算_cohen_kappa(rater_a: Sequence[int], rater_b: Sequence[int], weighted: bool = True) -> float:
    """Cohen's kappa（weighted=True 时为二次加权 kappa，适配有序李克特量表）。"""
    a = np.asarray(rater_a, dtype=int)
    b = np.asarray(rater_b, dtype=int)
    cats = int(max(a.max(), b.max()) - min(a.min(), b.min()) + 1)
    if cats < 2:
        return 1.0
    # 混淆矩阵
    n = len(a)
    cm = np.zeros((cats, cats), dtype=float)
    off = min(a.min(), b.min())
    for x, y in zip(a, b):
        cm[x - off, y - off] += 1
    po = np.trace(cm) / n
    if weighted:
        weights = np.zeros((cats, cats))
        for i in range(cats):
            for j in range(cats):
                weights[i, j] = (i - j) ** 2 / (cats - 1) ** 2
        pe = np.sum((cm.sum(axis=1) / n)[:, None] * (cm.sum(axis=0) / n)[None, :] * weights)
    else:
        pe = np.sum((cm.sum(axis=1) / n) * (cm.sum(axis=0) / n))
    return float((po - pe) / (1 - pe)) if (1 - pe) else 1.0


def 计算_人工评估(human_csv: str) -> pd.DataFrame:
    """读取人工评分 CSV 并计算各维度均值/标准差与评分者间一致性 kappa。

    CSV 约定：含维度列（如 流畅度/语义合理性/表达多样性/指代清晰度/口语自然度），
    以及两名评分者的列（rater1, rater2）。
    """
    hp = pd.read_csv(human_csv)
    维度列 = [c for c in hp.columns if c not in ("rater1", "rater2", "样本id", "sample_id")]
    if "rater1" in hp and "rater2" in hp:
        kappa = 计算_cohen_kappa(hp["rater1"].tolist(), hp["rater2"].tolist(), weighted=True)
    else:
        kappa = float("nan")
    rows = []
    for d in 维度列:
        vals = pd.to_numeric(hp[d], errors="coerce").dropna()
        rows.append({"维度": d, "均值": f"{vals.mean():.2f}", "标准差": f"{vals.std():.2f}"})
    if not math.isnan(kappa):
        rows.append({"维度": "评分者间一致性(Cohen's κ)", "均值": f"{kappa:.3f}", "标准差": "-"})
    return pd.DataFrame(rows, columns=["维度", "均值", "标准差"])


# ----------------------------------------------------------------------------
# 5.5.4 几何一致性与空间逻辑保真度校验（表 26）
# ----------------------------------------------------------------------------
方向模式: List[Tuple[str, str]] = [
    ("东北", "NE"), ("东南", "SE"), ("西南", "SW"), ("西北", "NW"),
    ("正东", "E"), ("正南", "S"), ("正西", "W"), ("正北", "N"),
    ("东", "E"), ("南", "S"), ("西", "W"), ("北", "N"),
]
_十六向转八向 = {
    "N": "N", "NNE": "N", "NE": "NE", "ENE": "E", "E": "E", "ESE": "E", "SE": "SE",
    "SSE": "S", "S": "S", "SSW": "S", "SW": "SW", "WSW": "W", "W": "W", "WNW": "W",
    "NW": "NW", "NNW": "N",
}
_八向角度 = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
拓扑词映射 = {
    "马路对面": "crosses", "对面": "crosses", "同侧": "adjacent", "相邻": "adjacent",
    "附近": "disjoint", "内部": "inside", "包含": "contains", "路口把角": "touches",
}


def 解析_方向(text: str) -> Optional[str]:
    for w, code in 方向模式:
        if w in (text or ""):
            return code
    return None


def 解析_距离(text: str) -> Optional[float]:
    """提取文本中的距离（米）。支持 米/m/M 与 千米/公里/km/KM。"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(千米|公里|km|KM|米|m|M)", text or "")
    if not m:
        return None
    v = float(m.group(1))
    if m.group(2) in ("千米", "公里", "km", "KM"):
        v *= 1000.0
    return v


def 解析_拓扑(text: str) -> Optional[str]:
    for w, t in 拓扑词映射.items():
        if w in (text or ""):
            return t
    return None


def haversine(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 地球半径_M * math.asin(min(1.0, math.sqrt(a)))


def 反推坐标(lng0: float, lat0: float, dir8: str, dist_m: float) -> Tuple[float, float]:
    """由参考地标坐标 + 方位(8向) + 距离(米) 反推目标点坐标（近似平面投影）。"""
    ang = math.radians(_八向角度.get(dir8, 0))
    dlat = dist_m * math.cos(ang) / 111_320.0
    dlng = dist_m * math.sin(ang) / (111_320.0 * math.cos(math.radians(lat0)))
    return lng0 + dlng, lat0 + dlat


def 方向编码_8(d: str) -> Optional[str]:
    return _十六向转八向.get((d or "").upper())


def 计算_几何一致性(
    df: pd.DataFrame,
    地标坐标: Dict,
    距离容差: float = 0.15,
    收敛最小样本数: int = 2,
) -> Dict[str, object]:
    """返回表 26 所需的全部几何一致性统计量。

    依赖：
      - 5.5.4(1) 文本-坐标误差：需 地标坐标（ref_landmark_id -> (lng,lat)）与
        target_lng/target_lat（或 target_x/target_y）真值。
      - 5.5.4(2) 方向/距离/拓扑一致性：需 direction / distance / topology 字段。
      - 5.5.4(3) 多描述收敛：需按目标点分组的反推坐标。
    """
    lng = df.get("target_lng") if "target_lng" in df else df.get("target_x")
    lat = df.get("target_lat") if "target_lat" in df else df.get("target_y")
    lng = pd.to_numeric(lng, errors="coerce") if lng is not None else None
    lat = pd.to_numeric(lat, errors="coerce") if lat is not None else None

    errs: List[float] = []
    方向一致 = 0
    方向可判 = 0
    距离一致 = 0
    距离可判 = 0
    拓扑一致 = 0
    拓扑可判 = 0
    分组: Dict[Tuple[float, float], List[Tuple[float, float]]] = defaultdict(list)

    for i, row in df.iterrows():
        desc = str(row.get("description", ""))
        # (2) 方向一致性
        pd8 = 解析_方向(desc)
        sd8 = 方向编码_8(str(row.get("direction", ""))) if "direction" in row else None
        if pd8 and sd8:
            方向可判 += 1
            if pd8 == sd8:
                方向一致 += 1
        # (2) 距离一致性
        pdist = 解析_距离(desc)
        sdist = pd.to_numeric(row.get("distance"), errors="coerce") if "distance" in row else np.nan
        if pdist is not None and not math.isnan(sdist) and sdist > 0:
            距离可判 += 1
            if abs(pdist - sdist) / sdist <= 距离容差:
                距离一致 += 1
        # (2) 拓扑一致性
        ptopo = 解析_拓扑(desc)
        stopo = str(row.get("topology", "")).lower() if "topology" in row else ""
        if ptopo:
            拓扑可判 += 1
            if ptopo == stopo:
                拓扑一致 += 1
        # (1)/(3) 反推坐标（需地标坐标）
        rid = str(row.get("ref_landmark_id", "")) if "ref_landmark_id" in row else str(row.get("ref_name", ""))
        coord0 = 地标坐标.get(rid) or 地标坐标.get(str(row.get("ref_name", "")))
        if coord0 is not None and pd8 and pdist is not None:
            xh, yh = 反推坐标(coord0[0], coord0[1], pd8, pdist)
            if lng is not None and lat is not None and not (math.isnan(lng.loc[i]) or math.isnan(lat.loc[i])):
                errs.append(haversine(xh, yh, float(lng.loc[i]), float(lat.loc[i])))
            if lng is not None and lat is not None and not (math.isnan(lng.loc[i]) or math.isnan(lat.loc[i])):
                分组[(float(lng.loc[i]), float(lat.loc[i]))].append((xh, yh))

    median_err = float(np.median(errs)) if errs else float("nan")
    p90_err = float(np.percentile(errs, 90)) if errs else float("nan")
    # (3) 多描述收敛半径（每组反推坐标到质心的平均距离）
    radii = []
    for grp in 分组.values():
        if len(grp) >= 收敛最小样本数:
            arr = np.array(grp)
            centroid = arr.mean(axis=0)
            radii.append(float(np.mean([haversine(x, y, centroid[0], centroid[1]) for x, y in grp])))
    conv_radius = float(np.mean(radii)) if radii else float("nan")

    return {
        "文本-坐标误差_中位数(m)": median_err,
        "文本-坐标误差_P90(m)": p90_err,
        "方向一致性率": (方向一致 / 方向可判) if 方向可判 else float("nan"),
        "方向可判样本数": 方向可判,
        "距离一致性率": (距离一致 / 距离可判) if 距离可判 else float("nan"),
        "距离可判样本数": 距离可判,
        "拓扑一致性率": (拓扑一致 / 拓扑可判) if 拓扑可判 else float("nan"),
        "拓扑可判样本数": 拓扑可判,
        "多描述收敛半径均值(m)": conv_radius,
        "可反推坐标样本数": len(errs),
    }


# ----------------------------------------------------------------------------
# 5.5.5 数据集完整性与合规性校验（异常率报告）
# ----------------------------------------------------------------------------
def 计算_完整性(df: pd.DataFrame) -> pd.DataFrame:
    """逐字段统计空值率、超短(<5字符)/超长(>200字符)文本占比；坐标边界内占比。

    注意：长度异常校验仅针对文本型描述字段（description / description_en）。
    ref_name 为参照地标名称，天然短小，不计入超短/超长统计；其仅报告空值率
    （地标名缺失才是真实完整性问题）。
    """
    rows = []
    # 文本型描述字段：执行完整的长度异常校验
    for col in ("description", "description_en"):
        if col not in df:
            continue
        s = df[col].astype(str)
        null_rate = df[col].isna().mean()
        short_rate = (s.str.len() < 5).mean()
        long_rate = (s.str.len() > 200).mean()
        rows.append({
            "字段": col,
            "空值率": f"{null_rate * 100:.3f}%",
            "超短(<5字符)率": f"{short_rate * 100:.3f}%",
            "超长(>200字符)率": f"{long_rate * 100:.3f}%",
        })
    # ref_name：仅报告空值率，长度校验设为 N/A（地标名本就短）
    if "ref_name" in df:
        null_rate = df["ref_name"].isna().mean()
        rows.append({
            "字段": "ref_name",
            "空值率": f"{null_rate * 100:.3f}%",
            "超短(<5字符)率": "N/A",
            "超长(>200字符)率": "N/A",
        })
    lng = df.get("target_lng") if "target_lng" in df else df.get("target_x")
    lat = df.get("target_lat") if "target_lat" in df else df.get("target_y")
    if lng is not None and lat is not None:
        lng = pd.to_numeric(lng, errors="coerce")
        lat = pd.to_numeric(lat, errors="coerce")
        inside = ((lng >= BJ_BBOX["lng_min"]) & (lng <= BJ_BBOX["lng_max"]) &
                  (lat >= BJ_BBOX["lat_min"]) & (lat <= BJ_BBOX["lat_max"]))
        rows.append({
            "字段": "目标坐标(target)",
            "空值率": f"{((lng.isna()) | (lat.isna())).mean() * 100:.3f}%",
            "超短(<5字符)率": "N/A",
            "超长(>200字符)率": f"北京边界内 {(inside.mean() * 100):.2f}%",
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 内置示例数据（覆盖 5.5.1~5.5.5 全部字段，无需数据库）
# ----------------------------------------------------------------------------
def _生成示例数据(n: int = 600) -> Tuple[pd.DataFrame, Dict]:
    rng = np.random.default_rng(42)
    地标名 = [f"地标{i:02d}" for i in range(12)]
    地标坐标 = {f"地标{i:02d}": (116.3 + 0.05 * i, 39.9 + 0.03 * i) for i in range(12)}
    场景 = ["urban", "suburb", "rural"]
    模糊 = ["EXACT", "SLIGHT", "MODERATE", "LARGE"]
    方位 = ["东", "南", "西", "北", "东北", "东南", "西南", "西北"]
    拓扑词 = ["", "", "附近", "相邻", "对面"]
    recs = []
    for _ in range(n):
        lm = str(rng.choice(地标名))
        d = int(rng.integers(50, 1000))
        ang = str(rng.choice(方位))
        topo = str(rng.choice(拓扑词))
        desc = f"在{lm}{ang}侧大约{d}米{topo}处"
        desc_en = f"about {d} meters {ang.lower()} of {lm}"
        dir8 = {"东": "E", "南": "S", "西": "W", "北": "N", "东北": "NE",
                "东南": "SE", "西南": "SW", "西北": "NW"}[ang]
        # 真值坐标：由地标坐标 + 方向 + 距离反推（含约 8 米级噪声）
        cx, cy = 地标坐标[lm]
        xh, yh = 反推坐标(cx, cy, dir8, float(d))
        noise_m = rng.normal(0, 8)
        target_lng = xh + noise_m / (111_320.0 * math.cos(math.radians(cy)))
        target_lat = yh + noise_m / 111_320.0
        topology = {"附近": "disjoint", "相邻": "adjacent", "对面": "crosses"}.get(topo, "adjacent")
        # 模板基线文本（4.4 节纯模板填充，无润色）：用于 5.5.3.1 相对改进对照
        desc_raw = f"{lm}{ang}侧{d}米{topo}"
        # 参考地标经纬度（对应 spatial_relations.landmark_x / landmark_y）
        recs.append(dict(
            id=_, description=desc, description_en=desc_en, description_raw=desc_raw, ref_name=lm,
            scene_code=str(rng.choice(场景)), ambiguity_level=str(rng.choice(模糊)),
            target_lng=target_lng, target_lat=target_lat,
            direction=dir8, distance=float(d), topology=topology,
            ref_landmark_id=lm, ref_landmark_name=lm,
            landmark_x=cx, landmark_y=cy,
        ))
    df_demo = pd.DataFrame(recs)
    # 制造少量"同一目标点被多条样本描述"以演示多描述收敛（5.5.4(3)）
    hub = [(116.40, 39.95), (116.55, 40.00), (116.70, 39.88)]
    sel = df_demo.sample(30, random_state=1).index
    for k, idx in enumerate(sel):
        df_demo.at[idx, "target_lng"] = hub[k % 3][0]
        df_demo.at[idx, "target_lat"] = hub[k % 3][1]
    return df_demo, 地标坐标


# ----------------------------------------------------------------------------
# 结果写出：xlsx（多工作表）/ json / csv
# ----------------------------------------------------------------------------
def _写出结果(path: str, out_json: Dict[str, object], 表们: Dict[str, pd.DataFrame]) -> None:
    """将全部统计结果写出到单一文件。

    - .xlsx：每个统计表写入一个工作表（表22~表26 + 表25(若有)），便于论文汇总；
    - .json：保留原始嵌套结构（标量 + 记录列表）；
    - 其他后缀：默认将表 23（含去地名掩码）导出为 CSV 供快速预览。
    """
    low = path.lower()
    if low.endswith(".xlsx"):
        try:
            import openpyxl  # 仅在写 xlsx 时惰性导入，未安装则优雅降级
        except ImportError:
            logger.warning("未检测到 openpyxl，无法写 xlsx；请执行 `pip install openpyxl`，或改用 .json 输出")
            _写出结果(path.rsplit(".", 1)[0] + ".json", out_json, 表们)
            return
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for sheet, frame in 表们.items():
                # Excel 工作表名上限 31 字符
                frame.to_excel(xw, sheet_name=sheet[:31], index=False)
        logger.info("结果已写入 Excel（共 %d 个工作表）：%s", len(表们), path)
    elif low.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2, default=str)
        logger.info("结果已写入 JSON：%s", path)
    else:
        # 默认 CSV：表 23 含全部语言统计（原始 + 去地名掩码）作示例
        表们.get("表23_语言统计", pd.DataFrame()).to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("结果已写入 CSV（表23 示例）：%s", path)


# ----------------------------------------------------------------------------
# 命令行入口：运行 5.5.1~5.5.5 全链路
# ----------------------------------------------------------------------------
def _main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="5.5 数据集统计特征与质量评估：全链路统计计算")
    parser.add_argument("--input", help="本地 CSV（含 geo_desc+spatial_relations 关联字段）")
    parser.add_argument("--database-url", help="PostGIS 连接串（缺省从 .env 读取）")
    parser.add_argument("--landmark-csv", help="地标坐标 CSV 覆盖（landmark_id/lng/lat 或 name/lng/lat）；缺省从 spatial_relations.landmark_x/landmark_y 自动构建")
    parser.add_argument("--template-col", default="description_raw", help="模板基线(中文)列名（表24 的\"模板基线\"列；默认 geo_desc.description_raw）")
    parser.add_argument("--human-csv", help="人工评分 CSV（用于 5.5.3.2）")
    parser.add_argument("--tokenizer-zh", default="auto", choices=["auto", "char", "jieba"])
    parser.add_argument("--distance-tol", type=float, default=0.15, help="距离一致性容差")
    parser.add_argument("--output", help="结果输出路径（.json 或 .csv）")
    parser.add_argument("--demo", action="store_true", help="使用内置示例数据（无需数据库/CSV）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.demo:
        df, 地标坐标 = _生成示例数据()
        logger.info("使用内置示例数据，样本数 = %d", len(df))
    else:
        df, 地标坐标 = 加载数据(数据库_URL=args.database_url, csv路径=args.input, landmark_csv=args.landmark_csv)

    out: Dict[str, object] = {}
    表们: Dict[str, pd.DataFrame] = {}

    print("\n===== 5.5.1 数据集总体规模与构成（表 22） =====")
    t22 = 计算_规模与构成(df)
    print(t22.to_string(index=False))
    out["表22_规模与构成"] = t22.to_dict(orient="records")
    表们["表22_规模与构成"] = t22

    print("\n===== 5.5.2 文本描述的语言统计特征（表 23，含去地名掩码） =====")
    t23 = 计算_语言统计(df, zh_mode=args.tokenizer_zh)
    print(t23.to_string(index=False))
    out["表23_语言统计"] = t23.to_dict(orient="records")
    表们["表23_语言统计"] = t23

    print("\n===== 5.5.3.1 自动化指标评估（表 24） =====")
    t24 = 计算_自动指标(df, template_col=args.template_col, zh_mode=args.tokenizer_zh)
    print(t24.to_string(index=False))
    out["表24_自动指标"] = t24.to_dict(orient="records")
    表们["表24_自动指标"] = t24

    t25 = None
    if args.human_csv:
        print("\n===== 5.5.3.2 人工抽样评估（表 25） =====")
        t25 = 计算_人工评估(args.human_csv)
        print(t25.to_string(index=False))
        out["表25_人工评估"] = t25.to_dict(orient="records")
        表们["表25_人工评估"] = t25

    print("\n===== 5.5.4 几何一致性与空间逻辑校验（表 26） =====")
    t26 = 计算_几何一致性(df, 地标坐标, 距离容差=args.distance_tol)
    for k, v in t26.items():
        print(f"  {k}: {v}")
    out["表26_几何一致性"] = t26
    t26_df = pd.DataFrame([{"指标": k, "数值": v} for k, v in t26.items()])
    表们["表26_几何一致性"] = t26_df

    print("\n===== 5.5.5 数据集完整性与合规性校验 =====")
    t55 = 计算_完整性(df)
    print(t55.to_string(index=False))
    out["表55_完整性"] = t55.to_dict(orient="records")
    表们["表55_完整性"] = t55

    if args.output:
        _写出结果(args.output, out, 表们)


if __name__ == "__main__":
    _main()
