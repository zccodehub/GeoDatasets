# -*- coding: utf-8 -*-
"""
5.5.2 文本描述的语言统计特征 —— 指标计算实现（PostGIS 数据源，双语对照表）

数据来源
========
  - 中文描述 ：PostGIS 表 geo_desc.description
  - 英文描述 ：PostGIS 表 geo_desc.description_en

实现的统计表（行 = 指标，列 = 语言）
====================================
  (1) 平均字符数 / 词数      mean of 字符数 / 词数（中文以字为主、英文以词为主，均双值呈现）
  (2) 中位字符数 / 词数      median of 字符数 / 词数
  (3) 句长 IQR              字符数分布的 四分位距 Q3 - Q1（句长按字符数口径，见 5.5.2）
  (4) TTR                   类符形符比 V / M
  (5) Distinct-1            n-gram 去重率（Li et al., 2016）
  (6) Distinct-2            n-gram 去重率（Li et al., 2016）
  (7) Self-BLEU             自洽性指标（越低多样性越强；Zhu et al., 2018），基于随机抽样的近似估计
  (8) 词汇熵                Shannon 熵 H（bit），并附归一化熵 H_norm

约定
====
  - 中文分词默认"auto"：若已安装 jieba 则优先词级（TTR / Distinct / 词汇熵以"词"
    为单位方有意义），否则退回字符级；可用 --tokenizer-zh char/jieba 显式指定。
  - 英文按正则切分为 word（小写、去标点）。
  - Self-BLEU 在大规模语料下采用"假设句抽样 × 参考池"的近似计算，避免 O(N^2) 开销；
    可通过 --self-bleu-hyp / --self-bleu-pool 调节精度与耗时。

依赖：pandas, numpy；（PostGIS 路径）sqlalchemy, psycopg2, python-dotenv；（可选）jieba
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 地标集总数：来自 4.2.4 节（若需计算地标覆盖率可传入；本表未直接使用，保留以备扩展）
DEFAULT_N_LANDMARK: int = 26_800

# 批次读取大小（遵循项目规范，流式读取避免一次性加载 47 万条导致 OOM）
CHUNK_SIZE: int = 10_000

# 英文分词正则：保留字母、数字与缩写撇号
_EN_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


# ----------------------------------------------------------------------------
# 分词
# ----------------------------------------------------------------------------
def 是否支持_jieba() -> bool:
    """检测运行环境是否已安装 jieba 分词库。"""
    try:
        import jieba  # noqa: F401
        return True
    except Exception:  # pragma: no cover - 依赖可选
        return False


def 字符分词(text: str, use_jieba: bool = False) -> List[str]:
    """将一条中文描述切分为词元（token）序列。

    参数
    ----
    text : str
        待切分的中文文本。
    use_jieba : bool
        是否使用 jieba 进行词级切分；为 False 时退化为字符级切分
        （每个非空白汉字为一个词元），保证无外部依赖即可运行。

    返回
    ----
    List[str]
        词元列表。文本为空时返回空列表。
    """
    text = (text or "").strip()
    if not text:
        return []
    if use_jieba and 是否支持_jieba():
        import jieba

        return [t for t in jieba.lcut(text) if t.strip()]
    # 字符级：去除空白后逐字切分（与"以字符数为单位"的句长口径一致）
    return [ch for ch in text if not ch.isspace()]


def 分词_英文(text: str) -> List[str]:
    """将一条英文描述切分为 word 序列（小写、去标点）。"""
    text = (text or "").lower()
    if not text.strip():
        return []
    return _EN_WORD_RE.findall(text)


def 选择分词函数(语言: str, zh_mode: str = "auto") -> Callable[[str], List[str]]:
    """根据语言与中文分词模式返回对应的分词函数。

    参数
    ----
    语言 : str
        "zh" 或 "en"。
    zh_mode : str
        中文分词模式："auto"（已装 jieba 则用词级，否则字符级）、
        "char"（强制字符级）、"jieba"（强制词级，未安装则报错）。
        说明：中文 TTR / Distinct / 词汇熵等以"词"为词表单位方有意义，
        故默认优先词级；英文一律按 word 切分。
    """
    if 语言 == "en":
        return 分词_英文
    if zh_mode == "char":
        return 字符分词
    if zh_mode == "jieba":
        if not 是否支持_jieba():
            raise RuntimeError("未安装 jieba，无法使用词级分词（请 pip install jieba 或改用 --tokenizer-zh char）")
        return lambda t: 字符分词(t, use_jieba=True)
    # auto：优先 jieba 词级，未安装则退回字符级
    if 是否支持_jieba():
        return lambda t: 字符分词(t, use_jieba=True)
    return 字符分词


def 字符数(text: str) -> int:
    """统计一条描述中的非空白字符数（中英文通用，作为"句长"口径）。"""
    return len([c for c in (text or "") if not c.isspace()])


# ----------------------------------------------------------------------------
# 基础指标（语言无关）
# ----------------------------------------------------------------------------
def 计算类符形符比(tokens: Sequence[str]) -> float:
    """计算类符形符比 TTR = V / M。词元为空时返回 0.0。"""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def 计算词汇熵(tokens: Sequence[str]) -> Tuple[float, float, float]:
    """计算词表分布的各项熵指标（Shannon 熵体系）。

    返回 (H, H_norm, perplexity)：
      - H          : Shannon 熵（bit）
      - H_norm     : 归一化熵 H / log2(V)
      - perplexity : 困惑度 2 ** H
    词元为空时返回 (0.0, 0.0, 1.0)。
    """
    if not tokens:
        return 0.0, 0.0, 1.0
    counts = Counter(tokens)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()], dtype=float)
    H = float(-np.sum(probs * np.log2(probs)))
    V = len(counts)
    H_norm = H / np.log2(V) if V > 1 else 0.0
    return H, float(H_norm), float(2.0 ** H)


def _抽取_ngram(tokens: List[str], n: int) -> List[tuple]:
    """从词元序列抽取全部长度为 n 的 n-gram。"""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def 计算_distinct_n(texts: Sequence[str], n: int, 分词函数: Callable[[str], List[str]]) -> float:
    """计算语料级 Distinct-n = 唯一 n-gram 数 / 全部 n-gram 数（Li et al., 2016）。"""
    all_ngrams: List[tuple] = []
    for t in texts:
        all_ngrams.extend(_抽取_ngram(分词函数(t), n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def _ngram计数字典(tokens: List[str], max_n: int) -> Dict[int, Counter]:
    """返回 {n: n-gram Counter}，n = 1..max_n。"""
    return {n: Counter(_抽取_ngram(tokens, n)) for n in range(1, max_n + 1)}


def _单句_bleu(hyp_toks: List[str], ref_data: List[Tuple[int, Dict[int, Counter]]], max_n: int) -> float:
    """计算单条假设句对参考池的 BLEU（多参考取各 n-gram 最大计数，closest 长度惩罚）。"""
    if not hyp_toks:
        return 0.0
    c = len(hyp_toks)
    ref_lens = [rd[0] for rd in ref_data]
    # 有效参考长度：与假设长度最接近者
    r = min(ref_lens, key=lambda L: (abs(L - c), L)) if ref_lens else 0
    bp = 1.0 if c >= r else float(np.exp(1.0 - r / c))
    logs: List[float] = []
    for n in range(1, max_n + 1):
        hyp_ng = Counter(_抽取_ngram(hyp_toks, n))
        if not hyp_ng:
            logs.append(float("-inf"))
            continue
        clipped = 0
        for gram, cnt in hyp_ng.items():
            max_ref = max((rd[1].get(n, {}).get(gram, 0) for rd in ref_data), default=0)
            clipped += min(cnt, max_ref)
        p = clipped / sum(hyp_ng.values())
        logs.append(np.log(p) if p > 0 else float("-inf"))
    if any(v == float("-inf") for v in logs):
        return 0.0
    return bp * float(np.exp(np.mean(logs)))


def 计算_self_bleu(
    texts: Sequence[str],
    分词函数: Callable[[str], List[str]],
    max_n: int = 4,
    假设样本数: int = 500,
    参考池大小: int = 200,
    random_state: int = 42,
) -> float:
    """计算语料 Self-BLEU（Zhu et al., 2018）。

    由于全量 O(N^2) 在约 47 万条语料上不可行，本实现采用"假设句抽样 × 参考池"
    的近似估计：从语料中抽取 假设样本数 条作为假设、另取 参考池大小 条作为参考池
    （二者不相交），对每条假设计算其对参考池的 BLEU，取均值。数值越低，样本间
    相互重复度越小、多样性越强。

    参数
    ----
    texts : Sequence[str]
        描述文本集合。
    分词函数 : Callable
        分词函数。
    max_n : int
        BLEU 最高 n-gram 阶数（默认 4）。
    假设样本数 : int
        参与计算的假设句数量（抽样上限）。
    参考池大小 : int
        每条假设的参考句数量（抽样上限）。
    random_state : int
        随机种子，保证结果可复现。

    返回
    ----
    float
        Self-BLEU 均值（区间 [0, 1]）。
    """
    N = len(texts)
    if N < 2:
        return 0.0
    rng = np.random.default_rng(random_state)
    total = min(N, 假设样本数 + 参考池大小)
    idx = rng.choice(N, size=total, replace=False)
    hyp_idx = list(idx[: min(假设样本数, total)])
    ref_idx = list(idx[min(假设样本数, total):])

    参考数据: List[Tuple[int, Dict[int, Counter]]] = []
    for i in ref_idx:
        toks = 分词函数(texts[i])
        参考数据.append((len(toks), _ngram计数字典(toks, max_n)))

    scores: List[float] = []
    for i in hyp_idx:
        scores.append(_单句_bleu(分词函数(texts[i]), 参考数据, max_n))
    return float(np.mean(scores)) if scores else 0.0


# ----------------------------------------------------------------------------
# 指标集（供双语对照表使用）
# ----------------------------------------------------------------------------
def 计算语言指标集(
    texts: Sequence[str],
    分词函数: Callable[[str], List[str]],
    *,
    假设样本数: int = 500,
    参考池大小: int = 200,
    random_state: int = 42,
) -> Dict[str, float]:
    """计算某一语言所需的全部行指标，返回以英文键组织的字典。

    键：avg_char, avg_word, med_char, med_word, iqr_char,
        ttr, distinct_1, distinct_2, self_bleu, vocab_entropy, vocab_entropy_norm
    """
    texts = [str(t) for t in texts]
    字符数列 = np.array([字符数(t) for t in texts], dtype=float)
    词数例 = np.array([len(分词函数(t)) for t in texts], dtype=float)
    all_tokens: List[str] = []
    for t in texts:
        all_tokens.extend(分词函数(t))

    H, H_norm, _ = 计算词汇熵(all_tokens)
    return {
        "avg_char": float(字符数列.mean()),
        "avg_word": float(词数例.mean()),
        "med_char": float(np.median(字符数列)),
        "med_word": float(np.median(词数例)),
        "iqr_char": float(np.percentile(字符数列, 75) - np.percentile(字符数列, 25)),
        "ttr": 计算类符形符比(all_tokens),
        "distinct_1": 计算_distinct_n(texts, 1, 分词函数),
        "distinct_2": 计算_distinct_n(texts, 2, 分词函数),
        "self_bleu": 计算_self_bleu(
            texts, 分词函数, max_n=4,
            假设样本数=假设样本数, 参考池大小=参考池大小, random_state=random_state,
        ),
        "vocab_entropy": H,
        "vocab_entropy_norm": H_norm,
    }


def 构建双语对照表(
    zh_texts: Sequence[str],
    en_texts: Sequence[str],
    *,
    zh_mode: str = "auto",
    假设样本数: int = 500,
    参考池大小: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """构建"行=指标、列=中文描述/英文描述"的对照表。

    参数
    ----
    zh_texts, en_texts : Sequence[str]
        中文 / 英文描述集合。
    zh_mode : str
        中文分词模式（"auto" / "char" / "jieba"）。
    假设样本数, 参考池大小 : int
        Self-BLEU 抽样的假设句数与参考池大小。
    random_state : int
        随机种子。

    返回
    ----
    pd.DataFrame
        三列：指标、中文描述、英文描述。
    """
    zh_metrics = 计算语言指标集(
        zh_texts, 选择分词函数("zh", zh_mode),
        假设样本数=假设样本数, 参考池大小=参考池大小, random_state=random_state,
    )
    en_metrics = 计算语言指标集(
        en_texts, 选择分词函数("en"),
        假设样本数=假设样本数, 参考池大小=参考池大小, random_state=random_state,
    )

    def 行(标签, zh_val, en_val) -> Dict[str, str]:
        return {"指标": 标签, "中文描述": zh_val, "英文描述": en_val}

    rows = [
        行("平均字符数 / 词数",
           f"{zh_metrics['avg_char']:.2f} 字 / {zh_metrics['avg_word']:.2f} 词",
           f"{en_metrics['avg_char']:.2f} 字 / {en_metrics['avg_word']:.2f} 词"),
        行("中位字符数 / 词数",
           f"{zh_metrics['med_char']:.2f} 字 / {zh_metrics['med_word']:.2f} 词",
           f"{en_metrics['med_char']:.2f} 字 / {en_metrics['med_word']:.2f} 词"),
        行("句长 IQR (字符)",
           f"{zh_metrics['iqr_char']:.2f}",
           f"{en_metrics['iqr_char']:.2f}"),
        行("TTR",
           f"{zh_metrics['ttr']:.4f}",
           f"{en_metrics['ttr']:.4f}"),
        行("Distinct-1",
           f"{zh_metrics['distinct_1']:.4f}",
           f"{en_metrics['distinct_1']:.4f}"),
        行("Distinct-2",
           f"{zh_metrics['distinct_2']:.4f}",
           f"{en_metrics['distinct_2']:.4f}"),
        行("Self-BLEU (越低越好)",
           f"{zh_metrics['self_bleu']:.4f}",
           f"{en_metrics['self_bleu']:.4f}"),
        行("词汇熵 (bit)",
           f"{zh_metrics['vocab_entropy']:.4f}",
           f"{en_metrics['vocab_entropy']:.4f}"),
    ]
    return pd.DataFrame(rows, columns=["指标", "中文描述", "英文描述"])


# ----------------------------------------------------------------------------
# 数据加载：PostGIS（主） / CSV（测试）
# ----------------------------------------------------------------------------
def 加载数据(
    *,
    数据库_URL: Optional[str] = None,
    csv路径: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """加载数据集。优先使用 csv路径（测试用），否则从 PostGIS 读取 geo_desc。

    返回含 description / description_en 两列的数据框。
    """
    if csv路径:
        logger.info("从 CSV 读取数据：%s", csv路径)
        return pd.read_csv(csv路径)

    from dotenv import load_dotenv
    from sqlalchemy import create_engine, text

    _env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(dotenv_path=os.path.abspath(_env_path))
    if 数据库_URL is None:
        数据库_URL = os.getenv("DATABASE_URL")
    if not 数据库_URL:
        raise ValueError("未找到 DATABASE_URL，请检查 .env 或传入 --database-url")

    engine = create_engine(数据库_URL, pool_pre_ping=True)
    sql = text("""
        SELECT description,
               description_en
        FROM geo_desc
        ORDER BY id
    """)
    logger.info("从 PostGIS 读取 geo_desc(description, description_en) ...")
    with engine.connect() as conn:
        reader = pd.read_sql(sql, conn, chunksize=chunk_size)
        df = pd.concat([chunk for chunk in reader], ignore_index=True)
    logger.info("读取完成，样本数 = %d", len(df))
    return df


# ----------------------------------------------------------------------------
# 演示数据（无需外部数据即可验证算法）
# ----------------------------------------------------------------------------
def _生成示例数据(n: int = 800) -> pd.DataFrame:
    """生成一份小规模合成数据集（含 description / description_en）。"""
    rng = np.random.default_rng(42)
    地标 = [f"地标{i:02d}" for i in range(20)]
    模板 = ["在{}东侧大约{}米处", "距{}东方约{}米", "从{}往东走{}米就到了", "{}东边{}米左右", "往{}走{}米便是"]
    模板_en = [
        "about {} meters east of {}",
        "roughly {} m to the east of {}",
        "{} is around {} meters east",
        "{} , approximately {} m east",
        "walk {} meters toward {}",
    ]
    rows = []
    for _ in range(n):
        lm = str(rng.choice(地标))
        d = int(rng.integers(50, 1000))
        rows.append(
            dict(
                description=rng.choice(模板).format(lm, d),
                description_en=rng.choice(模板_en).format(d, lm),
            )
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------------------
def _main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="5.5.2 语言统计特征：双语对照表（中文描述 vs 英文描述）"
    )
    parser.add_argument("--input", help="本地 CSV（测试用，含 description/description_en）")
    parser.add_argument("--database-url", help="PostGIS 连接串（缺省从 .env 读取 DATABASE_URL）")
    parser.add_argument("--tokenizer-zh", default="auto",
                        choices=["auto", "char", "jieba"],
                        help="中文分词模式：auto(优先 jieba 词级) / char(字符级) / jieba(强制词级)")
    parser.add_argument("--self-bleu-hyp", type=int, default=500, help="Self-BLEU 假设句抽样数")
    parser.add_argument("--self-bleu-pool", type=int, default=200, help="Self-BLEU 参考池大小")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--output", help="结果输出路径（.json 或 .csv）")
    parser.add_argument("--demo", action="store_true", help="使用内置示例数据（无需数据库/CSV）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.demo:
        df = _生成示例数据()
        logger.info("使用内置示例数据运行，样本数 = %d", len(df))
    else:
        df = 加载数据(数据库_URL=args.database_url, csv路径=args.input)

    if "description" not in df.columns or "description_en" not in df.columns:
        raise ValueError("数据需包含 description 与 description_en 两列")

    zh_texts = df["description"].astype(str).tolist()
    en_texts = df["description_en"].fillna("").astype(str).tolist()

    对照表 = 构建双语对照表(
        zh_texts, en_texts,
        zh_mode=args.tokenizer_zh,
        假设样本数=args.self_bleu_hyp, 参考池大小=args.self_bleu_pool,
        random_state=args.random_state,
    )
    print(对照表.to_string(index=False))

    if args.output:
        if args.output.lower().endswith(".json"):
            对照表.to_json(args.output, force_ascii=False, indent=2)
        else:
            对照表.to_csv(args.output, index=False, encoding="utf-8-sig")
        logger.info("结果已写入 %s", args.output)


if __name__ == "__main__":
    _main()
