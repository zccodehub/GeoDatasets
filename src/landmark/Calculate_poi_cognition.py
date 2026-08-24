import os
import re
import math
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import pandas as pd
import jieba
from collections import Counter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
BASE_DIR = Path(r"E:\Projects\PycharmProjects\Geocoding_dataset\data")

class CognitionProcessor:
    """负责POI认知显著性计算的类，封装所有核心逻辑。"""
    
    def __init__(self, input_file: str,output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.json_dir = os.path.join(BASE_DIR, "json")
        self.proper_nouns: List[Dict[str, Any]] = []
        self.all_suffixes: List[str] = []
        self.district_prefixes: List[str] = []
        self.flat_facilities: List[Dict[str, Any]] = []
        self._load_resources()

    def _load_json(self, filename: str) -> Any:
        with open(os.path.join(self.json_dir, filename), 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_resources(self):
        """加载所有辅助资源配置。"""
        # 加载行政区划
        self.district_prefixes = sorted(self._load_json("district_prefix.json")["stopwords"], key=len, reverse=True)
        
        # 加载专有名词
        pn_data = self._load_json("proper_nouns.json")
        for cat in pn_data["categories"]:
            for sub in cat["subcategories"]:
                self.proper_nouns.extend(sub["keywords"])

        # 加载后缀并扁平化
        suffix_groups = self._load_json("suffixes.json")["suffixes"]
        suffixes = set()
        for group in suffix_groups:
            for level in ['level_1', 'level_2', 'level_3']:
                suffixes.update(group.get(level, []))
        self.all_suffixes = sorted(list(suffixes), key=len, reverse=True)

        # 加载设施并扁平化
        fac_data = self._load_json("facilities.json")
        factors = {
            "high": fac_data["penalty_levels"]["high"]["factor"],
            "medium": fac_data["penalty_levels"]["medium"]["factor"],
            "low": fac_data["penalty_levels"]["low"]["factor"]
        }
        for cat in fac_data["categories"]:
            penalty = cat["penalty"]
            for kw in cat["keywords"]:
                self.flat_facilities.append({"word": kw, "factor": factors[penalty]})

    def clean_name(self, raw_name: str) -> str:
        """清洗 POI 名称。"""
        name = re.sub(r'[\(\（].*?[\)\）]', '', raw_name)
        for prefix in self.district_prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return re.sub(r'\s+', '', name)

    def calculate_knowledge_signal(self, clean_name: str, poi_type: str) -> float:
        """计算知识库收录迹象信号。"""
        len_good = 1 if 3 <= len(clean_name) <= 10 else 0
        no_suffix = 1 if not any(clean_name.endswith(s) for s in self.all_suffixes) else 0
        has_proper = 1 if any(kw['word'] in clean_name for kw in self.proper_nouns) else 0
        consistent = 1 if poi_type in clean_name else 0
        return (len_good + no_suffix + has_proper + consistent) / 4

    def calculate_brand_signal(self, clean_name: str) -> float:
        """计算品牌/专名强度信号。"""
        max_w = max([kw['weight'] for kw in self.proper_nouns if kw['word'] in clean_name], default=0.0)
        return min(max_w / 0.5, 1.0)

    def calculate_uniqueness(self, df: pd.DataFrame) -> pd.Series:
        # Uniqueness 计算
        all_tokens = [t for name in df['clean_name'] for t in self.tokenize(name)]
        counts = Counter(all_tokens)
        n = len(df)

        def get_uniqueness(name):
            tokens = self.tokenize(name)
            if not tokens: return 0
            return math.log10(1 + n / max([counts[t] for t in tokens]))

        uniqueness = df['clean_name'].apply(get_uniqueness)
        return (uniqueness - uniqueness.min()) / (uniqueness.max() - uniqueness.min())

    def tokenize(self, name: str) -> List[str]:
        """对POI名称进行分词预处理。"""
        words = jieba.lcut(name)
        return [w for w in words if w not in self.all_suffixes and len(w) >= 2 and not w.isdigit()]

    def calculate_penalty(self, raw_name: str) -> float:
        """计算功能设施惩罚因子。"""
        # 判断是否以地标白名单开头
        is_landmark_prefix = any(raw_name.startswith(kw['word']) for kw in self.proper_nouns)
        # 检查是否包含任何功能设施关键词
        matched_factor = None
        for f in self.flat_facilities:
            if f['word'] in raw_name:
                matched_factor = f['factor']  # 保存最后匹配的系数（或根据需求取最小/最大）
                break  # 找到第一个即可

        if is_landmark_prefix:
            # 地标附属设施 → 轻度惩罚（0.7）或免罚？根据设计选择
            return 0.7 if matched_factor is not None else 1.0
        else:
            # 非地标 → 按正常惩罚系数
            return matched_factor if matched_factor is not None else 1.0

    def calculate_cognition(self) :
        if not os.path.exists(self.input_file):
            logging.error(f"输入文件不存在: {self.input_file}")
            return

        df = pd.read_csv(self.input_file)
        logging.info("开始数据清洗与特征计算...")

        """计算认知显著性得分。"""
        df['clean_name'] = df['名称'].apply(self.clean_name)
        df['KnowledgeSignal'] = df.apply(lambda x: self.calculate_knowledge_signal(x['clean_name'], x['type_3']), axis=1)
        df['BrandSignal'] = df['clean_name'].apply(self.calculate_brand_signal)
        df['uniqueness'] = self.calculate_uniqueness(df)

        threshold = df['uniqueness'].quantile(0.9)
        star_ratios = df.groupby('type_3')[['uniqueness']].apply(lambda x: (x > threshold).mean())
        df['CategoryFameSignal'] = df['type_3'].map(star_ratios['uniqueness'])

        df['cognition_base'] = (0.2 * df['KnowledgeSignal'] + 0.35 * df['uniqueness'] +
                                0.25 * df['BrandSignal'] + 0.2 * df['CategoryFameSignal'])
        df['penalty'] = df['名称'].apply(self.calculate_penalty)
        df['cognition_score_raw'] =  df['cognition_base'] * df['penalty']
        df['cognition_score'] = (df['cognition_score_raw'] - df['cognition_score_raw'].min()) / (df['cognition_score_raw'].max() - df['cognition_score_raw'].min())

        df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        logging.info(f"处理完成，结果已保存: {self.output_file}")

if __name__ == "__main__":
    input_file = os.path.join(BASE_DIR, "Beijing_POI_IndividualScore.csv")
    output_file = os.path.join(BASE_DIR, "Beijing_POI_CognitionScore.csv")
    calc = CognitionProcessor(input_file, output_file)
    calc.calculate_cognition()

