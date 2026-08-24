import random
from typing import Optional, Dict

class DirectionGenerator:
    """
    方向词多粒度生成器（基于预计算的偏差角与模糊等级）
    输入参数直接来自 spatial_relations 表字段：
        - direction_8: 主方向（如 'East'）
        - deviation_angle: 预计算的偏离角度（0~22.5）
        - ambiguity_level: 预计算的模糊等级（'EXACT' / 'SLIGHT' / 'MODERATE'）
    """

    # 8方向基础词根映射
    DIRECTION_MAP = {
        'North': '北', 'Northeast': '东北', 'East': '东', 'Southeast': '东南',
        'South': '南', 'Southwest': '西南', 'West': '西', 'Northwest': '西北'
    }

    # 模糊副词池（直接按 ambiguity_level 索引）
    ADVERB_POOLS = {
        'EXACT': ['正', '直', ''],
        'SLIGHT': ['略', '稍', '微微'],
        'MODERATE': ['明显', '比较']
    }

    # 功能化后缀词库
    FUNCTIONAL_SUFFIXES = ['方向', '那边', '那头']

    # 粒度权重（可调）
    GRANULARITY_WEIGHTS = {
        'precise': 0.20,
        'vague': 0.50,
        'functional': 0.30
    }

    @staticmethod
    def _safe_str(val) -> str:
        """
        安全地将任意值转换为字符串，并处理常见空值情况
        """
        if val is None:
            return ''
        s = str(val)
        # 处理 pandas 读入的 NaN 或 None 字符串
        if s.lower() in ('nan', 'none', 'null', ''):
            return ''
        return s.strip()

    @classmethod
    def generate(
        cls,
        direction_8: str,
        deviation_angle: float,
        ambiguity_level: str,
        road_name: Optional[str] = None,
        landmark_name: Optional[str] = None,
        force_granularity: Optional[str] = None
    ) -> str:
        """
        生成方向描述文本
        """
        # 1. 基础校验
        if not direction_8 or direction_8 == 'None':
            return '附近'

        # 2. 判断是否为模糊状态
        is_ambiguous = (ambiguity_level != 'EXACT')

        # 3. 粒度选择（若未强制）
        if force_granularity:
            granularity = force_granularity
        else:
            granularity = cls._select_granularity(is_ambiguous)

        # 4. 功能化粒度特殊处理（安全转换后再使用）
        if granularity == 'functional':
            # 优先使用道路名
            road_str = cls._safe_str(road_name)
            if road_str:
                suffix = random.choice(cls.FUNCTIONAL_SUFFIXES)
                return f"{road_str}{suffix}"
            # 其次使用地标名
            # landmark_str = cls._safe_str(landmark_name)
            # if landmark_str:
            #     suffix = random.choice(cls.FUNCTIONAL_SUFFIXES)
            #     return f"{landmark_str}{suffix}"
            # 无参照，降级为模糊粒度
            granularity = 'vague'

        # 5. 根据粒度生成
        if granularity == 'precise':
            return cls._generate_precise(direction_8, deviation_angle, is_ambiguous)
        else:  # vague
            return cls._generate_vague(direction_8, ambiguity_level, is_ambiguous)

    # ---------- 以下辅助方法保持不变 ----------
    @classmethod
    def _select_granularity(cls, is_ambiguous: bool) -> str:
        weights = cls.GRANULARITY_WEIGHTS.copy()
        if is_ambiguous:
            weights['precise'] = 0.3
            weights['vague'] = 0.6
            weights['functional'] = 0.1
        else:
            weights['precise'] = 0.6
            weights['vague'] = 0.3
            weights['functional'] = 0.1
        total = sum(weights.values())
        rnd = random.random() * total
        cum = 0.0
        for key, w in weights.items():
            cum += w
            if rnd <= cum:
                return key
        return 'vague'

    @classmethod
    def _generate_precise(cls, direction_8: str, deviation: float, is_ambiguous: bool) -> str:
        base = cls.DIRECTION_MAP.get(direction_8, '')
        if not base:
            return ''
        if deviation <= 3:
            return f"正{base}"
        else:
            return f"{base}偏"

    @classmethod
    def _generate_vague(cls, direction_8: str, ambiguity_level: str, is_ambiguous: bool) -> str:
        base = cls.DIRECTION_MAP.get(direction_8, '')
        if not base:
            return ''
        adv_pool = cls.ADVERB_POOLS.get(ambiguity_level, cls.ADVERB_POOLS['EXACT'])
        if is_ambiguous:
            adv_pool = [a for a in adv_pool if a not in ['正', '直', '']]
            if not adv_pool:
                adv_pool = ['略', '稍']
        adv = random.choice(adv_pool)
        if adv in ['正', '直']:
            if random.random() < 0.3:
                suffix = random.choice(['边', '侧', '方'])
                return f"{adv}{base}{suffix}"
            return f"{adv}{base}"
        elif adv == '':
            return base
        else:
            if random.random() < 0.8:
                return f"{adv}偏{base}"
            else:
                return f"{adv}{base}"