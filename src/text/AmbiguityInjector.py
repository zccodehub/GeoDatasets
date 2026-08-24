import random
from typing import Dict, List, Optional

class AmbiguityInjector:
    """
    针对已预计算 ambiguity_level 和 deviation_angle 的副词动态注入协同器。
    职责：
        1. 根据 ambiguity_level 调整粒度选择权重（粒度层协同）
        2. 根据 ambiguity_level 返回副词候选池（副词层协同）
        3. 提供最终组合方法（方向词 + 副词 + 后缀）
    """

    # ---------- 基础词根映射 ----------
    DIRECTION_MAP = {
        'North': '北', 'Northeast': '东北', 'East': '东', 'Southeast': '东南',
        'South': '南', 'Southwest': '西南', 'West': '西', 'Northwest': '西北'
    }

    # ---------- 副词池（直接按 ambiguity_level 索引） ----------
    ADVERB_POOLS = {
        'EXACT': ['正', '直', ''],
        'SLIGHT': ['略', '稍', '微微'],
        'MODERATE': ['明显', '比较']
    }

    # ---------- 粒度基础权重 ----------
    BASE_GRANULARITY_WEIGHTS = {
        'precise': 0.20,
        'vague': 0.50,
        'functional': 0.30
    }

    @classmethod
    def adjust_granularity_weights(cls, ambiguity_level: str) -> Dict[str, float]:
        """
        协同机制第一步：根据 ambiguity_level 调整粒度权重

        当 ambiguity_level != 'EXACT' 时（即 SLIGHT 或 MODERATE）：
            - 精确粒度：大幅降低
            - 模糊粒度：大幅提升
            - 功能化粒度：小幅提升

        当 ambiguity_level == 'EXACT' 时：
            - 稍微偏向精确粒度
        """
        weights = cls.BASE_GRANULARITY_WEIGHTS.copy()
        if ambiguity_level != 'EXACT':
            # 边界/模糊状态：强制模糊主导
            weights['precise'] = 0.05
            weights['vague'] = 0.65
            weights['functional'] = 0.30
        else:
            # 精确状态：略微增加精确粒度权重
            weights['precise'] = 0.25
            weights['vague'] = 0.45
            weights['functional'] = 0.30
        return weights

    @classmethod
    def get_adverb_candidates(cls, ambiguity_level: str) -> List[str]:
        """
        协同机制第二步：根据 ambiguity_level 过滤副词候选池

        核心逻辑：
            - 若 ambiguity_level = 'EXACT'：保留完整池，但通过重复“正/直”实现加权
            - 若 ambiguity_level = 'SLIGHT' 或 'MODERATE'：强制移除“正/直”和空字符串，
              只保留模糊副词（略/稍/微微/明显/比较）
        """
        base_pool = cls.ADVERB_POOLS.get(ambiguity_level, cls.ADVERB_POOLS['EXACT'])

        if ambiguity_level == 'EXACT':
            # 非边界：重复“正/直”以提升概率
            weighted_pool = []
            for word in base_pool:
                if word in ['正', '直']:
                    # 精确词出现 3 次（约 3/5 ≈ 60% 概率）
                    weighted_pool.extend([word, word, word])
                else:
                    weighted_pool.append(word)
            return weighted_pool
        else:
            # SLIGHT 或 MODERATE：过滤掉“正/直”和空字符串
            filtered = [w for w in base_pool if w not in ['正', '直', '']]
            # 若过滤后为空（极端情况），用保底模糊词
            if not filtered:
                filtered = ['略', '稍']
            return filtered

    @classmethod
    def compose_direction_text(
        cls,
        direction_8: str,
        deviation_angle: float,
        ambiguity_level: str,
        granularity: str,
        use_suffix: bool = True
    ) -> str:
        """
        最终组合方法：根据粒度和 ambiguity_level 生成方向文本

        :param direction_8: 主方向（如 'East'）
        :param deviation_angle: 预计算的偏离角度（度）
        :param ambiguity_level: 预计算的模糊等级（'EXACT' / 'SLIGHT' / 'MODERATE'）
        :param granularity: 当前选中的粒度（'precise' / 'vague' / 'functional'）
        :param use_suffix: 是否随机添加“边/侧/方”后缀（增加多样性）
        :return: 组合后的方向描述
        """
        base = cls.DIRECTION_MAP.get(direction_8, '')
        if not base:
            return '附近'

        # 获取副词候选
        adv_candidates = cls.get_adverb_candidates(ambiguity_level)
        adv = random.choice(adv_candidates) if adv_candidates else ''

        is_ambiguous = (ambiguity_level != 'EXACT')

        # 根据粒度分支处理
        if granularity == 'precise':
            # 精确粒度：强制使用“正/直”
            if adv not in ['正', '直']:
                adv = random.choice(['正', '直'])
            # 若偏差 > 3°，虽然选了精确粒度但实际已偏，加“偏”字以维持诚实
            if deviation_angle > 3:
                return f"{adv}{base}偏"
            return f"{adv}{base}"

        elif granularity == 'functional':
            # 功能化粒度通常在外部处理（道路/地标名），此处仅做保底
            # 若落到这里，说明外部未提供道路/地标，直接返回方位词+后缀
            suffix = random.choice(['方向', '那边'])
            return f"{base}{suffix}"

        else:  # vague 模糊粒度
            # 模糊粒度：按照 ambiguity_level 和 adv 生成
            if not adv:
                return base

            # 若 adv 是“正/直”但 ambiguity_level 非 EXACT，强制替换为“略”（双重保险）
            if adv in ['正', '直'] and is_ambiguous:
                adv = random.choice(['略', '稍'])

            # 构造“副词偏方位”或“正方位”
            if adv in ['正', '直']:
                # 精确副词：不加“偏”，但可加后缀
                if use_suffix and random.random() < 0.3:
                    suffix = random.choice(['边', '侧'])
                    return f"{adv}{base}{suffix}"
                return f"{adv}{base}"
            else:
                # 模糊副词：加“偏”字（80%概率）或直接叠加（20%）
                if random.random() < 0.8:
                    return f"{adv}偏{base}"
                else:
                    return f"{adv}{base}"