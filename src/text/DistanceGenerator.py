import random
from typing import Optional


class DistanceGenerator:
    """
    距离词多粒度生成器（基于新等级定义）
    等级区间：
        VeryClose: [0, 50)
        Close:     [50, 200)
        Medium:    [200, 500)
        Far:       [500, 1000)
    """

    # ---------- 模糊粒度词袋（基于新区间） ----------
    VAGUE_POOLS = {
        'VeryClose': [
            '不到五十米', '几步路', '几十米', '近在咫尺',
            '走几步就到', '很近很近', '就在眼前'
        ],
        'Close': [
            '一百来米', '两百米内', '走两三分', '不到两百米',
            '一百多米', '近', '几分钟脚程'
        ],
        'Medium': [
            '三四百米', '五百米不到', '步行几分钟', '一里地左右',
            '约四百米', '不远', '走五六分钟'
        ],
        'Far': [
            '五百米开外', '七八百米', '近一公里', '八九百米',
            '一公里以内', '有些距离', '走十来分钟'
        ]
    }

    # ---------- 极模糊粒度定性词（基于新区间） ----------
    ULTRA_VAGUE_POOLS = {
        'VeryClose': ['就在跟前', '就在旁边', '触手可及', '近在眼前'],
        'Close': ['挺近的', '不算远', '很近', '没几步'],
        'Medium': ['不算远也不算近', '有些距离', '走一段'],
        'Far': ['有些远', '挺远的', '距离不近', '比较远']
    }

    # ---------- 语气词（极模糊时随机添加） ----------
    MODAL_PARTICLES = ['其实', '估摸着', '差不多', '大概', '应该']

    # ---------- 粒度权重 ----------
    GRANULARITY_WEIGHTS = {
        'precise': 0.5,
        'vague': 0.30,
        'ultra_vague': 0.1,
        'functional': 0.1
    }

    @classmethod
    def generate(cls,
                 exact_distance_m: float,
                 dist_level: str,
                 force_granularity: Optional[str] = None) -> str:
        """
        主入口：生成距离词文本

        :param exact_distance_m: 精确距离（米）
        :param dist_level: 距离等级 'VeryClose' | 'Close' | 'Medium' | 'Far'
        :param force_granularity: 强制粒度（可选）
        :return: 生成的距离描述文本
        """
        # 异常值保护
        if exact_distance_m < 0:
            exact_distance_m = 0.0
        if dist_level not in cls.VAGUE_POOLS:
            dist_level = 'Close'  # 保底

        # 粒度选择
        if force_granularity:
            granularity = force_granularity
        else:
            granularity = cls._select_granularity()

        # 路由
        if granularity == 'precise':
            return cls._generate_precise(exact_distance_m)
        elif granularity == 'functional':
            return cls._generate_functional(exact_distance_m, dist_level)
        elif granularity == 'ultra_vague':
            return cls._generate_ultra_vague(dist_level)
        else:  # vague
            return cls._generate_vague(dist_level)

    # ==================== 私有方法 ====================

    @classmethod
    def _select_granularity(cls) -> str:
        """按权重随机选择粒度"""
        weights = cls.GRANULARITY_WEIGHTS
        total = sum(weights.values())
        rnd = random.random() * total
        cum = 0.0
        for key, weight in weights.items():
            cum += weight
            if rnd <= cum:
                return key
        return 'vague'

    @classmethod
    def _generate_precise(cls, exact_distance_m: float) -> str:
        """
        精确粒度：
        - ≥1000m 转公里（但本定义最大1000，保留兼容）
        - <1000m 取整到十位（如487→约490米）
        - 0米特殊处理
        """
        if exact_distance_m == 0:
            return '0米（就在此处）'
        if exact_distance_m >= 1000:
            km = round(exact_distance_m / 1000, 1)
            if km.is_integer():
                return f"约{int(km)}公里"
            else:
                return f"约{km}公里"
        else:
            rounded = round(exact_distance_m / 10) * 10
            if rounded == 0:
                rounded = 10
            return f"约{rounded}米"

    @classmethod
    def _generate_vague(cls, dist_level: str) -> str:
        """模糊粒度：从对应等级词袋随机抽取"""
        pool = cls.VAGUE_POOLS.get(dist_level, cls.VAGUE_POOLS['Close'])
        return random.choice(pool)

    @classmethod
    def _generate_ultra_vague(cls, dist_level: str) -> str:
        """极模糊粒度：定性词 + 50%概率加语气词"""
        pool = cls.ULTRA_VAGUE_POOLS.get(dist_level, cls.ULTRA_VAGUE_POOLS['Close'])
        base = random.choice(pool)
        if random.random() < 0.5:
            modal = random.choice(cls.MODAL_PARTICLES)
            return f"{modal}{base}"
        return base

    @classmethod
    def _generate_functional(cls, exact_distance_m: float, dist_level: str) -> str:
        """
        功能化粒度（时间换算）：
        - 步行速度 80m/min，驾车速度 400m/min
        - >3000m 强制驾车，否则随机
        - <1分钟输出“不到1分钟”，>60分钟回退至极模糊
        """
        if exact_distance_m == 0:
            return '就在此处'

        walk_min = exact_distance_m / 80.0
        drive_min = exact_distance_m / 400.0

        if exact_distance_m > 3000:
            mode = 'drive'
        else:
            mode = random.choice(['walk', 'drive'])

        if mode == 'walk':
            minutes = walk_min
            mode_word = '步行'
        else:
            minutes = drive_min
            mode_word = '开车'

        if minutes < 1:
            return f"{mode_word}不到1分钟"

        rounded_min = round(minutes)
        if rounded_min > 60:
            # 回退至极模糊
            return cls._generate_ultra_vague(dist_level)

        approx = random.choice(['约', '大概'])
        return f"{mode_word}{approx}{rounded_min}分钟"