import random
import yaml
from typing import List, Dict, Any
# 假设以下类已定义在独立模块中
from DirectionGenerator import DirectionGenerator
from DistanceGenerator import DistanceGenerator
from AmbiguityInjector import AmbiguityInjector


class SlotFiller:
    """
    多样化槽位填充引擎，基于预计算的偏差角和模糊等级
    """

    def __init__(self, template_library_path: str):
        with open(template_library_path, 'r', encoding='utf-8') as f:
            self.library = yaml.safe_load(f)
        self.families = self.library.get('families', {})

    def get_templates_for_record(self, template_match_key: str) -> List[str]:
        """
        根据 template_match_key 获取模板列表（支持层级回退）
        template_match_key 格式示例：'SCENE-B_Opposite_Close_East_EXACT'
        回退顺序：精确键 → 场景+拓扑 → 仅场景 → 保底
        """
        # 1. 尝试完全匹配
        if template_match_key in self.families:
            return self.families[template_match_key].get('templates', [])

        parts = template_match_key.split('_')
        # 2. 尝试场景+拓扑（前两段）
        if len(parts) >= 2:
            family_key = f"{parts[0]}_{parts[1]}"
            if family_key in self.families:
                return self.families[family_key].get('templates', [])

        # 3. 尝试仅场景（第一段）
        if parts:
            scene_key = parts[0]
            if scene_key in self.families:
                return self.families[scene_key].get('templates', [])

        # 4. 保底
        fallback = self.families.get('FALLBACK_GENERIC', {})
        return fallback.get('templates', [])

    def generate_descriptions(self, record: Dict[str, Any], num_variants: int = 5) -> List[str]:
        """
        为单条记录生成多条描述

        record 必须包含字段：
            - template_match_key: 如 'SCENE-B_Opposite_Close_East_EXACT'
            - landmark_name, target_name, target_type（可选）
            - road_name（可选）
            - direction_8, deviation_angle, ambiguity_level
            - exact_distance_m, dist_level
        """
        templates = self.get_templates_for_record(record['template_match_key'])
        if not templates:
            return [f"{record.get('target_name', '目标')}位于{record.get('landmark_name', '参照点')}附近。"]

        # 获取方向粒度权重（由 AmbiguityInjector 根据 ambiguity_level 调整）
        dir_weights = AmbiguityInjector.adjust_granularity_weights(record['ambiguity_level'])
        # 距离粒度权重（直接从 DistanceGenerator 类属性获取）
        dist_weights = DistanceGenerator.GRANULARITY_WEIGHTS

        results = []
        for _ in range(num_variants):
            # 独立选择方向粒度和距离粒度
            dir_granularity = self._weighted_choice(dir_weights)
            dist_granularity = self._weighted_choice(dist_weights)

            # 生成方向文本
            dir_text = DirectionGenerator.generate(
                direction_8=record['direction_8'],
                deviation_angle=record['deviation_angle'],
                ambiguity_level=record['ambiguity_level'],
                road_name=record.get('road_name'),
                landmark_name=record.get('landmark_name'),
                force_granularity=dir_granularity
            )

            # 生成距离文本
            dist_text = DistanceGenerator.generate(
                exact_distance_m=record['exact_distance_m'],
                dist_level=record['dist_level'],
                force_granularity=dist_granularity
            )

            # 随机选一条模板
            template = random.choice(templates)

            # 填充槽位
            fill_values = {
                'landmark': record['landmark_name'],
                'target': record.get('target_name', '目标'),
                'target_type': record.get('target_type', '地点'),
                'road': record.get('road_name', '路'),
                'dir_text': dir_text,
                'dist_text': dist_text
            }

            # 处理模板中可能出现的额外槽位（如未提供则填空字符串）
            try:
                description = template.format(**fill_values)
            except KeyError as e:
                missing = str(e).strip("'")
                fill_values[missing] = ''
                description = template.format(**fill_values)

            results.append(description)

        return results

    @staticmethod
    def _weighted_choice(weights: Dict[str, float]) -> str:
        """根据权重字典随机选择键"""
        total = sum(weights.values())
        rnd = random.random() * total
        cum = 0.0
        for key, w in weights.items():
            cum += w
            if rnd <= cum:
                return key
        return list(weights.keys())[0]