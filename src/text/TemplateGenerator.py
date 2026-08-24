import os
import re
import yaml
import random
import logging
from typing import List, Dict, Optional
from openai import OpenAI  # 需安装 openai 库
from dotenv import load_dotenv
import shutil

# 在程序最开头强制加载 .env 文件
load_dotenv()
# ---------- 配置 ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 大模型 API 配置（使用环境变量）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")  # 可自定义
MODEL_NAME = "gpt-4.1"  # 或 "gpt-3.5-turbo"

# 文件路径
TEMPLATE_LIBRARY_PATH = "../../config/template_library.yaml"
TEMPLATE_LIBRARY_NEW_PATH = "../../config/template_library_new.yaml"


# 质量检查阈值
MIN_TEMPLATES_PER_FAMILY = 15  # 每个族至少保留模板数
SIMILARITY_THRESHOLD = 0.85  # 相似度阈值（未实现，保留接口）


# ---------- 工具函数 ----------
def load_template_library(yaml_path: str) -> Dict:
    """加载 YAML 模板库"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_template_library(yaml_path: str, library: Dict) -> None:
    """保存 YAML 模板库（保留格式）"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(library, f, allow_unicode=True, sort_keys=False, indent=2)


def extract_placeholders(template: str) -> List[str]:
    """提取模板中的所有占位符（如 {landmark}）"""
    return re.findall(r'\{(\w+)\}', template)


def is_hardcoded_direction(text: str) -> bool:
    """检测文本中是否包含硬编码的方位词（排除占位符内）"""
    # 先移除占位符内容，再检测
    cleaned = re.sub(r'\{[^}]*\}', '', text)
    # 方位词列表（含“左、右”等）
    direction_chars = ['东', '南', '西', '北', '左', '右', '前', '后', '上', '下']
    for ch in direction_chars:
        if ch in cleaned:
            return True
    return False


# ---------- 核心泛化类 ----------
class TemplateGeneralizer:
    def __init__(self, api_key: str, base_url: str = None, model: str = MODEL_NAME):
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model

    def generate_variants(self, seed_templates: List[str], family_key: str, required_slots: List[str]) -> List[str]:
        """
        调用大模型生成泛化变体
        :param seed_templates: 3条种子模板
        :param family_key: 逻辑族键（如 'SCENE_C'）
        :param required_slots: 必需占位符列表
        :return: 生成的模板列表（可能包含不合格项）
        """
        # 构建 prompt
        prompt = self._build_prompt(seed_templates, family_key, required_slots)

        # 调用 API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位精通中文地理描述的自然语言生成专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.85,  # 鼓励多样性
                max_tokens=1500,
                n=1
            )
            raw_output = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API 调用失败: {e}")
            return []

        # 解析输出，提取模板行（假设每行一个模板，或以编号开头）
        lines = raw_output.split('\n')
        generated = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 去除可能的编号前缀（如 "1. " 或 "- "）
            cleaned = re.sub(r'^(\d+\.\s*|-\s*)', '', line)
            # 检查是否包含占位符（简单过滤）
            if '{' in cleaned and '}' in cleaned:
                generated.append(cleaned)
        return generated

    def _build_prompt(self, seeds: List[str], family_key: str, required_slots: List[str]) -> str:
        """构建详细 Prompt"""
        # 构建占位符保护指令
        placeholders_str = ', '.join([f'{{{s}}}' for s in required_slots])

        prompt = f"""【任务】
                    你是地理语言学专家，请根据以下 {len(seeds)} 条种子模板，为逻辑族 "{family_key}"的每条种子分别生成 6-8 条句式各异但空间逻辑完全等价的泛化模板。
                    
                    【核心约束】
                    1. 必须保留所有占位符（{placeholders_str}），严禁替换、删除或硬编码具体方位、距离数值。
                    2. 生成的模板必须围绕参照点（{{landmark}}）展开，不得缺失参照点信息。
                    3. 语感必须符合中文口语指路习惯，避免欧化长句或生硬翻译腔。
                    
                    【多样性要求】
                    请从以下三个维度对每条种子进行变换：
                    - 视角切换：如“目标在参照点方位” ↔ “往参照点方位走就是目标”
                    - 语态/句式变换：陈述句 ↔ 把字句 ↔ 存现句（有…） ↔ 倒装句 ↔ 疑问句（你知道吗？）
                    - 口语化注入：适当加入“大概”、“估摸着”、“其实”、“顺着”等口语词，但不要过度。
                    
                    【种子模板】
                    {chr(10).join(f'{i + 1}. {s}' for i, s in enumerate(seeds))}
                    
                    【输出格式】
                    请直接返回生成的模板，每行一条，不要编号，不要额外解释。
                    """
        return prompt

    def quality_check(self, templates: List[str], family_config: Dict, family_key: str) -> List[str]:
        """
        质量过滤器（新版宽松逻辑）：
        1. 白名单校验：模板中所有占位符必须在 valid_slots 中。
        2. 核心强制校验：必须包含 core_required 中的全部占位符（默认为 ['landmark']）。
        """
        # 从配置中读取白名单和核心强制集
        valid_slots = set(family_config.get('valid_slots', []))
        core_required = set(family_config.get('core_required', ['landmark']))

        # 如果白名单为空（极端情况），放行所有占位符（但通常不会）
        if not valid_slots:
            logger.warning(f"逻辑族 {family_key} 未定义 valid_slots，跳空白名单检查")
            valid_slots = None

        valid_templates = []
        for tmpl in templates:
            # 提取占位符
            placeholders = set(extract_placeholders(tmpl))

            # 检查 1：核心强制项是否全部存在
            if not core_required.issubset(placeholders):
                missing = core_required - placeholders
                logger.debug(f"丢弃模板（缺失核心占位符 {missing}）: {tmpl}")
                continue

            # 检查 2：白名单校验（如果定义了白名单）
            if valid_slots is not None:
                if not placeholders.issubset(valid_slots):
                    invalid = placeholders - valid_slots
                    logger.debug(f"丢弃模板（含未定义占位符 {invalid}）: {tmpl}")
                    continue

            # 检查 3：硬编码方位词检测（保持不变）
            if is_hardcoded_direction(tmpl):
                logger.debug(f"丢弃模板（含硬编码方位词）: {tmpl}")
                continue

            # 检查 4：SCENE_B 特殊建议（但不强制，仅记录警告）
            if family_key.startswith('SCENE_B') and '{road}' not in placeholders:
                logger.debug(f"警告：SCENE_B 模板未包含 {{road}}，但已放行: {tmpl}")
                # 注意：这里只是警告，不再丢弃

            valid_templates.append(tmpl)

        return valid_templates

    def generalize_family(self, library: Dict, family_key: str, seed_indices: List[int] = None) -> Dict:
        """
        对单个逻辑族执行泛化流程
        :param library: 整个模板库字典
        :param family_key: 如 'SCENE_C'
        :param seed_indices: 指定种子模板索引（缺省则自动选择前3条）
        :return: 更新后的 library（未保存）
        """
        family = library['families'].get(family_key)
        if not family:
            logger.warning(f"逻辑族 {family_key} 不存在，跳过")
            return library

        existing_templates = family.get('templates', [])
        if len(existing_templates) < 3:
            logger.warning(f"逻辑族 {family_key} 模板数量不足3条，无法泛化")
            return library

        # 选择种子（若指定索引则使用，否则取前3条）
        if seed_indices:
            seeds = [existing_templates[i] for i in seed_indices if i < len(existing_templates)]
        else:
            seeds = existing_templates[:3]

        required_slots = family.get('required_slots', [])
        logger.info(f"开始泛化逻辑族 {family_key}，种子模板: {seeds}")

        # 生成变体
        generated = self.generate_variants(seeds, family_key, required_slots)
        if not generated:
            logger.warning(f"逻辑族 {family_key} 泛化未生成任何结果")
            return library

        # 质量过滤
        valid_generated = self.quality_check(generated, family, family_key)
        logger.info(f"逻辑族 {family_key} 生成 {len(generated)} 条，通过过滤 {len(valid_generated)} 条")

        if not valid_generated:
            logger.warning(f"逻辑族 {family_key} 没有通过质量检查的变体")
            return library

        # 人工抽检提示（实际项目中可将不合格项记录，并等待人工确认）
        # 此处模拟：记录日志，并提示
        logger.info(f"请人工抽检以下 {len(valid_generated)} 条生成模板（建议抽检 10%~15%）")
        for i, tmpl in enumerate(valid_generated[:3]):  # 仅打印前3条示例
            logger.info(f"  示例 {i + 1}: {tmpl}")

        # 合并去重（避免添加已存在的相同模板）
        existing_set = set(existing_templates)
        new_templates = [t for t in valid_generated if t not in existing_set]
        if new_templates:
            family['templates'].extend(new_templates)
            logger.info(f"逻辑族 {family_key} 新增 {len(new_templates)} 条泛化模板")
        else:
            logger.info(f"逻辑族 {family_key} 无新增模板（均已存在）")

        return library


# ---------- 主执行流程 ----------
def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("错误：未找到 OPENAI_API_KEY，请检查 .env 文件或系统环境变量！")
        exit(1)
    # 1. 加载模板库
    library = load_template_library(TEMPLATE_LIBRARY_PATH)

    # 2. 初始化泛化器
    generalizer = TemplateGeneralizer(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    # 3. 定义映射对象：每个逻辑族对应不同的种子索引列表
    #    格式：{ 'family_key': [index1, index2, ...] }
    # family_seed_map = {
    #     'SCENE-B_Opposite': [0, 2, 4, 5],
    #     'SCENE-B_SameSide': [0, 2, 3, 5],
    #     'SCENE-B_Intersection': [0, 1, 2, 3],
    #     'SCENE-A': [0, 2, 4],
    #     'SCENE-D': [0, 1, 2],
    #     'SCENE-C': [0,1,2,7,9,12],
    #     'FALLBACK_GENERIC': [0, 2, 3],
    #     # 可按需添加更多族及其对应的种子索引
    # }
    family_seed_map = {
        'SCENE-B_Opposite': [0, 1, 4],
        'SCENE-B_SameSide': [0, 2, 4],
        'SCENE-B_Intersection': [0, 1, 2],
        'SCENE-A': [0, 1, 2],
        'SCENE-D': [0, 2, 4],
        'SCENE-C': [0, 1, 3],
        'FALLBACK_GENERIC': [0, 2, 4],
    }

    # 4. 遍历映射，执行泛化
    for family_key, seed_indices in family_seed_map.items():
        if family_key not in library['families']:
            logger.warning(f"族 {family_key} 不存在，跳过")
            continue
        # 调用泛化方法，传入指定的种子索引
        library = generalizer.generalize_family(library, family_key, seed_indices=seed_indices)

    # 如果新文件已存在，先删除
    if os.path.exists(TEMPLATE_LIBRARY_NEW_PATH):
        os.remove(TEMPLATE_LIBRARY_NEW_PATH)
        logger.info(f"已删除旧的新文件 {TEMPLATE_LIBRARY_NEW_PATH}")

    save_template_library(TEMPLATE_LIBRARY_NEW_PATH, library)
    logger.info(f"模板库已更新保存至 {TEMPLATE_LIBRARY_NEW_PATH}")


if __name__ == "__main__":
    main()