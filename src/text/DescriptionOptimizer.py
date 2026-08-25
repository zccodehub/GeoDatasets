import os
import re
import random
import time
import logging
import json
import requests
from typing import List, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# ---------- 配置 ----------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("请设置环境变量 DATABASE_URL")

OLLAMA_URL = os.getenv("OLLAMA_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

BATCH_SIZE = 100                 # 每批处理记录数

# ---------- 日志 ----------
LOG_FILE = "../../logs/refine_geo_desc.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ---------- 数据库连接 ----------
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)


# ---------- 批量调用类方法 ----------
class QwenRefiner:
    def __init__(self, model=LLM_MODEL, url=OLLAMA_URL):
        self.model = model
        self.url = url

    def refine_batch(self, raw_texts: List[str]) -> List[str]:
        """
        批量润色多条文本，返回润色后的列表（顺序与输入一致）
        若解析失败，自动降级为逐条调用。
        """
        if not raw_texts:
            return []
        if len(raw_texts) == 1:
            # 单条直接调用单条方法
            return [self.refine(raw_texts[0])]

        # 构造批量Prompt
        system_prompt = (
            "你是一位地理描述润色专家。请将输入的“机器生成的地理描述”改写为“自然的中文指路口语”。\n"
            "改写铁律（必须严格遵守）：\n"
            "1. 【空间逻辑铁律】严禁改变任何方位词（东/南/西/北）、距离（米/公里）和具体地名（道路名、地标名）。\n"
            "2. 【去冗余】若句子中重复出现同一个地名（如“朝阳路”），第二次出现必须替换为“这条路”、“这儿”、“该处”或“其”。\n"
            "3. 【句式打散】禁止使用“在...的...方向，距离...”这种长定语结构。应拆分为短句，或改为“往...走”、“顺着...”的动作引导句式。\n"
            "4. 【口语化】适当加入“您呐”、“其实吧”、“顺着”、“拐个弯”等口语词，但不要过度。\n"
            "5. 【指代替换】若原文出现了两次相同的地名，第二次替换为“这儿”、“那儿”或“该处”。\n"
            "6. 【节奏控制】将一个长难句拆分为两个短句（逗号或句号分隔），增加停顿感。\n"
            "7. 【语句通顺】去掉部分重复意思的词，使整个语句保持通顺、无语病。\n"
            "8. 你必须以JSON数组格式返回润色后的文本列表，每个元素对应输入列表的一条文本。只输出JSON数组，不要有其他解释。\n"
            "待改写的文本列表（JSON数组格式）：\n"
            f"{json.dumps(raw_texts, ensure_ascii=False)}"
        )

        payload = {
            "model": self.model,
            "prompt": system_prompt,
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 800 + 50 * len(raw_texts),  # 根据文本长度动态调整
            "timeout": 120
        }

        try:
            response = requests.post(self.url, json=payload, timeout=240)
            if response.status_code != 200:
                logging.warning(f"批量LLM请求失败，状态码 {response.status_code}，降级为逐条处理")
                return [self.refine(t) for t in raw_texts]

            result = response.json()
            raw_output = result.get("response", "").strip()
            # 尝试解析JSON
            parsed = self._parse_json_array(raw_output)
            if parsed is not None and len(parsed) == len(raw_texts):
                return parsed
            else:
                logging.warning(f"批量解析结果数量不匹配，预期{len(raw_texts)}，实际{len(parsed) if parsed else 0}，降级为逐条处理")
                return [self.refine(t) for t in raw_texts]

        except Exception as e:
            logging.error(f"批量调用异常: {e}，降级为逐条处理")
            return [self.refine(t) for t in raw_texts]

    @staticmethod
    def _parse_json_array(text: str):
        """
        从可能包含Markdown或多余字符的文本中提取JSON数组
        """
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 移除Markdown代码块标记
        cleaned = re.sub(r'```json\s*', '', text)
        cleaned = re.sub(r'```\s*', '', cleaned)
        # 查找第一个'['和最后一个']'之间的内容
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        if start != -1 and end != -1 and start < end:
            json_str = cleaned[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # 尝试按行解析，但更稳健的方式是使用正则找出所有引号字符串
        # 这里仅作保底，尝试提取所有双引号片段（但不推荐）
        # 若以上均失败，返回None
        return None

    def refine(self, raw_text: str) -> str:
        """
        单条润色，支持解析 JSON 数组格式（兼容模型返回 ["..."] 的情况）
        """
        system_prompt = (
            "你是一位地理描述润色专家。请将输入的“机器生成的地理描述”改写为“自然的中文指路口语”。\n"
            "改写铁律（必须严格遵守）：\n"
            "1. 【空间逻辑铁律】严禁改变任何方位词（东/南/西/北）、距离（米/公里）和具体地名（道路名、地标名）。\n"
            "2. 【去冗余】若句子中重复出现同一个地名（如“朝阳路”），第二次出现必须替换为“这条路”、“这儿”、“该处”或“其”。\n"
            "3. 【句式打散】禁止使用“在...的...方向，距离...”这种长定语结构。应拆分为短句，或改为“往...走”、“顺着...”的动作引导句式。\n"
            "4. 【口语化】适当加入“您呐”、“其实吧”、“顺着”、“拐个弯”等口语词，但不要过度。\n"
            "5. 【指代替换】若原文出现了两次相同的地名，第二次替换为“这儿”、“那儿”或“该处”。\n"
            "6. 【节奏控制】将一个长难句拆分为两个短句（逗号或句号分隔），增加停顿感。\n"
            "7. 【语句通顺】去掉部分重复意思的词，使整个语句保持通顺、无语病。\n"
            "8. 你必须以JSON数组格式返回润色后的文本列表，每个元素对应输入列表的一条文本。只输出JSON数组，不要有其他解释。\n"
        )
        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n待改写的文本：{raw_text}",
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 200,
            "timeout": 60
        }
        try:
            response = requests.post(self.url, json=payload, timeout=70)
            if response.status_code == 200:
                result = response.json()
                raw_output = result.get("response", "").strip()
                # 尝试解析 JSON
                parsed = self._parse_json_array(raw_output)
                if parsed is not None and isinstance(parsed, list) and len(parsed) > 0:
                    return parsed[0]  # 取第一个元素
                else:
                    # 如果解析失败，直接返回原输出（可能是纯文本）
                    return raw_output
            else:
                logger.warning(f"LLM API 返回非200: {response.status_code}")
                return raw_text
        except Exception as e:
            logger.error(f"LLM 调用异常: {e}")
            return raw_text

# ---------- 前置清洗函数 ----------
def preprocess_remove_redundancy(text: str, road_name: str = None) -> str:
    """
    消除专有名词重复，修复常见标点和冗余介词。
    """
    if not text:
        return text
    # 1. 如果提供了道路名，将第二次及以后出现的道路名替换为“马路”或“该路”
    if road_name and len(road_name) > 1:
        parts = text.split(road_name)
        if len(parts) > 2:
            new_parts = [parts[0]]
            for i in range(1, len(parts)):
                if i == len(parts) - 1:
                    new_parts.append(parts[i])
                else:
                    new_parts.append("马路" + parts[i])
            text = road_name.join(new_parts)
    # 2. 修复“在...在...”等介词重复（简单正则）
    text = re.sub(r'在(.*?)在', r'在\1，', text)
    # 3. 去除多余的空格和标点
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- 逻辑后验校验 ----------
def validate_spatial_consistency(original: str, refined: str, expected_direction: str) -> bool:
    """
    简单校验：检查改写后的文本是否保留了原始文本中的核心方位词根。
    若原始文本有明确方位，改写后不能完全消失（除非被功能化“XX方向”替代）。
    """
    if not expected_direction:
        return True
    dir_keywords = ['东', '南', '西', '北']
    original_dirs = [d for d in dir_keywords if d in original]
    refined_dirs = [d for d in dir_keywords if d in refined]
    if original_dirs and not refined_dirs:
        if '方向' not in refined:
            return False
    # 期望方向检查
    dir_map = {'North':'北','South':'南','East':'东','West':'西'}
    expected_cn = dir_map.get(expected_direction, '')
    if expected_cn and expected_cn not in refined:
        if '方向' not in refined and '那边' not in refined:
            return False
    return True

def process_batch(records: List[Dict[str, Any]], refiner: QwenRefiner) -> List[Dict[str, Any]]:
    results = []
    cleaned_list = []
    need_llm_indices = []

    for idx, rec in enumerate(records):
        raw = rec['description_raw']
        if not raw:
            continue
        road_name = rec.get('road_name', '')
        cleaned = preprocess_remove_redundancy(raw, road_name)
        cleaned_list.append(cleaned)

        # ---------- 优化后的触发条件 ----------
        need = False
        key = rec.get('template_match_key', '')
        amb = rec.get('ambiguity_level', 'EXACT')
        dist = rec.get('dist_level', '')
        # 获取精确距离（如果可用），用于距离阈值判断
        exact_dist = rec.get('exact_distance_m', 0)  # 需在查询中增加该字段

        # 1. 拓扑复杂：路口（Intersection）始终触发
        if 'Intersection' in key:
            need = True

        # 2. 方位模糊：仅 MODERATE 触发，SLIGHT 不触发（因为轻微偏移可由规则处理）
        elif amb == 'MODERATE':
            need = True

        # 3. 远距离：仅当距离 > 800 米时触发（避免 500~800 米的中等偏远也触发）
        elif dist == 'Far' and exact_dist > 800:
            need = True

        # 4. 其他场景：若文本长度超过 40 字且包含复杂结构（如多个逗号/分句），触发
        else:
            # 统计逗号、句号、分号数量，大于2个分句且长度>40
            if len(cleaned) > 40 and cleaned.count('，') + cleaned.count('。') + cleaned.count('；') >= 2:
                need = True

        # 不再保留随机触发，以降低总调用量
        need_llm_indices.append(need)

    # 将需要LLM的文本按20~30条分批
    batch_size_llm = 20  # 可根据模型性能调整
    llm_indices = [i for i, flag in enumerate(need_llm_indices) if flag]
    llm_results = [None] * len(records)  # 预填充

    for start in range(0, len(llm_indices), batch_size_llm):
        batch_indices = llm_indices[start:start+batch_size_llm]
        batch_texts = [cleaned_list[i] for i in batch_indices]
        logging.info(f"批量调用LLM，数量: {len(batch_texts)}")
        refined_batch = refiner.refine_batch(batch_texts)
        # 将结果填入对应位置
        for idx_in_batch, orig_idx in enumerate(batch_indices):
            if idx_in_batch < len(refined_batch):
                refined_text = refined_batch[idx_in_batch]
                # 后验校验
                direction_8 = records[orig_idx].get('direction_8', '')
                if validate_spatial_consistency(cleaned_list[orig_idx], refined_text, direction_8):
                    llm_results[orig_idx] = refined_text
                    logging.warning(f"记录 {records[orig_idx]['id']} LLM润色后校验成功")
                else:
                    logging.warning(f"记录 {records[orig_idx]['id']} LLM润色后方位校验失败，回退至清洗文本")
                    llm_results[orig_idx] = cleaned_list[orig_idx]
            else:
                # 若返回数量不足，回退
                llm_results[orig_idx] = cleaned_list[orig_idx]

    # 构建最终结果
    for i, rec in enumerate(records):
        if need_llm_indices[i]:
            final_text = llm_results[i] if llm_results[i] is not None else cleaned_list[i]
        else:
            final_text = cleaned_list[i]
        results.append({
            'id': rec['id'],
            'description': final_text
        })
    return results

# ---------- 主程序 ----------
def main():
    refiner = QwenRefiner()

    # 确保 geo_desc 表有 description 列
    with engine.connect() as conn:
        conn.execute(text("""
            DO $$ 
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                               WHERE table_name='geo_desc' AND column_name='description') THEN
                    ALTER TABLE geo_desc ADD COLUMN description TEXT;
                END IF;
            END $$;
        """))
        conn.commit()

    # 获取待处理记录的最小ID（作为起始游标）
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT MIN(id) FROM geo_desc 
            WHERE description IS NULL OR description = ''
        """))
        last_id = result.scalar()
        if last_id is None:
            logger.info("没有待优化的记录。")
            return

    total = 0
    # 先统计总数用于进度显示
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(id) FROM geo_desc 
            WHERE description IS NULL OR description = ''
        """))
        total = result.scalar()
    logger.info(f"待优化记录总数: {total}")

    processed = 0
    while True:
        # 查询ID大于 last_id 且 description 为空的记录，按ID排序
        query = text("""
            SELECT 
                g.id,
                g.description_raw,
                g.template_match_key,
                s.road_name,
                s.direction_8,
                s.ambiguity_level,
                s.dist_level,
                s.exact_distance_m
            FROM geo_desc g
            LEFT JOIN spatial_relations s ON g.fid = s.id
            WHERE g.id >= :last_id 
              AND (g.description IS NULL OR g.description = '')
            ORDER BY g.id
            LIMIT :limit
        """)
        with engine.connect() as conn:
            rows = conn.execute(query, {'last_id': last_id, 'limit': BATCH_SIZE})
            records = [dict(row._mapping) for row in rows]

        if not records:
            break

        ids = [r['id'] for r in records]
        logger.info(f"正在处理记录 ID: {min(ids)} - {max(ids)}（共 {len(records)} 条）")
        updates = process_batch(records, refiner)

        # 批量更新 geo_desc.description
        if updates:
            with engine.connect() as conn:
                for item in updates:
                    conn.execute(
                        text("UPDATE geo_desc SET description = :desc WHERE id = :id"),
                        {'desc': item['description'], 'id': item['id']}
                    )
                conn.commit()

        processed += len(records)
        # 更新 last_id 为本批最大ID
        last_id = max(ids)
        logger.info(f"已处理 {processed}/{total} 条记录，当前游标ID: {last_id}")

    logger.info("所有记录优化完成！")

if __name__ == "__main__":
    main()