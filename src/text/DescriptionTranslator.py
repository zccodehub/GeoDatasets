import os
import re
import time
import json
import logging
import requests
from typing import List, Dict, Any
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ---------- 配置 ----------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("请设置环境变量 DATABASE_URL")

# Ollama 配置（可从环境变量读取，也可直接指定）
OLLAMA_URL = os.getenv("OLLAMA_URL")
LLM_EN_MODEL = os.getenv("LLM_MODEL")  # 替换为你实际的模型名

# 日志配置
LOG_FILE = "../../logs/translate_geo_desc.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 20              # 从数据库读取和翻译的批次大小

# ---------- 数据库连接 ----------
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ---------- 翻译器类 ----------
class Translator:
    @staticmethod
    def contains_chinese(text: str) -> bool:
        """检查文本是否包含中文字符或中文标点"""
        if text is None:
            return False
        # 检测Unicode中文范围：基本汉字、扩展汉字
        for char in text:
            # 基本汉字范围
            if '\u4e00' <= char <= '\u9fff':
                return True
            # 中文标点符号范围（部分）
            if '\u3000' <= char <= '\u303f':
                return True
            # 全角标点范围
            if '\uff00' <= char <= '\uffef':
                # 排除英文字母和数字的全角形式
                if not ('\uff21' <= char <= '\uff3a' or '\uff41' <= char <= '\uff5a' or '\uff10' <= char <= '\uff19'):
                    return True
        return False

    def __init__(self, model=LLM_EN_MODEL, url=OLLAMA_URL):
        self.model = model
        self.url = url

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        批量翻译多条文本，返回英文翻译列表（顺序一致）
        若解析失败，降级为逐条翻译。
        """
        if not texts:
            return []
        if len(texts) == 1:
            return [self.translate_single(texts[0])]

        # 构造批量翻译 Prompt
        system_prompt = (
            "你是一个专业的中英翻译引擎。请将以下中文文本逐条翻译为英文。\n\n"
            "输入是一个JSON数组格式的中文文本列表。\n"
            "要求：\n"
            "1. 只输出英文翻译，不添加任何解释、前缀或后缀。\n"
            "2. 输出格式：必须返回一个JSON数组，格式与输入完全相同，只是内容替换为英文翻译。\n"
            "3. 只需要输出JSON数组，不要有其他任何内容。\n\n"
            "输入JSON数组：\n"
            f"{json.dumps(texts, ensure_ascii=False)}"
        )

        payload = {
            "model": self.model,
            "prompt": system_prompt,
            "stream": False,
            "temperature": 0.1,          # 翻译任务用低温度，保证稳定性
            "top_p": 0.9,
            "max_tokens": 800 + 50 * len(texts),
            "timeout": 120
        }

        try:
            response = requests.post(self.url, json=payload, timeout=240)
            response.raise_for_status()  # 检查HTTP错误
            
            try:
                result = response.json()
            except Exception as json_error:
                logger.error(f"批量翻译响应不是有效的JSON: {json_error}")
                logger.debug(f"原始响应文本: {response.text[:200]}...")
                return [self.translate_single(t) for t in texts]
            
            raw_output = result.get("response", "").strip()
            logger.debug(f"批量翻译原始输出: {raw_output[:200]}...")
            parsed = self._parse_json_array(raw_output)
            if parsed is not None and len(parsed) == len(texts):
                # 检查翻译结果是否包含中文
                checked_translations = []
                for i, translation in enumerate(parsed):
                    if translation is not None and self.contains_chinese(translation):
                        logger.warning(f"批量翻译第 {i+1} 条包含中文: '{translation[:50]}...'，设置为NULL")
                        checked_translations.append(None)
                    else:
                        checked_translations.append(translation)
                return checked_translations
            else:
                 logger.warning(f"批量翻译解析结果数量不匹配，预期{len(texts)}，实际{len(parsed) if parsed else 0}")
                 logger.debug(f"解析失败的原始输出: {raw_output[:500]}...")
                 return [self.translate_single(t) for t in texts]

        except Exception as e:
            logger.error(f"批量翻译异常: {e}，降级为逐条翻译")
            return [self.translate_single(t) for t in texts]

    def translate_single(self, text: str) -> str:
        """单条翻译（降级方案）"""
        prompt = (
            f"你是一个专业的中英翻译引擎。请将以下中文文本翻译为英文：\n{text}\n"
            "要求：\n"
            "1. 只输出英文翻译，不添加任何解释、前缀或后缀。\n"
            "2. 输出格式：返回一个JSON数组，例如：[\"英文翻译\"]\n"
            "3. 只需要输出JSON数组，不要有其他任何内容。\n\n"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 500,
            "timeout": 60
        }
        try:
            response = requests.post(self.url, json=payload, timeout=70)
            response.raise_for_status()
            
            try:
                result = response.json()
            except Exception as json_error:
                logger.error(f"单条翻译响应不是有效的JSON: {json_error}")
                logger.debug(f"原始响应文本: {response.text[:200]}...")
                return None
            
            raw = result.get("response", "").strip()
            logger.debug(f"单条翻译原始输出: {raw[:200]}...")
            # 尝试解析 JSON（预防模型仍返回 ["..."] 格式）
            parsed = self._parse_json_array(raw)
            if parsed is not None and isinstance(parsed, list) and len(parsed) > 0:
                translation = parsed[0]
                if translation is not None and self.contains_chinese(translation):
                    logger.warning(f"单条翻译包含中文: '{translation[:50]}...'，设置为NULL")
                    return None
                return translation
            return None  # 返回 None 而不是原文
        except Exception as e:
            logger.error(f"单条翻译异常: {e}")
            return None  # 返回 None 而不是原文

    @staticmethod
    def _parse_json_array(text: str):
        """从可能包含 Markdown 或多余字符的文本中提取 JSON 数组"""
        if not text or text.strip() == '':
            logger.warning(f"_parse_json_array: 输入文本为空")
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 去除 Markdown 代码块
        cleaned = re.sub(r'```json\s*', '', text)
        cleaned = re.sub(r'```\s*', '', cleaned)
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        if start != -1 and end != -1 and start < end:
            json_str = cleaned[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 可能不是JSON格式，尝试其他格式
                pass
        
        # 尝试处理纯文本列表格式
        # 可能是每行一个翻译
        lines = text.strip().split('\n')
        if lines and len(lines) > 0:
            # 清理每行内容
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                # 移除列表标记如"1. ", "2. ", "- ", "* "等
                line = re.sub(r'^\d+\.\s+', '', line)
                line = re.sub(r'^[\-\*]\s+', '', line)
                line = re.sub(r'^["\'](.*)["\']$', r'\1', line)
                cleaned_lines.append(line)
            logger.debug(f"_parse_json_array: 处理为非JSON列表，找到 {len(cleaned_lines)} 行")
            return cleaned_lines
        return None
# ---------- 主处理流程 ----------
def main():
    translator = Translator()
    
    # 初始化统计计数器
    stats = {
        'total_processed': 0,
        'chinese_detected': 0,
        'translation_failed': 0,
        'null_filled': 0
    }

    # 1. 确保目标列存在
    with engine.connect() as conn:
        conn.execute(text("""
            DO $$ 
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                               WHERE table_name='geo_desc' AND column_name='description_en') THEN
                    ALTER TABLE geo_desc ADD COLUMN description_en TEXT;
                END IF;
            END $$;
        """))
        conn.commit()
        logger.info("确保 description_en 列存在")

    # 2. 统计待翻译记录数（description 不为空且 description_en 为空）
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(id) 
            FROM geo_desc 
            WHERE description IS NOT NULL 
              AND description != '' 
              AND (description_en IS NULL OR description_en = '')
        """))
        total = result.scalar()
    logger.info(f"待翻译记录总数: {total}")
    if total == 0:
        logger.info("所有记录已翻译，无需处理。")
        return

    # 3. 分批处理
    offset = 0
    processed = 0
    while offset < total:
        # 读取一批未翻译的记录
        query = text("""
            SELECT id, description
            FROM geo_desc
            WHERE description IS NOT NULL 
              AND description != '' 
              AND (description_en IS NULL OR description_en = '')
            ORDER BY id
            LIMIT :limit OFFSET :offset
        """)
        with engine.connect() as conn:
            rows = conn.execute(query, {'limit': BATCH_SIZE, 'offset': offset})
            records = [dict(row._mapping) for row in rows]

        if not records:
            break

        logger.info(f"正在处理第 {offset+1} - {offset+len(records)} 条")

        # 提取待翻译文本，过滤空文本
        texts = [rec['description'] for rec in records]
        if not texts:
            offset += BATCH_SIZE
            continue

        # 批量翻译（分批调用，避免单次请求过大）
        all_translations = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            logger.info(f"  翻译批次 {i//BATCH_SIZE + 1}/{ (len(texts)-1)//BATCH_SIZE + 1 }，条数 {len(batch_texts)}")
            translated = translator.translate_batch(batch_texts)
            all_translations.extend(translated)
            # 适当延迟，避免模型过载
            time.sleep(0.1)

    # 确保翻译数量与记录数量一致（若不一致，用NULL填充缺失）
        if len(all_translations) < len(records):
            logger.warning(f"翻译结果数量 ({len(all_translations)}) 少于记录数 ({len(records)})，用NULL填充")
            stats['null_filled'] += (len(records) - len(all_translations))
            all_translations.extend([None] * (len(records) - len(all_translations)))
        elif len(all_translations) > len(records):
            logger.warning(f"翻译结果数量 ({len(all_translations)}) 多于记录数 ({len(records)})，截断多余的部分")
            all_translations = all_translations[:len(records)]

        # 批量更新
        chinese_count = 0
        with engine.connect() as conn:
            for rec, trans in zip(records, all_translations):
                # 检查翻译是否包含中文（双重检查）
                if trans is not None and translator.contains_chinese(trans):
                    chinese_count += 1
                    trans = None

                conn.execute(
                    text("UPDATE geo_desc SET description_en = :en WHERE id = :id"),
                    {'en': trans, 'id': rec['id']}
                )
            conn.commit()

        if chinese_count > 0:
            stats['chinese_detected'] += chinese_count
            logger.info(f"  本批次检测到 {chinese_count} 条翻译包含中文，已设置为NULL")

        processed += len(records)
        stats['total_processed'] += len(records)
        offset += BATCH_SIZE
        logger.info(f"已处理 {processed}/{total} 条记录")

    logger.info("所有记录翻译完成！")
    logger.info(f"统计汇总:")
    logger.info(f"  总处理记录数: {stats['total_processed']}")
    logger.info(f"  检测到中文的翻译数: {stats['chinese_detected']}")
    logger.info(f"  用NULL填充的缺失翻译数: {stats['null_filled']}")

if __name__ == "__main__":
    main()
