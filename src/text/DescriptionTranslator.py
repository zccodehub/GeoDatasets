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
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_EN_MODEL = os.getenv("LLM_EN_MODEL", "ali6parmak/hy-mt1.5:1.8b")  # 替换为你实际的模型名

# 日志配置
LOG_FILE = "../../logs/translate_geo_desc.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 100              # 从数据库读取的记录批次大小（读取多，但翻译分批）
TRANSLATE_BATCH_SIZE = 20     # 每次调用 API 翻译的条数（根据模型性能调整）

# ---------- 数据库连接 ----------
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ---------- 翻译器类 ----------
class Translator:
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
            "你是一个专业的中英翻译引擎。请将以下中文文本逐条翻译为英文。\n"
            "要求：\n"
            "1. 只输出英文翻译，不添加任何解释、前缀或后缀。\n"
            "2. 翻译要准确、自然、地道，保留原文语义和风格。\n"
            "3. 输出格式：必须返回一个 JSON 数组，数组长度与输入列表相同，每个元素为对应序号的英文翻译。\n"
            "4. 只输出 JSON 数组，不要有其他内容。\n\n"
            "待翻译的中文文本列表（JSON 数组格式）：\n"
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
            if response.status_code != 200:
                logger.warning(f"批量翻译请求失败，状态码 {response.status_code}，降级为逐条翻译")
                return [self.translate_single(t) for t in texts]

            result = response.json()
            raw_output = result.get("response", "").strip()
            parsed = self._parse_json_array(raw_output)
            if parsed is not None and len(parsed) == len(texts):
                return parsed
            else:
                logger.warning(f"批量翻译解析结果数量不匹配，预期{len(texts)}，实际{len(parsed) if parsed else 0}，降级为逐条翻译")
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
            "2. 翻译要准确、自然、地道，保留原文语义和风格。\n"
            "3. 输出格式：必须返回一个 JSON 数组，数组长度与输入列表相同，每个元素为对应序号的英文翻译。\n"
            "4. 只输出 JSON 数组，不要有其他内容。\n\n"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 200,
            "timeout": 60
        }
        try:
            response = requests.post(self.url, json=payload, timeout=70)
            if response.status_code == 200:
                result = response.json()
                raw = result.get("response", "").strip()
                # 尝试解析 JSON（预防模型仍返回 ["..."] 格式）
                parsed = self._parse_json_array(raw)
                if parsed is not None and isinstance(parsed, list) and len(parsed) > 0:
                    return parsed[0]
                return raw
            else:
                logger.warning(f"单条翻译返回非200: {response.status_code}")
                return text  # 保底返回原文（但会标记）
        except Exception as e:
            logger.error(f"单条翻译异常: {e}")
            return text

    @staticmethod
    def _parse_json_array(text: str):
        """从可能包含 Markdown 或多余字符的文本中提取 JSON 数组"""
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
                pass
        return None

# ---------- 主处理流程 ----------
def main():
    translator = Translator()

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
        texts = [rec['description'] for rec in records if rec['description']]
        if not texts:
            offset += BATCH_SIZE
            continue

        # 批量翻译（分批调用，避免单次请求过大）
        all_translations = []
        for i in range(0, len(texts), TRANSLATE_BATCH_SIZE):
            batch_texts = texts[i:i+TRANSLATE_BATCH_SIZE]
            logger.info(f"  翻译批次 {i//TRANSLATE_BATCH_SIZE + 1}/{ (len(texts)-1)//TRANSLATE_BATCH_SIZE + 1 }，条数 {len(batch_texts)}")
            translated = translator.translate_batch(batch_texts)
            all_translations.extend(translated)
            # 适当延迟，避免模型过载
            time.sleep(0.1)

        # 确保翻译数量与记录数量一致（若不一致，用原文填充缺失）
        if len(all_translations) < len(records):
            logger.warning(f"翻译结果数量 ({len(all_translations)}) 少于记录数 ({len(records)})，用原文填充")
            all_translations.extend([records[len(all_translations)]['description']] * (len(records) - len(all_translations)))

        # 批量更新
        with engine.connect() as conn:
            for rec, trans in zip(records, all_translations):
                conn.execute(
                    text("UPDATE geo_desc SET description_en = :en WHERE id = :id"),
                    {'en': trans, 'id': rec['id']}
                )
            conn.commit()

        processed += len(records)
        offset += BATCH_SIZE
        logger.info(f"已处理 {processed}/{total} 条记录")

    logger.info("所有记录翻译完成！")

if __name__ == "__main__":
    main()