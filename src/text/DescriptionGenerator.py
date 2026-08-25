import os
import sys
import logging
import psycopg2
from openpyxl.styles.builtins import total
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
from SlotFiller import SlotFiller


load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
TEMPLATE_YAML_PATH = "../../config/template_library_new.yaml"
BATCH_SIZE = 200
VARIANTS_PER_RECORD = 1
OUTPUT_TABLE = "geo_desc"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_batch(engine, offset: int, limit: int):
    """使用 SQLAlchemy 引擎分批读取记录"""
    query = """
        SELECT 
            id AS fid,
            template_match_key,
            landmark_name,
            target_name,
            target_type,
            road_name,
            direction_8,
            deviation_angle,
            ambiguity_level,
            exact_distance_m,
            dist_level,
            target_x,
            target_y
        FROM spatial_relations
        ORDER BY id
        LIMIT %s OFFSET %s
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, con=conn, params=(limit, offset))
    return df.to_dict('records')

def insert_batch(conn, batch_data):
    if not batch_data:
        return
    insert_sql = f"""
        INSERT INTO {OUTPUT_TABLE} 
        (fid, description_raw, template_match_key, ref_name, target_x, target_y)
        VALUES %s
    """
    values = [
        (
            row['fid'],
           row['description_raw'],
           row['template_match_key'],
            row.get('ref_name', ''),
           row.get('target_x', ''),
           row.get('target_y', '')
        )
        for row in batch_data
    ]
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, values, page_size=1000)
    conn.commit()

def main():
    # 初始化引擎和原始连接（用于写入）
    engine = create_engine(DATABASE_URL)
    conn = psycopg2.connect(dsn=DATABASE_URL)

    # 确保输出表存在
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
                id SERIAL PRIMARY KEY,
                fid INTEGER NOT NULL,
                template_match_key TEXT,
                description_raw TEXT NOT NULL,
                description TEXT,
                description_en TEXT,
                ref_name TEXT,
                target_x TEXT,
                target_y TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_out_fid ON {OUTPUT_TABLE}(fid);
        """)
        conn.commit()

    # 获取总记录数
    # total = pd.read_sql("SELECT COUNT(*) FROM spatial_relations", engine).iloc[0, 0]
    total = 100
    filler = SlotFiller(TEMPLATE_YAML_PATH)
    offset = 0
    buffer = []
    processed = 0

    while offset < total:
        records = fetch_batch(engine, offset, BATCH_SIZE)
        if not records:
            break
        for rec in records:
            # 补全默认值
            rec.setdefault('target_type', '地点')
            rec.setdefault('road_name', '')
            descriptions = filler.generate_descriptions(rec, VARIANTS_PER_RECORD)
            for desc in descriptions:
                buffer.append({
                    'fid': rec['fid'],
                    'description_raw': desc,
                    'template_match_key': rec['template_match_key'],
                    'ref_name': rec.get('landmark_name', ''),
                    'target_x': rec['target_x'],
                    'target_y': rec['target_y']
                })
            if len(buffer) >= BATCH_SIZE * VARIANTS_PER_RECORD:
                insert_batch(conn, buffer)
                buffer.clear()
                processed += len(records)
                logger.info(f"已处理 {processed}/{total} 条记录")
        offset += BATCH_SIZE

    # 剩余数据
    if buffer:
        insert_batch(conn, buffer)
    conn.close()
    logger.info("全部处理完成！")

if __name__ == "__main__":
    main()
