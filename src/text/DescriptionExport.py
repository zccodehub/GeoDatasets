import os
import csv
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ---------- 配置 ----------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("请设置环境变量 DATABASE_URL")

OUTPUT_CSV = "../../output/geo_desc.csv"
BATCH_SIZE = 10000  # 每批读取记录数（避免内存溢出）

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- 连接数据库 ----------
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def export_to_csv():
    logger.info("开始导出 geo_desc 表到 CSV...")

    # 查询需要导出的字段，并重命名 description -> description_zh
    query = text("""
        SELECT 
            description AS description_zh,
            description_en,
            target_x,
            target_y
        FROM geo_desc
        ORDER BY id
    """)

    # 使用流式查询（yield_per）避免一次性加载所有数据到内存
    with engine.connect() as conn:
        result = conn.execute(query)
        # 获取列名（已重命名）
        columns = result.keys()
        logger.info(f"导出的列: {columns}")

        # 写入 CSV（带 BOM 以便 Excel 正确识别 UTF-8）
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(columns)  # 写入表头

            total = 0
            # 分批写入（使用 fetchmany）
            while True:
                rows = result.fetchmany(BATCH_SIZE)
                if not rows:
                    break
                writer.writerows(rows)
                total += len(rows)
                logger.info(f"已导出 {total} 条记录...")

    logger.info(f"导出完成！共导出 {total} 条记录，保存至 {OUTPUT_CSV}")


if __name__ == "__main__":
    export_to_csv()