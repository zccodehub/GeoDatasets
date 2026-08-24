import psycopg2
import logging
from psycopg2 import sql
import time
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv
import os


# 加载环境变量
load_dotenv()

# 数据库配置（使用环境变量）
DATABASE_URL = os.getenv("DATABASE_URL")

BATCH_SIZE = 500          # 每批处理的地标数
DISTANCE_LIMIT = 1000     # 只考虑1km范围内
LOG_LEVEL = logging.INFO

# ======================== 日志设置 ========================
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================== 数据库连接 ========================
def get_connection():
    # 1. 创建SQLAlchemy引擎（用于读取）
    engine = create_engine(DATABASE_URL)
    # 2. 创建psycopg2连接（用于更新）
    return psycopg2.connect(dsn=DATABASE_URL)

# ======================== 建表（若不存在） ========================
def init_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS spatial_relations (
                id SERIAL PRIMARY KEY,
                landmark_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                exact_distance_m NUMERIC,
                dist_level TEXT,
                azimuth_deg NUMERIC,
                direction_8 TEXT,
                direction_16 TEXT,
                topology_type TEXT,
                combo_class TEXT,
                is_ambiguous BOOLEAN,
                road_name TEXT,
                landmark_name TEXT,
                landmark_address TEXT,
                landmark_type TEXT,
                landmark_priority INTEGER,
                landmark_x TEXT,
                landmark_y TEXT,
                target_name TEXT,
                target_address TEXT,
                target_type TEXT,
                target_x TEXT, 
                target_y TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute("""
            DROP TABLE IF EXISTS temp_candidates;
            CREATE TEMP TABLE temp_candidates (
                landmark_id TEXT,
                target_id TEXT,
                exact_distance_m NUMERIC,
                dist_level TEXT,
                azimuth_deg NUMERIC,
                direction_8 TEXT,
                direction_16 TEXT,
                topology_type TEXT,
                combo_class TEXT,
                is_ambiguous BOOLEAN,
                road_name TEXT,
                poi_count INTEGER,
                max_keep INTEGER
            );
        """)
        cur.execute("CREATE INDEX idx_temp_landmark ON temp_candidates (landmark_id);")
        cur.execute("CREATE INDEX idx_temp_dist ON temp_candidates (exact_distance_m);")
        conn.commit()
        logger.info("Tables initialized.")

# ======================== 分批获取地标ID ========================
def get_all_landmark_ids(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT poi_id FROM landmarks WHERE priority > 0 ORDER BY priority DESC, poi_id;")
        return [row[0] for row in cur.fetchall()]

# ======================== 核心计算函数（每批） ========================
def process_batch(conn, landmark_ids):
    if not landmark_ids:
        return

    with conn.cursor() as cur:
        # ---------- 1. 清空临时表 ----------
        cur.execute("TRUNCATE temp_candidates;")

        # ---------- 2. 插入候选配对 ----------
        insert_candidates_sql = sql.SQL("""
            INSERT INTO temp_candidates (
                landmark_id, target_id, exact_distance_m, azimuth_deg
            )
            SELECT
                l.poi_id AS landmark_id,
                t.poi_id AS target_id,
                ST_Distance(l.geom, t.geom) AS dist_m,
                degrees(ST_Azimuth(l.geom, t.geom)) AS azim
            FROM
                landmarks l
                JOIN landmarks t ON ST_DWithin(l.geom, t.geom, %s)
            WHERE
                l.poi_id IN %s
                AND t.priority = 0
                AND l.poi_id != t.poi_id
        """)
        cur.execute(insert_candidates_sql, (DISTANCE_LIMIT, tuple(landmark_ids)))
        affected = cur.rowcount
        logger.info(f"Inserted {affected} candidate pairs for batch.")

        if affected == 0:
            return

        # ---------- 2.1 动态设置 max_keep ----------
        cur.execute("""
            WITH counts AS (
                SELECT landmark_id, COUNT(*) AS cnt
                FROM temp_candidates
                GROUP BY landmark_id
            )
            UPDATE temp_candidates
            SET max_keep = CASE
                WHEN c.cnt >= 1000 THEN 5
                WHEN c.cnt >= 500  THEN 4
                WHEN c.cnt >= 200  THEN 3
                ELSE 2
            END, poi_count = c.cnt
            FROM counts c
            WHERE temp_candidates.landmark_id = c.landmark_id;
        """)

        # ---------- 3. 距离等级 ----------
        cur.execute("""
            UPDATE temp_candidates
            SET dist_level = CASE
                WHEN exact_distance_m < 50 THEN 'VeryClose'
                WHEN exact_distance_m < 200 THEN 'Close'
                WHEN exact_distance_m < 500 THEN 'Medium'
                WHEN exact_distance_m <= 1000 THEN 'Far'
                ELSE NULL
            END;
        """)

        # ---------- 4. 8方向 ----------
        cur.execute("""
            UPDATE temp_candidates
            SET direction_8 = CASE
                WHEN azimuth_deg BETWEEN 337.5 AND 360 OR azimuth_deg BETWEEN 0 AND 22.5 THEN 'North'
                WHEN azimuth_deg BETWEEN 22.5 AND 67.5 THEN 'Northeast'
                WHEN azimuth_deg BETWEEN 67.5 AND 112.5 THEN 'East'
                WHEN azimuth_deg BETWEEN 112.5 AND 157.5 THEN 'Southeast'
                WHEN azimuth_deg BETWEEN 157.5 AND 202.5 THEN 'South'
                WHEN azimuth_deg BETWEEN 202.5 AND 247.5 THEN 'Southwest'
                WHEN azimuth_deg BETWEEN 247.5 AND 292.5 THEN 'West'
                WHEN azimuth_deg BETWEEN 292.5 AND 337.5 THEN 'Northwest'
                ELSE NULL
            END;
        """)

        # ---------- 5. 拓扑类型判定（按优先级依次更新，使用英文）  ----------
        # 5.1 重合 (Coincident)
        cur.execute("UPDATE temp_candidates SET topology_type = 'Coincident' WHERE exact_distance_m < 1;")
        # 5.2 贴边近邻 (Abutting)
        cur.execute("""
            UPDATE temp_candidates
            SET topology_type = 'Abutting'
            WHERE exact_distance_m BETWEEN 1 AND 30
              AND topology_type IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM road_centerlines r
                  WHERE ST_Intersects(
                      ST_MakeLine(
                          (SELECT geom FROM landmarks WHERE poi_id = landmark_id),
                          (SELECT geom FROM landmarks WHERE poi_id = target_id)
                      ),
                      r.geom
                  )
              );
        """)
        # 5.3 路口把角 (Intersection) — 获取交叉口名称
        cur.execute("""
            UPDATE temp_candidates
            SET topology_type = 'Intersection',
                road_name = COALESCE(
                    (SELECT name FROM road_intersects
                     WHERE ST_DWithin(
                         (SELECT geom FROM landmarks WHERE poi_id = target_id),
                         geom, 60
                     ) LIMIT 1),
                    (SELECT name FROM road_intersects
                     WHERE ST_DWithin(
                         (SELECT geom FROM landmarks WHERE poi_id = landmark_id),
                         geom, 60
                     ) LIMIT 1)
                )
            WHERE topology_type IS NULL
              AND (
                  EXISTS (
                      SELECT 1 FROM road_intersects ri
                      WHERE ST_DWithin(
                          (SELECT geom FROM landmarks WHERE poi_id = target_id),
                          ri.geom, 60
                      )
                  )
              );
        """)
        # 5.4 隔路正对 (Opposite)
        cur.execute("""
            WITH projected AS (
                SELECT
                    tc.landmark_id,
                    tc.target_id,
                    tc.exact_distance_m,
                    r.osm_id,
                    r.road_width AS est_width,
                    r.name AS road_name,
                    ST_LineMerge(r.geom) AS road_geom,
                    ST_Length(ST_LineMerge(r.geom)) AS road_length,
                    ST_ClosestPoint(ST_LineMerge(r.geom), l.geom) AS proj_landmark,
                    ST_ClosestPoint(ST_LineMerge(r.geom), t.geom) AS proj_target,
                    ST_LineLocatePoint(ST_LineMerge(r.geom), ST_ClosestPoint(ST_LineMerge(r.geom), l.geom)) AS pos_l,
                    ST_LineLocatePoint(ST_LineMerge(r.geom), ST_ClosestPoint(ST_LineMerge(r.geom), t.geom)) AS pos_t,
                    ST_Distance(ST_LineMerge(r.geom), l.geom) AS dist_l,
                    ST_Distance(ST_LineMerge(r.geom), t.geom) AS dist_t
                FROM temp_candidates tc
                JOIN landmarks l ON l.poi_id = tc.landmark_id
                JOIN landmarks t ON t.poi_id = tc.target_id
                CROSS JOIN LATERAL (
                    SELECT osm_id, road_width, name, geom
                    FROM road_centerlines
                    WHERE ST_DWithin(l.geom, geom, 50) OR ST_DWithin(t.geom, geom, 50)
                    ORDER BY ST_Distance(l.geom, geom) + ST_Distance(t.geom, geom)
                    LIMIT 1
                ) r
                WHERE tc.topology_type IS NULL
                  AND tc.exact_distance_m BETWEEN 30 AND 500
            )
            UPDATE temp_candidates
            SET topology_type = 'Opposite',
                road_name = p.road_name
            FROM projected p
            WHERE temp_candidates.landmark_id = p.landmark_id
              AND temp_candidates.target_id = p.target_id
              AND ABS(p.pos_l - p.pos_t) * p.road_length <= 20
              AND ABS(p.dist_l + p.dist_t - p.est_width) < 10
              AND p.dist_l > 0 AND p.dist_t > 0;
        """)
        # 5.5 临街同侧 (SameSide)
        cur.execute("""
            WITH projected AS (
                SELECT
                    tc.landmark_id,
                    tc.target_id,
                    r.name AS road_name,
                    ST_LineMerge(r.geom) AS road_geom,
                    ST_Length(ST_LineMerge(r.geom)) AS road_length,
                    ST_ClosestPoint(ST_LineMerge(r.geom), l.geom) AS proj_l,
                    ST_ClosestPoint(ST_LineMerge(r.geom), t.geom) AS proj_t,
                    ST_LineLocatePoint(ST_LineMerge(r.geom), ST_ClosestPoint(ST_LineMerge(r.geom), l.geom)) AS pos_l,
                    ST_LineLocatePoint(ST_LineMerge(r.geom), ST_ClosestPoint(ST_LineMerge(r.geom), t.geom)) AS pos_t,
                    ST_Distance(ST_LineMerge(r.geom), l.geom) AS dist_l,
                    ST_Distance(ST_LineMerge(r.geom), t.geom) AS dist_t
                FROM temp_candidates tc
                JOIN landmarks l ON l.poi_id = tc.landmark_id
                JOIN landmarks t ON t.poi_id = tc.target_id
                CROSS JOIN LATERAL (
                    SELECT name, geom
                    FROM road_centerlines
                    WHERE ST_DWithin(l.geom, geom, 80) OR ST_DWithin(t.geom, geom, 80)
                    ORDER BY ST_Distance(l.geom, geom) + ST_Distance(t.geom, geom)
                    LIMIT 1
                ) r
                WHERE tc.topology_type IS NULL
                  AND tc.exact_distance_m BETWEEN 50 AND 500
            )
            UPDATE temp_candidates
            SET topology_type = 'SameSide',
                road_name = p.road_name
            FROM projected p
            WHERE temp_candidates.landmark_id = p.landmark_id
              AND temp_candidates.target_id = p.target_id
              AND ABS(p.pos_l - p.pos_t) * p.road_length <= 200
              AND (p.dist_l > 0 AND p.dist_t > 0 AND p.dist_l * p.dist_t > 0)
              AND p.dist_l > 1 AND p.dist_t > 1;
        """)
        # 5.7 剩余未分类的设为 'Isolated'
        cur.execute("""
            UPDATE temp_candidates
            SET topology_type = 'Isolated'
            WHERE topology_type IS NULL;
        """)

        # ---------- 6. 16方向与模糊标记 ----------
        cur.execute("""
            UPDATE temp_candidates
            SET direction_16 = CASE
                WHEN (topology_type IN ('Opposite', 'Abutting') AND exact_distance_m < 200)
                     OR (topology_type = 'Intersection')
                THEN
                    CASE
                        WHEN azimuth_deg BETWEEN 348.75 AND 360 OR azimuth_deg BETWEEN 0 AND 11.25 THEN 'North'
                        WHEN azimuth_deg BETWEEN 11.25 AND 33.75 THEN 'NorthNortheast'
                        WHEN azimuth_deg BETWEEN 33.75 AND 56.25 THEN 'Northeast'
                        WHEN azimuth_deg BETWEEN 56.25 AND 78.75 THEN 'EastNortheast'
                        WHEN azimuth_deg BETWEEN 78.75 AND 101.25 THEN 'East'
                        WHEN azimuth_deg BETWEEN 101.25 AND 123.75 THEN 'EastSoutheast'
                        WHEN azimuth_deg BETWEEN 123.75 AND 146.25 THEN 'Southeast'
                        WHEN azimuth_deg BETWEEN 146.25 AND 168.75 THEN 'SouthSoutheast'
                        WHEN azimuth_deg BETWEEN 168.75 AND 191.25 THEN 'South'
                        WHEN azimuth_deg BETWEEN 191.25 AND 213.75 THEN 'SouthSouthwest'
                        WHEN azimuth_deg BETWEEN 213.75 AND 236.25 THEN 'Southwest'
                        WHEN azimuth_deg BETWEEN 236.25 AND 258.75 THEN 'WestSouthwest'
                        WHEN azimuth_deg BETWEEN 258.75 AND 281.25 THEN 'West'
                        WHEN azimuth_deg BETWEEN 281.25 AND 303.75 THEN 'WestNorthwest'
                        WHEN azimuth_deg BETWEEN 303.75 AND 326.25 THEN 'Northwest'
                        WHEN azimuth_deg BETWEEN 326.25 AND 348.75 THEN 'NorthNorthwest'
                        ELSE NULL
                    END
                ELSE direction_8
            END,
            is_ambiguous = CASE
                WHEN azimuth_deg % 22.5 BETWEEN 20 AND 25 THEN TRUE
                ELSE FALSE
            END;
        """)

        # ---------- 7. 组合分类 ----------
        cur.execute("""
            UPDATE temp_candidates
            SET combo_class = CASE
                WHEN topology_type IN ('Coincident', 'Abutting', 'Intersection') THEN 'Implicit'
                ELSE 'Explicit'
            END;
        """)

        # ---------- 8. 删除该批地标的旧数据（确保重置） ----------
        cur.execute("DELETE FROM spatial_relations WHERE landmark_id = ANY(%s::text[]);", (landmark_ids,))
        deleted = cur.rowcount
        if deleted:
            logger.info(f"Deleted {deleted} old records from spatial_relations for this batch.")

        # ---------- 8. 窗口函数筛选 Top K（使用动态 max_keep）并插入最终表 ----------
        insert_final_sql = """
            INSERT INTO spatial_relations (
                landmark_id, target_id, exact_distance_m, dist_level,
                azimuth_deg, direction_8, direction_16, topology_type,
                combo_class, is_ambiguous, road_name,
                landmark_name, landmark_address, landmark_type,landmark_priority,
                target_name, target_address, target_type,
                landmark_x, landmark_y, target_x, target_y
            )
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY landmark_id, topology_type, dist_level
                        ORDER BY random()
                    ) AS rn
                FROM temp_candidates
                WHERE topology_type IS NOT NULL AND dist_level IS NOT NULL
            )
            SELECT
                r.landmark_id, r.target_id, r.exact_distance_m, r.dist_level,
                r.azimuth_deg, r.direction_8, r.direction_16, r.topology_type,
                r.combo_class, r.is_ambiguous, r.road_name,
                l.名称 AS landmark_name,
                l.地址 AS landmark_address,
                l.type_3 AS landmark_type,
                l.priority AS landmark_priority,
                t.名称 AS target_name,
                t.地址 AS target_address,
                t.type_3 AS target_type,
                l.经度 AS landmark_x,
                l.纬度 AS landmark_y,
                t.经度 AS target_x,
                t.纬度 AS target_y
            FROM ranked r
            JOIN landmarks l ON l.poi_id = r.landmark_id
            JOIN landmarks t ON t.poi_id = r.target_id
            WHERE r.rn <= r.max_keep   -- 使用动态的 max_keep
            ON CONFLICT DO NOTHING;
        """
        cur.execute(insert_final_sql)
        inserted = cur.rowcount
        conn.commit()
        logger.info(f"Inserted {inserted} final records into spatial_relations.")


def export_temp_candidates(conn, filepath='temp_candidates_export.csv'):
    with conn.cursor() as cur:
        with open(filepath, 'w', encoding='utf-8') as f:
            cur.copy_expert("COPY temp_candidates TO STDOUT WITH CSV HEADER", f)
        logger.info(f"Exported temp_candidates to {filepath}")

# ======================== 主流程 ========================
def main():
    # 1. 创建SQLAlchemy引擎（用于读取）
    engine = create_engine(DATABASE_URL)
    # 2. 创建psycopg2连接（用于更新）
    conn = psycopg2.connect(dsn=DATABASE_URL)

    try:
        init_tables(conn)
        landmark_ids = get_all_landmark_ids(conn)
        total = len(landmark_ids)
        logger.info(f"Total landmarks: {total}")

        for start in range(0, total, BATCH_SIZE):
            batch = landmark_ids[start:start+BATCH_SIZE]
            logger.info(f"Processing batch {start//BATCH_SIZE + 1}, size {len(batch)}")
            process_batch(conn, batch)
            time.sleep(0.5)

        # start = 0
        # batch = landmark_ids[start:start + BATCH_SIZE]
        # logger.info(f"Processing batch {start // BATCH_SIZE + 1}, size {len(batch)}")
        # process_batch(conn, batch)
        #
        # # 如果是第一批，导出临时表内容
        # if start == 0:  # 需要将 start 传入或使用全局标志
        #     export_temp_candidates(conn, f'temp_batch_{start}.csv')

        logger.info("All batches processed successfully.")
    except Exception as e:
        logger.exception("Error occurred")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    main()