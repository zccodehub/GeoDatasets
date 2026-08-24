import psycopg2
import pandas as pd
import numpy as np
import logging
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# 加载环境变量
load_dotenv()

# 数据库配置（使用环境变量）
DATABASE_URL = os.getenv("DATABASE_URL")

TABLE_NAME = 'spatial_relations'
BATCH_SIZE = 10000  # 分批处理大小

# ---------- 日志 ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- 方位角处理工具 ----------
def normalize_azimuth(az):
    """将方位角归一化到 [0, 360)"""
    az = az % 360
    return az

def get_direction_8_center(direction_8):
    """返回8方向的标准中心角度（度）"""
    mapping = {
        'North': 0,
        'Northeast': 45,
        'East': 90,
        'Southeast': 135,
        'South': 180,
        'Southwest': 225,
        'West': 270,
        'Northwest': 315
    }
    return mapping.get(direction_8, None)

def calculate_deviation(azimuth, center):
    """计算偏离中心的最小角度差（度）"""
    diff = abs(azimuth - center) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

def determine_ambiguity_level(deviation):
    """根据偏离角度确定模糊等级"""
    if deviation <= 5:
        return 'EXACT'
    elif deviation <= 15:
        return 'SLIGHT'
    elif deviation <= 22.5:
        return 'MODERATE'
    else:
        return 'CROSS_BOUNDARY'

# ---------- 场景分型核心逻辑 ----------
def assign_scene(row):
    """为单行记录分配场景编码"""
    topo = row['topology_type']
    combo = row['combo_class']
    road = row.get('road_name', '')
    has_road = road is not None and str(road).strip() != ''

    # 优先级：道路参照 > 邻接 > 重合 > 纯方位 > 混合
    if topo in ('Opposite', 'SameSide', 'Intersection'):
        if has_road:
            return 'SCENE-B'
        else:
            return 'SCENE-E'  # 缺道路名，降级
    elif topo == 'Abutting' and combo == 'Explicit':
        return 'SCENE-A'
    elif topo == 'Coincident':
        return 'SCENE-D'
    elif topo in ('Isolated', 'Detour') or combo == 'Implicit':
        return 'SCENE-C'
    else:
        return 'SCENE-E'

def build_template_match_key(row):
    """
        生成唯一的五维模板检索键
        格式：SceneCode_TopologyType_DistLevel_Direction8_AmbiguityLevel
        """
    scene = row['scene_code']
    topo = row['topology_type']
    dist = row['dist_level']
    direc = row['direction_8'] if pd.notna(row['direction_8']) else 'None'
    amb = row['ambiguity_level'] if pd.notna(row['ambiguity_level']) else 'EXACT'

    return f"{scene}_{topo}_{dist}_{direc}_{amb}"

# ---------- 数据质量检查 ----------
def check_data_quality(row):
    """检查逻辑一致性，返回 'VALID' 或 'SUSPECT'"""
    dist = row['dist_level']
    dist_m = row['exact_distance_m']
    # 根据经验阈值（可根据实际分布调整）
    if dist == 'VeryClose' and dist_m > 50:
        return 'SUSPECT'
    elif dist == 'Close' and dist_m > 200:
        return 'SUSPECT'
    elif dist == 'Medium' and dist_m > 500:
        return 'SUSPECT'
    elif dist == 'Far' and dist_m < 500:
        return 'SUSPECT'  # Far 应大于500m
    # 方位角合法性
    az = row['azimuth_deg']
    if az is None or az < 0 or az >= 360:
        return 'SUSPECT'
    # 拓扑类型合法性
    valid_topos = ('Coincident','Abutting','Opposite','SameSide','Intersection','Detour','Isolated')
    if row['topology_type'] not in valid_topos:
        return 'SUSPECT'
    return 'VALID'

# ---------- 主处理函数 ----------
def spatial_relations_fractal():
    # 1. 创建SQLAlchemy引擎（用于读取）
    engine = create_engine(DATABASE_URL)
    # 2. 创建psycopg2连接（用于更新）
    conn = psycopg2.connect(dsn=DATABASE_URL)
    # 2. 读取数据（全量或分批）
    logger.info("读取数据...")
    query = f"SELECT id, landmark_id, target_id, landmark_name, target_name, landmark_address, target_address, exact_distance_m, dist_level, azimuth_deg, direction_8, topology_type, combo_class, is_ambiguous, road_name FROM {TABLE_NAME};"
    df = pd.read_sql(query, con=engine)
    logger.info(f"读取 {len(df)} 条记录")

    # 3. 数据清洗与特征工程
    # 标准化方位角
    df['azimuth_deg'] = df['azimuth_deg'].apply(normalize_azimuth)
    # 方位模糊度量化
    df['center_angle'] = df['direction_8'].apply(get_direction_8_center)
    # 只对非空方向计算偏离
    mask = df['center_angle'].notna()
    df.loc[mask, 'deviation_angle'] = df.loc[mask].apply(
        lambda r: calculate_deviation(r['azimuth_deg'], r['center_angle']), axis=1
    )
    df.loc[mask, 'ambiguity_level'] = df.loc[mask]['deviation_angle'].apply(determine_ambiguity_level)
    # 若无方向或中心角为空，设为默认值
    df.loc[~mask, 'deviation_angle'] = np.nan
    df.loc[~mask, 'ambiguity_level'] = None

    # 场景分型
    df['scene_code'] = df.apply(assign_scene, axis=1)
    df['template_match_key'] = df.apply(build_template_match_key, axis=1)
    df['has_road_ref'] = df['road_name'].notna() & (df['road_name'].str.strip() != '')

    # 4. 写回数据库（创建新列并更新）
    logger.info("写回数据库...")
    with conn.cursor() as cur:
        # 检查并添加新列（如果不存在）
        cur.execute("""
            DO $$ 
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='spatial_relations' AND column_name='scene_code') THEN
                    ALTER TABLE spatial_relations ADD COLUMN scene_code TEXT;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='spatial_relations' AND column_name='template_match_key') THEN
                    ALTER TABLE spatial_relations ADD COLUMN template_match_key TEXT;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='spatial_relations' AND column_name='has_road_ref') THEN
                    ALTER TABLE spatial_relations ADD COLUMN has_road_ref BOOLEAN;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='spatial_relations' AND column_name='deviation_angle') THEN
                    ALTER TABLE spatial_relations ADD COLUMN deviation_angle NUMERIC;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='spatial_relations' AND column_name='ambiguity_level') THEN
                    ALTER TABLE spatial_relations ADD COLUMN ambiguity_level TEXT;
                END IF;
            END $$;
        """)
        conn.commit()
        logger.info("新增列已创建（若不存在）")

    # 逐批更新（避免单条更新开销）
    logger.info("开始批量更新...")
    total = len(df)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = df.iloc[start:end]
        # 构造更新SQL（使用CASE WHEN 批量更新）
        # 此处使用临时表或逐行更新，简单起见采用逐行更新（可优化为批量）
        # 但为性能，我们使用psycopg2的execute_values或使用SQLAlchemy的update
        # 这里采用逐行更新，47万条可能较慢，可改用批量update via temporary table
        # 以下为简明的逐行更新示例，生产环境建议使用批量方法
        with conn.cursor() as cur:
            for idx, row in batch.iterrows():
                cur.execute("""
                    UPDATE spatial_relations 
                    SET scene_code = %s, template_match_key = %s, has_road_ref = %s,
                        deviation_angle = %s, ambiguity_level = %s
                    WHERE id = %s
                """, (
                    row['scene_code'],
                    row['template_match_key'],
                    row['has_road_ref'],
                    row['deviation_angle'],
                    row['ambiguity_level'],
                    row['id']
                ))
            conn.commit()
        logger.info(f"已更新 {end}/{total} 条")

    # 5. 创建索引（可选）
    with conn.cursor() as cur:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_template_match_key ON spatial_relations (template_match_key);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_scene_code ON spatial_relations (scene_code);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ambiguity ON spatial_relations (ambiguity_level);")
        conn.commit()
    logger.info("索引创建完成")

    conn.close()
    logger.info("处理完成！")

# ---------- 执行 ----------
if __name__ == "__main__":
    spatial_relations_fractal()