import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
from sklearn.neighbors import NearestNeighbors

# --- 配置参数 ---
# 使用 Windows 原始字符串路径格式
BASE_DIR = r"E:\Projects\PycharmProjects\Geocoding_dataset"
DATA_PATH = os.path.join(BASE_DIR, "data", "Beijing_POI_composite_3.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "final_landmarks.csv")

GRID_SIZE_METERS = 1000
ALPHA = 0.2
BETA = 0.8

# 空间覆盖参数
COVERAGE_RADIUS = 2000  # 2km 覆盖半径

# 请根据实际 CSV 文件头修改以下列名
COL_LAT = "纬度"
COL_LON = "经度"
COL_LIS = "composite_score"
COL_CIS = "cognition_score"
COL_SIS = "spatial_score"

def classify_priority(row, quantiles):
    sis, cis = row[COL_SIS], row[COL_CIS]
    if sis >= quantiles['SIS_P80'] and cis >= quantiles['CIS_P80']:
        return 1
    elif cis >= quantiles['CIS_P70'] and sis <= quantiles['SIS_P60']:
        return 2
    elif sis >= quantiles['SIS_P70'] and cis <= quantiles['CIS_P60']:
        return 3
    else:
        return 4

def run_landmark_extraction():
    # 1. 加载数据
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    geometry = [Point(xy) for xy in zip(df[COL_LON], df[COL_LAT])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs("EPSG:3857")
    
    # 2. 格网划分 (修正 IntCastingNaNError)
    print("Assigning grids...")
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # 计算格网坐标，使用 apply(np.floor) 确保处理浮点运算
    grid_x = ((gdf.geometry.x - minx) / GRID_SIZE_METERS).apply(np.floor)
    grid_y = ((gdf.geometry.y - miny) / GRID_SIZE_METERS).apply(np.floor)
    
    # 使用 Int64 可空类型存储，解决 NaN 到 int 转换报错
    gdf['grid_x'] = grid_x.astype('Int64')
    gdf['grid_y'] = grid_y.astype('Int64')
    gdf['grid_id'] = gdf['grid_x'].astype(str) + "_" + gdf['grid_y'].astype(str)
    
    # 3. 计算配额
    grid_stats = gdf.groupby('grid_id').size().reset_index(name='n_i')
    grid_stats['Q_i'] = grid_stats['n_i'].apply(lambda n: max(1, int(np.floor(ALPHA * (n ** BETA)))))

    print(f"格网的总数：{grid_stats.shape[0]}")
    # 4. 全局分位数计算
    quantiles = {
        'SIS_P60': gdf[COL_SIS].quantile(0.60), 'SIS_P70': gdf[COL_SIS].quantile(0.70), 'SIS_P80': gdf[COL_SIS].quantile(0.80),
        'CIS_P60': gdf[COL_CIS].quantile(0.60), 'CIS_P70': gdf[COL_CIS].quantile(0.70), 'CIS_P80': gdf[COL_CIS].quantile(0.80),
    }
    
    # 5. 格网内筛选与排序
    print("Processing grids...")
    gdf['landmark_priority'] = 0 
    
    for grid_id, group in gdf.groupby('grid_id'):
        quota = grid_stats.loc[grid_stats['grid_id'] == grid_id, 'Q_i'].values[0]
        
        # 本地P70过滤
        local_p70 = group[COL_LIS].quantile(0.70)
        candidates = group[group[COL_LIS] >= local_p70].copy()
        
        # 优先级判定
        candidates['priority'] = candidates.apply(lambda row: classify_priority(row, quantiles), axis=1)
        
        # 排序
        candidates = candidates.sort_values(by=['priority', COL_LIS], ascending=[True, False])
        
        # 截取
        selected = candidates.head(quota)
        
        # 将选中点的优先级回写到全局主表
        gdf.loc[selected.index, 'landmark_priority'] = selected['priority']

    # --- 新增分析：空间覆盖率与最近邻距离 ---
    landmarks = gdf[gdf['landmark_priority'] > 0]

    # 1. 空间覆盖率: 以地标为圆心构建缓冲区，计算其在所有POI范围内覆盖的比例
    print("Calculating spatial coverage...")
    # 构建地标覆盖区域 (Dissolve 多个缓冲区)
    landmark_buffers = landmarks.geometry.buffer(COVERAGE_RADIUS)
    union_buffer = landmark_buffers.union_all()

    # 获取所有POI范围的凸包或边界框作为研究区 (这里以全部POI的凸包近似北京区域)
    beijing_area = gdf.geometry.union_all().convex_hull

    # 计算覆盖比例
    coverage_ratio = union_buffer.area / beijing_area.area

    # 2. 平均最近邻距离 与 最大最近邻距离
    coords = np.array(list(landmarks.geometry.apply(lambda p: (p.x, p.y))))
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    # 取第二个距离 (第一个是点自身，距离为0)
    nn_distances = distances[:, 1]
    avg_nn_dist = np.mean(nn_distances)
    max_nn_dist = np.max(nn_distances)

    # --- 输出统计 ---
    print("\n--- 地标质量评估 ---")
    print(f"空间覆盖率: {coverage_ratio:.2%}")
    print(f"平均最近邻距离: {avg_nn_dist:.2f} 米")
    print(f"最大最近邻距离: {max_nn_dist:.2f} 米")

    priority_counts = gdf['landmark_priority'].value_counts().sort_index()
    print("\n--- 地标优先级统计 ---")
    for p in [0, 1, 2, 3, 4]:
        print(f"优先级 {p}: {priority_counts.get(p, 0)}")

    landmark_count = priority_counts.sum() - priority_counts.get(0, 0)
    landmark_ratio = (landmark_count - priority_counts.get(4, 0))/landmark_count
    print(f"地标总数： {landmark_count}")
    print(f"有效地标数： {landmark_ratio:.2%} ")
    print("----------------------\n")
    # 6. 输出结果与优先级统计
    print(f"Saving results to {OUTPUT_PATH}")
    
    # 转回 WGS84 坐标并保存 (使用 utf-8-sig 编码以适配 Excel)
    gdf.to_crs("EPSG:4326").drop(columns=['geometry', 'grid_x', 'grid_y', 'grid_id']).to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print("Done.")

if __name__ == "__main__":
    run_landmark_extraction()
