import geopandas as gpd
import pandas as pd

# ==================== 1. 定义道路类型 -> 宽度（米）的映射字典 ====================
# 参考《城市道路工程设计规范》及OSM常见分类，可根据北京市实际情况调整
WIDTH_MAPPING = {
    # 高速公路及城市快速路 (双向多车道，含隔离带)
    'motorway': 35,
    'motorway_link': 30,
    'trunk': 30,
    'trunk_link': 25,

    # 主干道 (城市主干路，一般双向6-8车道)
    'primary': 25,
    'primary_link': 20,

    # 次干道 (城市次干路，一般双向4-6车道)
    'secondary': 15,
    'secondary_link': 12,

    # 支线/三级道路 (一般双向2-4车道)
    'tertiary': 10,
    'tertiary_link': 8,

    # 居住区道路、未分类道路 (一般双向2车道或单车道)
    'residential': 6,
    'unclassified': 5,
    'road': 5,

    # 服务道路、生活街道 (较窄，可能有停车)
    'service': 3.5,
    'living_street': 4,

    # 乡村道路、土路、机耕道
    'track': 2.5,

    # 人行道、自行车道、步行道
    'footway': 1.5,
    'cycleway': 1.5,
    'pedestrian': 2,
    'path': 1,

    # 其他特殊类型
    'steps': 1,          # 台阶
    'bridleway': 1.5,    # 骑马道
    'construction': 5,   # 施工道路
    'proposed': 5,       # 规划道路
    'busway': 3.5,  #公交专用道
    'track_grade1': 3.0,  # 一级土路
    'track_grade2': 2.5,  # 二级土路
    'track_grade3': 2.0,  # 三级土路
    'track_grade4': 1.5,  # 四级土路
    'track_grade5': 1.0,  # 五级土路
}

# 默认宽度（当fclass未在映射表中时使用）
DEFAULT_WIDTH = 4

# ==================== 2. 读取OSM路网数据 ====================
# 支持多种格式：.gpkg, .shp, .geojson 等
input_path = r"F:\360安全云盘同步版\02-我的论文\数据集构建\Data\QGIS数据\地标库\Beijing_OSM_Roads_UTM.gpkg"

try:
    gdf = gpd.read_file(input_path)
    print(f"成功读取数据，共 {len(gdf)} 条记录")
    print(f"数据坐标系：{gdf.crs}")
except Exception as e:
    print(f"读取数据失败：{e}")
    exit()

# ==================== 3. 检查fclass字段是否存在 ====================
if 'fclass' not in gdf.columns:
    print("错误：数据中没有 'fclass' 字段。请检查字段名（可能是 'highway' 或其他）。")
    # 若字段名不同，可在此处修改或让用户指定
    # 例如：fclass_col = 'highway'
    exit()

# 提取所有非空的fclass唯一值
unique_fclasses = gdf['fclass'].dropna().unique()
unique_fclasses = sorted(unique_fclasses)  # 排序便于查看

print(f"\n共发现 {len(unique_fclasses)} 种道路类型：")
print(unique_fclasses)

# ==================== 4. 构建映射结果表 ====================
rows = []
for fclass in unique_fclasses:
    # 如果映射表中存在，则使用；否则使用默认值
    width = WIDTH_MAPPING.get(fclass, DEFAULT_WIDTH)
    rows.append({'fclass': fclass, 'road_width': width})

# 创建DataFrame
df_width = pd.DataFrame(rows)

# ==================== 5. 保存为CSV ====================
output_csv = "../../output/road_width_mapping.csv"  #输出文件名
df_width.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"\n映射表已保存至：{output_csv}")
print("内容预览：")
print(df_width.head(10))

# ==================== 6. (可选) 显示未在映射表中的类型 ====================
unmapped = [f for f in unique_fclasses if f not in WIDTH_MAPPING]
if unmapped:
    print(f"\n注意：以下类型在映射表中未定义，已使用默认宽度 {DEFAULT_WIDTH} 米：")
    print(unmapped)
else:
    print("\n所有道路类型均已定义宽度。")