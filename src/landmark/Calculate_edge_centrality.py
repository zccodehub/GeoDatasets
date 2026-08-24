import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
import warnings
import time

warnings.filterwarnings('ignore')

# ==================== 1. 配置路径和参数 ====================
INPUT_GPKG = r"F:\360安全云盘同步版\02-我的论文\数据集构建\Data\QGIS数据\地标库\Beijing_major_roads_singleparts.gpkg"
OUTPUT_GPKG = r"F:\360安全云盘同步版\02-我的论文\数据集构建\Data\QGIS数据\地标库\Beijing_major_roads_with_centrality.gpkg"

UNIQUE_ID_FIELD = "fid"
GEOMETRY_FIELD = "geometry"

# ==================== 2. 读取路网数据 ====================
print("正在读取路网数据...")
roads_gdf = gpd.read_file(INPUT_GPKG)

if roads_gdf.crs and roads_gdf.crs.is_geographic:
    print("检测到地理坐标系，自动投影到 UTM 50N (EPSG:32650)...")
    roads_gdf = roads_gdf.to_crs("EPSG:32650")

print(f"路网数据共 {len(roads_gdf)} 条记录")

# ==================== 3. 构建 NetworkX 图 ====================
print("正在构建网络图...")
G = nx.Graph()
edge_to_index = {}
skipped_count = 0

for idx, row in roads_gdf.iterrows():
    geom = row[GEOMETRY_FIELD]
    if geom is None or geom.is_empty:
        skipped_count += 1
        continue

    coords = list(geom.coords)
    if len(coords) < 2:
        skipped_count += 1
        continue

    start_node = coords[0]
    end_node = coords[-1]

    # 检查坐标是否有效
    if not np.isfinite(start_node).all() or not np.isfinite(end_node).all():
        skipped_count += 1
        continue

    G.add_edge(start_node, end_node, fid=idx, weight=geom.length)
    edge_to_index[(start_node, end_node)] = idx

print(f"网络图构建完成，包含 {G.number_of_nodes():,} 个节点，{G.number_of_edges():,} 条边")
if skipped_count > 0:
    print(f"跳过 {skipped_count} 条无效记录")

# ==================== 4. 检查路网连通性 ====================
if nx.is_connected(G):
    print("✅ 路网是连通的")
else:
    print(f"⚠️ 路网不连通，包含 {nx.number_connected_components(G)} 个连通分量")
    # 只保留最大连通分量
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"提取最大连通分量后: {G.number_of_nodes():,} 个节点, {G.number_of_edges():,} 条边")

# ==================== 5. 计算边介数中心性 ====================
print("正在计算边介数中心性...")
start_time = time.time()

edge_centrality = nx.edge_betweenness_centrality(
    G,
    weight='weight',
    normalized=True  # 归一化到0-1
)

elapsed = time.time() - start_time
print(f"边介数中心性计算完成，耗时 {elapsed:.2f} 秒")
print(f"共计算 {len(edge_centrality)} 条边")

# 输出统计信息
values = list(edge_centrality.values())
if values:
    print(f"  最小值: {min(values):.6f}")
    print(f"  最大值: {max(values):.6f}")
    print(f"  平均值: {np.mean(values):.6f}")
    print(f"  中位数: {np.median(values):.6f}")

# ==================== 6. 将结果合并回 GeoDataFrame ====================
print("正在将结果合并回路网数据...")

# 构建查找字典
centrality_by_index = {}
matched_count = 0

for (u, v), centrality in edge_centrality.items():
    idx = edge_to_index.get((u, v))
    if idx is None:
        idx = edge_to_index.get((v, u))
    if idx is not None:
        centrality_by_index[idx] = centrality
        matched_count += 1

print(f"成功匹配: {matched_count} 条边")
print(f"未匹配: {len(edge_centrality) - matched_count} 条边")

# 为每条记录赋值
roads_gdf["edge_centrality"] = roads_gdf.index.map(
    lambda i: centrality_by_index.get(i, 0.0)
)

positive_count = (roads_gdf["edge_centrality"] > 0).sum()
print(f"赋值结果:")
print(f"  >0 的记录数: {positive_count:,} ({positive_count/len(roads_gdf)*100:.1f}%)")
print(f"  =0 的记录数: {len(roads_gdf) - positive_count:,}")

# ==================== 7. 抽样验证 ====================
print("\n=== 抽样验证（中心性最高的5条路段）===")
top_5 = roads_gdf.nlargest(5, "edge_centrality")

# 只显示中心性值，不显示ID
cols = ["edge_centrality"]
if "name" in roads_gdf.columns:
    cols = ["name"] + cols

print(top_5[cols].to_string(index=False))

# ==================== 8. 保存结果 ====================
print(f"\n正在保存结果到: {OUTPUT_GPKG}")
roads_gdf.to_file(OUTPUT_GPKG, driver="GPKG")
print("✅ 全部完成！")