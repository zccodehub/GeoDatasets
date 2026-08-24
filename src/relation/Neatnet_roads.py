import geopandas as gpd
import neatnet

# ==================== 1. 读取数据 ====================
# 从 GeoPackage 读取 OSM 路网数据
input_gpkg = r"F:\360安全云盘同步版\02-我的论文\数据集构建\Data\QGIS数据\空间关系\Beijing_major_roads_with_width.gpkg"
roads = gpd.read_file(input_gpkg)

print(f"原始道路数量: {len(roads)}")
print(f"原始坐标系: {roads.crs}")

# ==================== 2. 坐标系检查与投影 ====================
# neatnet 要求输入数据为投影坐标系（以米为单位）[reference:3]
# 如果数据是经纬度（WGS84, EPSG:4326），需要先投影
if roads.crs is None or roads.crs.is_geographic:
    # 北京地区推荐使用 CGCS2000 / 3-degree Gauss-Kruger zone 120 (EPSG:4547)
    # 或 UTM zone 50N (EPSG:32650)
    target_crs = "EPSG:4547"  # 可根据需要调整
    roads = roads.to_crs(target_crs)
    print(f"已投影至: {roads.crs}")

# ==================== 3. 筛选道路数据（可选） ====================
# 如果数据中包含非道路要素，建议按 highway 字段筛选
if "highway" in roads.columns:
    roads = roads[roads["highway"].notna()]
    print(f"筛选后道路数量: {len(roads)}")


# 1. 闭合间隙：将断开的线段在指定阈值内 snapping 连接
roads.geometry = neatnet.close_gaps(roads, tolerance=0.25)

# 2. 移除插值节点：删除线段上不必要的中间节点
roads = neatnet.remove_interstitial_nodes(roads)

# 将所有列中类型为 list 的值转换为字符串
for col in roads.columns:
    if roads[col].apply(lambda x: isinstance(x, list)).any():
        roads[col] = roads[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)

# ==================== 4. 执行中心线简化 ====================
# neatify() 是 neatnet 的核心函数，自动处理双线道路、环岛等复杂几何[reference:4]
simplified = neatnet.neatify(roads)

print(f"简化后道路数量: {len(simplified)}")

# ==================== 5. 保存结果 ====================
output_gpkg = r"F:\360安全云盘同步版\02-我的论文\数据集构建\Data\QGIS数据\空间关系\Beijing_Roads_Centerlines.gpkg"
simplified.to_file(output_gpkg, driver="GPKG")
print(f"中心线已保存至: {output_gpkg}")