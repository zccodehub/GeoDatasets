import sys
import os

# 获取项目根目录（假设当前文件在 src/landmark/ 下）
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # 回到项目根目录

# 添加到Python搜索路径的最前面
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.PercentileScaler import PercentileScaler

# 配置参数
INPUT_FILE = r"F:\360安全云盘同步版\02-我的论文\数据集构建\Data\QGIS数据\地标库\Beijing_POI_centrality_2.gpkg"
OUTPUT_FILE = r"F:\360安全云盘同步版\02-我的论文\数据集构建\Data\QGIS数据\地标库\Beijing_POI_centrality.gpkg"
FIELD_NAME = "edge_centrality"  # 要处理的字段名
OUTPUT_FIELD = "centrality_scaled"  # 处理后的新字段名

# 创建工具类实例并执行
scaler = PercentileScaler(
    input_path=INPUT_FILE,
    output_path=OUTPUT_FILE,
    field_name=FIELD_NAME,
    output_field_name=OUTPUT_FIELD
)

result_gdf = scaler.run()