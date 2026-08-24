import geopandas as gpd
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')


class PercentileScaler:
    """
    将 GeoPackage 文件中指定字段的 >0 值，通过百分位秩映射到 50-100 区间。

    参数:
        input_path (str): 输入 GeoPackage 文件路径
        output_path (str): 输出 GeoPackage 文件路径
        field_name (str): 需要处理的字段名（如 max_centrality）
        output_field_name (str): 处理后新字段的名称（如 centrality_scaled）
    """

    def __init__(self, input_path, output_path, field_name, output_field_name):
        self.input_path = input_path
        self.output_path = output_path
        self.field_name = field_name
        self.output_field_name = output_field_name
        self.gdf = None
        self.stats = {}

    def read_data(self):
        """读取 GeoPackage 数据"""
        print(f"正在读取数据: {self.input_path}")
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"输入文件不存在: {self.input_path}")

        self.gdf = gpd.read_file(self.input_path)
        print(f"数据读取成功，共 {len(self.gdf)} 条记录")
        return self

    def validate_field(self):
        """验证字段是否存在"""
        if self.field_name not in self.gdf.columns:
            raise ValueError(
                f"字段 '{self.field_name}' 不存在于数据中。"
                f"可用字段: {list(self.gdf.columns)}"
            )
        return self

    def compute_stats(self):
        """计算并输出原始数据统计信息"""
        total = len(self.gdf)
        positive_mask = self.gdf[self.field_name] > 0
        positive_count = positive_mask.sum()
        zero_count = total - positive_count
        max_val = self.gdf[self.field_name].max()
        min_val = self.gdf[self.field_name].min()

        self.stats = {
            'total': total,
            'positive_count': positive_count,
            'positive_ratio': positive_count / total if total > 0 else 0,
            'zero_count': zero_count,
            'max_value': max_val,
            'min_value': min_val
        }

        print(f"\n=== 原始 {self.field_name} 字段统计 ===")
        print(f"总记录数: {self.stats['total']:,}")
        print(f"> 0 的记录数: {self.stats['positive_count']:,} ({self.stats['positive_ratio'] * 100:.1f}%)")
        print(f"= 0 的记录数: {self.stats['zero_count']:,} ({self.stats['zero_count'] / total * 100:.1f}%)")
        print(f"最大值: {self.stats['max_value']:.6f}")
        print(f"最小值: {self.stats['min_value']:.6f}")

        return self

    def transform(self):
        """
        将 > 0 的值通过百分位秩映射到 50-100 区间，
        = 0 或 NULL 的值映射到 0
        """
        print(f"\n正在将 {self.field_name} 映射到 50-100 区间...")

        # 初始化新字段为 0
        self.gdf[self.output_field_name] = 0.0

        # 获取 > 0 的记录
        positive_mask = self.gdf[self.field_name] > 0
        positive_count = positive_mask.sum()

        if positive_count == 0:
            print("警告：没有找到 > 0 的记录，所有值将被设为 0")
            return self

        # 对 > 0 的记录按原始值升序排序
        positive_data = self.gdf[positive_mask].copy()
        positive_data_sorted = positive_data.sort_values(self.field_name)
        n = positive_data_sorted.shape[0]

        # 计算百分位秩（0-100）
        if n > 1:
            percentile_ranks = np.arange(n) / (n - 1) * 100
        else:
            percentile_ranks = np.array([50])

        # 映射到 50-100
        mapped_scores = 50 + percentile_ranks / 2

        # 按索引写回
        positive_indices = positive_data_sorted.index
        self.gdf.loc[positive_indices, self.output_field_name] = mapped_scores

        # 打印转换后的统计信息
        scaled_positive = (self.gdf[self.output_field_name] > 0).sum()
        print(f"转换后 > 0 的记录数: {scaled_positive:,}")
        print(f"转换后最大值: {self.gdf[self.output_field_name].max():.4f}")
        print(f"转换后最小值: {self.gdf[self.output_field_name].min():.4f}")

        # 一致性检查
        if positive_count != scaled_positive:
            print(f"⚠️ 警告：原始 > 0 的记录数 ({positive_count:,}) 与转换后 > 0 的记录数 ({scaled_positive:,}) 不一致！")
        else:
            print(f"✅ 数据一致性检查通过：{positive_count:,} 条 > 0 的记录全部保留。")

        return self

    def sample_verify(self, n=5):
        """抽样验证转换结果"""
        print(f"\n=== 抽样验证（前 {n} 条原始值最小的 > 0 记录）===")
        positive_data = self.gdf[self.gdf[self.field_name] > 0]
        if len(positive_data) > 0:
            sample_small = positive_data.nsmallest(n, self.field_name)[
                [self.field_name, self.output_field_name]
            ]
            print(sample_small.to_string(index=False))

        print(f"\n=== 抽样验证（前 {n} 条原始值最大的 > 0 记录）===")
        if len(positive_data) > 0:
            sample_large = positive_data.nlargest(n, self.field_name)[
                [self.field_name, self.output_field_name]
            ]
            print(sample_large.to_string(index=False))

        return self

    def save(self):
        """保存结果到新的 GeoPackage 文件"""
        print(f"\n正在保存结果到: {self.output_path}")

        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.gdf.to_file(self.output_path, driver="GPKG")
        print("✅ 保存完成！")
        return self

    def run(self):
        """执行完整的处理流程"""
        try:
            self.read_data() \
                .validate_field() \
                .compute_stats() \
                .transform() \
                .sample_verify() \
                .save()
            print("\n🎉 处理成功完成！")
            return self.gdf
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            raise


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = r"F:\360安全云盘同步版\02-我的论文\数据集构建\Data\地图数据\地标库\Beijing_poi_crossroads.gpkg"
    OUTPUT_FILE = r"F:\360安全云盘同步版\02-我的论文\数据集构建\Data\地图数据\地标库\Beijing_poi_crossroads_exposure.gpkg"
    FIELD_NAME = "fid_count"  # 要处理的字段名
    OUTPUT_FIELD = "exposure"  # 处理后的新字段名

    # 创建工具类实例并执行
    scaler = PercentileScaler(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        field_name=FIELD_NAME,
        output_field_name=OUTPUT_FIELD
    )

    result_gdf = scaler.run()

    # 如果需要，可以继续使用 result_gdf 进行后续操作
    # print(result_gdf.head())