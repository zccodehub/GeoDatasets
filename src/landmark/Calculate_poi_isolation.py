import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings('ignore')


class IsolationScorer:
    """
    计算每个POI在其类别中的孤立度（到最近3个同类POI的平均距离）
    支持类型字段包含 "|" 分割的多类型（自动拆分为多行计算，然后按fid聚合取平均值）
    最终结果映射到 0 或 50-100 区间

    参数:
        input_path (str): 输入 GeoPackage 文件路径
        output_path (str): 输出 GeoPackage 文件路径
        category_field (str): 类别字段名（如 'type'），支持 "|" 分割
        id_field (str): 唯一标识字段名，默认 'fid'
        output_field_name (str): 输出字段名，默认 'isolation_score'
        n_neighbors (int): 最近邻居数量，默认 3
        aggregate_func (str): 聚合方式，'mean' 或 'max'，默认 'mean'
    """

    def __init__(self, input_path, output_path, category_field,
                 id_field='fid', output_field_name='isolation_score',
                 n_neighbors=3, aggregate_func='mean'):
        self.input_path = input_path
        self.output_path = output_path
        self.category_field = category_field
        self.id_field = id_field
        self.output_field_name = output_field_name
        self.n_neighbors = n_neighbors
        self.aggregate_func = aggregate_func
        self.gdf = None
        self.stats = {}
        self.original_columns = []  # 保存原始字段列表

    def read_and_split_data(self):
        """读取数据，拆分多类型字段为多行"""
        print(f"正在读取数据: {self.input_path}")
        self.gdf = gpd.read_file(self.input_path)

        # 保存原始字段列表（用于后续恢复）
        self.original_columns = [col for col in self.gdf.columns if col != 'geometry']

        # 确保 fid 字段存在且为整数类型
        if self.id_field not in self.gdf.columns:
            if self.gdf.index.name == self.id_field or self.gdf.index.name is None:
                self.gdf = self.gdf.reset_index()
                if 'index' in self.gdf.columns:
                    self.gdf = self.gdf.rename(columns={'index': self.id_field})
                else:
                    self.gdf[self.id_field] = range(1, len(self.gdf) + 1)
            else:
                for col in self.gdf.columns:
                    if col.lower() == self.id_field.lower():
                        self.gdf = self.gdf.rename(columns={col: self.id_field})
                        break
                else:
                    self.gdf[self.id_field] = range(1, len(self.gdf) + 1)
        else:
            # 确保 fid 是整数类型
            self.gdf[self.id_field] = self.gdf[self.id_field].astype(int)

        print(f"原始数据: {len(self.gdf)} 条记录")

        # 检查坐标系
        if self.gdf.crs and self.gdf.crs.is_geographic:
            print("检测到地理坐标系，正在投影到 UTM 50N (EPSG:32650)...")
            self.gdf = self.gdf.to_crs("EPSG:32650")

        # 拆分类型字段（支持 "|" 分割）
        if self.category_field in self.gdf.columns:
            # 处理空值
            self.gdf[self.category_field] = self.gdf[self.category_field].fillna('unknown')
            # 将类型字段拆分为列表
            self.gdf['_type_list'] = self.gdf[self.category_field].str.split('|')
            # 展开为多行
            self.gdf = self.gdf.explode('_type_list')
            # 重命名展开后的类型字段
            self.gdf = self.gdf.rename(columns={self.category_field: 'original_type'})
            self.gdf = self.gdf.rename(columns={'_type_list': self.category_field})
            # 去除可能存在的空字符串或空白
            self.gdf = self.gdf[self.gdf[self.category_field].str.strip() != '']
            self.gdf = self.gdf[self.gdf[self.category_field].notna()]
        else:
            print(f"警告：字段 '{self.category_field}' 不存在，使用全量数据计算")
            self.gdf['_temp_type'] = 'all'
            self.gdf = self.gdf.rename(columns={'_temp_type': self.category_field})

        print(f"拆分后: {len(self.gdf)} 条记录")
        print(f"类别数量: {self.gdf[self.category_field].nunique()}")
        return self

    def _compute_isolation_for_category(self, coords, indices):
        """计算单个类别中每个点的孤立度"""
        n = len(coords)
        results = {}

        if n == 0:
            return results

        if n == 1:
            results[indices[0]] = np.inf
            return results

        if n == 2:
            if len(indices) >= 2:
                dist = np.linalg.norm(coords[0] - coords[1])
                results[indices[0]] = dist
                results[indices[1]] = dist
            else:
                results[indices[0]] = 0
            return results

        tree = cKDTree(coords)

        for i, idx in enumerate(indices):
            k = min(self.n_neighbors + 1, n)
            distances, neighbor_indices = tree.query(coords[i], k=k)

            valid_distances = []
            for d, ni in zip(distances, neighbor_indices):
                if d > 1e-9 and ni != i:
                    valid_distances.append(d)
                if len(valid_distances) >= self.n_neighbors:
                    break

            if len(valid_distances) < self.n_neighbors:
                avg_dist = np.mean(valid_distances) if valid_distances else 0
            else:
                avg_dist = np.mean(valid_distances[:self.n_neighbors])

            results[idx] = avg_dist

        return results

    def compute_isolation(self):
        """计算所有POI的孤立度"""
        print(f"\n正在计算孤立度（每个POI到最近{self.n_neighbors}个同类POI的平均距离）...")

        self.gdf['_isolation_raw'] = 0.0

        categories = self.gdf[self.category_field].unique()
        print(f"共有 {len(categories)} 个类别")

        processed = 0
        all_results = {}
        single_category_count = 0

        for cat in categories:
            mask = self.gdf[self.category_field] == cat
            cat_indices = self.gdf[mask].index.tolist()

            if len(cat_indices) == 0:
                continue

            if len(cat_indices) == 1:
                all_results[cat_indices[0]] = np.inf
                single_category_count += 1
                processed += 1
                continue

            coords = np.array([
                [geom.x, geom.y]
                for geom in self.gdf.loc[cat_indices].geometry
            ])

            if len(coords) != len(cat_indices):
                min_len = min(len(coords), len(cat_indices))
                coords = coords[:min_len]
                cat_indices = cat_indices[:min_len]

            cat_results = self._compute_isolation_for_category(coords, cat_indices)
            all_results.update(cat_results)
            processed += len(cat_indices)

            if processed % 10000 == 0:
                print(f"已处理 {processed:,} 条记录...")

        if single_category_count > 0:
            print(f"共有 {single_category_count} 个POI属于只有1个POI的类别")

        for idx, value in all_results.items():
            self.gdf.loc[idx, '_isolation_raw'] = value

        inf_count = (self.gdf['_isolation_raw'] == np.inf).sum()
        if inf_count > 0:
            print(f"处理 {inf_count} 个无限值（只有1个POI的类别）...")
            max_finite = self.gdf[self.gdf['_isolation_raw'] < np.inf]['_isolation_raw'].max()
            if max_finite > 0 and np.isfinite(max_finite):
                self.gdf.loc[self.gdf['_isolation_raw'] == np.inf, '_isolation_raw'] = max_finite * 2
            else:
                self.gdf.loc[self.gdf['_isolation_raw'] == np.inf, '_isolation_raw'] = 1000

        print(f"孤立度计算完成，共 {processed:,} 条记录")
        return self

    def aggregate_by_fid(self):
        """
        按 fid 聚合，保留所有原始属性
        取孤立度的平均值或最大值
        """
        print(f"\n正在按 {self.id_field} 聚合（{self.aggregate_func}）...")

        # 确保 fid 是整数类型
        self.gdf[self.id_field] = self.gdf[self.id_field].astype(int)

        # 提取几何列名
        geom_col = 'geometry'
        if geom_col not in self.gdf.columns:
            raise ValueError("数据中没有 geometry 列，无法进行空间操作")

        # 保存 fid 的原始值（用于后续验证）
        original_fids = self.gdf[self.id_field].copy()

        # 构建聚合规则
        # 所有非几何、非孤立度字段都用 'first'
        agg_rules = {}
        for col in self.gdf.columns:
            if col == geom_col:
                continue
            elif col == '_isolation_raw':
                agg_rules[col] = self.aggregate_func
            elif col == self.id_field:
                continue  # fid 作为分组键，不聚合
            else:
                agg_rules[col] = 'first'

        # 执行分组聚合，不包含 fid 字段（作为分组键）
        aggregated = self.gdf.groupby(self.id_field).agg(agg_rules).reset_index()

        # 确保 reset_index() 后的列名正确
        # 如果 reset_index 产生了 'index' 列，重命名为 fid
        if 'index' in aggregated.columns and self.id_field not in aggregated.columns:
            aggregated = aggregated.rename(columns={'index': self.id_field})

        # 处理 geometry 列
        # 单独提取每个 fid 的第一个几何
        geom_agg = self.gdf.groupby(self.id_field)[geom_col].first().reset_index()
        geom_agg.columns = [self.id_field, geom_col]

        # 合并属性数据和几何数据
        final_df = pd.merge(aggregated, geom_agg, on=self.id_field, how='inner')

        # 创建 GeoDataFrame
        self.gdf = gpd.GeoDataFrame(final_df, crs=self.gdf.crs, geometry=geom_col)

        # 验证 fid 是否正确
        print(f"聚合后: {len(self.gdf)} 条记录")
        print(f"字段列表: {list(self.gdf.columns)}")
        if self.id_field in self.gdf.columns:
            print(f"fid 范围: {self.gdf[self.id_field].min()} - {self.gdf[self.id_field].max()}")

        return self

    def compute_stats(self):
        """计算并输出原始数据统计信息"""
        raw = self.gdf['_isolation_raw']
        positive_mask = raw > 0
        positive_count = positive_mask.sum()
        total = len(self.gdf)

        self.stats = {
            'total': total,
            'positive_count': positive_count,
            'positive_ratio': positive_count / total if total > 0 else 0,
            'max_value': raw.max(),
            'min_value': raw[raw > 0].min() if positive_count > 0 else 0,
            'mean_value': raw[raw > 0].mean() if positive_count > 0 else 0,
            'median_value': raw[raw > 0].median() if positive_count > 0 else 0
        }

        print(f"\n=== 原始孤立度统计（聚合后） ===")
        print(f"总记录数: {self.stats['total']:,}")
        print(f"距离 > 0 的记录数: {self.stats['positive_count']:,} ({self.stats['positive_ratio'] * 100:.1f}%)")
        print(f"距离 = 0 的记录数: {self.stats['total'] - self.stats['positive_count']:,}")
        print(f"最大值: {self.stats['max_value']:.2f} 米")
        if self.stats['positive_count'] > 0:
            print(f"最小值: {self.stats['min_value']:.2f} 米")
            print(f"平均值: {self.stats['mean_value']:.2f} 米")
            print(f"中位数: {self.stats['median_value']:.2f} 米")

        return self

    def scale_to_50_100(self):
        """将原始距离通过百分位秩映射到 50-100 区间"""
        print(f"\n正在将孤立度映射到 50-100 区间...")

        self.gdf[self.output_field_name] = 0.0

        positive_mask = self.gdf['_isolation_raw'] > 0
        positive_count = positive_mask.sum()

        if positive_count == 0:
            print("警告：没有找到距离 > 0 的记录，所有值将被设为 0")
            return self

        positive_data = self.gdf[positive_mask].copy()
        positive_data_sorted = positive_data.sort_values('_isolation_raw')
        n = positive_data_sorted.shape[0]

        if n > 1:
            percentile_ranks = np.arange(n) / (n - 1) * 100
        else:
            percentile_ranks = np.array([50])

        mapped_scores = 50 + percentile_ranks / 2

        positive_indices = positive_data_sorted.index
        self.gdf.loc[positive_indices, self.output_field_name] = mapped_scores

        scaled_positive = (self.gdf[self.output_field_name] > 0).sum()
        print(f"转换后 > 0 的记录数: {scaled_positive:,}")
        print(f"转换后最大值: {self.gdf[self.output_field_name].max():.4f}")
        print(f"转换后最小值: {self.gdf[self.output_field_name].min():.4f}")
        print(f"转换后平均值: {self.gdf[self.output_field_name].mean():.4f}")

        if positive_count != scaled_positive:
            print(f"⚠️ 警告：原始 > 0 的记录数 ({positive_count:,}) 与转换后 > 0 的记录数 ({scaled_positive:,}) 不一致！")
        else:
            print(f"✅ 数据一致性检查通过：{positive_count:,} 条 > 0 的记录全部保留。")

        return self

    def sample_verify(self, n=5):
        """抽样验证转换结果"""
        print(f"\n=== 抽样验证 ===")
        positive_data = self.gdf[self.gdf['_isolation_raw'] > 0]

        if len(positive_data) > 0:
            print(f"距离最小的 {n} 条记录（最扎堆）：")
            cols_to_show = [col for col in [self.id_field, '_isolation_raw', self.output_field_name] if
                            col in self.gdf.columns]
            sample_small = positive_data.nsmallest(n, '_isolation_raw')[cols_to_show]
            print(sample_small.to_string(index=False))

            print(f"\n距离最大的 {n} 条记录（最孤立）：")
            sample_large = positive_data.nlargest(n, '_isolation_raw')[cols_to_show]
            print(sample_large.to_string(index=False))

        return self

    def clean_temp_fields(self):
        """删除临时字段"""
        temp_fields = ['_isolation_raw', 'original_type', '_type_list']
        for field in temp_fields:
            if field in self.gdf.columns:
                self.gdf = self.gdf.drop(columns=[field])
        return self

    def save(self):
        """保存结果到新的 GeoPackage 文件"""
        print(f"\n正在保存结果到: {self.output_path}")

        import os
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.gdf.to_file(self.output_path, driver="GPKG")
        print("✅ 保存完成！")
        return self

    def run(self):
        """执行完整的处理流程"""
        try:
            self.read_and_split_data() \
                .compute_isolation() \
                .aggregate_by_fid() \
                .compute_stats() \
                .scale_to_50_100() \
                .sample_verify() \
                .save()
            print("\n🎉 孤立度计算完成！")
            return self.gdf
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            raise


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = r"F:\360安全云盘同步版\02-我的论文\数据集构建\Data\QGIS数据\Beijing_POI_WGS84_UTM.gpkg"
    OUTPUT_FILE = r"F:\360安全云盘同步版\02-我的论文\数据集构建\Data\QGIS数据\地标库\Beijing_POI_with_isolation.gpkg"

    CATEGORY_FIELD = "类型编码"  #类型字段名（支持 "|" 分割）
    ID_FIELD = "fid"  # 唯一标识字段名

    # 创建工具类实例并执行
    scorer = IsolationScorer(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        category_field=CATEGORY_FIELD,
        id_field=ID_FIELD,
        output_field_name="isolation_scaled",
        n_neighbors=3,
        aggregate_func='mean'  # 'mean' 或 'max'
    )

    result_gdf = scorer.run()