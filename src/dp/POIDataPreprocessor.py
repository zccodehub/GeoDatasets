import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from typing import Tuple, Optional, Set


class POIDataPreprocessor:
    """
    POI数据预处理器
    包含数据清洗、质量过滤、标准化等完整流程
    """

    def __init__(self, file_path: str = '', adcode:str = '', del_type_file:str = ''):
        """
        初始化预处理器

        Args:
            file_path: POI数据Excel文件路径
        """
        self.file_path = file_path
        self.df = None
        self.adcode = adcode
        self.del_type_file = del_type_file
        self.del_types = set()  # 需要删除的类型编码集合
        self.filtered_records = []  # 记录被过滤的数据及其原因
        self.quality_report = {}  # 质量报告

    # ==================== 1. 数据加载与初步探查 ====================

    def load_data(self) -> pd.DataFrame:
        """加载Excel数据"""
        self.df = pd.read_excel(self.file_path, sheet_name=0)
        print(f"✅ 数据加载成功，共 {len(self.df)} 条记录")
        return self.df

    def inspect_data(self) -> dict:
        """初步探查数据结构"""
        report = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'sample': self.df.head(3).to_dict('records')
        }
        print(f"📊 数据列数: {report['total_columns']}")
        print(f"📊 各列缺失值: {report['missing_values']}")
        return report

    def load_del_types(self) -> Set[str]:
        """
        从Excel文件中加载需要删除的类型编码

        Returns:
            需要删除的类型编码集合
        """
        print(f"目录是{self.del_type_file}")
        if not self.del_type_file or not os.path.exists(self.del_type_file):
            print("⚠️ 未指定类型删除文件或文件不存在，跳过类型过滤")
            return set()

        try:
            # 读取poi_types_del表
            df_del = pd.read_excel(self.del_type_file, sheet_name=0)

            # 检查是否存在NEW_TYPE列
            if 'NEW_TYPE' not in df_del.columns:
                # 尝试查找可能的列名
                possible_cols = ['NEW_TYPE', '类型编码', '编码', 'code']
                found_col = None
                for col in possible_cols:
                    if col in df_del.columns:
                        found_col = col
                        break

                if found_col is None:
                    print(f"⚠️ 未找到NEW_TYPE列，当前列名: {df_del.columns.tolist()}")
                    return set()

                del_types = set(df_del[found_col].astype(str).str.strip().tolist())
            else:
                del_types = set(df_del['NEW_TYPE'].astype(str).str.strip().tolist())

            print(f"✅ 加载需要删除的类型编码: {len(del_types)} 个")
            # 打印前10个作为示例
            sample = list(del_types)[:10]
            print(f"   示例: {sample}")
            return del_types

        except Exception as e:
            print(f"⚠️ 加载类型删除文件失败: {e}")
            return set()

    # ==================== 2. 重复值处理 ====================

    def deduplicate_by_poi_id(self) -> int:
        """
        基于POI ID去重

        Returns:
            移除的重复记录数
        """
        before = len(self.df)
        # 记录被过滤的POI ID
        # dup_ids = self.df[self.df['POI ID'].duplicated(keep=False)]['POI ID'].unique()
        # for dup_id in dup_ids:
        #     self._add_filter_record('POI ID重复', dup_id, f'重复ID: {dup_id}')

        # 找到所有重复的POI ID
        duplicate_mask = self.df['POI ID'].duplicated(keep=False)
        duplicate_ids = self.df[duplicate_mask]['POI ID'].unique()

        # 对每个重复ID，只记录除第一条外的记录
        for dup_id in duplicate_ids:
            # 获取该ID的所有行
            id_rows = self.df[self.df['POI ID'] == dup_id]
            # 跳过第一条，记录后面的所有重复项
            for idx in id_rows.index[1:]:  # 注意：跳过第一条
                self._add_filter_record(
                    'POI ID重复',
                    dup_id,
                    f'重复ID: {dup_id} (保留第一条)'
                )

        self.df = self.df.drop_duplicates(subset=['POI ID'], keep='first')
        removed = before - len(self.df)
        print(f"🗑️ 基于POI ID去重: 移除 {removed} 条重复记录")
        return removed

    def deduplicate_by_location(self, threshold_meters: float = 1.0) -> int:
        """
        基于地理位置去重（坐标距离小于阈值的视为重复）

        Args:
            threshold_meters: 距离阈值（米），默认1米

        Returns:
            移除的重复记录数
        """
        from geopy.distance import geodesic

        before = len(self.df)
        to_drop = set()

        # 按名称分组，降低计算复杂度
        grouped = self.df.groupby('名称')

        # for name, group in grouped:
        #     if len(group) <= 1:
        #         continue
        #
        #     coords = group[['纬度', '经度']].values
        #     indices = group.index.tolist()
        #
        #     # 两两比较同一名称的POI距离
        #     for i in range(len(coords)):
        #         for j in range(i + 1, len(coords)):
        #             dist = geodesic(coords[i], coords[j]).meters
        #             if dist <= threshold_meters:
        #                 # 保留第一个，标记后续为重复
        #                 to_drop.add(indices[j])
        #                 self._add_filter_record(
        #                     '位置重复',
        #                     self.df.loc[indices[j], 'POI ID'],
        #                     f'与 "{name}" 距离 {dist:.2f}m'
        #                 )

        for name, group in grouped:
            if len(group) <= 1:
                continue

            coords = group[['纬度', '经度']].values
            indices = group.index.tolist()

            # 记录每个索引是否已经被标记
            marked = set()

            # 两两比较
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    dist = geodesic(coords[i], coords[j]).meters
                    if dist <= threshold_meters:
                        # 标记后续为重复
                        if indices[j] not in marked:
                            to_drop.add(indices[j])
                            marked.add(indices[j])
                            self._add_filter_record(
                                '位置重复',
                                self.df.loc[indices[j], 'POI ID'],
                                f'与 "{name}" 距离 {dist:.2f}m'
                            )

        self.df = self.df.drop(index=to_drop)
        removed = before - len(self.df)
        print(f"🗑️ 基于位置去重: 移除 {removed} 条重复记录")
        return removed

    # ==================== 3. 坐标异常值过滤 ====================

    def filter_coordinate_nulls(self) -> int:
        """过滤经纬度为空值的记录"""
        before = len(self.df)
        mask = self.df['经度'].isnull() | self.df['纬度'].isnull()
        null_ids = self.df[mask]['POI ID'].tolist()
        for pid in null_ids:
            self._add_filter_record('坐标为空', pid, '经度或纬度为NULL')

        self.df = self.df[~mask]
        removed = before - len(self.df)
        print(f"🗑️ 过滤坐标空值: 移除 {removed} 条记录")
        return removed

    def filter_zero_coordinates(self) -> int:
        """过滤经纬度为(0,0)的记录"""
        before = len(self.df)
        mask = (self.df['经度'] == 0) & (self.df['纬度'] == 0)
        zero_ids = self.df[mask]['POI ID'].tolist()
        for pid in zero_ids:
            self._add_filter_record('坐标为(0,0)', pid, '经度=0且纬度=0')

        self.df = self.df[~mask]
        removed = before - len(self.df)
        print(f"🗑️ 过滤零坐标: 移除 {removed} 条记录")
        return removed

    def filter_coordinate_range(self) -> int:
        """
        过滤坐标超出北京市合理范围的记录
        北京范围: 经度115.4-117.5°E, 纬度39.4-41.0°N
        """
        before = len(self.df)

        # 北京范围校验
        beijing_mask = (
                (self.df['经度'] < 115.4) | (self.df['经度'] > 117.5) |
                (self.df['纬度'] < 39.4) | (self.df['纬度'] > 41.0)
        )

        invalid_ids = self.df[beijing_mask]['POI ID'].tolist()
        for pid in invalid_ids:
            row = self.df[self.df['POI ID'] == pid].iloc[0]
            self._add_filter_record(
                '坐标超出范围',
                pid,
                f'经度={row["经度"]}, 纬度={row["纬度"]}'
            )

        self.df = self.df[~beijing_mask]
        removed = before - len(self.df)
        print(f"🗑️ 过滤坐标超范围: 移除 {removed} 条记录")
        return removed

    # ==================== 4. 地址清洗 ====================

    def filter_address_nulls(self) -> int:
        """过滤地址为空值的记录"""
        before = len(self.df)
        mask = self.df['地址'].isnull()
        null_ids = self.df[mask]['POI ID'].tolist()
        for pid in null_ids:
            self._add_filter_record('地址为空', pid, '地址字段为NULL')

        self.df = self.df[~mask]
        removed = before - len(self.df)
        print(f"🗑️ 过滤地址空值: 移除 {removed} 条记录")
        return removed

    def normalize_address(self) -> None:
        """
        地址规范化处理
        - 去除首尾空格
        - 替换"北京城区"为"北京市"
        - 提取地标信息（可选）
        """

        def clean_addr(addr: str) -> str:
            if pd.isna(addr):
                return addr
            # 去除首尾空格
            addr = addr.strip()
            # 替换冗余表述
            addr = addr.replace('北京城区', '北京市')
            # 去除多余空格
            addr = re.sub(r'\s+', ' ', addr)
            return addr

        self.df['地址'] = self.df['地址'].apply(clean_addr)
        print("✅ 地址规范化完成")

    def extract_district_info(self) -> None:
        """
        从地址中提取区域信息（如街道、商圈）
        可作为后续特征
        """
        # 预定义区域关键词
        districts = ['中关村', '上地', '五棵松', '四季青', '清河', '西三旗',
                     '万柳', '世纪城', '牡丹园', '学院路', '北太平庄', '甘家口']

        def extract_region(addr: str) -> str:
            if pd.isna(addr):
                return ''
            for region in districts:
                if region in addr:
                    return region
            return ''

        self.df['区域'] = self.df['地址'].apply(extract_region)
        print("✅ 区域信息提取完成")

    # ==================== 5. 停业状态过滤 ====================

    def filter_closed_business(self) -> int:
        """
        过滤名称中标记为停业/关闭的POI

        关键词: 暂停营业, 已停业, 关闭, 装修中, 暂停服务
        """
        before = len(self.df)

        # 定义停业关键词
        closed_keywords = ['暂停营业', '已停业', '关闭', '装修中', '暂停服务']

        # 构建正则表达式
        pattern = '|'.join(closed_keywords)
        mask = self.df['名称'].str.contains(pattern, na=False)

        closed_ids = self.df[mask]['POI ID'].tolist()
        for pid in closed_ids:
            name = self.df[self.df['POI ID'] == pid].iloc[0]['名称']
            self._add_filter_record('已停业', pid, f'名称包含停业关键词: {name}')

        self.df = self.df[~mask]
        removed = before - len(self.df)
        print(f"🗑️ 过滤停业POI: 移除 {removed} 条记录")
        return removed

    # ==================== 6. 类型字段解析 ====================

    def parse_category(self) -> None:
        """
        解析类型字段，拆分为三级分类
        格式: "一级;二级;三级"
        """

        def split_category(cat_str: str) -> Tuple[str, str, str]:
            if pd.isna(cat_str) or cat_str == '':
                return ('', '', '')

            parts = cat_str.split(';')
            # 处理最多三级
            levels = ['', '', '']
            for i, part in enumerate(parts[:3]):
                levels[i] = part.strip()

            # 如果只有一级，其他为空
            return tuple(levels)

        # 使用apply拆分为三列
        self.df[['type_1', 'type_2', 'type_3']] = self.df['类型'].apply(
            lambda x: pd.Series(split_category(x))
        )
        print("✅ 类型字段解析完成")

    def filter_empty_category(self) -> int:
        """过滤类型字段完全为空的记录"""
        before = len(self.df)
        mask = self.df['类型'].isnull() | (self.df['类型'] == '')
        empty_ids = self.df[mask]['POI ID'].tolist()
        for pid in empty_ids:
            self._add_filter_record('类型为空', pid, '类型字段为空')

        self.df = self.df[~mask]
        removed = before - len(self.df)
        print(f"🗑️ 过滤类型为空: 移除 {removed} 条记录")
        return removed

    # ==================== 7. 无效POI过滤 ====================

    def filter_meaningless_poi(self) -> int:
        """
        过滤名称无意义的POI（如纯机构名且无具体服务内容）
        """
        before = len(self.df)

        # 定义无意义关键词
        meaningless_keywords = ['测试', 'test', '中国邮政', '国家电网', '中国移动', '中国联通', '中国电信']

        # 需要结合类型判断，如果类型也非常泛化则过滤
        mask = pd.Series([False] * len(self.df))
        for keyword in meaningless_keywords:
            # 名称包含关键词，且type_2为空或为"公司"
            name_mask = self.df['名称'].str.contains(keyword, na=False)
            type_mask = self.df['type_2'].isnull() | (self.df['type_2'] == '') | (self.df['type_2'] == '公司')
            mask = mask | (name_mask & type_mask)

        filtered_ids = self.df[mask]['POI ID'].tolist()
        for pid in filtered_ids:
            name = self.df[self.df['POI ID'] == pid].iloc[0]['名称']
            self._add_filter_record('无意义POI', pid, f'名称: {name}')

        self.df = self.df[~mask]
        removed = before - len(self.df)
        print(f"🗑️ 过滤无意义POI: 移除 {removed} 条记录")
        return removed

    # ==================== 8. 数据标准化 ====================

    def standardize_data_types(self) -> None:
        """标准化各列数据类型"""
        # 坐标列转换为float64
        self.df['经度'] = self.df['经度'].astype('float64')
        self.df['纬度'] = self.df['纬度'].astype('float64')

        # 字符串列转换为string类型
        string_cols = ['POI ID', '名称', '地址', '类型', 'type_1', 'type_2', 'type_3', '区域']
        for col in string_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('string')

        print("✅ 数据类型标准化完成")

    def round_coordinates(self, decimals: int = 6) -> None:
        """保留经纬度指定小数位数"""
        self.df['经度'] = self.df['经度'].round(decimals)
        self.df['纬度'] = self.df['纬度'].round(decimals)
        print(f"✅ 经纬度保留 {decimals} 位小数")

    def add_status_column(self) -> None:
        """添加数据状态标记列"""
        self.df['data_status'] = 'active'
        print("✅ 添加状态标记列")

    # ==================== 9. 按类型编码过滤 ====================
    def filter_by_type_code(self, del_types: Set[str] = None) -> int:
        """
        根据类型编码过滤记录
        如果类型编码包含在删除列表中，则移除该记录

        Args:
            del_types: 需要删除的类型编码集合，如果为None则使用self.del_types

        Returns:
            移除的记录数
        """
        if del_types is None:
            del_types = self.del_types

        if not del_types:
            print("⚠️ 没有需要删除的类型编码，跳过类型过滤")
            return 0

        before = len(self.df)

        # 检查是否存在类型编码列
        if '类型编码' not in self.df.columns:
            print("⚠️ 数据中不存在'类型编码'列，跳过类型过滤")
            return 0

        # 定义判断函数：如果类型编码包含任何需要删除的编码，则返回True（需要删除）
        def should_remove(type_codes) -> bool:
            if pd.isna(type_codes):
                return False
            # 按 | 分隔多个编码
            codes = str(type_codes).split('|')
            for code in codes:
                code_cleaned = code.strip()
                if code_cleaned in del_types:
                    return True
            return False

        # 找出需要删除的记录
        remove_mask = self.df['类型编码'].apply(should_remove)
        remove_ids = self.df[remove_mask]['POI ID'].tolist()

        # 记录被过滤的详细信息
        for pid in remove_ids:
            row = self.df[self.df['POI ID'] == pid].iloc[0]
            type_codes = row.get('类型编码', '')
            # 找出具体匹配的编码
            matched_codes = []
            if not pd.isna(type_codes):
                codes = str(type_codes).split('|')
                for code in codes:
                    code_cleaned = code.strip()
                    if code_cleaned in del_types:
                        matched_codes.append(code_cleaned)
            self._add_filter_record(
                '类型编码被删除',
                pid,
                f'匹配的类型编码: {", ".join(matched_codes)} (原类型编码: {type_codes})'
            )

        # 执行删除
        self.df = self.df[~remove_mask]
        removed = before - len(self.df)
        print(f"🗑️ 按类型编码过滤: 移除 {removed} 条记录 (删除类型编码数: {len(del_types)})")
        return removed

    # ==================== 10. 辅助函数 ====================

    def _add_filter_record(self, reason: str, poi_id: str, detail: str) -> None:
        """记录被过滤的数据"""
        self.filtered_records.append({
            'POI ID': poi_id,
            '过滤原因': reason,
            '详情': detail
        })

    def _add_column(self) -> None:
        self.df['citycode'] = '110000'
        self.df['adcode'] = self.adcode

    def _update_quality_report(self, step_name: str, removed_count: int) -> None:
        """更新质量报告"""
        self.quality_report[step_name] = removed_count

    # ==================== 11. 完整流程执行 ====================

    def run_full_pipeline(self) -> pd.DataFrame:
        """
        执行完整的预处理流程

        Returns:
            清洗后的DataFrame
        """
        print("=" * 60)
        print("🚀 开始POI数据预处理")
        print("=" * 60)

        # 1. 加载数据
        self.load_data()
        original_count = len(self.df)

        # 2. 初步探查
        self.inspect_data()

        # 3. 重复值处理
        self.deduplicate_by_poi_id()
        self.deduplicate_by_location()

        # 4. 坐标异常过滤
        self.filter_coordinate_nulls()
        self.filter_zero_coordinates()
        self.filter_coordinate_range()

        # 5. 地址清洗
        self.filter_address_nulls()
        self.normalize_address()
        #self.extract_district_info()

        # 6. 停业状态过滤
        self.filter_closed_business()

        # 7. 类型解析
        self.filter_empty_category()
        self.parse_category()

        # 8. 无效POI过滤
        self.filter_meaningless_poi()

        # 9. 按类型编码过滤
        self.filter_by_type_code()

        # 10. 数据标准化
        #self.standardize_data_types()
        #self.round_coordinates(decimals=6)
        #self.add_status_column()
        self._add_column()

        # 11. 生成报告
        final_count = len(self.df)
        print("=" * 60)
        print(f"✅ 预处理完成!")
        print(f"📊 原始记录数: {original_count}")
        print(f"📊 清洗后记录数: {final_count}")
        print(f"📊 移除记录数: {original_count - final_count}")
        print(f"📊 过滤记录详情: {len(self.filtered_records)} 条被标记")
        print("=" * 60)

        return self.df

    # ==================== 11. 结果导出 ====================

    def save_cleaned_data(self, output_path: str) -> None:
        """
        保存清洗后的数据

        Args:
            output_path: 输出文件路径（支持.csv或.xlsx）
        """
        if output_path.endswith('.csv'):
            self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif output_path.endswith('.xlsx'):
            self.df.to_excel(output_path, index=False)
        else:
            self.df.to_csv(f'{output_path}.csv', index=False, encoding='utf-8-sig')
        print(f"💾 清洗数据已保存至: {output_path}")

    def save_filtered_log(self, output_path: str) -> None:
        """
        保存被过滤的记录日志

        Args:
            output_path: 日志文件路径
        """
        if not self.filtered_records:
            print("⚠️ 没有被过滤的记录")
            return

        log_df = pd.DataFrame(self.filtered_records)
        if output_path.endswith('.csv'):
            log_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif output_path.endswith('.xlsx'):
            log_df.to_excel(output_path, index=False)
        else:
            log_df.to_csv(f'{output_path}.csv', index=False, encoding='utf-8-sig')
        print(f"📋 过滤日志已保存至: {output_path}")

    def print_quality_report(self) -> None:
        """打印质量报告"""
        print("\n" + "=" * 60)
        print("📋 数据质量报告")
        print("=" * 60)

        original_count = len(self.df) + len(self.filtered_records)
        final_count = len(self.df)

        print(f"原始记录数: {original_count}")
        print(f"清洗后记录数: {final_count}")
        print(f"有效数据率: {final_count / original_count * 100:.2f}%")

        print("\n各字段缺失情况:")
        for col in ['名称', '地址', '经度', '纬度', '类型', 'type_1', 'type_2', 'type_3']:
            if col in self.df.columns:
                missing = self.df[col].isnull().sum()
                print(f"  {col}: {missing} 条缺失")

        print("\n各分类数量:")
        if 'type_1' in self.df.columns:
            print(self.df['type_1'].value_counts().head(10).to_string())

        print("=" * 60)

    # ==================== 12. 处理整个目录 ====================

    def load_adcode_mapping(self,adcode_file_path):
        """
        加载adcode映射表

        参数:
            adcode_file_path: adcode Excel文件路径

        返回:
            dict: 区名到adcode的映射字典
        """
        try:
            df = pd.read_excel(adcode_file_path)

            # 检查必要的列是否存在
            if '中文名' not in df.columns or 'adcode' not in df.columns:
                print(f"错误: Excel文件中缺少'中文名'或'adcode'列")
                print(f"当前列: {df.columns.tolist()}")
                return {}

            # 创建映射字典
            adcode_map = {}
            for _, row in df.iterrows():
                district_name = str(row['中文名']).strip()
                adcode = str(row['adcode']).strip()
                adcode_map[district_name] = adcode

            print(f"成功加载adcode映射，共 {len(adcode_map)} 个区")
            return adcode_map

        except Exception as e:
            print(f"加载adcode文件时出错: {e}")
            return {}

    def extract_district_from_filename(self,file_path):
        """
        从文件名中提取区名

        参数:
            file_path: 文件路径

        返回:
            str: 提取的区名，如果提取失败则返回None
        """
        # 获取文件名（不含扩展名）
        filename = Path(file_path).stem

        # 定义匹配模式：以区名结尾，后面跟着"_POI数据"
        # 匹配模式：任意字符 + _POI数据
        pattern = r'(.+?)_POI数据$'
        match = re.search(pattern, filename)

        if match:
            district = match.group(1)
            print(f"从文件名 '{filename}' 中提取区名: {district}")
            return district

        # 尝试其他可能的模式
        # 模式2: 区名直接作为文件名（不含_POI数据后缀）
        # 检查文件名是否包含"区"字
        if '区' in filename:
            # 尝试提取包含"区"的部分
            district_match = re.search(r'([\u4e00-\u9fa5]+区)', filename)
            if district_match:
                district = district_match.group(1)
                print(f"从文件名 '{filename}' 中提取区名: {district}")
                return district

        print(f"无法从文件名 '{filename}' 中提取区名")
        return None

    def generate_output_path(self,input_file_path, suffix='_已清洗'):
        """
        生成输出文件路径

        参数:
            input_file_path: 输入文件路径
            suffix: 要添加的后缀

        返回:
            str: 输出文件路径
        """
        # 获取文件所在目录、文件名和扩展名
        file_dir = Path(input_file_path).parent
        file_stem = Path(input_file_path).stem
        file_ext = Path(input_file_path).suffix

        # 构建输出文件名: 原文件名 + 后缀 + 扩展名
        output_filename = f"{file_stem}{suffix}{file_ext}"
        output_path = file_dir / output_filename

        return str(output_path)

    def process_poi_files(self,directory_path, adcode_file_path, del_type_file=None):
        """
        遍历目录处理所有POI文件

        参数:
            directory_path: 包含POI文件的目录路径
            adcode_file_path: adcode映射Excel文件路径
        返回:
            dict: 处理结果统计
        """
        # 加载adcode映射
        adcode_map = self.load_adcode_mapping(adcode_file_path)
        if not adcode_map:
            print("错误: 无法加载adcode映射")
            return {}

        # 设置删除类型文件
        self.del_type_file = del_type_file
        self.del_types = self.load_del_types()

        # 获取所有POI文件
        poi_files = []
        for file in os.listdir(directory_path):
            if file.endswith('_POI数据.xlsx') or file.endswith('_POI数据.xls'):
                file_path = os.path.join(directory_path, file)
                poi_files.append(file_path)

        print(f"找到 {len(poi_files)} 个POI文件")

        # 处理结果统计
        results = {
            'total_files': len(poi_files),
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'details': []
        }

        # 遍历处理每个文件
        for file_path in poi_files:
            print(f"\n" + "=" * 50)
            print(f"处理文件: {os.path.basename(file_path)}")

            try:
                # 从文件名提取区名
                district = self.extract_district_from_filename(file_path)

                if not district:
                    print(f"跳过文件: 无法提取区名")
                    results['skipped'] += 1
                    results['details'].append({
                        'file': file_path,
                        'status': 'skipped',
                        'reason': '无法提取区名'
                    })
                    continue

                # 从映射中获取adcode
                if district not in adcode_map:
                    print(f"跳过文件: 区名 '{district}' 不在adcode映射中")
                    results['skipped'] += 1
                    results['details'].append({
                        'file': file_path,
                        'status': 'skipped',
                        'reason': f'区名 "{district}" 不在adcode映射中'
                    })
                    continue

                adcode = adcode_map[district]
                print(f"区名: {district}, adcode: {adcode}")

                # 1. 初始化预处理器
                self.file_path = file_path
                self.adcode = adcode
                cleaned_data_path = self.generate_output_path(file_path, '_已清洗')
                filter_log_path = self.generate_output_path(file_path, '_过滤日志');

                self.filtered_records = []
                self.quality_report = {}

                # 2. 执行完整预处理流程
                cleaned_df = self.run_full_pipeline()

                # 3. 保存结果
                self.save_cleaned_data(cleaned_data_path)
                self.save_filtered_log(filter_log_path)

                # 4. 打印质量报告
                self.print_quality_report()

            except Exception as e:
                print(f"处理文件时出错: {e}")
                results['errors'] += 1
                results['details'].append({
                    'file': file_path,
                    'status': 'error',
                    'error': str(e)
                })

        # 打印总结
        print(f"\n" + "=" * 50)
        print("处理完成!")
        print(f"总文件数: {results['total_files']}")
        print(f"成功处理: {results['processed']}")
        print(f"跳过: {results['skipped']}")
        print(f"错误: {results['errors']}")

        return results


if __name__ == "__main__":
    import os

    # 获取当前脚本所在目录的父目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '../../data')  # 根据你的实际目录结构调整

    # 构建文件路径
    file_path_dir = os.path.join(current_dir, '../../output')
    adcode_file_path = os.path.join(data_dir, 'adcode_beijing.xlsx')
    del_type_file = os.path.join(data_dir, 'poi_types_del.xlsx') # 需要删除的类型编码文件

    preprocessor = POIDataPreprocessor()
    preprocessor.process_poi_files(file_path_dir,adcode_file_path,del_type_file)


