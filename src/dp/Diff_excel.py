import pandas as pd
import numpy as np


def find_different_rows(file1, file2, sheet_name=0, key_columns=None, output_file=None):
    """
    找出两个Excel文件中不同的行

    参数:
        file1: 第一个Excel文件路径
        file2: 第二个Excel文件路径
        sheet_name: 工作表名称或索引，默认为第一个工作表
        key_columns: 用于比较的关键列列表，如果为None则比较所有列
        output_file: 输出文件路径，如果为None则返回DataFrame

    返回:
        包含差异信息的字典
    """

    # 读取两个Excel文件
    print(f"正在读取: {file1}")
    df1 = pd.read_excel(file1, sheet_name=sheet_name)

    print(f"正在读取: {file2}")
    df2 = pd.read_excel(file2, sheet_name=sheet_name)

    print(f"文件1形状: {df1.shape}, 文件2形状: {df2.shape}")

    # 如果指定了关键列，只使用这些列进行比较
    if key_columns is not None:
        # 检查关键列是否都存在
        missing_cols1 = [col for col in key_columns if col not in df1.columns]
        missing_cols2 = [col for col in key_columns if col not in df2.columns]
        if missing_cols1:
            print(f"警告: 文件1中缺少列: {missing_cols1}")
        if missing_cols2:
            print(f"警告: 文件2中缺少列: {missing_cols2}")

        # 只保留关键列（以及序号列用于标识）
        cols_to_keep = key_columns
        df1_compare = df1[cols_to_keep].copy()
        df2_compare = df2[cols_to_keep].copy()
    else:
        df1_compare = df1.copy()
        df2_compare = df2.copy()

    # 添加原始行索引以便追溯
    df1_compare['_原始行号'] = df1.index + 1
    df2_compare['_原始行号'] = df2.index + 1

    # 使用merge找出差异
    # 方法1: 找出在df1中但不在df2中的行（基于所有列）
    # 方法2: 找出在df2中但不在df1中的行

    # 为了比较，将所有列转换为字符串（处理NaN和不同数据类型）
    df1_str = df1_compare.astype(str)
    df2_str = df2_compare.astype(str)

    # 创建唯一标识键（用于比较）
    # 如果有关键列，使用关键列；否则使用所有列
    if key_columns is not None:
        compare_cols = key_columns
    else:
        compare_cols = [col for col in df1_compare.columns if col != '_原始行号']

    # 创建用于比较的元组
    df1_keys = df1_str[compare_cols].apply(lambda row: tuple(row), axis=1)
    df2_keys = df2_str[compare_cols].apply(lambda row: tuple(row), axis=1)

    # 找出差异
    only_in_df1_indices = []
    only_in_df2_indices = []

    # 创建df2的键集合
    df2_key_set = set(df2_keys)

    # 找出在df1中不在df2中的行
    for idx, key in enumerate(df1_keys):
        if key not in df2_key_set:
            only_in_df1_indices.append(idx)

    # 创建df1的键集合
    df1_key_set = set(df1_keys)

    # 找出在df2中不在df1中的行
    for idx, key in enumerate(df2_keys):
        if key not in df1_key_set:
            only_in_df2_indices.append(idx)

    # 提取差异行
    df1_only = df1.iloc[only_in_df1_indices].copy() if only_in_df1_indices else pd.DataFrame()
    df2_only = df2.iloc[only_in_df2_indices].copy() if only_in_df2_indices else pd.DataFrame()

    # 添加差异类型标识
    if not df1_only.empty:
        df1_only['_差异类型'] = '仅在文件1中'
    if not df2_only.empty:
        df2_only['_差异类型'] = '仅在文件2中'

    # 统计信息
    result = {
        'df1_shape': df1.shape,
        'df2_shape': df2.shape,
        'rows_only_in_file1': len(df1_only),
        'rows_only_in_file2': len(df2_only),
        'total_different_rows': len(df1_only) + len(df2_only),
        'df1_only': df1_only,
        'df2_only': df2_only
    }

    # 输出结果
    print(f"\n差异统计:")
    print(f"  仅在文件1中的行数: {len(df1_only)}")
    print(f"  仅在文件2中的行数: {len(df2_only)}")
    print(f"  总差异行数: {len(df1_only) + len(df2_only)}")

    # 显示部分差异行
    if not df1_only.empty:
        print(f"\n仅在文件1中的前5行:")
        print(df1_only.head())

    if not df2_only.empty:
        print(f"\n仅在文件2中的前5行:")
        print(df2_only.head())

    # 保存结果到文件
    if output_file:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            if not df1_only.empty:
                df1_only.to_excel(writer, sheet_name='仅在文件1中', index=False)
            if not df2_only.empty:
                df2_only.to_excel(writer, sheet_name='仅在文件2中', index=False)

            # 创建汇总信息表
            summary_data = {
                '指标': ['文件1行数', '文件2行数', '仅在文件1中行数', '仅在文件2中行数', '总差异行数'],
                '数值': [df1.shape[0], df2.shape[0], len(df1_only), len(df2_only), len(df1_only) + len(df2_only)]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='汇总信息', index=False)

        print(f"\n结果已保存到: {output_file}")

    return result


def find_different_rows_detailed(file1, file2, sheet_name=0, key_columns=None, output_file=None):
    """
    更详细的差异比较，包括逐行对比
    """

    print(f"正在读取: {file1}")
    df1 = pd.read_excel(file1, sheet_name=sheet_name)

    print(f"正在读取: {file2}")
    df2 = pd.read_excel(file2, sheet_name=sheet_name)

    print(f"文件1形状: {df1.shape}, 文件2形状: {df2.shape}")

    # 确保两个DataFrame有相同的列
    all_columns = list(set(df1.columns) | set(df2.columns))

    # 填充缺失的列
    for col in all_columns:
        if col not in df1.columns:
            df1[col] = np.nan
        if col not in df2.columns:
            df2[col] = np.nan

    # 按指定列排序以便比较
    if key_columns is not None:
        # 检查关键列是否存在
        existing_keys = [col for col in key_columns if col in all_columns]
        if existing_keys:
            df1 = df1.sort_values(by=existing_keys).reset_index(drop=True)
            df2 = df2.sort_values(by=existing_keys).reset_index(drop=True)

    # 创建比较用的DataFrame
    df1_compare = df1[all_columns].copy()
    df2_compare = df2[all_columns].copy()

    # 将NaN填充为特殊值
    df1_compare = df1_compare.fillna('__NaN__')
    df2_compare = df2_compare.fillna('__NaN__')

    # 转换为字符串进行比较
    df1_str = df1_compare.astype(str)
    df2_str = df2_compare.astype(str)

    # 找出不同的行
    diff_rows = []

    # 如果两个文件行数不同，找出多出的行
    min_rows = min(len(df1_str), len(df2_str))

    # 比较共同的行的差异
    for i in range(min_rows):
        row1 = df1_str.iloc[i]
        row2 = df2_str.iloc[i]

        if not row1.equals(row2):
            diff_cols = []
            for col in all_columns:
                if row1[col] != row2[col]:
                    diff_cols.append({
                        '列名': col,
                        '文件1值': row1[col],
                        '文件2值': row2[col]
                    })

            diff_rows.append({
                '行号': i + 1,
                '差异列数': len(diff_cols),
                '差异详情': diff_cols
            })

    # 处理文件行数不同的情况
    if len(df1_str) > len(df2_str):
        for i in range(min_rows, len(df1_str)):
            diff_rows.append({
                '行号': i + 1,
                '差异列数': '全部',
                '差异详情': '仅在文件1中'
            })
    elif len(df2_str) > len(df1_str):
        for i in range(min_rows, len(df2_str)):
            diff_rows.append({
                '行号': i + 1,
                '差异列数': '全部',
                '差异详情': '仅在文件2中'
            })

    print(f"\n发现 {len(diff_rows)} 行存在差异")

    # 保存详细结果
    if output_file and diff_rows:
        # 创建详细差异报告
        detail_data = []
        for diff in diff_rows:
            if isinstance(diff['差异详情'], list):
                for col_diff in diff['差异详情']:
                    detail_data.append({
                        '行号': diff['行号'],
                        '列名': col_diff['列名'],
                        '文件1值': col_diff['文件1值'],
                        '文件2值': col_diff['文件2值']
                    })
            else:
                detail_data.append({
                    '行号': diff['行号'],
                    '列名': '整行',
                    '文件1值': diff['差异详情'],
                    '文件2值': diff['差异详情']
                })

        detail_df = pd.DataFrame(detail_data)

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            detail_df.to_excel(writer, sheet_name='详细差异', index=False)

            # 汇总信息
            summary_df = pd.DataFrame({
                '指标': ['总行数差异', '差异行数'],
                '文件1': [df1.shape[0], len([d for d in diff_rows if '仅在文件1中' in str(d['差异详情'])])],
                '文件2': [df2.shape[0], len([d for d in diff_rows if '仅在文件2中' in str(d['差异详情'])])]
            })
            summary_df.to_excel(writer, sheet_name='汇总', index=False)

        print(f"详细差异报告已保存到: {output_file}")

    return diff_rows, df1, df2


if __name__ == "__main__":
    # 使用示例

    # 示例1: 基础比较 - 找出所有不同的行
    result = find_different_rows(
        file1='data\\高德POI分类与编码（中英文）_V1.06_20230208.xlsx',
        file2='data\\poi_types.xlsx',
        output_file='data\\差异报告_基础比较.xlsx'
    )

    # 示例2: 使用关键列进行比较
    # 找出NEW_TYPE列不同的行
    result_key = find_different_rows(
        file1='data\\高德POI分类与编码（中英文）_V1.06_20230208.xlsx',
        file2='data\\poi_types.xlsx',
        key_columns=['NEW_TYPE'],  # 只比较NEW_TYPE列
        output_file='data\\差异报告_按NEW_TYPE比较.xlsx'
    )

    # 示例3: 详细比较，查看具体哪些列不同
    # diff_details, df1, df2 = find_different_rows_detailed(
    #     file1='高德POI分类与编码（中英文）_V1.06_20230208.xlsx',
    #     file2='poi_types.xlsx',
    #     key_columns=['序号'],
    #     output_file='详细差异报告.xlsx'
    # )