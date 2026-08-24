"""
数据清洗脚本：处理 Beijing_POI_WGS84_UTM.csv 中的 type_3 字段

功能：
1. 清洗 type_3 字段，移除与 type_1 和 type_2 混用的内容
2. 移除品牌名称和通用描述词
3. 确保类型层级清晰：type_1（一级）< type_2（二级）< type_3（三级）
4. 输出清洗后的新文件

作者：GeoCoder
日期：2026-07-27
"""

import pandas as pd
import os
from typing import List, Optional


def clean_type3_value(
    type3_value: str,
    type1_value: str,
    type2_value: str,
    brand_keywords: List[str] = None,
) -> str:
    """
    清洗单个 type_3 值

    参数:
        type3_value: 原始 type_3 值
        type1_value: type_1 值
        type2_value: type_2 值
        brand_keywords: 品牌关键词列表（可选）

    返回:
        清洗后的 type_3 值
    """
    if pd.isna(type3_value) or not isinstance(type3_value, str):
        return "未分类"

    # 标准化输入：去除首尾空格，按 | 分割
    type3_value = type3_value.strip()
    subcategories = [sub.strip() for sub in type3_value.split("|") if sub.strip()]

    if not subcategories:
        return "未分类"

    # 如果未提供品牌关键词，使用默认列表
    if brand_keywords is None:
        brand_keywords = [
            "中国石化",
            "中国石油",
            "利安顺",
            "千里马",
            "途虎",
            "宝子",
            "圳兴",
            "博睿",
            "博众",
            "靓点",
            "赛轮",
        ]

    # 筛选候选类别：排除与 type_1、type_2 完全相同的，排除品牌关键词
    candidates = []
    for sub in subcategories:
        # 排除与 type_1 或 type_2 完全相同的值
        if sub == type1_value or sub == type2_value:
            continue

        # 排除品牌关键词
        if any(brand in sub for brand in brand_keywords):
            continue

        candidates.append(sub)

    # 如果没有候选，尝试使用最长的子类别
    if not candidates:
        candidates = subcategories

    # 选择最具体的类别：优先选择描述最长的（通常更具体）
    if candidates:
        # 如果有多个候选，选择最长的
        selected = max(candidates, key=len)
        return selected

    return "未分类"


def process_csv(
    input_path: str,
    output_path: str,
    brand_keywords: List[str] = None,
) -> pd.DataFrame:
    """
    处理 CSV 文件，清洗 type_3 字段

    参数:
        input_path: 输入文件路径
        output_path: 输出文件路径
        brand_keywords: 品牌关键词列表（可选）

    返回:
        处理后的 DataFrame
    """
    # 检查文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 读取 CSV 文件
    print(f"正在读取文件: {input_path}")
    df = pd.read_csv(input_path)

    # 显示原始数据统计
    print("\n=== 原始数据统计 ===")
    print(f"总记录数: {len(df)}")
    print(f"type_3 值的唯一数量: {df['type_3'].nunique()}")
    print(f"\ntype_3 值的前 20 个:")
    print(df['type_3'].value_counts().head(20))

    # 应用清洗函数
    print("\n正在清洗 type_3 字段...")
    df['type_3_cleaned'] = df.apply(
        lambda row: clean_type3_value(
            row['type_3'],
            row['type_1'],
            row['type_2'],
            brand_keywords,
        ),
        axis=1,
    )

    # 显示清洗后的数据统计
    print("\n=== 清洗后数据统计 ===")
    print(f"type_3_cleaned 值的唯一数量: {df['type_3_cleaned'].nunique()}")
    print(f"\ntype_3_cleaned 值的前 20 个:")
    print(df['type_3_cleaned'].value_counts().head(20))

    # 检查清洗前后的变化
    print("\n=== 数据变化分析 ===")
    changed = df[df['type_3'] != df['type_3_cleaned']]
    print(f"被修改的记录数: {len(changed)} ({len(changed) / len(df) * 100:.2f}%)")

    # 显示一些被修改的示例
    if len(changed) > 0:
        print("\n=== 修改示例（前 10 条）===")
        for idx, row in changed.head(10).iterrows():
            print(
                f"ID: {row['POI ID']}, 原值: {row['type_3']}, "
                f"清洗后: {row['type_3_cleaned']}"
            )

    # 保存到新文件
    print(f"\n正在保存清洗后的数据到: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"保存完成！共 {len(df)} 条记录。")

    return df


def main():
    """主函数"""
    # 文件路径配置
    input_file = r"F:\360安全云盘同步版\02-我的论文\数据集构建\Data\QGIS数据\Beijing_POI_WGS84_UTM.csv"
    output_file = r"F:\360安全云盘同步版\02-我的论文\数据集构建\Data\QGIS数据\Beijing_POI_WGS84_UTM_cleaned.csv"

    # 可选：自定义品牌关键词
    custom_brands = [
        "中国石化",
        "中国石油",
        "利安顺",
        "千里马",
        "途虎",
        "宝子",
        "圳兴",
        "博睿",
        "博众",
        "靓点",
        "赛轮",
    ]

    try:
        # 处理文件
        df_cleaned = process_csv(input_file, output_file, custom_brands)

        print("\n=== 处理完成 ===")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        print(f"总记录数: {len(df_cleaned)}")
        print(f"清洗后的唯一类型数: {df_cleaned['type_3_cleaned'].nunique()}")

    except Exception as e:
        print(f"\n错误: {e}")
        raise


if __name__ == "__main__":
    main()
