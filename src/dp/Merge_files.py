import pandas as pd
import os
from pathlib import Path


def merge_poi_excel_files(input_dir, output_file, file_pattern="*.xlsx"):
    """
    合并目录下所有结构相同的POI数据Excel文件

    参数:
        input_dir: 包含Excel文件的目录路径
        output_file: 输出合并文件的路径
        file_pattern: 文件匹配模式，默认为*.xlsx
    """
    # 获取目录下所有匹配的Excel文件
    excel_files = []
    for file_path in Path(input_dir).glob(file_pattern):
        # 跳过输出文件本身（如果存在于目录中）
        if os.path.basename(file_path) == os.path.basename(output_file):
            continue
        excel_files.append(file_path)

    if not excel_files:
        print(f"在目录 {input_dir} 中未找到匹配的Excel文件")
        return

    print(f"找到 {len(excel_files)} 个Excel文件")

    # 读取所有Excel文件并合并
    all_dfs = []
    for file_path in excel_files:
        try:
            df = pd.read_excel(file_path, engine='openpyxl')

            # 添加来源文件名列（便于追溯）
            df['来源文件'] = file_path.name

            all_dfs.append(df)
            print(f"成功读取: {file_path.name}, 行数: {len(df)}")
        except Exception as e:
            print(f"读取文件 {file_path.name} 时出错: {e}")

    if not all_dfs:
        print("没有成功读取任何文件")
        return

    # 合并所有DataFrame
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # 检查是否有重复数据（基于POI ID）
    if 'POI ID' in merged_df.columns:
        duplicate_count = merged_df['POI ID'].duplicated().sum()
        if duplicate_count > 0:
            print(f"发现 {duplicate_count} 条重复的POI ID记录")
            # 可选：保留第一次出现的记录
            merged_df = merged_df.drop_duplicates(subset=['POI ID'], keep='first')

    print(f"合并完成，总行数: {len(merged_df)}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存合并后的文件
    try:
        # 根据文件扩展名选择保存格式
        if output_file.endswith('.xlsx'):
            merged_df.to_excel(output_file, index=False, engine='openpyxl')
        elif output_file.endswith('.csv'):
            merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        else:
            merged_df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"合并文件已保存到: {output_file}")
    except Exception as e:
        print(f"保存文件时出错: {e}")


def main():
    """
    主函数 - 使用示例
    """
    # 配置参数
    input_directory = "../../data/poi_data"  # 包含多个POI Excel文件的目录
    output_filename = "../../output/merged_poi_data.xlsx"  # 合并后的输出文件名

    # 如果当前目录下没有poi_data文件夹，使用当前目录
    if not os.path.exists(input_directory):
        print(f"目录 {input_directory} 不存在，使用当前目录")
        input_directory = "."
        output_filename = "./merged_poi_data.xlsx"

    # 执行合并
    merge_poi_excel_files(
        input_dir=input_directory,
        output_file=output_filename,
        file_pattern="*.xlsx"  # 可以改为 "昌平区*.xlsx" 匹配特定文件
    )


if __name__ == "__main__":
    main()