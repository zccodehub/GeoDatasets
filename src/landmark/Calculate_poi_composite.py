import pandas as pd
import numpy as np
from pathlib import Path

def calculate_composite_significance(input_path: str, output_path: str):
    """
    根据空间指标（路网中心性、孤立度、路口暴露度）及认知显著性得分，
    计算空间综合显著性及最终的综合显著性指标。
    """
    # 读取数据
    df = pd.read_csv(input_path)
    
    # 确保必要列存在
    required_cols = ['exposure_scaled', 'centrality_scaled', 'isolation_scaled', 'cognition_score']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"输入数据缺少必要列: {required_cols}")

    # 第一步：计算 spatial_score
    # 三项指标中至少两项为0时，得分强制为0；否则按加权公式计算。
    # 权重：exposure(0.4), centrality(0.35), isolation(0.25)
    # 注意：孤立度通常为负向指标（越孤立越不显著），此处假设原始scaled已做正向处理。

    # 统计零值个数
    # zero_count = (df['exposure_scaled'] == 0).astype(int)  + (df['centrality_scaled'] == 0).astype(int) + (df['isolation_scaled'] == 0).astype(int)
    #
    # # 向量化条件赋值
    # df['spatial_score'] = np.where(
    #     zero_count >= 2,
    #     0.0,
    #     (0.4 * df['exposure_scaled'] + 0.35 * df['centrality_scaled'] + 0.25 * df['isolation_scaled'])/100
    # )

    spatial_score = (0.4 * df['exposure_scaled'] + 0.4 * df['centrality_scaled'] + 0.2 * df['isolation_scaled'])

    df['spatial_score'] = (spatial_score - spatial_score.min()) / (spatial_score.max() - spatial_score.min())
    # 第二步：计算 composite_score
    # 1. 构建融合信号 (空间得分归一化到0-1，认知得分在0-1)
    raw_signal = 0.2 * df['spatial_score']  + 0.8 * df['cognition_score']
    
    # 2. Sigmoid 映射
    k = 8
    composite_norm = 1 / (1 + np.exp(-k * (raw_signal - 0.5)))
    
    # 3. 映射到 [0, 100]
    # df['composite_score'] = composite_norm * 100
    df['composite_score'] = composite_norm
    # 输出结果
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"计算完成，结果已保存至: {output_path}")

if __name__ == "__main__":
    BASE_DIR = Path(r"E:\Projects\PycharmProjects\Geocoding_dataset")
    INPUT_FILE = BASE_DIR / "data" / "Beijing_POI_CognitionScore.csv"
    OUTPUT_FILE = BASE_DIR / "data" / "Beijing_POI_composite_3.csv"
    
    if INPUT_FILE.exists():
        calculate_composite_significance(str(INPUT_FILE), str(OUTPUT_FILE))
    else:
        print(f"找不到输入文件: {INPUT_FILE}")
