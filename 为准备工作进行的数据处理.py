import pandas as pd
import numpy as np
import os
from sklearn.decomposition import TruncatedSVD

# 1. 加载数据
try:
    # 使用经过 text_scoring_model 处理过的全量评分数据
    if os.path.exists('Cleaned_Data_Scored.csv'):
        df_main = pd.read_csv('Cleaned_Data_Scored.csv')
        print("成功读取 Cleaned_Data_Scored.csv (包含 Text_Model_Score 和全量数据)")
    else:
        # Fallback: 如果没有 scored 数据，尝试读取 With_Negative 并警告
        print("警告: Cleaned_Data_Scored.csv 未找到，尝试读取 Cleaned_Data_With_Negative.csv...")
        df_main = pd.read_csv('Cleaned_Data_With_Negative.csv')
        # 这里应该调用模型评分，但在本脚本中简化处理，若无分值则给随机值（不推荐）
        if 'Text_Model_Score' not in df_main.columns:
            print("注意: 数据缺少 Text_Model_Score，将使用随机值填充（请先运行 text_scoring_model.py）")
            df_main['Text_Model_Score'] = np.random.uniform(0.3, 0.8, size=len(df_main))

    df_images = pd.read_csv('Cleaned_Images_v3.csv')
    print("成功读取 Cleaned_Images_v3.csv")
except Exception as e:
    print(f"读取失败: {e}")
    exit()

# --- 核心修复：清洗 GlobalID 确保匹配 ---
def clean_id(s):
    if pd.isna(s): return s
    return str(s).replace('{', '').replace('}', '').strip().upper()

df_main['Match_ID'] = df_main['GlobalID'].apply(clean_id)
# 注意：Cleaned_Data_Scored.csv 可能已经有 clean id，但为了保险再做一次
if 'GlobalID_Clean' in df_main.columns:
    df_main['Match_ID'] = df_main['GlobalID_Clean']

df_images['Match_ID'] = df_images['GlobalID'].apply(clean_id)

# 2. 移除时间筛选，处理整个 12 个月的数据
# 注意：Cleaned_Data_Scored.csv 已经是全量数据 (4440条)
df_filtered = df_main.copy()
if 'Submission Date' in df_filtered.columns:
    df_filtered['Submission Date'] = pd.to_datetime(df_filtered['Submission Date'])

# 3. 统计图片 (如果 Scored 数据中已有 Has_Image，可以跳过，但为了 SVD 仍需 Image 数据)
# 确保 Has_Image 列存在
if 'Has_Image' not in df_filtered.columns:
    image_counts = df_images.groupby('Match_ID').size().reset_index(name='img_count')
    df_filtered = pd.merge(df_filtered, image_counts, on='Match_ID', how='left')
    df_filtered['Has_Image'] = df_filtered['img_count'].fillna(0) > 0

# 4. SVD 特征提取
image_pivot = pd.crosstab(df_images['Match_ID'], df_images['FileType'])
if not image_pivot.empty:
    svd = TruncatedSVD(n_components=min(10, image_pivot.shape[1]), random_state=42)
    svd_features = svd.fit_transform(image_pivot)
    svd_score_map = pd.Series(np.linalg.norm(svd_features, axis=1), index=image_pivot.index).to_dict()
else:
    svd_score_map = {}

# 5. 构造最终输出
final_df = pd.DataFrame({
    'Data_ID': range(1, len(df_filtered) + 1),
    'Global_ID': df_filtered['GlobalID'],
    'Timestamp': df_filtered['Submission Date'].dt.strftime('%Y-%m-%d') if 'Submission Date' in df_filtered.columns else '',
    'Lat': df_filtered['Latitude'],
    'Lon': df_filtered['Longitude'],
    'Status': df_filtered['Lab Status'],
    'Text_Model_Score': df_filtered['Text_Model_Score'], # 使用真实评分
    'Has_Image': df_filtered['Has_Image'],
    'Has_Video': False # 暂时默认为 False
})

# 填充基于 SVD 的分值
final_df['Img_Model_Score'] = df_filtered['Match_ID'].map(svd_score_map).fillna(0.1)
# 线性映射到 0.1-0.9
if final_df['Img_Model_Score'].max() > 0.1:
    m_min, m_max = final_df['Img_Model_Score'].min(), final_df['Img_Model_Score'].max()
    final_df['Img_Model_Score'] = 0.1 + (final_df['Img_Model_Score'] - m_min) / (m_max - m_min) * 0.8
final_df['Img_Model_Score'] = final_df['Img_Model_Score'].round(4)

# 7. 保存结果为 CSV 供测试使用
final_df.to_csv('data.csv', index=False)

print(f"清洗完成！")
print(f"12个月记录总数: {len(final_df)}")
print(f"带图片记录数: {final_df['Has_Image'].sum()}")