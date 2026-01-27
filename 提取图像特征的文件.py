"""
MCM Project: Image Feature Extraction & Centroid Matching
=========================================================
功能说明：
本脚本负责处理图像数据，提取 **可解释的手工特征 (Handcrafted Features)** 并计算与“标准亚洲大黄蜂”的相似度。
适用于小样本 (Small Data) 场景，避免了深度学习模型对大量训练数据的依赖。

核心算法 (Feature Engineering):
1.  **纹理特征 (GLCM - Gray Level Co-occurrence Matrix)**:
    -   **Contrast**: 反映腹部条纹边缘的清晰度。
    -   **Entropy**: 反映纹理的复杂度（黄蜂条纹具有特定的规则性）。
    -   **Correlation**: 反映像素间的线性相关性。
    -   **Energy**: 反映纹理的均匀性。
    *注意：由于缺少 skimage 库，本脚本包含 GLCM 的手动 Numpy 实现。*

2.  **颜色特征 (HSV Histogram)**:
    -   在 HSV 空间 (Hue, Saturation, Value) 统计均值和标准差。
    -   捕捉亚洲大黄蜂独特的“橙色头部”和“深色胸部”特征。

3.  **重心匹配模型 (Centroid Matching Model)**:
    -   利用少量 Positive ID 图像计算特征空间中的“重心”向量 $\\vec{C}$。
    -   计算新图像 $\\vec{x}$ 到重心的欧氏距离 $D$。
    -   相似度得分 $S = 1 - (D / D_{max})$。
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import math

# Configuration
IMAGE_DIR = r"d:\wzm\python_math_model\2021MCM_ProblemC_Files"
OUTPUT_CSV = r"d:\wzm\python_math_model\Image_Features_Scored.csv"
CLEANED_IMAGES_CSV = r"d:\wzm\python_math_model\Cleaned_Images_v3.csv"
CLEANED_DATA_CSV = r"d:\wzm\python_math_model\Cleaned_Data_Scored.csv"
GLCM_LEVELS = 16  # Quantize to 16 levels for efficiency and robustness

def load_data_map():
    """Load image mapping and lab status."""
    print("Loading data mapping...")
    try:
        df_imgs = pd.read_csv(CLEANED_IMAGES_CSV)
        df_data = pd.read_csv(CLEANED_DATA_CSV)
        
        # Merge to get Lab Status for each image
        # Cleaned_Images_v3.csv: FileName, GlobalID
        # Cleaned_Data_Scored.csv: GlobalID, Lab Status
        merged = pd.merge(df_imgs, df_data[['GlobalID', 'Lab Status']], on='GlobalID', how='left')
        return merged
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return None

def compute_glcm_features(img_gray):
    """
    Compute GLCM features manually using Numpy (since skimage is missing).
    Features: Contrast, Energy, Correlation, Entropy
    Distance=1, Angle=0 (Horizontal)
    """
    # Resize for speed if too large
    if img_gray.width > 512:
        scale = 512 / img_gray.width
        img_gray = img_gray.resize((512, int(img_gray.height * scale)))
    
    arr = np.array(img_gray)
    
    # Quantize to GLCM_LEVELS
    bins = np.linspace(0, 256, GLCM_LEVELS+1)
    arr_quantized = np.digitize(arr, bins) - 1
    arr_quantized = np.clip(arr_quantized, 0, GLCM_LEVELS-1)
    
    # Create GLCM (Horizontal: right neighbor)
    # Flatten pairs: (current_pixel, right_neighbor)
    # Value = row * LEVELS + col
    current_pixels = arr_quantized[:, :-1].flatten()
    right_pixels = arr_quantized[:, 1:].flatten()
    
    # Compute counts using bincount
    indices = current_pixels * GLCM_LEVELS + right_pixels
    counts = np.bincount(indices, minlength=GLCM_LEVELS**2)
    glcm = counts.reshape((GLCM_LEVELS, GLCM_LEVELS))
    
    # Normalize
    glcm_sum = glcm.sum()
    if glcm_sum == 0:
        return 0.0, 0.0, 0.0, 0.0
    P = glcm / glcm_sum
    
    # Create indices matrices
    rows, cols = np.indices((GLCM_LEVELS, GLCM_LEVELS))
    
    # 1. Contrast: sum(P[i,j] * (i-j)^2)
    contrast = np.sum(P * (rows - cols)**2)
    
    # 2. Energy: sum(P[i,j]^2)
    energy = np.sum(P**2)
    
    # 3. Entropy: -sum(P[i,j] * log(P[i,j]))
    mask = P > 0
    entropy = -np.sum(P[mask] * np.log(P[mask]))
    
    # 4. Correlation
    mu_i = np.sum(rows * P)
    mu_j = np.sum(cols * P)
    sigma_i = np.sqrt(np.sum(P * (rows - mu_i)**2))
    sigma_j = np.sqrt(np.sum(P * (cols - mu_j)**2))
    
    if sigma_i * sigma_j == 0:
        correlation = 0.0
    else:
        correlation = np.sum(P * (rows - mu_i) * (cols - mu_j)) / (sigma_i * sigma_j)
        
    return contrast, energy, correlation, entropy

def compute_hsv_histogram(img):
    """
    Compute HSV histogram features.
    Returns: Mean and Std of H, S, V channels (6 features)
    """
    img_hsv = img.convert('HSV')
    arr = np.array(img_hsv)
    
    # H: 0-255, S: 0-255, V: 0-255 in PIL
    h = arr[:,:,0].flatten()
    s = arr[:,:,1].flatten()
    v = arr[:,:,2].flatten()
    
    return (
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v)
    )

def extract_features_for_file(filepath):
    try:
        with Image.open(filepath) as img:
            # GLCM (needs grayscale)
            img_gray = img.convert('L')
            contrast, energy, correlation, entropy = compute_glcm_features(img_gray)
            
            # HSV
            h_mean, h_std, s_mean, s_std, v_mean, v_std = compute_hsv_histogram(img)
            
            return {
                'GLCM_Contrast': contrast,
                'GLCM_Energy': energy,
                'GLCM_Correlation': correlation,
                'GLCM_Entropy': entropy,
                'HSV_H_Mean': h_mean, 'HSV_H_Std': h_std,
                'HSV_S_Mean': s_mean, 'HSV_S_Std': s_std,
                'HSV_V_Mean': v_mean, 'HSV_V_Std': v_std,
                'Success': True
            }
    except Exception as e:
        # print(f"Error processing {filepath}: {e}")
        return {'Success': False}

def main():
    print(">>> Starting Feature Extraction...")
    
    # 1. Load Data Map
    df_map = load_data_map()
    if df_map is None:
        return
    
    # 2. Identify Positive ID Images for Centroid
    positive_df = df_map[df_map['Lab Status'] == 'Positive ID']
    positive_files = positive_df['FileName'].unique()
    print(f"Found {len(positive_files)} Positive ID images for training centroid.")
    
    # 3. Process All Images
    feature_list = []
    
    # Get all jpg/png files in directory
    all_files = glob.glob(os.path.join(IMAGE_DIR, "*.[jp][pn]g")) # Matches .jpg, .png
    # Also case insensitive check if needed, but glob is usually OS dependent. 
    # Let's assume standard extensions.
    
    print(f"Found {len(all_files)} files in directory.")
    
    # Dictionary to store features for centroid calculation
    positive_features = []
    
    for idx, filepath in enumerate(all_files):
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(all_files)}...")
            
        filename = os.path.basename(filepath)
        
        # Check mapping
        row = df_map[df_map['FileName'] == filename]
        global_id = row['GlobalID'].values[0] if not row.empty else "Unknown"
        lab_status = row['Lab Status'].values[0] if not row.empty else "Unknown"
        
        # Extract Features
        feats = extract_features_for_file(filepath)
        
        if feats['Success']:
            record = {
                'FileName': filename,
                'GlobalID': global_id,
                'Lab Status': lab_status,
                **feats
            }
            del record['Success']
            feature_list.append(record)
            
            # Collect Positive ID features
            if lab_status == 'Positive ID':
                # Convert features to list for calculation
                feat_vector = [
                    feats['GLCM_Contrast'], feats['GLCM_Energy'], 
                    feats['GLCM_Correlation'], feats['GLCM_Entropy'],
                    feats['HSV_H_Mean'], feats['HSV_H_Std'],
                    feats['HSV_S_Mean'], feats['HSV_S_Std'],
                    feats['HSV_V_Mean'], feats['HSV_V_Std']
                ]
                positive_features.append(feat_vector)
    
    # 4. Calculate Centroid
    if not positive_features:
        print("Warning: No Positive ID images found/processed. Cannot compute Centroid.")
        centroid = np.zeros(10)
    else:
        positive_matrix = np.array(positive_features)
        centroid = np.mean(positive_matrix, axis=0)
        print("Centroid computed:", centroid)
    
    # 5. Calculate Distances and Scores
    print("Calculating scores...")
    results = []
    feature_cols = [
        'GLCM_Contrast', 'GLCM_Energy', 'GLCM_Correlation', 'GLCM_Entropy',
        'HSV_H_Mean', 'HSV_H_Std', 'HSV_S_Mean', 'HSV_S_Std', 'HSV_V_Mean', 'HSV_V_Std'
    ]
    
    max_dist = 0
    
    # First pass: compute distances
    for item in feature_list:
        vec = np.array([item[col] for col in feature_cols])
        dist = np.linalg.norm(vec - centroid)
        item['Centroid_Distance'] = dist
        if dist > max_dist:
            max_dist = dist
            
    # Second pass: normalize scores (1 = Close to Centroid, 0 = Far)
    for item in feature_list:
        if max_dist > 0:
            # Score: Closer is better (higher)
            # Simple linear scaling: 1 - (dist / max_dist)
            # Or exponential decay: exp(-dist)
            # Using linear for now as requested "0-1"
            item['Image_Score'] = 1.0 - (item['Centroid_Distance'] / max_dist)
        else:
            item['Image_Score'] = 0.0
        results.append(item)
        
    # 6. Save
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved scored features to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
