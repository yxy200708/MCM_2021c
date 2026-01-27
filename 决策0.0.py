"""
MCM Project: Decision Model Prototype (Legacy/Initial Version)
==============================================================
功能说明：
本脚本 (`决策0.0.py`) 是决策模型的 **早期原型**。
它实现了基础的 Haversine 距离计算和简单的加权逻辑，用于初步跑通数据流程。

注意：
-   **正式模型** 请参考 `决策+扩散+优先级invasion_model.py`。
-   本脚本保留用于对比或作为简化版逻辑的参考。
-   它包含基础的“用户信誉”更新逻辑和简单的阈值判定。
"""

import pandas as pd
import numpy as np

# --- 1. 核心计算函数 (Haversine公式) ---
def haversine(lat1, lon1, lat2, lon2):
    r = 6371  # 地球半径公里
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# --- 2. 处理流程函数 ---
def process_reports(df):
    # 初始化状态
    initial_points = [(49.0, -122.5)] # 这里的初始点可以根据你数据的大致范围微调
    certified_points = initial_points.copy()
    user_reputation = {} # {Global_ID: [正确数, 总数]}
    results = []
    
    # 确保时间是日期类型并排序
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp')
    
    print(f"正在处理 {len(df)} 条记录...")

    for index, row in df.iterrows():
        g_id = row['Global_ID']
        
        # A. 计算用户信誉分
        if g_id not in user_reputation:
            user_reputation[g_id] = [1, 2] 
        s, n = user_reputation[g_id]
        r_score = s / n
        
        # B. 逻辑决策
        if row['Status'] == 'Positive ID':
            is_true = True
            final_score = 1.0
            certified_points.append((row['Lat'], row['Lon']))
            user_reputation[g_id][0] += 1 
            user_reputation[g_id][1] += 1 
        else:
            # 第一层：空间逻辑 (L)
            distances = [haversine(row['Lat'], row['Lon'], pt[0], pt[1]) for pt in certified_points]
            min_dist = min(distances) if distances else 999
            l_score = 1.0 if min_dist <= 30 else 0.2
            
            # 第二层：文本权重 (T)
            t_score = row['Text_Model_Score']
            
            # 第三层：图片/视频 (V)
            if row['Has_Video'] == True: # 显式检查
                v_score = 0.8
            else:
                v_score = row['Img_Model_Score'] if row['Has_Image'] else 0.1
            
            # 综合评分 (权重: L:0.3, T:0.3, V:0.4)
            final_score = r_score * (l_score * 0.3 + t_score * 0.3 + v_score * 0.4)
            
            # 判定阈值
            if final_score > 0.75:
                is_true = True
                certified_points.append((row['Lat'], row['Lon']))
                user_reputation[g_id][0] += 1
            else:
                is_true = False
            user_reputation[g_id][1] += 1
            
        # C. 记录结果
        alert_status = "High" if final_score > 0.75 else ("Medium" if final_score > 0.4 else "Low")
        results.append({
            'Data_ID': row['Data_ID'],
            'Global_ID': g_id,
            'Final_Score': round(final_score, 3),
            'Is_True': is_true,
            'Alert_Level': alert_status
        })
        
    return pd.DataFrame(results)

# --- 3. 运行与展示 ---
try:
    # 加载你刚刚洗好的数据
    input_df = pd.read_csv('data_test.csv')
    
    # 运行模型
    output_df = process_reports(input_df)
    
    # 展示结果
    print("\n--- 决策模型运行结果 (前10条) ---")
    print(output_df.head(10).to_string(index=False))
    
    # 简单统计
    print("\n--- 预警级别分布 ---")
    print(output_df['Alert_Level'].value_counts())

    # 保存结果
    output_df.to_csv('decision_results.csv', index=False)
    print("\n结果已保存至 decision_results.csv")

except FileNotFoundError:
    print("错误：未找到 data_test.csv 文件，请先运行数据清洗脚本。")