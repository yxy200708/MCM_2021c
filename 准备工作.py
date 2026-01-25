import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 函数：计算两个经纬度之间的距离 (Haversine公式) ---
def haversine(lat1, lon1, lat2, lon2):
    r = 6371  # 地球半径，单位公里
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# --- 1. 初始化数据 ---
# 假设这是你的初始点位（经纬度）
initial_points = [(30.0, 120.0), (30.5, 120.5)] 
# 已认证的真点集合，初始为初始点
certified_points = initial_points.copy()

# 用户信誉字典 {Global_ID: [正确数, 总数]}
user_reputation = {}

# --- 2. 处理流程函数 ---
def process_reports(df):
    results = []
    
    # 按时间排序，确保模拟真实发生顺序
    df = df.sort_values(by='Timestamp')
    
    for index, row in df.iterrows():
        g_id = row['Global_ID']
        
        # A. 计算用户信誉分 (Bayesian)
        if g_id not in user_reputation:
            user_reputation[g_id] = [1, 2] # 初始默认 0.5
        s, n = user_reputation[g_id]
        r_score = s / n
        
        # B. 如果是 Positive ID，直接认证
        if row['Status'] == 'Positive ID':
            is_true = True
            final_score = 1.0
            certified_points.append((row['Lat'], row['Lon']))
            user_reputation[g_id][0] += 1 # 正确数+1
            user_reputation[g_id][1] += 1 # 总数+1
        
        # C. 如果是 Unverified，进入三层决策模型
        else:
            # 第一层：空间逻辑 (L)
            # 计算到所有已认证点的最短距离
            min_dist = min([haversine(row['Lat'], row['Lon'], pt[0], pt[1]) for pt in certified_points])
            l_score = 1.0 if min_dist <= 30 else 0.2
            
            # 第二层：文本/词频权重 (T)
            t_score = row['Text_Model_Score'] # 填入你预设的模型分
            
            # 第三层：图片/视频 (V)
            if row['Has_Video']:
                v_score = 0.8 # 视频待人工，暂给高分挂起
            else:
                v_score = row['Img_Model_Score'] if row['Has_Image'] else 0.1
            
            # 综合计算总分
            final_score = r_score * (l_score * 0.3 + t_score * 0.3 + v_score * 0.4)
            
            # 判定真假
            if final_score > 0.75:
                is_true = True
                certified_points.append((row['Lat'], row['Lon']))
                user_reputation[g_id][0] += 1
            else:
                is_true = False
            user_reputation[g_id][1] += 1
            
        # D. 处理半衰期 (针对疑似真 0.4~0.75)
        alert_status = "High" if final_score > 0.75 else ("Medium" if final_score > 0.4 else "Low")
        
        results.append({
            'Data_ID': row['Data_ID'],
            'Final_Score': round(final_score, 2),
            'Is_True': is_true,
            'Alert_Level': alert_status
        })
        
    return pd.DataFrame(results)

print("模型逻辑已就绪。")