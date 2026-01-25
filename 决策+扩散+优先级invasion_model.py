import pandas as pd
import numpy as np
from haversine import haversine  # 注意：这个库的函数参数是 (纬度,经度) 的元组

# --- 配置参数 ---
DISPERSAL_SPEED = 5.0  # 物种每月扩散速度 (km)
FRONTIER_BUFFER = 5.0  # 前沿探测带宽度 (km)
CORE_RADIUS = 30.0     # 初始逻辑圆半径 (km)

# --- 核心判定逻辑 ---
def run_decision_engine(df, initial_points):
    # 1. 初始化
    certified_points = initial_points.copy()
    user_db = {} # 存储 {Global_ID: [Success, Total]}
    results = []

    # 按时间排序，模拟真实扩散过程
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')

    for _, row in df.iterrows():
        uid = row['Global_ID']
        # 获取用户信誉分 (Bayesian)
        if uid not in user_db: user_db[uid] = [1, 2]
        r_score = user_db[uid][0] / user_db[uid][1]

        # 计算到最近认证点的距离
        min_dist = min([haversine(row['Lat'], row['Lon'], p[0], p[1]) for p in certified_points])

        # --- 优先级判定 (Priority Layer) ---
        if min_dist > (CORE_RADIUS + FRONTIER_BUFFER):
            priority = "P1: Frontier (Critical)"  # 扩散前沿，必须严审
        elif min_dist <= 10.0:
            priority = "P3: Core (Data Only)"    # 重灾区，仅作记录
        else:
            priority = "P2: Known Area"          # 已知区

        # --- 综合评分逻辑 ---
        if row['Status'] == 'Positive ID':
            final_score = 1.0
            is_true = True
        else:
            # L: 空间分, T: 词频分, V: 图像分
            l_score = 1.0 if min_dist <= CORE_RADIUS else 0.4
            t_score = row['Text_Weight_Score'] 
            v_score = row['Img_Weight_Score'] if row['Has_Image'] else 0.1
            
            # 综合计算
            final_score = r_score * (l_score * 0.3 + t_score * 0.3 + v_score * 0.4)
            is_true = True if final_score > 0.7 else False

        # 更新认证库和用户信誉
        if is_true:
            certified_points.append((row['Lat'], row['Lon']))
            user_db[uid][0] += 1
        user_db[uid][1] += 1

        results.append({
            'Data_ID': row['Data_ID'],
            'Final_Score': round(final_score, 2),
            'Is_True': is_true,
            'Priority': priority,
            'Dist_to_Source': round(min_dist, 2)
        })

    return pd.DataFrame(results)