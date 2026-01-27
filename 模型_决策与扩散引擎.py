"""
MCM Project: Multi-Modal Invasion Detection Model (Final Decision Engine)
=========================================================================
功能说明：
本脚本实现了 **多模态融合 (Multi-Modal Fusion)** 和 **时空动态更新 (Spatio-Temporal Update)** 的最终决策逻辑。
它模拟了随着时间推移，基于用户提交的报告不断更新“确信点 (Certified Points)”和“用户信誉 (User Reputation)”的过程。

核心模型 (The "Invasion Model"):
1.  **多模态融合 (Evidence Fusion)**:
    -   $ Score_{final} = R_{user} \\times (w_L \\cdot S_{loc} + w_T \\cdot S_{text} + w_V \\cdot S_{img}) $
    -   结合了 **空间邻近度 (Location)**、**文本描述 (Text)** 和 **图像证据 (Image)**。
    -   $R_{user}$: 用户信誉系数，随历史准确率动态更新，体现了“众包数据的自我清洗”能力。

2.  **动态优先级 (Dynamic Priority)**:
    -   根据报告点与当前确信入侵前沿的距离，将报告划分为不同优先级：
        -   **P1 Frontier (Critical)**: 扩散前沿，最具预警价值，需人工严查。
        -   **P2 Known Area**: 已知感染区，价值较低。
        -   **P3 Core**: 核心重灾区。

3.  **时空扩散模拟**:
    -   `certified_points` 列表随时间动态增长，模拟了入侵范围的扩大，从而改变后续报告的空间得分 ($S_{loc}$)。
"""

import pandas as pd
import numpy as np
from haversine import haversine  # 注意：这个库的函数参数是 (纬度,经度) 的元组

# --- 配置参数 ---
DISPERSAL_SPEED = 5.0  # 物种每月扩散速度 (km)
FRONTIER_BUFFER = 5.0  # 前沿探测带宽度 (km)
CORE_RADIUS = 30.0     # 初始逻辑圆半径 (km)

# --- 核心判定逻辑 ---
def run_decision_engine(df, initial_points, weights={'L': 0.3, 'T': 0.3, 'V': 0.4}, threshold=0.7):
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
        min_dist = min([haversine((row['Lat'], row['Lon']), (p[0], p[1])) for p in certified_points])

        # --- 优先级判定 (Priority Layer) ---
        if min_dist > (CORE_RADIUS + FRONTIER_BUFFER):
            priority = "P1: Frontier (Critical)"  # 扩散前沿，必须严审
        elif min_dist <= 10.0:
            priority = "P3: Core (Data Only)"    # 重灾区，仅作记录
        else:
            priority = "P2: Known Area"          # 已知区

        # --- 综合评分逻辑 ---
        # 即使是 Positive ID，我们也计算分数用于验证，但 Is_True 强制为 True
        # L: 空间分, T: 词频分, V: 图像分
        l_score = 1.0 if min_dist <= CORE_RADIUS else 0.4
        t_score = row['Text_Weight_Score'] 
        v_score = row['Img_Weight_Score'] if row['Has_Image'] else 0.1
        
        # 综合计算 (使用传入的权重参数)
        w_L = weights.get('L', 0.3)
        w_T = weights.get('T', 0.3)
        w_V = weights.get('V', 0.4)
        
        final_score = r_score * (l_score * w_L + t_score * w_T + v_score * w_V)
        
        # 判定真假
        if row['Status'] == 'Positive ID':
            # 训练集中已确认为真的，反馈给系统以更新信誉和扩散模型
            is_true = True
            # 注意：这里的 final_score 是模型给出的分，而不是 1.0，这样可以用于后续评估模型对此样本的置信度
        else:
            is_true = True if final_score > threshold else False

        # 更新认证库和用户信誉 (仅当判定为真时)
        if is_true:
            certified_points.append((row['Lat'], row['Lon']))
            user_db[uid][0] += 1
        user_db[uid][1] += 1

        results.append({
            'Data_ID': row['Data_ID'],
            'Final_Score': round(final_score, 4),
            'Is_True': is_true,
            'Priority': priority,
            'Dist_to_Source': round(min_dist, 2)
        })

    return pd.DataFrame(results)