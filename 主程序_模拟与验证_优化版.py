
"""
MCM Project: 主程序 - 模拟与验证 (优化版)
=========================================
功能说明：
本脚本是 2021 MCM Problem C 的核心运行程序。
它整合了文本、图像和地理位置数据，通过【网格搜索】自动寻找最优权重参数，
运行时空扩散模型，并生成详细的中文评估报告和可视化图表。

主要功能：
1. 数据加载：读取整合后的 `最终整合数据_完整版.csv`。
2. 参数优化：自动寻找最佳的 文本(Text)、图像(Image)、位置(Location) 权重组合。
3. 模拟运行：使用最优参数运行入侵检测模型。
4. 结果评估：计算召回率(Recall)、特异度(Specificity)和F1分数。
5. 鲁棒性分析：测试模型对参数变化的敏感度。
6. 可视化：生成中文标注的得分分布图和参数敏感性图。

使用方法：
直接运行本脚本即可。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from importlib.machinery import SourceFileLoader
import itertools
import os

# --- 动态导入模型模块 ---
# 注意：文件名必须与实际存在的文件名一致
model_path = r"d:\wzm\python_math_model\模型_决策与扩散引擎.py"
invasion_model = SourceFileLoader("invasion_model", model_path).load_module()
run_decision_engine = invasion_model.run_decision_engine

# --- 配置 ---
DATA_PATH = r"d:\wzm\python_math_model\最终整合数据_完整版.csv"
OUTPUT_RESULT_PATH = r"d:\wzm\python_math_model\结果_最终预测.csv"

# --- 绘图中文支持 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载并预处理数据"""
    print(f"正在加载数据: {DATA_PATH} ...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    # 确保列名映射正确 (根据整合脚本的输出)
    # 需要: Global_ID, Timestamp, Lat, Lon, Status, Text_Weight_Score, Img_Weight_Score, Has_Image, Data_ID
    # 原始列名可能不同，需要映射
    rename_map = {
        'GlobalID': 'Global_ID',
        'Detection Date': 'Timestamp',
        'Latitude': 'Lat',
        'Longitude': 'Lon',
        'Lab Status': 'Status',
        'Text_Model_Score': 'Text_Weight_Score',
        'Image_Score': 'Img_Weight_Score'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # 确保 Data_ID
    if 'Data_ID' not in df.columns:
        df['Data_ID'] = df.index
        
    # 处理日期错误
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    
    return df

def calculate_metrics(df_result, df_truth):
    """
    计算评估指标
    df_result: 模型输出的结果 (含 Is_True, Final_Score)
    df_truth: 原始带标签数据 (含 Status)
    """
    merged = pd.merge(df_result, df_truth[['Data_ID', 'Status']], on='Data_ID', how='left')
    
    # 1. 召回率 (Recall) - 针对 Positive ID
    # 目标：尽可能多地发现 Positive ID
    pos_mask = merged['Status'] == 'Positive ID'
    pos_total = pos_mask.sum()
    if pos_total > 0:
        # Is_True 为 True 且 实际为 Positive ID
        pos_correct = merged[pos_mask]['Is_True'].sum()
        recall = pos_correct / pos_total
    else:
        recall = 0.0

    # 2. 特异度 (Specificity) - 针对 Negative ID
    # 目标：不要把 Negative ID 误报为 True
    neg_mask = merged['Status'] == 'Negative ID'
    neg_total = neg_mask.sum()
    if neg_total > 0:
        # Is_True 为 False 且 实际为 Negative ID
        neg_correct = (~merged[neg_mask]['Is_True']).sum()
        specificity = neg_correct / neg_total
    else:
        specificity = 0.0
        
    # 3. F1 Score (综合指标)
    # 这里我们定义一个加权 F1，假设 Recall 和 Specificity 同等重要
    if (recall + specificity) > 0:
        f1 = 2 * (recall * specificity) / (recall + specificity)
    else:
        f1 = 0.0
        
    return recall, specificity, f1

def optimize_parameters(df):
    """
    网格搜索寻找最优权重参数
    """
    print("\n--- 开始参数优化 (Grid Search) ---")
    
    # 1. 确定初始确信点 (2020年2月之前)
    cutoff_date = pd.Timestamp('2020-02-01')
    initial_mask = (df['Status'] == 'Positive ID') & (df['Timestamp'] < cutoff_date)
    initial_points = df[initial_mask][['Lat', 'Lon']].values.tolist()
    
    # 2. 定义参数网格
    # 权重: L(位置), T(文本), V(图像)
    # 约束: L + T + V ≈ 1.0 (可以稍微放宽，因为只是比例)
    w_range = [0.2, 0.3, 0.4, 0.5]
    thresholds = [0.6, 0.7, 0.8]
    
    best_score = -1
    best_params = {}
    history = []
    
    # 生成权重组合
    for w_l, w_t in itertools.product(w_range, w_range):
        w_v = round(1.0 - w_l - w_t, 1)
        if w_v < 0.1 or w_v > 0.6: continue # 跳过不合理的组合
        
        for th in thresholds:
            weights = {'L': w_l, 'T': w_t, 'V': w_v}
            
            # 运行模型
            res_df = run_decision_engine(df, initial_points, weights=weights, threshold=th)
            
            # 计算指标
            rec, spec, f1 = calculate_metrics(res_df, df)
            
            # 记录
            history.append({
                'w_L': w_l, 'w_T': w_t, 'w_V': w_v, 'Th': th,
                'Recall': rec, 'Specificity': spec, 'F1': f1
            })
            
            # 更新最优
            if f1 > best_score:
                best_score = f1
                best_params = {'weights': weights, 'threshold': th}
                
    print(f"最优参数找到: 权重={best_params['weights']}, 阈值={best_params['threshold']}")
    print(f"最佳 F1分数: {best_score:.4f}")
    
    return best_params, pd.DataFrame(history)

def plot_analysis(df_res, df_history, best_th):
    """生成可视化图表"""
    print("\n--- 生成可视化图表 ---")
    
    # 图1: 参数敏感性 (F1 Score vs Threshold)
    # 聚合不同权重下的表现
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_history, x='Th', y='F1', marker='o', label='F1 Score')
    sns.lineplot(data=df_history, x='Th', y='Recall', marker='o', linestyle='--', label='Recall (召回率)')
    sns.lineplot(data=df_history, x='Th', y='Specificity', marker='o', linestyle='--', label='Specificity (特异度)')
    plt.title('图表1: 模型鲁棒性与参数敏感性分析\n(Robustness & Sensitivity Analysis)')
    plt.xlabel('决策阈值 (Decision Threshold)')
    plt.ylabel('性能指标 (Performance Metrics)')
    plt.grid(True, alpha=0.3)
    plt.savefig(r'd:\wzm\python_math_model\图表_鲁棒性分析.png')
    print("已保存: 图表_鲁棒性分析.png")
    
    # 图2: 最终得分分布 (按类别)
    # 合并状态标签
    df_plot = pd.merge(df_res, df_res[['Data_ID']], left_index=True, right_index=True, suffixes=('', '_dup'))
    # 注意：run_decision_engine 返回的 df_res 可能没有 Status 列，需要从原始数据 merge，
    # 或者我们在 main 中 merge 好了再传进来。
    # 这里我们在 main 中处理。
    
def main():
    # 1. 加载数据
    df = load_data()
    print(f"数据加载完成，共 {len(df)} 条记录。")
    
    # 2. 参数优化
    best_params, df_history = optimize_parameters(df)
    
    # 3. 使用最优参数运行最终模拟
    print("\n--- 运行最终模拟 ---")
    cutoff_date = pd.Timestamp('2020-02-01')
    initial_mask = (df['Status'] == 'Positive ID') & (df['Timestamp'] < cutoff_date)
    initial_points = df[initial_mask][['Lat', 'Lon']].values.tolist()
    
    final_res = run_decision_engine(df, initial_points, 
                                   weights=best_params['weights'], 
                                   threshold=best_params['threshold'])
    
    # 合并原始信息以便分析
    final_full = pd.merge(final_res, df[['Data_ID', 'Global_ID', 'Status', 'Timestamp']], on='Data_ID', how='left')
    
    # 4. 最终评估报告
    rec, spec, f1 = calculate_metrics(final_res, df)
    print("\n" + "="*40)
    print("       最终模型评估报告 (Final Report)")
    print("="*40)
    print(f"最优权重配置: 空间(L)={best_params['weights']['L']}, 文本(T)={best_params['weights']['T']}, 图像(V)={best_params['weights']['V']}")
    print(f"决策阈值: {best_params['threshold']}")
    print("-" * 40)
    print(f"Positive ID 召回率 (Recall):       {rec:.2%} (目标: 100%)")
    print(f"Negative ID 特异度 (Specificity):  {spec:.2%} (目标: 尽量高)")
    print(f"综合 F1 分数 (F1 Score):           {f1:.4f}")
    
    # 统计 Unverified 的结果
    unv_mask = final_full['Status'] == 'Unverified'
    unv_pos = final_full[unv_mask]['Is_True'].sum()
    unv_total = unv_mask.sum()
    print("-" * 40)
    print(f"未验证数据 (Unverified) 预测结果:")
    print(f"  - 判定为阳性 (Positive): {unv_pos} 条")
    print(f"  - 判定为阴性 (Negative): {unv_total - unv_pos} 条")
    print("="*40)
    
    # 5. 保存结果
    final_full.to_csv(OUTPUT_RESULT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n详细预测结果已保存至: {OUTPUT_RESULT_PATH}")
    
    # 6. 可视化
    plot_analysis(final_full, df_history, best_params['threshold'])
    
    # 图2: 得分分布
    plt.figure(figsize=(10, 6))
    # 过滤掉 NaNs
    plot_data = final_full.dropna(subset=['Final_Score', 'Status'])
    
    # 绘制直方图
    sns.histplot(data=plot_data, x='Final_Score', hue='Status', 
                 element="step", stat="density", common_norm=False, palette='Set2')
    
    plt.axvline(best_params['threshold'], color='r', linestyle='--', linewidth=2, 
                label=f'决策阈值 ({best_params["threshold"]})')
    
    plt.title('图表2: 不同类别样本的最终得分分布\n(Score Distribution by Status)')
    plt.xlabel('模型综合得分 (Final Probability Score)')
    plt.ylabel('密度 (Density)')
    plt.legend()
    plt.savefig(r'd:\wzm\python_math_model\图表_得分分布.png')
    print("已保存: 图表_得分分布.png")

if __name__ == "__main__":
    main()
