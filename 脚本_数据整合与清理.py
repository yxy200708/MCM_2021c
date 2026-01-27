
import pandas as pd
import os
import shutil
import glob

# 路径配置
BASE_DIR = r"d:\wzm\python_math_model"

# 文件路径
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "Cleaned_Data_Scored.csv")
IMAGE_SCORES_PATH = os.path.join(BASE_DIR, "图像打分.csv")
ORIGINAL_DATASET = os.path.join(BASE_DIR, "2021MCMProblemC_DataSet.xlsx")
ORIGINAL_IMAGES_INDEX = os.path.join(BASE_DIR, "2021MCM_ProblemC_Images_by_GlobalID.xlsx")

# 目标文件路径
INTEGRATED_DATA_PATH = os.path.join(BASE_DIR, "最终整合数据_完整版.csv")

def integrate_data():
    print("正在整合数据...")
    if not os.path.exists(CLEANED_DATA_PATH):
        print(f"警告：未找到 {CLEANED_DATA_PATH}，无法进行完整整合。")
        return

    df_main = pd.read_csv(CLEANED_DATA_PATH)
    
    # 加载图像分数
    if os.path.exists(IMAGE_SCORES_PATH):
        df_img = pd.read_csv(IMAGE_SCORES_PATH)
        # 选取需要的列，避免重复
        # 假设我们使用 'New_Prob_Score' 作为最新的图像分数
        if 'New_Prob_Score' in df_img.columns:
            img_subset = df_img[['GlobalID', 'New_Prob_Score']].copy()
            img_subset.rename(columns={'New_Prob_Score': 'Image_Score'}, inplace=True)
        elif 'Image_Score' in df_img.columns:
            img_subset = df_img[['GlobalID', 'Image_Score']].copy()
        else:
            img_subset = pd.DataFrame(columns=['GlobalID', 'Image_Score'])
        
        # 去重，每个GlobalID只保留一个分数（取平均或第一个）
        img_subset = img_subset.groupby('GlobalID').mean().reset_index()
        
        # 合并
        df_merged = pd.merge(df_main, img_subset, on='GlobalID', how='left')
    else:
        print("未找到图像分数文件，跳过图像分数合并。")
        df_merged = df_main
        df_merged['Image_Score'] = 0.5 # 默认值

    # 填充缺失值
    df_merged['Image_Score'] = df_merged['Image_Score'].fillna(0.1)
    
    # 保存整合后的文件
    df_merged.to_csv(INTEGRATED_DATA_PATH, index=False, encoding='utf-8-sig')
    print(f"数据已整合至: {INTEGRATED_DATA_PATH}")

def rename_and_cleanup():
    print("正在执行重命名和清理...")
    
    # 1. 重命名原始数据文件 (保留并添加介绍)
    rename_map = {
        "2021MCMProblemC_DataSet.xlsx": "原数据_2021MCM_C题_数据集.xlsx",
        "2021MCM_ProblemC_Images_by_GlobalID.xlsx": "原数据_2021MCM_C题_图片索引表.xlsx",
        "run_simulation_and_validation.py": "主程序_模拟与验证_优化版.py",
        "generate_eval_notebook.py": "工具_生成评估报告.py",
        "决策+扩散+优先级invasion_model.py": "模型_决策与扩散引擎.py",
        "Image_Model_Evaluation.ipynb": "报告_图像模型评估.ipynb",
        "SVD奇异值处理图片.R": "脚本_SVD图片处理.R",
        "cmd.txt": "说明_命令行记录.txt",
        "requirements.txt": "配置_依赖库列表.txt"
    }

    for old_name, new_name in rename_map.items():
        old_path = os.path.join(BASE_DIR, old_name)
        new_path = os.path.join(BASE_DIR, new_name)
        if os.path.exists(old_path):
            try:
                os.rename(old_path, new_path)
                print(f"已重命名: {old_name} -> {new_name}")
            except Exception as e:
                print(f"重命名失败 {old_name}: {e}")
    
    # 2. 删除可再生/无关的CSV和图片 (保留刚生成的整合文件)
    files_to_delete = [
        "Cleaned_Data_Scored.csv",
        "Cleaned_Data_No_Negative.csv",
        "Cleaned_Data_With_Negative.csv",
        "word_weights.csv",
        "Final_Prediction_Results.csv",
        "New_Image_Scores_Prob.csv",
        "图像打分.csv",
        "Full_Image_Features_SVD.csv",
        "robustness_score_dist.png"
    ]
    
    for fname in files_to_delete:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                print(f"已删除: {fname}")
            except Exception as e:
                print(f"删除失败 {fname}: {e}")

    # 3. 检查是否有其他可视化图片需要清理
    # (这里比较谨慎，只删明确的)

def main():
    integrate_data()
    rename_and_cleanup()
    print("整合与清理完成。")

if __name__ == "__main__":
    main()
