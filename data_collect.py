import os
import sys
import subprocess

# --- 0. 环境自动修复 ---
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_and_import('openpyxl')
import pandas as pd

# --- 1. 定位桌面路径 ---
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
dataset_name = '2021MCMProblemC_DataSet.xlsx'
images_name = '2021MCM_ProblemC_ Images_by_GlobalID.xlsx'

dataset_path = os.path.join(desktop_path, dataset_name)
images_path = os.path.join(desktop_path, images_name)

if not os.path.exists(dataset_path):
    print(f"❌ 报错：桌面上没找到 {dataset_name}")
else:
    print("正在读取并清洗数据（已自动处理缺失日期）...")
    
    # 读取 Excel
    df_set = pd.read_excel(dataset_path, engine='openpyxl')
    df_img = pd.read_excel(images_path, engine='openpyxl')

    # --- 2. DataSet 清洗 (重点修复日期解析) ---
    # errors='coerce' 会将无法解析的错误日期（如 <Null>）转为 NaT (空时间对象)
    df_set['Detection Date'] = pd.to_datetime(df_set['Detection Date'], errors='coerce')
    df_set['Submission Date'] = pd.to_datetime(df_set['Submission Date'], errors='coerce')

    # 剔除日期为空的行（因为无法判断月份且无法排序）
    df_set = df_set.dropna(subset=['Detection Date', 'Submission Date'])

    # 剔除 12、1、2 月观测数据
    df_set_cleaned = df_set[~df_set['Detection Date'].dt.month.isin([12, 1, 2])]

    # 按 Submission Date 排序
    df_set_cleaned = df_set_cleaned.sort_values(by='Submission Date')

    # --- 3. Images 清洗 ---
    def get_prefix(filename):
        filename = str(filename)
        return filename.split('_')[0] if '_' in filename else filename[:10]

    df_img['Prefix'] = df_img['FileName'].apply(get_prefix)
    df_img_cleaned = df_img.drop_duplicates(subset=['GlobalID', 'Prefix'], keep='first')
    df_img_cleaned = df_img_cleaned.drop(columns=['Prefix'])

    # --- 4. 保存结果 ---
    out_set = os.path.join(desktop_path, 'Cleaned_DataSet.csv')
    out_img = os.path.join(desktop_path, 'Cleaned_Images.csv')

    # 使用 utf-8-sig 确保 Excel 打开不乱码
    df_set_cleaned.to_csv(out_set, index=False, encoding='utf-8-sig')
    df_img_cleaned.to_csv(out_img, index=False, encoding='utf-8-sig')

    print("\n" + "="*30)
    print("✅ 处理成功！请在桌面查看：")
    print(f"1. DataSet 清洗结果 (剩余 {len(df_set_cleaned)} 条)")
    print(f"2. Images 清洗结果 (剩余 {len(df_img_cleaned)} 条)")
    print("="*30)