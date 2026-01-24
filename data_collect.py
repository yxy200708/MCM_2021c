import os
import pandas as pd

# --- 1. 定位数据文件路径 ---
# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

dataset_name = '2021MCMProblemC_DataSet.xlsx'
images_name = '2021MCM_ProblemC_Images_by_GlobalID.xlsx'

dataset_path = os.path.join(current_dir, dataset_name)
images_path = os.path.join(current_dir, images_name)

# 检查原始文件是否存在
if not os.path.exists(dataset_path):
    print(f"❌ 错误：在文件夹中找不到 {dataset_name}")
    print("请确保 Excel 文件和此 Python 脚本放在同一个文件夹里。")
else:
    print("正在处理数据并保存到本地...")

    # --- 2. 读取数据 ---
    df_set = pd.read_excel(dataset_path, engine='openpyxl')
    df_img = pd.read_excel(images_path, engine='openpyxl')

    # --- 3. DataSet 清洗 ---
    # A. 转换日期格式
    df_set['Detection Date'] = pd.to_datetime(df_set['Detection Date'], errors='coerce')
    df_set['Submission Date'] = pd.to_datetime(df_set['Submission Date'], errors='coerce')

    # B. 剔除日期为空的行
    df_set = df_set.dropna(subset=['Detection Date', 'Submission Date'])

    # C. 剔除 Lab Status 为 Unverified 的行
    df_set_cleaned = df_set[df_set['Lab Status'] != 'Unverified']

    # --- 4. Images 清洗 (去重处理) ---
    # 针对同一个 GlobalID 的多张照片进行去重，保证数据简洁
    df_img_cleaned = df_img.drop_duplicates(subset=['GlobalID'], keep='first')

    # --- 5. 保存结果到本地 ---
    output_set_path = os.path.join(current_dir, 'Cleaned_DataSet.csv')
    output_img_path = os.path.join(current_dir, 'Cleaned_Images.csv')

    # 使用 utf-8-sig 编码，确保用 Excel 打开时中文和符号不乱码
    df_set_cleaned.to_csv(output_set_path, index=False, encoding='utf-8-sig')
    df_img_cleaned.to_csv(output_img_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*40)
    print("✅ 两个文件已生成在当前文件夹：")
    print(f"1. {os.path.basename(output_set_path)} (剩余 {len(df_set_cleaned)} 条记录)")
    print(f"2. {os.path.basename(output_img_path)} (剩余 {len(df_img_cleaned)} 条记录)")
    print("="*40)