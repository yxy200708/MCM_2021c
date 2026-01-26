import pandas as pd
import numpy as np
import os

# --- Configuration ---
DATA_FILE = '2021MCMProblemC_DataSet.xlsx'
IMG_FILE = '2021MCM_ProblemC_Images_by_GlobalID.xlsx'
OUTPUT_WITH_NEG = 'Cleaned_Data_With_Negative.csv'
OUTPUT_NO_NEG = 'Cleaned_Data_No_Negative.csv'

def clean_id(s):
    if pd.isna(s): return s
    # Standardize GlobalID: remove braces, trim whitespace, uppercase
    return str(s).replace('{', '').replace('}', '').strip().upper()

def process_data():
    print("--- Starting Raw Data Processing ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE) or not os.path.exists(IMG_FILE):
        print(f"Error: Source files not found.\n{DATA_FILE}\n{IMG_FILE}")
        return

    print(f"Loading {DATA_FILE}...")
    df_main = pd.read_excel(DATA_FILE)
    print(f"Loading {IMG_FILE}...")
    df_imgs = pd.read_excel(IMG_FILE)

    # 2. Standardize IDs
    print("Standardizing GlobalIDs...")
    df_main['GlobalID_Clean'] = df_main['GlobalID'].apply(clean_id)
    df_imgs['GlobalID_Clean'] = df_imgs['GlobalID'].apply(clean_id)

    # 3. Merge Image Info
    print("Merging Image Data...")
    # Count images per GlobalID
    img_counts = df_imgs.groupby('GlobalID_Clean').size().reset_index(name='Image_Count')
    # Get image file types (optional, for SVD later)
    # For now, just boolean Has_Image
    
    df_merged = pd.merge(df_main, img_counts, on='GlobalID_Clean', how='left')
    df_merged['Has_Image'] = df_merged['Image_Count'].fillna(0) > 0
    
    # 4. Save Version 1: With Negative (Full Dataset)
    print(f"Saving {OUTPUT_WITH_NEG} (All Data)...")
    df_merged.to_csv(OUTPUT_WITH_NEG, index=False)
    print(f"Saved {len(df_merged)} rows. Images matched: {df_merged['Has_Image'].sum()}")

    # 5. Save Version 2: No Negative (Filtered)
    print(f"Saving {OUTPUT_NO_NEG} (No Negative ID)...")
    # Filter out 'Negative ID'
    df_no_neg = df_merged[df_merged['Lab Status'] != 'Negative ID'].copy()
    df_no_neg.to_csv(OUTPUT_NO_NEG, index=False)
    print(f"Saved {len(df_no_neg)} rows.")

    print("--- Processing Complete ---")

if __name__ == "__main__":
    process_data()
