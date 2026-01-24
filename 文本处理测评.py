import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def clean_tokenize(text):
    if pd.isna(text): return []
    # 识别否定词并处理语义反转
    words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
    res, negate = [], False
    for w in words:
        if w in ['not', 'no', 'never', 'neither', 'none']:
            negate = True
            continue
        res.append(f"not_{w}" if negate else w)
        negate = False
    return res

def evaluate_performance():
    # --- 自动寻找文件逻辑 ---
    possible_files = [
        '2021MCMProblemC_DataSet.xlsx - Sheet1.csv',
        '2021MCMProblemC_DataSet.csv',
        '2021MCMProblemC_DataSet.xlsx'
    ]
    df = None
    for f in possible_files:
        if os.path.exists(f):
            print(f"📂 正在读取文件: {f}")
            df = pd.read_excel(f) if f.endswith('.xlsx') else pd.read_csv(f)
            break
    
    if df is None:
        print("❌ 错误：找不到数据集文件！请确保文件与脚本在同一文件夹下。")
        return

    # 准备测评数据
    eval_df = df[df['Lab Status'].isin(['Positive ID', 'Negative ID'])].copy()
    eval_df['target'] = eval_df['Lab Status'].apply(lambda x: 1 if x == 'Positive ID' else 0)
    
    # 划分训练集和测试集
    train_set, test_set = train_test_split(eval_df, test_size=0.2, random_state=42)
    
    # 训练逻辑
    pos_total = len(train_set[train_set['target'] == 1])
    neg_total = len(train_set[train_set['target'] == 0])
    word_map = {}
    
    for _, row in train_set.iterrows():
        for w in set(clean_tokenize(row['Notes'])):
            if w not in word_map: word_map[w] = [0, 0]
            word_map[w][0 if row['target'] == 1 else 1] += 1
            
    # 计算权重
    temp_weights = {w: (h[0]/(pos_total+1) - h[1]/(neg_total+1))*10 for w, h in word_map.items()}
    
    # 对测试集打分
    test_set['score'] = test_set['Notes'].apply(lambda x: sum(temp_weights.get(w, 0) for w in clean_tokenize(x)))
    
    try:
        auc = roc_auc_score(test_set['target'], test_set['score'])
        print("\n" + "="*30)
        print(f"📊 测评报告")
        print(f"当前算法 AUC: {auc:.4f}")
        print("注：AUC > 0.5 说明模型逻辑已经修正为正向识别。")
        print("="*30)
    except ValueError:
        print("⚠ 测评失败：测试集中正样本数量太少，无法计算AUC。")

if __name__ == "__main__":
    evaluate_performance()
    