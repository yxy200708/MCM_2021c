import pandas as pd
import re
from collections import Counter
import numpy as np
import os

# --- 第一部分：加载数据（优化读取逻辑） ---
DATA_FILE = '2021MCMProblemC_DataSet.xlsx'

def load_data(file_path):
    if not os.path.exists(file_path):
        # 尝试寻找同名的 CSV 文件
        csv_path = file_path.replace('.xlsx', '.csv')
        if os.path.exists(csv_path):
            print(f"未找到 {file_path}，正在读取 {csv_path}...")
            # 增加编码兼容性
            try:
                return pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(csv_path, encoding='gbk')
        else:
            raise FileNotFoundError(f"找不到数据文件：{file_path} 或 {csv_path}")
    
    print(f"正在读取 {file_path}...")
    if file_path.endswith('.xlsx'):
    
        return pd.read_excel(file_path)
    else:
        return pd.read_csv(file_path)

try:
    df = load_data(DATA_FILE)
except Exception as e:
    print(f"读取失败: {e}")
    # 提供备选方案或退出
    exit()

# --- 第二部分：计算样本平衡系数 ---
# 检查列名是否存在，增加容错
REQUIRED_COLS = ['Lab Status', 'Lab Comments', 'Notes']
for col in REQUIRED_COLS:
    if col not in df.columns:
        # 模糊匹配列名（处理空格或大小写）
        matches = [c for c in df.columns if col.lower().replace(' ', '') in c.lower().replace(' ', '')]
        if matches:
            df.rename(columns={matches[0]: col}, inplace=True)
            print(f"列名重映射: {matches[0]} -> {col}")
        else:
            print(f"警告：找不到必需的列 {col}")

total_pos = len(df[df['Lab Status'] == 'Positive ID'])
total_neg = len(df[df['Lab Status'] == 'Negative ID'])
balance_factor = total_pos / total_neg if total_neg > 0 else 1

# --- 第二部分：PDF 理论加成定义 ---
PDF_KEYWORDS = {
    'decapitated': 2.0, 'beheading': 2.0, 'sparrow': 2.0, 'mandarinia': 2.0,
    'orange': 1.5, 'mandibles': 1.5, 'slaughter': 1.5, 'inch': 1.0, 'massive': 1.0
}

def clean_text(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-zA-Z\s]', '', str(text).lower())

def get_row_base_score(row):
    status = row['Lab Status']
    comment = clean_text(row['Lab Comments'])
    pos_s = ['match', 'consistent', 'likely', 'typical', 'correct', 'confirmed']
    neg_s = ['not', 'cicada', 'european', 'wasp', 'bee', 'small', 'common', 'yellowjacket']
    
    score = 0
    if status == 'Positive ID': score = 5.0
    elif status == 'Negative ID': score = -5.0 * balance_factor
    elif status == 'Unverified': score = 0.5
    
    for s in pos_s:
        if s in comment: score += 2.0
    for s in neg_s:
        if s in comment: score -= 2.0 * balance_factor
    return score

# --- 第三部分：特征测量 ---
word_total_scores = Counter()
word_occurrences = Counter()
word_in_pos_notes = Counter()
word_in_neg_notes = Counter()

for _, row in df.iterrows():
    row_weight = get_row_base_score(row)
    note_content = clean_text(row['Notes'])
    words = set(re.findall(r'\b[a-z]{4,}\b', note_content))
    
    for word in words:
        word_total_scores[word] += row_weight
        word_occurrences[word] += 1
        if row['Lab Status'] == 'Positive ID':
            word_in_pos_notes[word] += 1
        elif row['Lab Status'] == 'Negative ID':
            word_in_neg_notes[word] += 1

# --- 第四部分：核心逻辑修改（置信度过滤与平滑） ---
final_weights = []
for word, count in word_occurrences.items():
    if count > 2:
        # 1. 专家与状态基础分
        avg_expert_score = word_total_scores[word] / count
        
        # 2. 修改点 A：阳性率门槛对理论奖励的影响
        theory_bonus = PDF_KEYWORDS.get(word, 0)
        # 如果该关键词从未在阳性样本中出现过，奖励减半
        if theory_bonus > 0 and word_in_pos_notes[word] == 0:
            theory_bonus *= 0.5
        
        # 3. 修改点 B：动态放大系数（基于词频平滑）
        # 总次数越多，系数适当降低。使用对数平滑：10 / log(count + e)
        # 这样低频词系数大（约8.0），极高频词系数小（趋向2.0-3.0）
        dynamic_factor = 10.0 / np.log(count + 2.718)
        
        pos_freq = word_in_pos_notes[word] / (total_pos + 1)
        neg_freq = (word_in_neg_notes[word] * balance_factor) / (total_neg + 1)
        
        witness_bonus = (pos_freq - neg_freq) * dynamic_factor
        
        final_weights.append({
            'word': word,
            'final_weight': round(avg_expert_score + theory_bonus + witness_bonus, 4),
            'pos_count': word_in_pos_notes[word],
            'total_count': count,
            'dynamic_factor': round(dynamic_factor, 2)
        })

weight_df = pd.DataFrame(final_weights).sort_values(by='final_weight', ascending=False)
try:
    weight_df.to_csv('word_weights.csv', index=False)
    print("结果已保存至 word_weights.csv")
except PermissionError:
    new_filename = 'word_weights_new.csv'
    weight_df.to_csv(new_filename, index=False)
    print(f"警告：word_weights.csv 被占用，结果已保存至 {new_filename}")
print(weight_df.head(15))