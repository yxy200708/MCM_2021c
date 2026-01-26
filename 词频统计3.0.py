import pandas as pd
import re
from collections import Counter
import numpy as np
import os

# --- 第一部分：加载数据（优化读取逻辑） ---
DATA_FILE = '2021MCMProblemC_DataSet.xlsx'
IMG_FEATURE_FILE = 'Small_Sample_Result.xlsx'

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
            return None
    
    print(f"正在读取 {file_path}...")
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        return pd.read_csv(file_path)

try:
    df = load_data(DATA_FILE)
    if df is None: raise FileNotFoundError(f"找不到主数据文件：{DATA_FILE}")
    
    # 融合图片特征 (已禁用，专注于纯文本特征优化)
    # img_df = load_data(IMG_FEATURE_FILE)
    # ... (代码移除) ...
    print(f"数据加载完成，维度: {df.shape}")

except Exception as e:
    print(f"读取失败: {e}")
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
# 优化平衡系数：使用平滑后的比例，避免负样本权重过低
balance_factor = np.sqrt(total_pos / total_neg) if total_neg > 0 else 1

# --- 第二部分：PDF 理论加成定义 (基于 Vespa mandarinia 特征) ---
# 参考 PDF 文件特征：
# 1. 头部橙色 (Orange head) vs 许多其他蜂类是黑色头部
# 2. 体型巨大 (1.5-2英寸)
# 3. 筑巢于地下 (Ground nest) vs 树上纸巢
# 4. 攻击蜜蜂 (Slaughter, Decapitated bees)
PDF_KEYWORDS = {
    # 强特征 (Administrative/Confirmed - Data Driven)
    'wsda': 10.0, 'verified': 8.0, 'confirmed': 8.0, 
    'provincial': 6.0, 'government': 6.0, 'specimen': 6.0, 
    'collected': 5.0, 'thank': 4.0, 'submission': 4.0,
    'blaine': 4.0, 'nanaimo': 4.0, 'colony': 4.0,
    # 理论特征 (Theory/PDF - Physical & Behavioral)
    'mandarinia': 4.0, 'vespa': 3.0, 'murder': 3.0,
    'slaughter': 3.5, 'decapitated': 3.5, 'beheading': 3.5, 'severed': 3.0,
    'pile': 2.5, 'ground': 3.0, 'hole': 2.5, 'dirt': 2.0,
    # 形态特征 (Morphological)
    'orange': 2.0, 'head': 1.5, 'stripe': 1.0, 'band': 1.0,
    'inch': 1.5, 'huge': 1.5, 'massive': 1.5, 'giant': 1.5, 'monster': 1.5,
    'large': 1.0, 
}

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

def get_row_base_score(row):
    status = row['Lab Status']
    comment = clean_text(row.get('Lab Comments', ''))
    notes = clean_text(row.get('Notes', ''))
    
    # --- 1. ID 状态赋分 ---
    score = 0
    if status == 'Positive ID': score = 30.0 # 提高阳性样本基础分
    elif status == 'Negative ID': score = -15.0 * balance_factor
    elif status == 'Unverified': score = -2.0
    
    # --- 2. 专家评论 (Lab Comments) 赋分 ---
    pos_s_strong = ['confirmed', 'positive', 'match', 'correct', 'is a vespa', 'is asian', 'wsda', 'verified']
    pos_s_weak = ['likely', 'probable', 'possible', 'consistent', 'looks like', 'thank']
    neg_s_strong = ['not a', 'negative', 'cicada', 'bald', 'yellowjacket', 'bumblebee', 'paper', 'native', 'european', 'ichneumon']
    neg_s_weak = ['unlikely', 'doubtful', 'too small', 'wrong color']

    for s in pos_s_strong:
        if s in comment: score += 15.0
    for s in pos_s_weak:
        if s in comment: score += 5.0
    for s in neg_s_strong:
        if s in comment: score -= 15.0 * balance_factor
    for s in neg_s_weak:
        if s in comment: score -= 5.0 * balance_factor

    # --- 3. 文本内容 (Notes) 负面特征惩罚 ---
    misid_features = ['nest in tree', 'paper nest', 'hanging', 'black head', 'white face', 'cicada', 'sawfly']
    for feat in misid_features:
        if feat in notes: score -= 10.0
    
    return score

# --- 第三部分：特征测量 ---
word_total_scores = Counter()
word_occurrences = Counter()
word_in_pos_notes = Counter()
word_in_neg_notes = Counter()

# 停用词列表 (排除通用无意义词汇)
STOP_WORDS = {
    'the', 'and', 'was', 'this', 'that', 'with', 'from', 'have', 'for', 'are', 
    'saw', 'see', 'found', 'just', 'like', 'about', 'very', 'some', 'what',
    'there', 'they', 'them', 'when', 'where', 'photo', 'picture', 'image',
    'sent', 'submit', 'submission', 'bug', 'insect', 'bee', 'wasp', # bee/wasp 太通用
    'my', 'in', 'on', 'at', 'to', 'of', 'is', 'it'
}

for _, row in df.iterrows():
    row_weight = get_row_base_score(row)
    note_content = clean_text(row.get('Notes', ''))
    words = [w for w in re.findall(r'\b[a-z]{3,}\b', note_content) if w not in STOP_WORDS]
    # 增加 Lab Comments 的词汇
    comment_content = clean_text(row.get('Lab Comments', ''))
    comment_words = [w for w in re.findall(r'\b[a-z]{3,}\b', comment_content) if w not in STOP_WORDS]
    
    all_words = set(words + comment_words)
    
    for word in all_words: #每条记录每个词只计一次权重
        word_total_scores[word] += row_weight
        word_occurrences[word] += 1
        if row['Lab Status'] == 'Positive ID':
            word_in_pos_notes[word] += 1
        elif row['Lab Status'] == 'Negative ID':
            word_in_neg_notes[word] += 1

# --- 第四部分：核心权重计算 ---
final_weights = []
total_samples = total_pos + total_neg

for word, count in word_occurrences.items():
    if count >= 2: # 降低频次阈值，因为样本极少
        # 1. 专家基础分
        avg_base_score = word_total_scores[word] / count
        
        # 2. 理论加成
        theory = PDF_KEYWORDS.get(word, 0)
        
        # 3. 统计区分度 (Log Odds Ratio with Smoothing)
        p_pos = (word_in_pos_notes[word] + 0.1) / (total_pos + 1.0)
        p_neg = (word_in_neg_notes[word] + 0.1) / (total_neg + 1.0)
        
        lod = np.log(p_pos / p_neg)
        
        # 4. 最终权重融合
        # 如果是理论词汇，给予额外权重
        if theory > 0:
            final_score = avg_base_score + (lod * 5.0) + (theory * 3.0)
        else:
            final_score = avg_base_score + (lod * 2.0)
        
        final_weights.append({
            'word': word,
            'final_weight': round(final_score, 4),
            'pos_count': word_in_pos_notes[word],
            'total_count': count,
            'base_score': round(avg_base_score, 2),
            'lod_score': round(lod, 2)
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