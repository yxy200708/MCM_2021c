import pandas as pd
import re
from collections import Counter

# 加载数据
df = pd.read_csv('2021MCMProblemC_DataSet.xlsx - Sheet1.csv')

# --- 第一部分：计算样本平衡系数 ---
total_pos = len(df[df['Lab Status'] == 'Positive ID'])
total_neg = len(df[df['Lab Status'] == 'Negative ID'])
balance_factor = total_pos / total_neg if total_neg > 0 else 1

# --- 第二部分：定义 PDF 理论加成 ---
PDF_KEYWORDS = {
    'decapitated': 2.0, 'beheading': 2.0, 'sparrow': 2.0, 'mandarinia': 2.0,
    'orange': 1.5, 'mandibles': 1.5, 'slaughter': 1.5, 'inch': 1.0, 'massive': 1.0
}

def clean_text(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-zA-Z\s]', '', str(text).lower())

# --- 第三部分：核心评分函数（整合 Status + Comments） ---
def get_row_base_score(row):
    status = row['Lab Status']
    comment = clean_text(row['Lab Comments'])
    
    pos_s = ['match', 'consistent', 'likely', 'typical', 'correct', 'confirmed']
    neg_s = ['not', 'cicada', 'european', 'wasp', 'bee', 'small', 'common', 'yellowjacket']
    
    score = 0
    if status == 'Positive ID':
        score = 5.0  # 原始 Positive 标签极高分
    elif status == 'Negative ID':
        score = -5.0 * balance_factor
    elif status == 'Unverified':
        score = 0.5
    
    for s in pos_s:
        if s in comment: score += 2.0
    for s in neg_s:
        if s in comment: score -= 2.0 * balance_factor
            
    return score

# --- 第四部分：统计与特征测量 ---
word_total_scores = Counter()
word_occurrences = Counter()
# 新增：记录词汇在正/负 Note 中的分布
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

# --- 第五部分：最终权重合成公式 ---
final_weights = []
for word, count in word_occurrences.items():
    if count > 2:
        # 1. 专家与状态均分
        avg_expert_score = word_total_scores[word] / count
        
        # 2. PDF 理论奖励
        theory_bonus = PDF_KEYWORDS.get(word, 0)
        
        # 3. 目击者共性奖励 (新加入)
        # 如果该词在正样本 Note 中出现的频率显著高于负样本
        pos_freq = word_in_pos_notes[word] / (total_pos + 1)
        neg_freq = (word_in_neg_notes[word] * balance_factor) / (total_neg + 1)
        witness_bonus = (pos_freq - neg_freq) * 5.0 # 放大共性特征
        
        final_weights.append({
            'word': word,
            'final_weight': round(avg_expert_score + theory_bonus + witness_bonus, 4),
            'pos_frequency': word_in_pos_notes[word],
            'total_count': count
        })

weight_df = pd.DataFrame(final_weights).sort_values(by='final_weight', ascending=False)
weight_df.to_csv('agh_final_optimized_weights.csv', index=False)
print(weight_df.head(15))