import pandas as pd
import re
from collections import Counter

# 1. 注入 PDF 科研权重（金标准加成）
PDF_KEYWORDS = {
    'decapitated': 2.0, 'beheading': 2.0, 'sparrow': 2.0, 'mandarinia': 2.0,
    'orange': 1.5, 'mandibles': 1.5, 'slaughter': 1.5, 'inch': 1.0, 'massive': 1.0
}

# 2. 预处理与数据加载
df = pd.read_csv('2021MCMProblemC_DataSet.xlsx - Sheet1.csv')

# --- 动态平衡系数计算 ---
total_pos = len(df[df['Lab Status'] == 'Positive ID'])
total_neg = len(df[df['Lab Status'] == 'Negative ID'])
# 计算平衡因子：如果负样本多，则按比例缩小负向评分的影响力
# 这样 1 个 Positive 的权重分量将等同于所有 Negative 的平均分量权重
balance_factor = total_pos / total_neg if total_neg > 0 else 1

def clean_text(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-zA-Z\s]', '', str(text).lower())

def get_weight_score_ultimate(row):
    status = row['Lab Status']
    comment = clean_text(row['Lab Comments'])
    
    # 专家评价信号词库
    pos_s = ['match', 'consistent', 'likely', 'typical', 'correct', 'confirmed', 'verified']
    neg_s = ['not', 'cicada', 'european', 'wasp', 'bee', 'small', 'common', 'yellowjacket', 'horntail']
    
    score = 0
    # A. 基础状态分判定（应用平衡因子）
    if status == 'Positive ID':
        score = 5.0  # 正样本赋予极高基础分
    elif status == 'Negative ID':
        score = -5.0 * balance_factor  # 负样本分值根据样本比例被大幅压缩
    elif status == 'Unverified':
        score = 0.5  # 未证实样本保持微弱正向倾向
    
    # B. Lab Comments 深度判定（应用平衡因子）
    for s in pos_s:
        if s in comment:
            score += 2.0  # 专家评价正面，大幅加分
            
    for s in neg_s:
        if s in comment:
            score -= 2.0 * balance_factor # 专家评价负面，扣分受平衡因子限制
            
    return score

# 3. 统计词频并应用权重
word_total_scores = Counter()
word_occurrences = Counter()

for _, row in df.iterrows():
    row_weight = get_weight_score_ultimate(row)
    # 提取4位以上有效词
    words = set(re.findall(r'\b[a-z]{4,}\b', str(row['Notes']).lower()))
    
    for word in words:
        word_total_scores[word] += row_weight
        word_occurrences[word] += 1

# 4. 汇总最终权重词典
final_weights = []
for word, count in word_occurrences.items():
    if count > 2:  # 过滤低频偶然词
        # 实战权重均值
        avg_expert_score = word_total_scores[word] / count
        # PDF 理论加成
        bonus = PDF_KEYWORDS.get(word, 0)
        
        final_weights.append({
            'word': word,
            'final_weight': round(avg_expert_score + bonus, 4),
            'count': count
        })

# 5. 结果导出与展示
weight_df = pd.DataFrame(final_weights).sort_values(by='final_weight', ascending=False)
weight_df.to_csv('agh_balanced_weights.csv', index=False)

print(f"检测到样本比例 - Positive: {total_pos}, Negative: {total_neg}")
print(f"动态平衡因子已设定为: {round(balance_factor, 4)}")
print("\n--- 最终模型高权重词 (Top 15) ---")
print(weight_df.head(15).to_string(index=False))