import pandas as pd
import re
from collections import Counter

# 1. 核心权重计算公式实现
def get_expert_sentiment_weight(row):
    status = row['Lab Status']
    comment = str(row['Lab Comments']).lower()
    
    # 定义专家的“倾向性”信号
    pos_signals = ['match', 'consistent', 'likely', 'typical', 'correct', 'confirmed']
    neg_signals = ['not', 'cicada', 'european', 'wasp', 'bee', 'small', 'common']
    
    score = 0
    if status == 'Positive ID': score = 2.0
    elif status == 'Negative ID': score = -1.5
    elif status == 'Unverified': score = 0.5  # 默认给一点基础分
    
    # 根据专家评论进行动态加权修正
    for s in pos_signals:
        if s in comment: score += 1.0
    for s in neg_signals:
        if s in comment: score -= 1.0
        
    return score

# 2. 运行分析
df = pd.read_csv('2021MCMProblemC_DataSet.xlsx - Sheet1.csv')
word_stats = Counter()
word_counts = Counter()

for _, row in df.iterrows():
    w_weight = get_expert_sentiment_weight(row)
    # 提取描述中的关键词（长度大于3的词）
    words = set(re.findall(r'\b[a-z]{4,}\b', str(row['Notes']).lower()))
    for w in words:
        word_stats[w] += w_weight
        word_counts[w] += 1

# 3. 生成最终权重字典
final_weights = {w: round(word_stats[w]/word_counts[w], 4) for w in word_stats if word_counts[w] > 2}

# 4. 判定函数：计算任意文本的“专家分”
def calculate_agh_score(text):
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    total_score = sum(final_weights.get(w, 0) for w in words)
    return round(total_score, 2)

# --- 测试 ---
test_note = "Found a massive hornet, about 2 inches, it was decapitating honeybees."
print(f"该描述的专家判定得分为: {calculate_agh_score(test_note)}")