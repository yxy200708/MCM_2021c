import pandas as pd
import re
from collections import Counter

# 1. 注入 PDF 科研权重
PDF_KEYWORDS = {
    'decapitated': 1.5, 'beheading': 1.5, 'sparrow': 1.5, 'mandarinia': 1.5,
    'orange': 1.0, 'mandibles': 1.0, 'slaughter': 1.2, 'inch': 1.0
}

def get_weight_score(row):
    status = row['Lab Status']
    comment = str(row['Lab Comments']).lower()
    
    # 专家评价信号
    pos_s = ['match', 'consistent', 'likely', 'typical', 'correct']
    neg_s = ['not', 'cicada', 'european', 'wasp', 'bee', 'small']
    
    score = 0.5 # 初始基础分
    if status == 'Positive ID': score = 2.5
    elif status == 'Negative ID': score = -2.0
    
    for s in pos_s:
        if s in comment: score += 1.2
    for s in neg_s:
        if s in comment: score -= 1.2
    return score

# 2. 统计并计算
df = pd.read_csv('2021MCMProblemC_DataSet.xlsx - Sheet1.csv')
word_scores = Counter()
word_counts = Counter()

for _, row in df.iterrows():
    w = get_weight_score(row)
    # 正则清洗：只取4位以上纯字母
    words = set(re.findall(r'\b[a-z]{4,}\b', str(row['Notes']).lower()))
    for word in words:
        word_scores[word] += w
        word_counts[word] += 1

# 3. 最终加权汇总
final_data = []
for word, count in word_counts.items():
    if count > 2:
        expert_avg = word_scores[word] / count
        bonus = PDF_KEYWORDS.get(word, 0) # PDF 奖励
        final_data.append({'word': word, 'final_weight': round(expert_avg + bonus, 4)})

# 4. 导出结果
weight_df = pd.DataFrame(final_data).sort_values(by='final_weight', ascending=False)
weight_df.to_csv('final_weights_expert_pdf.csv', index=False)
print("模型词典生成成功！Top权重词：\n", weight_df.head(10))