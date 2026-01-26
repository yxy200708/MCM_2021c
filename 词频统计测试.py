import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve, classification_report

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 数据加载与预处理 ---
def load_data(file_path):
    if not os.path.exists(file_path):
        csv_path = file_path.replace('.xlsx', '.csv')
        if os.path.exists(csv_path):
            try: return pd.read_csv(csv_path, encoding='utf-8')
            except: return pd.read_csv(csv_path, encoding='gbk')
        sheet_csv = file_path + " - Sheet1.csv"
        if os.path.exists(sheet_csv):
            return pd.read_csv(sheet_csv, encoding='latin1')
        raise FileNotFoundError(f"找不到数据文件：{file_path}")
    
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)

def clean_text(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-zA-Z\s]', '', str(text).lower())

# --- 2. 核心算法移植 (来自词频统计3.0) ---
PDF_KEYWORDS = {
    'decapitated': 2.0, 'beheading': 2.0, 'sparrow': 2.0, 'mandarinia': 2.0,
    'orange': 1.5, 'mandibles': 1.5, 'slaughter': 1.5, 'inch': 1.0, 'massive': 1.0
}

def train_word_weights(train_df):
    """根据训练集生成权重表 (完全复刻 3.0 逻辑)"""
    # 统一列名
    REQUIRED_COLS = {'Lab Status': 'Lab Status', 'Notes': 'Notes', 'Lab Comments': 'Lab Comments'}
    for target, standard in REQUIRED_COLS.items():
        if standard not in train_df.columns:
            matches = [c for c in train_df.columns if standard.lower().replace(' ', '') in c.lower().replace(' ', '')]
            if matches: train_df = train_df.rename(columns={matches[0]: standard})

    total_pos = len(train_df[train_df['Lab Status'] == 'Positive ID'])
    total_neg = len(train_df[train_df['Lab Status'] == 'Negative ID'])
    # 样本平衡系数
    balance_factor = total_pos / total_neg if total_neg > 0 else 1.0
    
    word_total_scores = Counter()
    word_occurrences = Counter()
    word_in_pos_notes = Counter()
    word_in_neg_notes = Counter()
    
    # 定义行基础分计算函数
    def get_row_base_score(row):
        status = row['Lab Status']
        comment = clean_text(row.get('Lab Comments', ''))
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

    # 遍历训练集统计词频和得分
    for _, row in train_df.iterrows():
        row_weight = get_row_base_score(row)
        note_content = clean_text(row.get('Notes', ''))
        words = set(re.findall(r'\b[a-z]{4,}\b', note_content))
        
        for word in words:
            word_total_scores[word] += row_weight
            word_occurrences[word] += 1
            if row['Lab Status'] == 'Positive ID':
                word_in_pos_notes[word] += 1
            elif row['Lab Status'] == 'Negative ID':
                word_in_neg_notes[word] += 1
                
    # 计算最终权重
    weights = {}
    for word, count in word_occurrences.items():
        if count > 2:
            avg_expert_score = word_total_scores[word] / count
            
            # 理论奖励
            theory_bonus = PDF_KEYWORDS.get(word, 0)
            if theory_bonus > 0 and word_in_pos_notes[word] == 0:
                theory_bonus *= 0.5
            
            # 动态平滑
            dynamic_factor = 10.0 / np.log(count + 2.718)
            
            pos_freq = word_in_pos_notes[word] / (total_pos + 1)
            neg_freq = (word_in_neg_notes[word] * balance_factor) / (total_neg + 1)
            
            witness_bonus = (pos_freq - neg_freq) * dynamic_factor
            
            weights[word] = avg_expert_score + theory_bonus + witness_bonus
            
    return weights, balance_factor

# --- 3. 交叉验证与评估 ---
try:
    df = load_data('2021MCMProblemC_DataSet.xlsx')
except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

# 统一列名
if 'Lab Status' not in df.columns:
    matches = [c for c in df.columns if 'status' in c.lower()]
    if matches: df.rename(columns={matches[0]: 'Lab Status'}, inplace=True)

# 过滤 Unverified
df_clean = df[df['Lab Status'].isin(['Positive ID', 'Negative ID'])].copy()
print(f"已过滤 Unverified 数据，剩余样本: {len(df_clean)}")
print(f"Positive: {len(df_clean[df_clean['Lab Status']=='Positive ID'])}, Negative: {len(df_clean[df_clean['Lab Status']=='Negative ID'])}")

# 准备数据
X = df_clean.index.values
y = (df_clean['Lab Status'] == 'Positive ID').astype(int).values

# 使用分层 K 折交叉验证 (解决正负样本不平衡)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
conf_matrices = []

plt.figure(figsize=(12, 10))

print("\n开始 5-Fold 交叉验证...")

all_y_true = []
all_y_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    train_data = df_clean.iloc[train_idx].copy()
    test_data = df_clean.iloc[test_idx].copy()
    
    # 1. 在训练集上训练权重
    current_weights, _ = train_word_weights(train_data)
    
    # 2. 在测试集上预测
    def apply_score(text):
        words = set(re.findall(r'\b[a-z]{4,}\b', clean_text(text)))
        raw_score = sum(current_weights.get(w, 0) for w in words)
        # Sigmoid 归一化
        return 1 / (1 + np.exp(-0.5 * raw_score)) 
    
    note_col = 'Notes' if 'Notes' in df_clean.columns else [c for c in df_clean.columns if 'note' in c.lower()][0]
    y_score = test_data[note_col].apply(apply_score).values
    y_test = y[test_idx]
    
    all_y_true.extend(y_test)
    all_y_scores.extend(y_score)
    
    # 3. 计算指标
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    
    # 寻找最佳阈值 (Youden's J statistic)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    
    y_pred = (y_score >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    conf_matrices.append(cm)
    
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold+1} (AUC = {roc_auc:.2f})')
    
    print(f"Fold {fold+1}: AUC={roc_auc:.4f}, Best Threshold={best_thresh:.4f}")

# --- 4. 可视化与汇总 ---

# A. 平均 ROC 曲线
plt.subplot(2, 2, 1)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})', lw=2, alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (5-Fold CV)')
plt.legend(loc="lower right")

# B. Precision-Recall 曲线 (针对全部数据)
precision, recall, _ = precision_recall_curve(all_y_true, all_y_scores)
pr_auc = auc(recall, precision)

plt.subplot(2, 2, 2)
plt.plot(recall, precision, color='purple', lw=2, label=f'PR Curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

# C. 混淆矩阵 (汇总)
total_cm = sum(conf_matrices)
plt.subplot(2, 2, 3)
sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Neg', 'Predicted Pos'], yticklabels=['Actual Neg', 'Actual Pos'])
plt.title('Confusion Matrix (Aggregated)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# D. 文本报告
plt.subplot(2, 2, 4)
plt.axis('off')
y_pred_all = (np.array(all_y_scores) >= 0.5).astype(int) # 使用默认 0.5 阈值做报告
report = classification_report(all_y_true, y_pred_all, target_names=['Negative', 'Positive'], output_dict=True)
text_report = f"Classification Report (Threshold=0.5):\n"
text_report += f"Accuracy:  {report['accuracy']:.4f}\n"
text_report += f"Precision (Pos): {report['Positive']['precision']:.4f}\n"
text_report += f"Recall (Pos):    {report['Positive']['recall']:.4f}\n"
text_report += f"F1-Score (Pos):  {report['Positive']['f1-score']:.4f}\n"
text_report += f"\nTotal Positive Samples: {sum(all_y_true)}\n"
text_report += f"Total Negative Samples: {len(all_y_true) - sum(all_y_true)}"

plt.text(0.1, 0.5, text_report, fontsize=12, family='monospace')
plt.title('Performance Summary')

plt.tight_layout()
plt.savefig('model_verification_result.png')
print(f"\n验证完成！结果已保存为 'model_verification_result.png'")
print("请查看生成的图片以分析模型性能。")


# --- 3. 结果分析 (修复 KeyError) ---
# 由于之前的循环已经计算了所有指标，这里不需要再次访问 df['Blind_Test_Score']
# 直接使用收集到的 all_y_true 和 all_y_scores

y_true = np.array(all_y_true)
y_score = np.array(all_y_scores)

auc_val = roc_auc_score(y_true, y_score)
print(f"\n模型总盲测 AUC 得分: {auc_val:.4f}")
print("注：AUC 越接近 1.0 表示模型区分真假报告的能力越强（>0.8 即为非常优秀）")

# 保存带盲测评分的结果（可选）
# df_clean['Blind_Test_Score'] = ... (如果需要保存到文件再解开，但要注意对齐索引)
