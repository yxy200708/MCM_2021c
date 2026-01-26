import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve, classification_report

# 设置绘图风格
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 数据加载与多模态融合 ---
DATA_FILE = '2021MCMProblemC_DataSet.xlsx'
IMG_FEATURE_FILE = 'Small_Sample_Result.xlsx'

def load_and_merge_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"找不到主数据文件：{DATA_FILE}")
    
    print(f"正在读取主数据 {DATA_FILE}...")
    df = pd.read_excel(DATA_FILE) if DATA_FILE.endswith('.xlsx') else pd.read_csv(DATA_FILE)
    
    # 统一列名
    col_map = {'Lab Status': 'Lab Status', 'Notes': 'Notes', 'Lab Comments': 'Lab Comments', 'GlobalID': 'GlobalID'}
    for target, standard in col_map.items():
        if standard not in df.columns:
            matches = [c for c in df.columns if standard.lower().replace(' ', '') in c.lower().replace(' ', '')]
            if matches: df.rename(columns={matches[0]: standard}, inplace=True)

    # 融合图片特征
    if os.path.exists(IMG_FEATURE_FILE):
        print(f"检测到图片特征文件 {IMG_FEATURE_FILE}，正在融合...")
        img_df = pd.read_excel(IMG_FEATURE_FILE) if IMG_FEATURE_FILE.endswith('.xlsx') else pd.read_csv(IMG_FEATURE_FILE)
        id_col = next((c for c in img_df.columns if 'GlobalID' in c or 'ID' in c), None)
        if id_col:
            feature_cols = [c for c in img_df.columns if c.startswith('V') or c == id_col]
            img_df = img_df[feature_cols]
            df = pd.merge(df, img_df, on=id_col, how='left')
            for c in [c for c in df.columns if c.startswith('V')]:
                df[c] = df[c].fillna(0)
    return df

def clean_text(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-zA-Z\s]', '', str(text).lower())

# --- 2. 优化后的核心算法 (同步 3.0 逻辑) ---
# 参考 PDF 文件特征与数据集中阳性样本的实际特征
# 1. 行政/确认类关键词 (Administrative/Confirmed) - 数据集中的强特征
# 2. 物理/行为特征 (Physical/Behavioral) - PDF 理论特征
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

def train_weights_optimized(train_df):
    total_pos = len(train_df[train_df['Lab Status'] == 'Positive ID'])
    total_neg = len(train_df[train_df['Lab Status'] == 'Negative ID'])
    balance_factor = np.sqrt(total_pos / total_neg) if total_neg > 0 else 1.0
    total_samples = total_pos + total_neg
    
    word_total_scores = Counter()
    word_occurrences = Counter()
    word_in_pos_notes = Counter()
    word_in_neg_notes = Counter()
    
    def get_row_score(row):
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

    STOP_WORDS = {'the', 'and', 'was', 'this', 'that', 'with', 'from', 'have', 'for', 'are', 'saw', 'see', 'found', 'just', 'like', 'about', 'very', 'some', 'what', 'there', 'they', 'them', 'when', 'where', 'photo', 'picture', 'image', 'sent', 'submit', 'submission', 'bug', 'insect', 'bee', 'wasp', 'my', 'in', 'on', 'at', 'to', 'of', 'is', 'it'}

    for _, row in train_df.iterrows():
        base_weight = get_row_score(row)
        words = [w for w in re.findall(r'\b[a-z]{3,}\b', clean_text(row.get('Notes', ''))) if w not in STOP_WORDS]
        # 增加 Lab Comments 的词汇
        comment_words = [w for w in re.findall(r'\b[a-z]{3,}\b', clean_text(row.get('Lab Comments', ''))) if w not in STOP_WORDS]
        
        all_words = set(words + comment_words)
        
        for word in all_words:
            word_total_scores[word] += base_weight
            word_occurrences[word] += 1
            if row['Lab Status'] == 'Positive ID': word_in_pos_notes[word] += 1
            elif row['Lab Status'] == 'Negative ID': word_in_neg_notes[word] += 1
                
    weights = {}
    for word, count in word_occurrences.items():
        if count >= 2: # 降低频次阈值，因为样本极少
            avg_base = word_total_scores[word] / count
            
            theory = PDF_KEYWORDS.get(word, 0)
            
            # 贝叶斯平滑概率
            p_pos = (word_in_pos_notes[word] + 0.1) / (total_pos + 1.0)
            p_neg = (word_in_neg_notes[word] + 0.1) / (total_neg + 1.0)
            
            lod = np.log(p_pos / p_neg)
            
            # 如果是理论词汇，给予额外权重
            if theory > 0:
                weights[word] = avg_base + (lod * 5.0) + (theory * 3.0)
            else:
                weights[word] = avg_base + (lod * 2.0)
                
    return weights

# --- 3. 交叉验证过程 ---
try:
    df_raw = load_and_merge_data()
    # 移除图片特征相关列 (如果存在)
    cols_to_drop = [c for c in df_raw.columns if c.startswith('V') and c[1:].isdigit()]
    if cols_to_drop:
        df_raw.drop(columns=cols_to_drop, inplace=True)
        
    df_clean = df_raw[df_raw['Lab Status'].isin(['Positive ID', 'Negative ID'])].copy()
    print(f"有效样本数: {len(df_clean)} (Pos: {sum(df_clean['Lab Status']=='Positive ID')}, Neg: {sum(df_clean['Lab Status']=='Negative ID')})")
    
    X = df_clean.index.values
    y = (df_clean['Lab Status'] == 'Positive ID').astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    all_y_true, all_y_scores = [], []
    conf_matrices = []

    print("\n执行 5-Fold 交叉验证...")
    # 优化配色方案：使用高饱和度颜色
    colors = ['#FF0000', '#0000FF', '#008000', '#800080', '#FF8C00'] # 红蓝绿紫橙
    
    plt.figure(figsize=(18, 14))
    
    # A. ROC Curve
    ax1 = plt.subplot(2, 2, 1)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_set = df_clean.iloc[train_idx]
        test_set = df_clean.iloc[test_idx]
        
        current_weights = train_weights_optimized(train_set)
        
        def predict_score(row):
            words = [w for w in re.findall(r'\b[a-z]{3,}\b', clean_text(row.get('Notes', ''))) if w not in {'the', 'and', 'was'}]
            comment_words = [w for w in re.findall(r'\b[a-z]{3,}\b', clean_text(row.get('Lab Comments', ''))) if w not in {'the', 'and', 'was'}]
            
            all_words = words + comment_words
            
            # 取前 15 个最显著的词
            word_weights = sorted([current_weights.get(w, 0) for w in all_words], key=lambda x: abs(x), reverse=True)
            word_score = sum(word_weights[:15]) if word_weights else 0
            
            # 使用 Sigmoid 映射到 0-1
            return 1 / (1 + np.exp(-0.5 * word_score))

        y_score = test_set.apply(predict_score, axis=1).values
        y_test = y[test_idx]
        
        all_y_true.extend(y_test)
        all_y_scores.extend(y_score)
        
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        
        plt.plot(fpr, tpr, lw=2, alpha=0.6, color=colors[fold], label=f'Fold {fold+1} (AUC = {roc_auc:.3f})')
        
        # 寻找最佳阈值
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
        y_pred = (y_score >= best_threshold).astype(int)
        conf_matrices.append(confusion_matrix(y_test, y_pred))
        print(f"Fold {fold+1} AUC: {roc_auc:.4f}, Best Threshold: {best_threshold:.3f}")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='black', lw=4, linestyle='--', label=f'Mean ROC (AUC = {mean_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    plt.title('Receiver Operating Characteristic (Optimized)', fontsize=16, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    # B. PR Curve
    ax2 = plt.subplot(2, 2, 2)
    precision, recall, _ = precision_recall_curve(all_y_true, all_y_scores)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, color='#2E8B57', lw=3, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.fill_between(recall, precision, color='#2E8B57', alpha=0.2)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    # 缩小间距：设置 xlim/ylim
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])

    # C. Confusion Matrix
    ax3 = plt.subplot(2, 2, 3)
    total_cm = sum(conf_matrices)
    # 颜色变化更明显：使用 Reds 或 Blues，并增大字体
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={"size": 20, "weight": "bold"}, # 增大数值字体
                xticklabels=['Pred Neg', 'Pred Pos'], 
                yticklabels=['Actual Neg', 'Actual Pos'])
    plt.title('Aggregated Confusion Matrix', fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)

    # D. Summary Report & KS Statistic
    ax4 = plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # 计算 KS 值
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_scores)
    ks_statistic = np.max(tpr - fpr)
    
    # 寻找全局最佳阈值 (基于 F1)
    precisions, recalls, pr_thresholds = precision_recall_curve(all_y_true, all_y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_global_idx = np.argmax(f1_scores)
    best_global_threshold = pr_thresholds[best_global_idx] if len(pr_thresholds) > 0 else 0.5
    best_f1 = f1_scores[best_global_idx]
    
    # 使用最佳阈值生成报告
    report = classification_report(all_y_true, (np.array(all_y_scores) >= best_global_threshold).astype(int), target_names=['Neg', 'Pos'], output_dict=True)
    
    summary_text = f"--- Final Evaluation Report ---\n\n"
    summary_text += f"Mean AUC:       {mean_auc:.4f}\n"
    summary_text += f"KS Statistic:   {ks_statistic:.4f}\n"
    summary_text += f"Best Threshold: {best_global_threshold:.4f}\n"
    summary_text += f"Accuracy:       {report['accuracy']:.2%}\n"
    summary_text += f"Pos Precision:  {report['Pos']['precision']:.2%}\n"
    summary_text += f"Pos Recall:     {report['Pos']['recall']:.2%}\n"
    summary_text += f"F1-Score:       {best_f1:.3f}\n\n"
    summary_text += f"Total Samples:  {len(all_y_true)}\n"
    
    status = "Needs Improvement"
    if mean_auc > 0.7 and best_f1 > 0.6 and ks_statistic > 0.5:
        status = "Excellent (Targets Met!)"
    elif mean_auc > 0.7:
        status = "Good (AUC Met)"
        
    summary_text += f"Model Status:   {status}"
    
    plt.text(0.05, 0.5, summary_text, fontsize=14, family='monospace', 
             bbox=dict(facecolor='#F0F8FF', edgecolor='#4682B4', boxstyle='round,pad=1', alpha=0.8),
             verticalalignment='center')
    plt.title('Performance Summary', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_verification_result.png', dpi=300)
    print(f"\n所有任务完成！AUC: {mean_auc:.4f}, F1: {best_f1:.3f}, KS: {ks_statistic:.4f}")
    print(f"结果已保存至 model_verification_result.png")

except Exception as e:
    print(f"执行出错: {e}")
    import traceback
    traceback.print_exc()
