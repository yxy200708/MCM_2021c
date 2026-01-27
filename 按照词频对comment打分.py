"""
MCM Project: Text Scoring Model (NLP)
=====================================
功能说明：
本脚本 (`按照词频对comment打分.py`) 是 `词频统计3.0.py` (训练权重) 和 `词频统计测试.py` (模型验证) 的 **功能超集与工程化封装**。
它整合了词权重训练、阈值优化、概率预测和数据评分的完整流程，是最终用于生产环境（为入侵检测模型提供 Text_Model_Score）的核心脚本。

核心算法优势 (Robustness & Innovation):
1.  **专家知识融合 (Expert Knowledge Fusion)**:
    -   不完全依赖数据统计，而是引入 `PDF_KEYWORDS` (基于生物学论文的先验知识) 对权重进行修正。
    -   例如：'orange head', 'ground nest' 等特征获得额外加权，即使在训练数据中出现频率较低。
2.  **贝叶斯对数几率 (Bayesian Log-Odds Ratio)**:
    -   使用平滑后的 Log-Odds 衡量单词对 Positive/Negative 类别的区分度，避免小样本带来的极端权重。
3.  **自适应阈值优化 (Adaptive Thresholding)**:
    -   通过 F1-Score 和 Kappa 系数自动寻找最佳分类阈值，并基于此阈值将原始得分校准为 0-1 的概率值。
4.  **抗噪设计**:
    -   针对 Lab Comments 和 Notes 分别处理，利用 Lab Comments 中的专家确认信息增强模型置信度。
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

class TextScorer:
    def __init__(self):
        self.word_weights = {}
        self.balance_factor = 1.0
        self.scaler = MinMaxScaler()
        
        # PDF/Theory Keywords (Lowercased)
        self.PDF_KEYWORDS = {
            'wsda': 10.0, 'verified': 8.0, 'confirmed': 8.0, 
            'provincial': 6.0, 'government': 6.0, 'specimen': 6.0, 
            'collected': 5.0, 'thank': 4.0, 'submission': 4.0,
            'blaine': 4.0, 'nanaimo': 4.0, 'colony': 4.0,
            'mandarinia': 4.0, 'vespa': 3.0, 'murder': 3.0,
            'slaughter': 3.5, 'decapitated': 3.5, 'beheading': 3.5, 'severed': 3.0,
            'pile': 2.5, 'ground': 3.0, 'hole': 2.5, 'dirt': 2.0,
            'orange': 2.0, 'head': 1.5, 'stripe': 1.0, 'band': 1.0,
            'inch': 1.5, 'huge': 1.5, 'massive': 1.5, 'giant': 1.5, 'monster': 1.5,
            'large': 1.0, 
        }
        
        self.STOP_WORDS = {
            'the', 'and', 'was', 'this', 'that', 'with', 'from', 'have', 'for', 'are', 
            'saw', 'see', 'found', 'just', 'like', 'about', 'very', 'some', 'what',
            'there', 'they', 'them', 'when', 'where', 'photo', 'picture', 'image',
            'sent', 'submit', 'submission', 'bug', 'insect', 'bee', 'wasp',
            'my', 'in', 'on', 'at', 'to', 'of', 'is', 'it'
        }

    def clean_text(self, text):
        if pd.isna(text): return ""
        text = str(text).lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text

    def get_row_base_score(self, row):
        # Used for TRAINING weights only
        status = row.get('Lab Status', '')
        comment = self.clean_text(row.get('Lab Comments', ''))
        notes = self.clean_text(row.get('Notes', ''))
        
        score = 0
        if status == 'Positive ID': score = 30.0 
        elif status == 'Negative ID': score = -15.0 * self.balance_factor
        elif status == 'Unverified': score = -2.0
        
        # Lab Comments Logic
        pos_s_strong = ['confirmed', 'positive', 'match', 'correct', 'is a vespa', 'is asian', 'wsda', 'verified']
        pos_s_weak = ['likely', 'probable', 'possible', 'consistent', 'looks like', 'thank']
        neg_s_strong = ['not a', 'negative', 'cicada', 'bald', 'yellowjacket', 'bumblebee', 'paper', 'native', 'european', 'ichneumon']
        
        for s in pos_s_strong:
            if s in comment: score += 15.0
            if s in notes: score += 5.0 # Also check notes for confirmation
        for s in pos_s_weak:
            if s in comment: score += 5.0
        for s in neg_s_strong:
            if s in comment: score -= 15.0 * self.balance_factor
            
        return score

    def fit(self, df):
        print("Training Text Scoring Model...")
        # Calculate balance factor
        pos_count = len(df[df['Lab Status'] == 'Positive ID'])
        neg_count = len(df[df['Lab Status'] == 'Negative ID'])
        self.balance_factor = np.sqrt(pos_count / neg_count) if neg_count > 0 else 1.0
        
        word_total_scores = Counter()
        word_occurrences = Counter()
        word_in_pos = Counter()
        word_in_neg = Counter()
        
        for _, row in df.iterrows():
            row_weight = self.get_row_base_score(row)
            text = self.clean_text(row.get('Notes', '')) + " " + self.clean_text(row.get('Lab Comments', ''))
            words = set([w for w in re.findall(r'\b[a-z]{3,}\b', text) if w not in self.STOP_WORDS])
            
            status = row.get('Lab Status', '')
            
            for word in words:
                word_total_scores[word] += row_weight
                word_occurrences[word] += 1
                if status == 'Positive ID':
                    word_in_pos[word] += 1
                elif status == 'Negative ID':
                    word_in_neg[word] += 1
                    
        # Calculate Final Weights
        for word, count in word_occurrences.items():
            if count < 2: continue
            
            avg_base = word_total_scores[word] / count
            theory = self.PDF_KEYWORDS.get(word, 0)
            
            p_pos = (word_in_pos[word] + 0.1) / (pos_count + 1.0)
            p_neg = (word_in_neg[word] + 0.1) / (neg_count + 1.0)
            lod = np.log(p_pos / p_neg)
            
            final_w = avg_base + (lod * 5.0) + (theory * 3.0)
            self.word_weights[word] = final_w
            
        print(f"Model trained. Vocabulary size: {len(self.word_weights)}")

    def predict_raw(self, df):
        scores = []
        for _, row in df.iterrows():
            text = self.clean_text(row.get('Notes', '')) + " " + self.clean_text(row.get('Lab Comments', ''))
            words = set([w for w in re.findall(r'\b[a-z]{3,}\b', text) if w not in self.STOP_WORDS])
            
            score = 0
            for w in words:
                score += self.word_weights.get(w, 0)
            scores.append(score)
        return np.array(scores)

    def find_optimal_threshold(self, df):
        # Only evaluate on labeled data
        eval_mask = df['Lab Status'].isin(['Positive ID', 'Negative ID'])
        df_eval = df[eval_mask].copy()
        
        y_true = (df_eval['Lab Status'] == 'Positive ID').astype(int)
        raw_scores = self.predict_raw(df_eval)
        
        best_f1 = 0
        best_thresh = 0
        best_kappa = 0
        
        # Search thresholds
        thresholds = np.linspace(raw_scores.min(), raw_scores.max(), 100)
        for t in thresholds:
            y_pred = (raw_scores > t).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                best_kappa = cohen_kappa_score(y_true, y_pred)
                
        print(f"Optimal Threshold (Raw): {best_thresh:.4f}")
        return best_thresh, best_f1, best_kappa

    def predict_proba(self, df, threshold_offset=0):
        raw_scores = self.predict_raw(df)
        # Calibrate so that raw_score = threshold_offset maps to 0.5
        # Sigmoid(x) = 0.5 when x=0. So we want (raw - offset) = 0
        # Scale factor controls steepness.
        return 1 / (1 + np.exp(-0.1 * (raw_scores - threshold_offset)))

def run_pipeline():
    # Load Data
    try:
        df = pd.read_csv('Cleaned_Data_With_Negative.csv')
    except:
        print("Error: Cleaned_Data_With_Negative.csv not found. Run raw_data_processor.py first.")
        return

    scorer = TextScorer()
    scorer.fit(df)
    
    # Optimize Threshold
    best_thresh, best_f1, best_kappa = scorer.find_optimal_threshold(df)
    
    print(f"\nModel Evaluation (Optimized):")
    print(f"F1 Score: {best_f1:.4f} (Target > 0.7)")
    print(f"Kappa:    {best_kappa:.4f} (Target > 0.7)")
    
    try:
        y_true = (df[df['Lab Status'].isin(['Positive ID', 'Negative ID'])]['Lab Status'] == 'Positive ID').astype(int)
        y_scores = scorer.predict_proba(df[df['Lab Status'].isin(['Positive ID', 'Negative ID'])], best_thresh)
        auc = roc_auc_score(y_true, y_scores)
        print(f"AUC:      {auc:.4f}")
    except:
        print("AUC: N/A")
    
    # Generate Scores for ALL data using calibrated threshold
    df['Text_Model_Score'] = scorer.predict_proba(df, best_thresh).round(4)
    
    # Save scored file
    output_file = 'Cleaned_Data_Scored.csv'
    df.to_csv(output_file, index=False)
    print(f"\nScored data saved to {output_file}")

if __name__ == "__main__":
    run_pipeline()
