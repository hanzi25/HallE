import json
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

json_path = "/raid_sdd/zzy/experiments/halle/train/exp10_llava_verifier_logits_scalar_frozen_1.0_joint_6+3+1k_1ep_16bz_3e5/eval/prob_score_new_2/1.0/prob_score_llava_verifier.jsonl"

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化存储数据的列表
verified_lengths = []     # 单词数
v_hallucinations = []
original_lengths = []     # 单词数
o_hallucinations = []

for result in data['results']:
    # Verified Caption处理
    v_caption = result['verified_caption']
    v_words = re.findall(r'\w+', v_caption)
    verified_lengths.append(len(v_words))     # 单词数
    v_hallucinations.append(result['metrics']['hallucinated_words_count_v'])
    
    # Original Caption处理
    o_caption = result['original_caption']
    o_words = re.findall(r'\w+', o_caption)
    original_lengths.append(len(o_words))     # 单词数
    o_hallucinations.append(result['metrics']['hallucinated_words_count_o'])

# ===== 新增统计量计算 =====
def calc_stats(values):
    """计算均值和样本方差"""
    mean = np.mean(values)
    var = np.var(values, ddof=1)  # 无偏方差
    return mean, var

# Verified Caption统计
v_word_mean, v_word_var = calc_stats(verified_lengths)
v_hal_mean, v_hal_var = calc_stats(v_hallucinations)
# Original Caption统计
o_word_mean, o_word_var = calc_stats(original_lengths)
o_hal_mean, o_hal_var = calc_stats(o_hallucinations)
# ===== 打印统计结果 =====
print("=== 文本统计量 ===")
print("Verified Caption:")
print(f"  Seq Len: μ = {v_word_mean:.1f} ± σ² = {v_word_var:.1f}\n")
print(f"  Word Count: μ = {v_word_mean:.1f} ± σ² = {v_word_var:.1f}\n")

print("Original Caption:")
print(f"  Seq Len: μ = {o_word_mean:.1f} ± σ² = {o_word_var:.1f}\n")

# ===== 保留原有分析代码 =====
# 计算相关系数
corr_pearson_v, p_pearson_v = pearsonr(verified_lengths, v_hallucinations)
corr_spearman_v, p_spearman_v = spearmanr(verified_lengths, v_hallucinations)
corr_pearson_o, p_pearson_o = pearsonr(original_lengths, o_hallucinations)
corr_spearman_o, p_spearman_o = spearmanr(original_lengths, o_hallucinations)

# 打印相关分析结果
print("=== 相关性分析 ===")
print(f"Verified Caption: Pearson r = {corr_pearson_v:.3f}, Spearman ρ = {corr_spearman_v:.3f}")
print(f"Original Caption: Pearson r = {corr_pearson_o:.3f}, Spearman ρ = {corr_spearman_o:.3f}")

# 绘制散点图（保持原有可视化代码不变）
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(verified_lengths, v_hallucinations, alpha=0.6, color='blue')
plt.title(f'Verified Caption\nPearson: {corr_pearson_v:.2f} | Spearman: {corr_spearman_v:.2f}')
plt.xlabel('Word Count')
plt.ylabel('Hallucinated Words (V)')

plt.subplot(1, 2, 2)
plt.scatter(original_lengths, o_hallucinations, alpha=0.6, color='green')
plt.title(f'Original Caption\nPearson: {corr_pearson_o:.2f} | Spearman: {corr_spearman_o:.2f}')
plt.xlabel('Word Count')
plt.ylabel('Hallucinated Words (O)')

plt.tight_layout()
plt.savefig('co.png', dpi=300)
plt.show()