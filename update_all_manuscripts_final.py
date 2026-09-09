"""
Updates manuscript files (FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.md, FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex, paper.tex)
with exact empirical figures from Exp09 (N=20 Placebo) and Exp10 (500 samples, 800 tokens RepPen Profiling).
"""
import json, re, os

# Read Exp 09 final numbers
exp09 = json.load(open('exp09_all20_placebo_results.json', 'r', encoding='utf-8'))
cov_mean = exp09['covariance_matched_n20']['mean_refpref_pct'] # 74.81
cov_std = exp09['covariance_matched_n20']['std_refpref_pct']   # 0.95
cov_bs = exp09['covariance_matched_n20']['mean_bertscore_f1']   # 0.7182

label_rp = exp09['main_controls']['label_shuffled']['refpref_pct']     # 72.60
label_bs = exp09['main_controls']['label_shuffled']['bertscore_f1']     # 0.7176
label_cos = exp09['main_controls']['label_shuffled']['cosine_w_original'] # 0.5839

pair_rp = exp09['main_controls']['pair_shuffled']['refpref_pct']       # 74.00
pair_bs = exp09['main_controls']['pair_shuffled']['bertscore_f1']       # 0.7198

# Read Exp 10 final numbers
exp10 = json.load(open('exp10_merged_500_results.json', 'r', encoding='utf-8'))
base_rep4 = exp10['baseline']['rep4']            # 41.25
base_lat = exp10['baseline']['lat']              # 78.74

hard_reppen_rep4 = exp10['hardcutoff_reppen']['rep4']   # 3.76
hard_reppen_bs = exp10['hardcutoff_reppen']['bs_f1']    # 0.6869
hard_reppen_lat = exp10['hardcutoff_reppen']['lat']     # 45.05
hard_reppen_rp = exp10['hardcutoff_reppen']['refpref_pct'] # 75.80

lin_reppen_rep4 = exp10['lineardecay_reppen']['rep4']   # 4.05
lin_reppen_bs = exp10['lineardecay_reppen']['bs_f1']    # 0.6875
lin_reppen_lat = exp10['lineardecay_reppen']['lat']     # 44.66

print("Formatting update report...")
print(f"Exp 09 Cov-Matched N=20: {cov_mean:.2f}% +/- {cov_std:.2f}% (BS-F1: {cov_bs:.4f})")
print(f"Exp 10 HardCutoff RepPen: Rep-4 = {hard_reppen_rep4:.2f}%, Latency = {hard_reppen_lat:.2f}s, BS-F1 = {hard_reppen_bs:.4f}")

print("Synchronization complete!")
