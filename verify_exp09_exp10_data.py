"""
Script to update paper.tex, FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex, and FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.md
with the exact final empirical results from Exp 09 (N=20 Covariance Matched Placebo) and Exp 10 (500-sample 800-token Rep Penalty & Profiling).
"""
import json, os, re

# Load Exp 09 final merged JSON
with open('exp09_all20_placebo_results.json', 'r', encoding='utf-8') as f:
    exp09_data = json.load(f)

# Load Exp 10 final merged JSON
with open('exp10_merged_500_results.json', 'r', encoding='utf-8') as f:
    exp10_data = json.load(f)

print("Exp09 final summary:", exp09_data['summary']['cov_matched'])
print("Exp10 final summary:", exp10_data['summary'])

# Prepare LaTeX table strings & numbers
cov_mean = exp09_data['summary']['cov_matched']['mean_refpref']
cov_std = exp09_data['summary']['cov_matched']['std_refpref']
cov_bs = exp09_data['summary']['cov_matched']['mean_bertscore_f1']

label_rp = exp09_data['summary']['label_shuffled']['refpref_pct']
label_bs = exp09_data['summary']['label_shuffled']['bertscore_f1']
label_cos = exp09_data['summary']['label_shuffled']['cosine']

# Exp 10 numbers
exp10_base = exp10_data['conditions']['Baseline']
exp10_hard_nopen = exp10_data['conditions']['HardCutoff_NoPen']
exp10_hard_reppen = exp10_data['conditions']['HardCutoff_RepPen']
exp10_lin_nopen = exp10_data['conditions']['LinearDecay_NoPen']
exp10_lin_reppen = exp10_data['conditions']['LinearDecay_RepPen']

print(f"Exp 10 Hard Cutoff RepPen 800 tok: Rep-4={exp10_hard_reppen['rep4']:.2f}%, RefPref={exp10_hard_reppen['refpref_pct']:.2f}%")

print("All metrics loaded cleanly for manuscript update.")
