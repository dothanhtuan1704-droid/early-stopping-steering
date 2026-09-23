import json, glob, sys, gc
import numpy as np
import torch
from bert_score import BERTScorer
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

print("Initializing BERTScorer and Qwen2.5 Tokenizer once...")
scorer = BERTScorer(lang='vi', rescale_with_baseline=False)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

schedules = [
    ("exp02_part4*continuous_800tok*.json", "Continuous (K=inf)"),
    ("exp02_part1*early_1_16_800tok*.json", "Hard Cutoff (K=16)"),
    ("exp04_part1*baseline_800tok*.json", "Unsteered Baseline")
]

prefix_lengths = [50, 100, 200, 400, 800]

def get_prefix_text(text, max_toks):
    toks = tokenizer.encode(text, add_special_tokens=False)
    if len(toks) <= max_toks:
        return text
    truncated_toks = toks[:max_toks]
    return tokenizer.decode(truncated_toks)

prefix_results = {}

for file_pattern, label in schedules:
    files_A = glob.glob(file_pattern.replace("*", "A_*0_250"))
    files_B = glob.glob(file_pattern.replace("*", "B_*250_500"))
    
    if not files_A or not files_B:
        files = glob.glob(file_pattern)
        if len(files) < 2: continue
        files_A, files_B = files[:1], files[1:]

    with open(files_A[0], 'r', encoding='utf-8') as f:
        data_A = json.load(f)
    with open(files_B[0], 'r', encoding='utf-8') as f:
        data_B = json.load(f)
        
    all_500 = (data_A.get('records', []) if isinstance(data_A, dict) else data_A) + \
              (data_B.get('records', []) if isinstance(data_B, dict) else data_B)
    all_500.sort(key=lambda x: x['global_idx'])
    
    print(f"\nProcessing {label} (N={len(all_500)})...")
    
    schedule_scores = {}
    for T in prefix_lengths:
        prefix_texts = [get_prefix_text(r['gen_text'], T) for r in all_500]
        refs_pos = [r['pos_ref'] for r in all_500]
        refs_neg = [r['neg_ref'] for r in all_500]
        
        P_pos, R_pos, F1_pos = scorer.score(prefix_texts, refs_pos)
        P_neg, R_neg, F1_neg = scorer.score(prefix_texts, refs_neg)
        
        pref_counts = sum(1 for p, n in zip(F1_pos.tolist(), F1_neg.tolist()) if p > n)
        refpref_pct = (pref_counts / len(all_500)) * 100.0
        
        schedule_scores[T] = round(refpref_pct, 2)
        print(f"  T={T:3d} tokens: RefPref = {refpref_pct:.2f}% ({pref_counts}/500)")
        
    prefix_results[label] = schedule_scores
    gc.collect()

print("\n" + "="*80)
print("PREFIX LENGTH DISENTANGLEMENT MATRIX (RefPref % by Prefix Length T):")
print("="*80)
print(f"{'Schedule':<25} | {'T=50':<8} | {'T=100':<8} | {'T=200':<8} | {'T=400':<8} | {'T=800':<8}")
print("-" * 75)
for label, scores in prefix_results.items():
    row_str = f"{label:<25} | " + " | ".join([f"{scores.get(T, 0.0):6.2f}%" for T in prefix_lengths])
    print(row_str)

with open('prefix_disentanglement_results.json', 'w', encoding='utf-8') as f:
    json.dump(prefix_results, f, indent=2, ensure_ascii=False)
print("\nSaved prefix_disentanglement_results.json!")
