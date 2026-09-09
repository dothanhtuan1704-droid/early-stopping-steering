import json, os, glob
import numpy as np

try:
    from bert_score import score as bert_score_eval
    from rouge_score import rouge_scorer
    HAS_EVAL_LIBS = True
except ImportError:
    HAS_EVAL_LIBS = False
    print("⚠️ bert_score / rouge_score not installed locally. Will merge raw outputs.")

def calc_rep4(texts):
    total_4grams = 0
    dup_4grams = 0
    for text in texts:
        tokens = text.split()
        if len(tokens) < 4: continue
        fourgrams = [tuple(tokens[i:i+4]) for i in range(len(tokens)-3)]
        total_4grams += len(fourgrams)
        dup_4grams += (len(fourgrams) - len(set(fourgrams)))
    return (dup_4grams / max(1, total_4grams)) * 100.0

def process_merge_for_config(window_name, max_tokens):
    partA_pattern = f"exp02_*A_{window_name.lower()}*_{max_tokens}tok*.json"
    partB_pattern = f"exp02_*B_{window_name.lower()}*_{max_tokens}tok*.json"
    
    files_A = glob.glob(partA_pattern)
    files_B = glob.glob(partB_pattern)
    
    if not files_A or not files_B:
        return None
        
    with open(files_A[0], 'r', encoding='utf-8') as f:
        data_A = json.load(f)
    with open(files_B[0], 'r', encoding='utf-8') as f:
        data_B = json.load(f)
        
    records_A = data_A.get('records', [])
    records_B = data_B.get('records', [])
    all_500 = records_A + records_B
    all_500.sort(key=lambda x: x['global_idx'])
    
    print(f"\nMerging {window_name} [{max_tokens}tok]: Loaded {len(records_A)} (Part A) + {len(records_B)} (Part B) = {len(all_500)} total samples.")
    
    gen_texts = [r['gen_text'] for r in all_500]
    refs_pos = [r['pos_ref'] for r in all_500]
    refs_neg = [r['neg_ref'] for r in all_500]
    gen_lengths = [r['gen_length'] for r in all_500]
    
    if HAS_EVAL_LIBS:
        print("  Evaluating BERTScore over full 500 merged samples...")
        P_pos, R_pos, F1_pos = bert_score_eval(gen_texts, refs_pos, lang='vi', verbose=False)
        P_neg, R_neg, F1_neg = bert_score_eval(gen_texts, refs_neg, lang='vi', verbose=False)
        
        pref_counts = sum(1 for p, n in zip(F1_pos.tolist(), F1_neg.tolist()) if p > n)
        refpref_pct = (pref_counts / len(all_500)) * 100.0
        bert_f1_mean = float(np.mean(F1_pos.tolist()))
        
        rep4_val = calc_rep4(gen_texts)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(r, g)['rougeL'].fmeasure for r, g in zip(refs_pos, gen_texts)]
        rouge_l_mean = float(np.mean(rouge_scores)) * 100.0
    else:
        refpref_pct = 0.0
        bert_f1_mean = 0.0
        rep4_val = calc_rep4(gen_texts)
        rouge_l_mean = 0.0
        pref_counts = 0
        
    eos_hits = data_A.get('eos_hits', 0) + data_B.get('eos_hits', 0)
    
    summary = {
        "window_name": window_name,
        "max_new_tokens": max_tokens,
        "refpref_pct": round(refpref_pct, 2),
        "pref_count": pref_counts,
        "total_samples": len(all_500),
        "bertscore_f1": round(bert_f1_mean, 4),
        "rouge_l_pct": round(rouge_l_mean, 2),
        "rep4_pct": round(rep4_val, 2),
        "eos_hit_rate_pct": round((eos_hits / len(all_500)) * 100.0, 2),
        "mean_gen_length": round(float(np.mean(gen_lengths)), 2)
    }
    return summary

windows = ["early_1_16", "delayed_17_32", "later_33_48", "continuous"]
merged_results_200 = {}
merged_results_800 = {}

for w in windows:
    res200 = process_merge_for_config(w, 200)
    if res200: merged_results_200[w] = res200
    
    res800 = process_merge_for_config(w, 800)
    if res800: merged_results_800[w] = res800

if merged_results_200:
    out200 = 'exp02_delayed_injection_window_results.json'
    with open(out200, 'w', encoding='utf-8') as f:
        json.dump(merged_results_200, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Merged 200tok unified summary saved to '{out200}'.")

if merged_results_800:
    out800 = 'exp02_delayed_injection_window_800tok_results.json'
    with open(out800, 'w', encoding='utf-8') as f:
        json.dump(merged_results_800, f, indent=2, ensure_ascii=False)
    print(f"✅ Merged 800tok unified summary saved to '{out800}'.")
