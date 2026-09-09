import os, json, glob
import numpy as np

xls_dir = 'XLS'
for f in glob.glob(os.path.join(xls_dir, 'exp04*.json')):
    fname = os.path.basename(f)
    with open(f, 'r', encoding='utf-8') as src:
        data = json.load(src)
    with open(fname, 'w', encoding='utf-8') as dst:
        json.dump(data, dst, indent=2, ensure_ascii=False)
    print(f"Copied {fname} from XLS to root directory.")

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

def calc_overlap_score(gen, ref):
    g_toks = set(gen.lower().split())
    r_toks = set(ref.lower().split())
    if not r_toks: return 0.0
    return len(g_toks.intersection(r_toks)) / len(r_toks)

print("================================================================================")
print("EXPERIMENT 4 (SYNERGISTIC RAG + STEERING) ANALYSIS")
print("================================================================================")

# 1. Condition 1: Baseline (Part 1A + Part 1B)
f1A = 'exp04_part1A_baseline_800tok_0_250.json'
f1B = 'exp04_part1B_baseline_250_500.json'

if os.path.exists(f1A) and os.path.exists(f1B):
    with open(f1A, 'r', encoding='utf-8') as f: d1A = json.load(f)
    with open(f1B, 'r', encoding='utf-8') as f: d1B = json.load(f)
    
    recs = d1A['records'] + d1B['records']
    recs.sort(key=lambda x: x['global_idx'])
    
    gen_texts = [r['gen_text'] for r in recs]
    refs_pos = [r.get('pos_ref', r.get('right_answer', '')) for r in recs]
    refs_neg = [r.get('neg_ref', r.get('hallucinated_answer', '')) for r in recs]
    gen_lens = [r['gen_length'] for r in recs]
    
    pref_counts = 0
    for g, p, n in zip(gen_texts, refs_pos, refs_neg):
        score_p = calc_overlap_score(g, p)
        score_n = calc_overlap_score(g, n)
        if score_p > score_n:
            pref_counts += 1
            
    overlap_refpref = (pref_counts / len(recs)) * 100.0
    rep4_val = calc_rep4(gen_texts)
    eos_hits = d1A.get('eos_hits', 0) + d1B.get('eos_hits', 0)
    
    summary_c1 = {
        "condition": "Exp04 Condition 1: Vanilla Baseline (No RAG, No Steering)",
        "max_new_tokens": 800,
        "total_samples": len(recs),
        "pref_count": f"{pref_counts}/500",
        "overlap_refpref_pct": round(overlap_refpref, 2),
        "rep4_pct": round(rep4_val, 2),
        "eos_hit_rate_pct": round((eos_hits / len(recs)) * 100.0, 2),
        "mean_token_length": round(float(np.mean(gen_lens)), 2)
    }
    
    print("\n--- [CONDITION 1: VANILLA BASELINE (FULL 500 SAMPLES)] ---")
    print(json.dumps(summary_c1, indent=2))
