import os, json, glob
import numpy as np

xls_dir = 'XLS'
for f in glob.glob(os.path.join(xls_dir, 'exp02*.json')):
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

# 1. Evaluate Merged Early (1-16) 800tok (500 samples)
fA = 'exp02_part1A_early_1_16_800tok_0_250.json'
fB = 'exp02_part1B_early_1_16_800tok_250_500.json'

if os.path.exists(fA) and os.path.exists(fB):
    with open(fA, 'r', encoding='utf-8') as f: dA = json.load(f)
    with open(fB, 'r', encoding='utf-8') as f: dB = json.load(f)
    
    recs = dA['records'] + dB['records']
    recs.sort(key=lambda x: x['global_idx'])
    
    gen_texts = [r['gen_text'] for r in recs]
    refs_pos = [r['pos_ref'] for r in recs]
    refs_neg = [r['neg_ref'] for r in recs]
    gen_lens = [r['gen_length'] for r in recs]
    
    print(f"\n================================================================================")
    print(f"MERGED RESULTS FOR: Early (1-16) at 800 tokens (N=500 samples)")
    print(f"================================================================================")
    
    pref_counts = 0
    for g, p, n in zip(gen_texts, refs_pos, refs_neg):
        score_p = calc_overlap_score(g, p)
        score_n = calc_overlap_score(g, n)
        if score_p > score_n:
            pref_counts += 1
            
    overlap_refpref = (pref_counts / len(recs)) * 100.0
    rep4_val = calc_rep4(gen_texts)
    eos_hits = dA.get('eos_hits', 0) + dB.get('eos_hits', 0)
    
    summary_early = {
        "condition": "Early (1-16)",
        "max_new_tokens": 800,
        "total_samples": len(recs),
        "overlap_refpref_pct": round(overlap_refpref, 2),
        "rep4_pct": round(rep4_val, 2),
        "eos_hit_rate_pct": round((eos_hits / len(recs)) * 100.0, 2),
        "mean_token_length": round(float(np.mean(gen_lens)), 2)
    }
    
    print(json.dumps(summary_early, indent=2))

# 2. Evaluate Part A of Delayed (17-32) 800tok (250 samples)
fA_del = 'exp02_part2A_delayed_17_32_800tok_0_250.json'
if os.path.exists(fA_del):
    with open(fA_del, 'r', encoding='utf-8') as f: dDelA = json.load(f)
    recs = dDelA['records']
    gen_texts = [r['gen_text'] for r in recs]
    refs_pos = [r['pos_ref'] for r in recs]
    refs_neg = [r['neg_ref'] for r in recs]
    gen_lens = [r['gen_length'] for r in recs]
    
    print(f"\n================================================================================")
    print(f"PARTIAL RESULTS FOR: Delayed (17-32) Part A at 800 tokens (N=250 samples)")
    print(f"================================================================================")
    
    pref_counts = 0
    for g, p, n in zip(gen_texts, refs_pos, refs_neg):
        score_p = calc_overlap_score(g, p)
        score_n = calc_overlap_score(g, n)
        if score_p > score_n:
            pref_counts += 1
            
    overlap_refpref = (pref_counts / len(recs)) * 100.0
    rep4_val = calc_rep4(gen_texts)
    
    summary_delayed_A = {
        "condition": "Delayed (17-32) Part A (0-250)",
        "max_new_tokens": 800,
        "total_samples": len(recs),
        "overlap_refpref_pct": round(overlap_refpref, 2),
        "rep4_pct": round(rep4_val, 2),
        "eos_hit_rate_pct": round((dDelA.get('eos_hits', 0) / len(recs)) * 100.0, 2),
        "mean_token_length": round(float(np.mean(gen_lens)), 2)
    }
    
    print(json.dumps(summary_delayed_A, indent=2))
