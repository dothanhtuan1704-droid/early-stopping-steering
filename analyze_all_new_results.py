import os, json, glob
import numpy as np

# Copy all JSON files from XLS to root
xls_dir = 'XLS'
for f in glob.glob(os.path.join(xls_dir, '*.json')):
    if 'huggingface' in f: continue
    fname = os.path.basename(f)
    with open(f, 'r', encoding='utf-8') as src:
        data = json.load(src)
    with open(fname, 'w', encoding='utf-8') as dst:
        json.dump(data, dst, indent=2, ensure_ascii=False)

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

def evaluate_recs(recs, cond_name, max_tok, eos_hits):
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
    
    return {
        "condition": cond_name,
        "max_new_tokens": max_tok,
        "total_samples": len(recs),
        "pref_count": f"{pref_counts}/{len(recs)}",
        "overlap_refpref_pct": round(overlap_refpref, 2),
        "rep4_pct": round(rep4_val, 2),
        "eos_hit_rate_pct": round((eos_hits / len(recs)) * 100.0, 2),
        "mean_token_length": round(float(np.mean(gen_lens)), 2)
    }

print("================================================================================")
print("1. EXPERIMENT 2: INJECTION WINDOW SHIFT CONTROL (800 TOKENS - FULL MATRIX)")
print("================================================================================")

configs_exp02 = [
    (1, "Early (1-16)", "early_1_16"),
    (2, "Delayed (17-32)", "delayed_17_32"),
    (3, "Later (33-48)", "later_33_48"),
    (4, "Continuous (1-inf)", "continuous")
]

exp02_full_summary = {}

for part_num, name, kw in configs_exp02:
    fA = f"exp02_part{part_num}A_{kw}_800tok_0_250.json"
    fB = f"exp02_part{part_num}B_{kw}_800tok_250_500.json"
    
    if os.path.exists(fA) and os.path.exists(fB):
        with open(fA, 'r', encoding='utf-8') as f: dA = json.load(f)
        with open(fB, 'r', encoding='utf-8') as f: dB = json.load(f)
        recs = dA['records'] + dB['records']
        recs.sort(key=lambda x: x['global_idx'])
        eos = dA.get('eos_hits', 0) + dB.get('eos_hits', 0)
        res = evaluate_recs(recs, name, 800, eos)
        exp02_full_summary[name] = res
        print(f"[{name} - 800tok FULL 500]: Rep-4: {res['rep4_pct']}% | EOS: {res['eos_hit_rate_pct']}% | Overlap RefPref: {res['overlap_refpref_pct']}% | Avg Len: {res['mean_token_length']} tok")
    else:
        print(f"[{name} - 800tok]: PENDING (Missing Part A or Part B)")

with open("exp02_delayed_injection_window_800tok_results.json", "w", encoding='utf-8') as f:
    json.dump(exp02_full_summary, f, indent=2, ensure_ascii=False)


print("\n================================================================================")
print("2. EXPERIMENT 4: SYNERGISTIC RAG + STEERING (800 TOKENS - FULL MATRIX)")
print("================================================================================")

configs_exp04 = [
    (1, "Vanilla Baseline (No RAG, No Steering)", "baseline"),
    (2, "Steering Alone (No RAG)", "steering_alone"),
    (3, "Hybrid RAG Alone (No Steering)", "hybrid_rag_alone"),
    (4, "Synergistic Hybrid RAG + Steering", "hybrid_rag_plus_steering")
]

exp04_full_summary = {}

for part_num, name, kw in configs_exp04:
    fA_matches = glob.glob(f"exp04_*A_{kw}*0_250.json") + glob.glob(f"exp04_*A_{kw}*.json")
    fB_matches = glob.glob(f"exp04_*B_{kw}*250_500.json") + glob.glob(f"exp04_*B_{kw}*.json")
    
    fA = fA_matches[0] if fA_matches else None
    fB = fB_matches[0] if fB_matches else None
    
    if fA and fB and os.path.exists(fA) and os.path.exists(fB):
        with open(fA, 'r', encoding='utf-8') as f: dA = json.load(f)
        with open(fB, 'r', encoding='utf-8') as f: dB = json.load(f)
        recs = dA['records'] + dB['records']
        recs.sort(key=lambda x: x['global_idx'])
        eos = dA.get('eos_hits', 0) + dB.get('eos_hits', 0)
        res = evaluate_recs(recs, name, 800, eos)
        exp04_full_summary[name] = res
        print(f"[{name} - 800tok FULL 500]: Overlap RefPref: {res['overlap_refpref_pct']}% | Rep-4: {res['rep4_pct']}% | EOS: {res['eos_hit_rate_pct']}% | Avg Len: {res['mean_token_length']} tok")
    else:
        print(f"[{name} - 800tok]: PENDING (Missing Part A or Part B)")

with open("exp04_synergistic_rag_steering_results.json", "w", encoding='utf-8') as f:
    json.dump(exp04_full_summary, f, indent=2, ensure_ascii=False)

print("\nAnalysis & JSON export completed!")
