import os, json, glob
import numpy as np

xls_dir = 'XLS'
for f in glob.glob(os.path.join(xls_dir, 'exp02*.json')):
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

def evaluate_records(recs, cond_name, max_tok, eos_hits):
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
print("COMPREHENSIVE ANALYSIS OF ALL COMPLETED EXPERIMENT 2 RUNS")
print("================================================================================")

# --- 1. MỐC 200 TOKENS (UNIFIED 500 SAMPLES) ---
print("\n--- [MOC 200 TOKENS (BOUNDED STRESS TEST)] ---")
results_200 = []

for part_num, name in [(1, "Early (1-16)"), (2, "Delayed (17-32)"), (3, "Later (33-48)"), (4, "Continuous (1-inf)")]:
    fn = f"exp02_part{part_num}_{name.split()[0].lower()}_*200tok_results.json"
    matches = glob.glob(fn)
    if not matches:
        fn_alt = f"exp02_part{part_num}_*200tok_results.json"
        matches = glob.glob(fn_alt)
        
    if matches:
        with open(matches[0], 'r', encoding='utf-8') as f:
            d = json.load(f)
        recs = d.get('records', [])
        if not recs and 'refpref_pct' in d:
            # Ready summary json
            print(f"[{name} - 200tok]: {d['total_samples']} samples | Rep-4: {d.get('rep4_pct')}% | EOS: {d.get('eos_hit_rate_pct')}%")
        else:
            res = evaluate_records(recs, name, 200, d.get('eos_hits', 0))
            results_200.append(res)
            print(f"[{name} - 200tok]: {res['total_samples']} samples | Overlap RefPref: {res['overlap_refpref_pct']}% | Rep-4: {res['rep4_pct']}% | EOS: {res['eos_hit_rate_pct']}% | Avg Len: {res['mean_token_length']} tok")
    else:
        print(f"[{name} - 200tok]: [PENDING...]")

# --- 2. MỐC 800 TOKENS (NATURAL COMPLETION) ---
print("\n--- [MOC 800 TOKENS (NATURAL COMPLETION)] ---")
results_800 = []

configs_800 = [
    (1, "Early (1-16)", "early_1_16"),
    (2, "Delayed (17-32)", "delayed_17_32"),
    (3, "Later (33-48)", "later_33_48"),
    (4, "Continuous (1-inf)", "continuous")
]

for part_num, name, kw in configs_800:
    fA = f"exp02_part{part_num}A_{kw}_800tok_0_250.json"
    fB = f"exp02_part{part_num}B_{kw}_800tok_250_500.json"
    
    hasA = os.path.exists(fA)
    hasB = os.path.exists(fB)
    
    if hasA and hasB:
        with open(fA, 'r', encoding='utf-8') as f: dA = json.load(f)
        with open(fB, 'r', encoding='utf-8') as f: dB = json.load(f)
        recs = dA['records'] + dB['records']
        eos = dA.get('eos_hits', 0) + dB.get('eos_hits', 0)
        res = evaluate_records(recs, f"{name} (FULL 500)", 800, eos)
        results_800.append(res)
        print(f"[{name} - 800tok FULL]: 500 samples | Overlap RefPref: {res['overlap_refpref_pct']}% | Rep-4: {res['rep4_pct']}% | EOS: {res['eos_hit_rate_pct']}% | Avg Len: {res['mean_token_length']} tok")
    elif hasA:
        with open(fA, 'r', encoding='utf-8') as f: dA = json.load(f)
        recs = dA['records']
        eos = dA.get('eos_hits', 0)
        res = evaluate_records(recs, f"{name} (Part A 0-250)", 800, eos)
        results_800.append(res)
        print(f"[{name} - 800tok Part A]: 250 samples | Overlap RefPref: {res['overlap_refpref_pct']}% | Rep-4: {res['rep4_pct']}% | EOS: {res['eos_hit_rate_pct']}% | Avg Len: {res['mean_token_length']} tok [Part B Pending]")
    else:
        print(f"[{name} - 800tok]: [PENDING...]")

print("\nAnalysis complete!")
