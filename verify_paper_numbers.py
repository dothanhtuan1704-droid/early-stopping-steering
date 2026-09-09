import json, pandas as pd, numpy as np, sys, os
sys.stdout.reconfigure(encoding='utf-8')

print('='*80)
print('CROSS-VERIFICATION 1: PHASE 6B (200 tokens)')
print('Source: phase6b_v2_bertscore_clinical_results.json')
print('='*80)

with open('phase6b_v2_bertscore_clinical_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

base_samples = d['baseline_per_sample']
steer_samples = d['steered_per_sample']
N = len(base_samples)
print(f'N_test = {N}')

base_correct = sum(1 for s in base_samples if s['bertscore_vs_ref'] > s['bertscore_vs_hal'])
steer_correct = sum(1 for s in steer_samples if s['bertscore_vs_ref'] > s['bertscore_vs_hal'])
print(f'Baseline RefPref: {base_correct}/{N} = {100*base_correct/N:.2f}%')
print(f'Steered RefPref:  {steer_correct}/{N} = {100*steer_correct/N:.2f}%')
print(f'Delta: +{100*(steer_correct-base_correct)/N:.2f} pp')

base_bs = [s['bertscore_vs_ref'] for s in base_samples]
steer_bs = [s['bertscore_vs_ref'] for s in steer_samples]
print(f'Baseline BERTScore F1 (vs ref): {np.mean(base_bs):.4f}')
print(f'Steered BERTScore F1 (vs ref):  {np.mean(steer_bs):.4f}')

# McNemar contingency table
a, b, c, dd = 0, 0, 0, 0
for bs_s, st_s in zip(base_samples, steer_samples):
    bc = bs_s['bertscore_vs_ref'] > bs_s['bertscore_vs_hal']
    sc = st_s['bertscore_vs_ref'] > st_s['bertscore_vs_hal']
    if bc and sc: a += 1
    elif bc and not sc: b += 1
    elif not bc and sc: c += 1
    else: dd += 1
print(f'McNemar 2x2: a={a}, b={b}, c={c}, d={dd}')
print(f'  Check: a+b={a+b} (should = baseline correct {base_correct})')
print(f'  Check: a+c={a+c} (should = steered correct {steer_correct})')
print(f'  b+c = {b+c} discordant pairs')

from scipy.stats import binomtest, wilcoxon
result = binomtest(b, b+c, 0.5)
print(f'  Exact binomial McNemar p = {result.pvalue:.6f}')

# Wilcoxon signed-rank test on BERTScore F1
diffs = [st['bertscore_vs_ref'] - bs['bertscore_vs_ref'] for bs, st in zip(base_samples, steer_samples)]
w_stat, w_p = wilcoxon(diffs)
print(f'  Wilcoxon signed-rank p = {w_p:.4f}')

# Latency
base_lat = [s['elapsed_ms']/1000.0 for s in base_samples]
steer_lat = [s['elapsed_ms']/1000.0 for s in steer_samples]
print(f'Baseline Latency: {np.mean(base_lat):.2f}s +/- {np.std(base_lat):.2f}s')
print(f'Steered Latency:  {np.mean(steer_lat):.2f}s +/- {np.std(steer_lat):.2f}s')

# Category-wise
cats = {}
for s in base_samples:
    cat = s.get('category', 'unknown')
    if cat not in cats:
        cats[cat] = {'base_correct': 0, 'steer_correct': 0, 'n': 0}
    cats[cat]['n'] += 1
    if s['bertscore_vs_ref'] > s['bertscore_vs_hal']:
        cats[cat]['base_correct'] += 1
for s in steer_samples:
    cat = s.get('category', 'unknown')
    if s['bertscore_vs_ref'] > s['bertscore_vs_hal']:
        cats[cat]['steer_correct'] += 1
print('\nCategory-wise breakdown:')
for cat, v in sorted(cats.items()):
    ba = 100*v['base_correct']/v['n']
    sa = 100*v['steer_correct']/v['n']
    n = v['n']
    bc2 = v['base_correct']
    sc2 = v['steer_correct']
    print(f'  {cat}: N={n}, Base={bc2}/{n} ({ba:.2f}%), Steer={sc2}/{n} ({sa:.2f}%), Delta={sa-ba:+.2f}pp')

print('\n' + '='*80)
print('CROSS-VERIFICATION 2: PLACEBO N=100')
print('Source: expanded_100_placebo_results.csv (from Output moi/)')
print('='*80)

df_placebo = pd.read_csv('expanded_100_placebo_results.csv')
print(f'Total seeds: {len(df_placebo)}')
print(f'Seed range: {df_placebo["seed"].min()} to {df_placebo["seed"].max()}')
accs = df_placebo['accuracy']
print(f'Mean: {accs.mean():.2f}% +/- {accs.std(ddof=1):.2f}%')
print(f'Min: {accs.min():.2f}%, Max: {accs.max():.2f}%')
z = (77.20 - accs.mean()) / accs.std(ddof=1)
print(f'Z-score of +v_steer (77.20%): +{z:.2f} sigma')
p_emp = (sum(accs >= 77.20) + 1) / (len(accs) + 1)
print(f'Empirical p-value: {p_emp:.4f}')
print(f'Any vector >= 77.20%? {sum(accs >= 77.20)} out of {len(accs)}')

print('\n' + '='*80)
print('CROSS-VERIFICATION 3: RAG BASELINE')
print('Source: rag_evaluation_outputs.csv')
print('='*80)

df_rag = pd.read_csv('rag_evaluation_outputs.csv')
print(f'N_test = {len(df_rag)}')
print(f'Top-1 Recall (retrieval_hit): {df_rag["retrieval_hit"].mean()*100:.2f}%')
print(f'Recall@3: {df_rag["retrieval_hit_top3"].mean()*100:.2f}%')
print(f'Recall@5: {df_rag["retrieval_hit_top5"].mean()*100:.2f}%')
print(f'MRR: {df_rag["mrr"].mean():.4f}')
print(f'Mean prompt_tokens: {df_rag["prompt_tokens"].mean():.1f}')
print(f'Mean generation_latency: {df_rag["generation_latency"].mean():.2f}s +/- {df_rag["generation_latency"].std():.2f}s')

# RAG RefPref = BS_positive > BS_oracle? Need to understand columns
# BS_positive = BERTScore of RAG-generated text vs positive reference
# BS_oracle = BERTScore of oracle-generated text vs positive reference
# Actually for RAG RefPref, we need BS vs ref > BS vs hal
# Let's check if there are hal columns
print(f'Columns: {df_rag.columns.tolist()}')

print('\n' + '='*80)
print('CROSS-VERIFICATION 4: PHASE 6B SUMMARY CSV')
print('Source: phase6b_v2_summary.csv')
print('='*80)

df_sum = pd.read_csv('phase6b_v2_summary.csv')
print(df_sum.to_string(index=False))

# Paper claims vs data
print('\n' + '='*80)
print('PAPER CLAIMS vs VERIFIED DATA')
print('='*80)
claims = [
    ('Baseline RefPref 200tok', '73.00% (365/500)', f'{100*base_correct/N:.2f}% ({base_correct}/{N})'),
    ('Steered RefPref 200tok', '77.20% (386/500)', f'{100*steer_correct/N:.2f}% ({steer_correct}/{N})'),
    ('Delta', '+4.20 pp', f'+{100*(steer_correct-base_correct)/N:.2f} pp'),
    ('McNemar b,c', 'b=16, c=37', f'b={b}, c={c}'),
    ('McNemar p-value', '0.0055', f'{result.pvalue:.4f}'),
    ('Wilcoxon p', '0.017', f'{w_p:.4f}'),
    ('Baseline BERTScore F1', '0.7133', f'{np.mean(base_bs):.4f}'),
    ('Steered BERTScore F1', '0.7150', f'{np.mean(steer_bs):.4f}'),
    ('Placebo Mean', '55.82% +/- 4.15%', f'{accs.mean():.2f}% +/- {accs.std(ddof=1):.2f}%'),
    ('Placebo Z-score', '+5.15 sigma', f'+{z:.2f} sigma'),
    ('Placebo p', '0.0099', f'{p_emp:.4f}'),
    ('RAG Top-1 Recall', '19.20%', f'{df_rag["retrieval_hit"].mean()*100:.2f}%'),
    ('RAG Recall@3', '28.40%', f'{df_rag["retrieval_hit_top3"].mean()*100:.2f}%'),
    ('RAG Recall@5', '32.80%', f'{df_rag["retrieval_hit_top5"].mean()*100:.2f}%'),
    ('RAG MRR', '0.2433', f'{df_rag["mrr"].mean():.4f}'),
    ('RAG prompt_tokens', '75.4', f'{df_rag["prompt_tokens"].mean():.1f}'),
    ('RAG latency', '14.45s +/- 3.83s', f'{df_rag["generation_latency"].mean():.2f}s +/- {df_rag["generation_latency"].std():.2f}s'),
    ('Baseline latency', '7.32s +/- 0.84s', f'{np.mean(base_lat):.2f}s +/- {np.std(base_lat):.2f}s'),
]

for name, paper_val, data_val in claims:
    match = 'OK' if paper_val.strip() == data_val.strip() else 'MISMATCH!'
    print(f'  {match:10s} | {name:30s} | Paper: {paper_val:25s} | Data: {data_val}')
