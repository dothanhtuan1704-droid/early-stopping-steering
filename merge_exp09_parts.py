"""
Merge Exp09 Part 1 (Main Placebo Controls) + Part 2 (Covariance Matched Controls)
"""
import json, os, glob

f1 = glob.glob('*exp09_main_placebo_results*.json') or glob.glob('*exp09*main*.json')
f2 = glob.glob('*exp09_covariance_matched_results*.json') or glob.glob('*exp09*cov*.json')

if not f1 or not f2:
    print('Missing Part 1 or Part 2 JSON files for Exp 09!')
    print(f'Part 1: {f1}')
    print(f'Part 2: {f2}')
    exit(1)

with open(f1[0], 'r', encoding='utf-8') as f:
    d1 = json.load(f)
with open(f2[0], 'r', encoding='utf-8') as f:
    d2 = json.load(f)

res1 = d1['results']
res2 = d2['results']

print('='*80)
print('EXPERIMENT 09 — FINAL MERGED PLACEBO RESULTS (500 samples)')
print('='*80)
print(f'{"Control Type":<40} {"RefPref (%)":<15} {"BERTScore F1":<15}')
print('-'*80)
print(f'{"Original v_steer":<40} {res1["original"]["refpref_pct"]:<15.2f} {res1["original"]["bertscore_f1"]:<15.4f}')
print(f'{"Label-Shuffled":<40} {res1["label_shuffled"]["refpref_pct"]:<15.2f} {res1["label_shuffled"]["bertscore_f1"]:<15.4f}')
print(f'{"Pair-Shuffled":<40} {res1["pair_shuffled"]["refpref_pct"]:<15.2f} {res1["pair_shuffled"]["bertscore_f1"]:<15.4f}')
print(f'{"Cov-Matched (N=3) Mean+/-SD":<40} {res2["cov_matched"]["mean_refpref"]:.2f}+/-{res2["cov_matched"]["std_refpref"]:.2f}{"":5} {res2["cov_matched"]["mean_bertscore_f1"]:<15.4f}')
print(f'{"Isotropic Gaussian (N=100) [paper]":<40} {"55.82+/-4.15":<15} {"0.6945":<15}')
print('='*80)

d_label = res1["original"]["refpref_pct"] - res1["label_shuffled"]["refpref_pct"]
d_pair = res1["original"]["refpref_pct"] - res1["pair_shuffled"]["refpref_pct"]
d_cov = res1["original"]["refpref_pct"] - res2["cov_matched"]["mean_refpref"]

print(f'\nDelta(Original - Label-Shuffled): {d_label:+.2f} pp')
print(f'Delta(Original - Pair-Shuffled):  {d_pair:+.2f} pp')
print(f'Delta(Original - Cov-Matched):    {d_cov:+.2f} pp')

merged_output = {
    'experiment': 'Experiment_09_Stronger_Placebo_Merged_Complete',
    'results': {
        'original_v_steer': res1['original'],
        'label_shuffled': res1['label_shuffled'],
        'pair_shuffled': res1['pair_shuffled'],
        'covariance_matched_n3': res2['cov_matched']
    },
    'deltas': {'vs_label': d_label, 'vs_pair': d_pair, 'vs_cov': d_cov}
}

with open('exp09_final_merged_results.json', 'w', encoding='utf-8') as f:
    json.dump(merged_output, f, indent=2, ensure_ascii=False)

print('\nSaved: exp09_final_merged_results.json')
