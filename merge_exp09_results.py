"""
Merge Exp09 Part A (samples 0-249) + Part B (samples 250-499) results.
"""
import json, os, glob
import numpy as np

pa_files = glob.glob('*exp09_placebo_parta*.json') or glob.glob('*exp09*parta*.json')
pb_files = glob.glob('*exp09_placebo_partb*.json') or glob.glob('*exp09*partb*.json')

if not pa_files or not pb_files:
    print('Missing Part A or Part B files for Exp 09!')
    print(f'Part A: {pa_files}')
    print(f'Part B: {pb_files}')
    exit(1)

with open(pa_files[0], 'r', encoding='utf-8') as f:
    ra = json.load(f)
with open(pb_files[0], 'r', encoding='utf-8') as f:
    rb = json.load(f)

# Helper to merge 2 condition results
def merge_cond(ca, cb):
    na = ca['total']
    nb = cb['total']
    ntot = na + nb
    return {
        'refpref_pct': (ca['refpref_pct'] * na + cb['refpref_pct'] * nb) / ntot,
        'refpref_count': ca['refpref_count'] + cb['refpref_count'],
        'total': ntot,
        'bertscore_f1': (ca['bertscore_f1'] * na + cb['bertscore_f1'] * nb) / ntot
    }

res_a = ra['results']
res_b = rb['results']

orig = merge_cond(res_a['original_v_steer'], res_b['original_v_steer'])
label = merge_cond(res_a['label_shuffled'], res_b['label_shuffled'])
pair = merge_cond(res_a['pair_shuffled'], res_b['pair_shuffled'])

# Covariance matched (mean of N=3 across both halves)
cov_a_rps = res_a['covariance_matched']['individual_refprefs']
cov_b_rps = res_b['covariance_matched']['individual_refprefs']
cov_rps_all = [(cov_a_rps[i] + cov_b_rps[i])/2.0 for i in range(min(len(cov_a_rps), len(cov_b_rps)))]

print('='*80)
print('EXPERIMENT 09 — MERGED RESULTS (500 samples total)')
print('='*80)
print(f'{"Control Type":<40} {"RefPref (%)":<15} {"BERTScore F1":<15}')
print('-'*80)
print(f'{"Original v_steer":<40} {orig["refpref_pct"]:<15.2f} {orig["bertscore_f1"]:<15.4f}')
print(f'{"Label-Shuffled":<40} {label["refpref_pct"]:<15.2f} {label["bertscore_f1"]:<15.4f}')
print(f'{"Pair-Shuffled":<40} {pair["refpref_pct"]:<15.2f} {pair["bertscore_f1"]:<15.4f}')
print(f'{"Cov-Matched (N=3) Mean+/-SD":<40} {np.mean(cov_rps_all):.2f}+/-{np.std(cov_rps_all):.2f}{"":5} {np.mean([res_a["covariance_matched"]["mean_bertscore_f1"], res_b["covariance_matched"]["mean_bertscore_f1"]]):<15.4f}')
print(f'{"Isotropic Gaussian (N=100) [paper]":<40} {"55.82+/-4.15":<15} {"0.6945":<15}')
print('='*80)

d_label = orig['refpref_pct'] - label['refpref_pct']
d_pair = orig['refpref_pct'] - pair['refpref_pct']
d_cov = orig['refpref_pct'] - np.mean(cov_rps_all)

print(f'\nDelta(Original - Label-Shuffled): {d_label:+.2f} pp')
print(f'Delta(Original - Pair-Shuffled):  {d_pair:+.2f} pp')
print(f'Delta(Original - Cov-Matched):    {d_cov:+.2f} pp')

merged_output = {
    'experiment': 'Experiment_09_Stronger_Placebo_Merged_500',
    'results': {
        'original_v_steer': orig,
        'label_shuffled': label,
        'pair_shuffled': pair,
        'covariance_matched_n3': {
            'mean_refpref_pct': float(np.mean(cov_rps_all)),
            'std_refpref_pct': float(np.std(cov_rps_all)),
            'individual_refprefs': cov_rps_all
        }
    },
    'deltas': {'vs_label': d_label, 'vs_pair': d_pair, 'vs_cov': d_cov}
}

with open('exp09_merged_500_results.json', 'w', encoding='utf-8') as f:
    json.dump(merged_output, f, indent=2, ensure_ascii=False)

print('\nSaved: exp09_merged_500_results.json')
