"""
Merge Exp 9 Main Placebo Results + All 7 Covariance Matched Batches (N=20 vectors total)
Usage: Copy all exp09_main_placebo_results.json and exp09_covmatch_batch1..7.json files here, then run.
"""
import json, os, glob
import numpy as np

OUTPUT_DIR = '.'

# 1. Main controls (Original, Label-Shuffled, Pair-Shuffled)
main_files = glob.glob('*exp09_main_placebo_results*.json') or glob.glob('*exp09*main*.json')
if not main_files:
    print('WARNING: exp09_main_placebo_results.json not found yet')
    d_main = None
else:
    with open(main_files[0], 'r', encoding='utf-8') as f:
        d_main = json.load(f)

# 2. Covariance Matched Batches (1..7)
all_cov_results = []
all_cov_refprefs = []
all_cov_bscores = []
all_cosines = []

for b in range(1, 8):
    bfiles = glob.glob(f'*exp09_covmatch_batch{b}.json') or glob.glob(f'*batch{b}.json')
    if bfiles:
        with open(bfiles[0], 'r', encoding='utf-8') as f:
            bdata = json.load(f)
        for r in bdata.get('results', []):
            all_cov_results.append(r)
            all_cov_refprefs.append(r['refpref_pct'])
            all_cov_bscores.append(r['bertscore_f1'])
        all_cosines.extend(bdata.get('cosines', []))
        print(f'Batch {b}: Loaded {len(bdata.get("results", []))} vectors')
    else:
        print(f'Batch {b}: Not loaded yet')

print('='*80)
print('EXPERIMENT 09 — ULTIMATE PLACEBO REPORT (N=20 COVARIANCE MATCHED VECTORS)')
print('='*80)
print(f'{"Control Type":<45} {"RefPref (%)":<15} {"BERTScore F1":<15}')
print('-'*80)

if d_main:
    res = d_main['results']
    print(f'{"Original v_steer":<45} {res["original"]["refpref_pct"]:<15.2f} {res["original"]["bertscore_f1"]:<15.4f}')
    print(f'{"Label-Shuffled":<45} {res["label_shuffled"]["refpref_pct"]:<15.2f} {res["label_shuffled"]["bertscore_f1"]:<15.4f}')
    print(f'{"Pair-Shuffled":<45} {res["pair_shuffled"]["refpref_pct"]:<15.2f} {res["pair_shuffled"]["bertscore_f1"]:<15.4f}')

if all_cov_refprefs:
    m_rp = np.mean(all_cov_refprefs)
    s_rp = np.std(all_cov_refprefs)
    m_bs = np.mean(all_cov_bscores)
    print(f'{"Cov-Matched (N=" + str(len(all_cov_refprefs)) + ") Mean+/-SD":<45} {m_rp:.2f}+/-{s_rp:.2f}{"":5} {m_bs:<15.4f}')

print(f'{"Isotropic Gaussian (N=100) [paper]":<45} {"55.82+/-4.15":<15} {"0.6945":<15}')
print('='*80)

final_summary = {
    'experiment': 'Experiment_09_All_20_Covariance_Matched_Controls',
    'main_controls': d_main['results'] if d_main else None,
    'covariance_matched_n20': {
        'count': len(all_cov_refprefs),
        'mean_refpref_pct': float(np.mean(all_cov_refprefs)) if all_cov_refprefs else None,
        'std_refpref_pct': float(np.std(all_cov_refprefs)) if all_cov_refprefs else None,
        'min_refpref_pct': float(np.min(all_cov_refprefs)) if all_cov_refprefs else None,
        'max_refpref_pct': float(np.max(all_cov_refprefs)) if all_cov_refprefs else None,
        'mean_bertscore_f1': float(np.mean(all_cov_bscores)) if all_cov_bscores else None,
        'individual_refprefs': all_cov_refprefs,
        'cosines_with_original': all_cosines
    }
}

out_path = 'exp09_all20_placebo_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(final_summary, f, indent=2, ensure_ascii=False)
print(f'\nSaved complete results: {out_path}')
