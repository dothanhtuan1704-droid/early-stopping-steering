"""
Merge Exp10 Part A + Part B results into combined 500-sample results.
Usage: Copy all exp10_*_parta.json and exp10_*_partb.json files here, then run.
"""
import json, os, glob
import numpy as np

OUTPUT_DIR = '.'  # or wherever the JSONs are

conditions = [
    'baseline',
    'hardcutoff_nopen', 
    'hardcutoff_reppen',
    'lineardecay_nopen',
    'lineardecay_reppen'
]

merged_results = {}

for cond in conditions:
    pa_file = glob.glob(f'*{cond}*parta*.json') or glob.glob(f'*{cond}*partA*.json')
    pb_file = glob.glob(f'*{cond}*partb*.json') or glob.glob(f'*{cond}*partB*.json')
    
    if not pa_file or not pb_file:
        print(f'WARNING: Missing files for {cond}')
        print(f'  Part A: {pa_file}')
        print(f'  Part B: {pb_file}')
        continue
    
    with open(pa_file[0], 'r', encoding='utf-8') as f:
        ra = json.load(f)
    with open(pb_file[0], 'r', encoding='utf-8') as f:
        rb = json.load(f)
    
    # Merge numeric metrics (weighted average)
    na = ra['total']
    nb_total = rb['total']
    n_total = na + nb_total
    
    merged = {
        'name': cond,
        'total': n_total,
        'refpref_pct': (ra['refpref_pct'] * na + rb['refpref_pct'] * nb_total) / n_total,
        'refpref_n': ra['refpref_n'] + rb['refpref_n'],
        'bs_f1': (ra['bs_f1'] * na + rb['bs_f1'] * nb_total) / n_total,
        'rep4': (ra['rep4'] * na + rb['rep4'] * nb_total) / n_total,
        'eos_pct': (ra['eos_pct'] * na + rb['eos_pct'] * nb_total) / n_total,
        'eos_n': ra['eos_n'] + rb['eos_n'],
        'lat': (ra['lat'] * na + rb['lat'] * nb_total) / n_total,
        'tok': (ra['tok'] * na + rb['tok'] * nb_total) / n_total,
        'peak_gpu': max(ra.get('peak_gpu', 0), rb.get('peak_gpu', 0)),
        'rep_pen': ra.get('rep_pen', 1.0),
        'part_a': ra,
        'part_b': rb
    }
    
    merged_results[cond] = merged
    print(f'{cond}: RefPref={merged["refpref_pct"]:.2f}% ({merged["refpref_n"]}/{n_total}), '
          f'BS-F1={merged["bs_f1"]:.4f}, Rep4={merged["rep4"]:.2f}%, '
          f'EOS={merged["eos_pct"]:.1f}%')

# Print comparison table
print(f'\n{"="*100}')
print(f'EXPERIMENT 10 — MERGED RESULTS (500 samples, 800 tokens)')
print(f'{"="*100}')
print(f'{"Condition":<25} {"RefPref%":<10} {"BS-F1":<10} {"Rep4%":<10} {"EOS%":<8} {"Lat(s)":<10} {"Tokens":<8}')
print('-'*100)
for cond in conditions:
    if cond in merged_results:
        r = merged_results[cond]
        print(f'{cond:<25} {r["refpref_pct"]:<10.2f} {r["bs_f1"]:<10.4f} {r["rep4"]:<10.2f} '
              f'{r["eos_pct"]:<8.1f} {r["lat"]:<10.2f} {r["tok"]:<8.1f}')

# Key comparisons
if 'hardcutoff_nopen' in merged_results and 'hardcutoff_reppen' in merged_results:
    h_no = merged_results['hardcutoff_nopen']
    h_rp = merged_results['hardcutoff_reppen']
    print(f'\n--- HARD CUTOFF: rep_penalty=1.15 effect ---')
    print(f'  RefPref: {h_no["refpref_pct"]:.2f}% -> {h_rp["refpref_pct"]:.2f}% ({h_rp["refpref_pct"]-h_no["refpref_pct"]:+.2f}pp)')
    print(f'  Rep-4:   {h_no["rep4"]:.2f}% -> {h_rp["rep4"]:.2f}% ({h_rp["rep4"]-h_no["rep4"]:+.2f}pp)')

if 'lineardecay_nopen' in merged_results and 'lineardecay_reppen' in merged_results:
    l_no = merged_results['lineardecay_nopen']
    l_rp = merged_results['lineardecay_reppen']
    print(f'\n--- LINEAR DECAY: rep_penalty=1.15 effect ---')
    print(f'  RefPref: {l_no["refpref_pct"]:.2f}% -> {l_rp["refpref_pct"]:.2f}% ({l_rp["refpref_pct"]-l_no["refpref_pct"]:+.2f}pp)')
    print(f'  Rep-4:   {l_no["rep4"]:.2f}% -> {l_rp["rep4"]:.2f}% ({l_rp["rep4"]-l_no["rep4"]:+.2f}pp)')

# Save merged
with open('exp10_merged_500_results.json', 'w', encoding='utf-8') as f:
    json.dump(merged_results, f, indent=2, ensure_ascii=False)
print(f'\nSaved: exp10_merged_500_results.json')
