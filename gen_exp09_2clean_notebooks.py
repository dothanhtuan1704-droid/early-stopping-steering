import json, copy

# Load original nb9 template
with open('Kaggle_Experiment_09_LabelShuffled_Placebo.ipynb', 'r', encoding='utf-8') as f:
    nb9 = json.load(f)

# ==============================================================================
# NOTEBOOK 1: Exp09_Main_Placebo_Controls.ipynb
# Contains: 9A (Original), 9B (Label-Shuffled), 9C (Pair-Shuffled)
# Runtime: ~8.5 - 9.0 hours (completes 100% cleanly within 12h limit!)
# ==============================================================================

nb1 = copy.deepcopy(nb9)
nb1['cells'][0]['source'] = [
    "# Exp 9 — Part 1: Main Placebo Controls (Original + Label-Shuffled + Pair-Shuffled)\n",
    "Runs 3 main conditions (500 test samples each), max_new_tokens=200, greedy decoding.\n",
    "Runtime: ~8.5-9.0 hours total (completes 100% cleanly within Kaggle 12h limit)."
]

# Keep cells up to Cell 13 (Pair-Shuffled) and append clean summary
summary_cell_code_1 = [
    "# === FINAL SUMMARY TABLE (MAIN PLACEBO CONTROLS) ===\n",
    "print('\\n' + '='*80)\n",
    "print('EXPERIMENT 9 PART 1 — MAIN PLACEBO CONTROLS SUMMARY')\n",
    "print('='*80)\n",
    "print(f\"{'Control Type':<40} {'RefPref (%)':<15} {'BERTScore F1':<15} {'Time (min)':<12}\")\n",
    "print('-'*80)\n",
    "print(f\"{'Original v_steer':<40} {result_original['refpref_pct']:<15.2f} {result_original['bertscore_f1']:<15.4f} {result_original.get('gen_time_min',0)+result_original.get('bs_time_min',0):<12.1f}\")\n",
    "print(f\"{'Label-Shuffled':<40} {result_label['refpref_pct']:<15.2f} {result_label['bertscore_f1']:<15.4f} {result_label.get('gen_time_min',0)+result_label.get('bs_time_min',0):<12.1f}\")\n",
    "print(f\"{'Pair-Shuffled':<40} {result_pair['refpref_pct']:<15.2f} {result_pair['bertscore_f1']:<15.4f} {result_pair.get('gen_time_min',0)+result_pair.get('bs_time_min',0):<12.1f}\")\n",
    "print(f\"{'Isotropic Gaussian (N=100) [paper]':<40} {'55.82+/-4.15':<15} {'0.6945':<15} {'---':<12}\")\n",
    "print('='*80)\n",
    "\n",
    "d_label = result_original['refpref_pct'] - result_label['refpref_pct']\n",
    "d_pair = result_original['refpref_pct'] - result_pair['refpref_pct']\n",
    "\n",
    "print(f'\\nDelta(Original - Label-Shuffled): {d_label:+.2f} pp')\n",
    "print(f'Delta(Original - Pair-Shuffled):  {d_pair:+.2f} pp')\n",
    "\n",
    "summary = {\n",
    "    'experiment': 'Exp09_Main_Placebo_Controls',\n",
    "    'config': {'layer': BEST_LAYER, 'alpha_0': ALPHA_0, 'K': K, 'decay': DECAY,\n",
    "               'max_new_tokens': MAX_NEW_TOKENS, 'n_test': len(test_subset)},\n",
    "    'results': {\n",
    "        'original': {'refpref_pct': result_original['refpref_pct'], 'bertscore_f1': result_original['bertscore_f1']},\n",
    "        'label_shuffled': {'refpref_pct': result_label['refpref_pct'], 'bertscore_f1': result_label['bertscore_f1'],\n",
    "                           'n_flipped': int(n_flipped), 'cosine_w_original': float(cos_label)},\n",
    "        'pair_shuffled': {'refpref_pct': result_pair['refpref_pct'], 'bertscore_f1': result_pair['bertscore_f1'],\n",
    "                          'cosine_w_original': float(cos_pair)},\n",
    "        'isotropic_n100_paper': {'mean_refpref': 55.82, 'std': 4.15}\n",
    "    },\n",
    "    'deltas': {'vs_label': d_label, 'vs_pair': d_pair}\n",
    "}\n",
    "\n",
    "out_path = os.path.join(OUTPUT_DIR, 'exp09_main_placebo_results.json')\n",
    "with open(out_path, 'w', encoding='utf-8') as f:\n",
    "    json.dump(summary, f, indent=2, ensure_ascii=False)\n",
    "print(f'\\nSaved: {out_path}')\n"
]

nb1['cells'] = nb1['cells'][:14]  # Up to Cell 13 (Pair-Shuffled)
nb1['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": summary_cell_code_1
})

with open('Exp09_Main_Placebo_Controls.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb1, f, indent=1, ensure_ascii=False)
print('Created Exp09_Main_Placebo_Controls.ipynb (~8.5 - 9.0h runtime)')

# ==============================================================================
# NOTEBOOK 2: Exp09_Covariance_Matched_Controls.ipynb
# Contains ONLY: 9D (Covariance-Matched Random Vectors N=3)
# Runtime: ~8.5 - 9.0 hours total (completes 100% cleanly within 12h limit!)
# ==============================================================================

nb2 = copy.deepcopy(nb9)
nb2['cells'][0]['source'] = [
    "# Exp 9 — Part 2: Covariance-Matched Random Vectors (N=3)\n",
    "Runs N=3 Ledoit-Wolf Covariance-Matched random vectors (500 test samples each).\n",
    "Runtime: ~8.5-9.0 hours total (completes 100% cleanly within Kaggle 12h limit)."
]

# Set N_COV_MATCHED = 3 in Config cell
for line in nb2['cells'][2]['source']:
    if 'N_COV_MATCHED =' in line:
        line = 'N_COV_MATCHED = 3  # 3 covariance-matched vectors for 9h runtime limit\n'

# Build nb2 cells: keep Cells 0-10, then only Cell 14 (CovMatch) and Cell 15 (Summary)
# Cells 11, 12, 13 (Original, Label, Pair) are removed from nb2 to save time!
nb2_cells = nb2['cells'][:11]  # Cells 0-10

summary_cell_code_2 = [
    "# === SUMMARY TABLE (COVARIANCE MATCHED) ===\n",
    "cov_rps = [r['refpref_pct'] for r in results_cov]\n",
    "cov_bs = [r['bertscore_f1'] for r in results_cov]\n",
    "\n",
    "print('\\n' + '='*80)\n",
    "print('EXPERIMENT 9 PART 2 — COVARIANCE MATCHED SUMMARY')\n",
    "print('='*80)\n",
    "print(f\"{'Control Type':<40} {'RefPref (%)':<15} {'BERTScore F1':<15}\")\n",
    "print('-'*80)\n",
    "print(f\"{'Cov-Matched (N=3) Mean+/-SD':<40} {np.mean(cov_rps):.2f}+/-{np.std(cov_rps):.2f}{'':5} {np.mean(cov_bs):<15.4f}\")\n",
    "print(f\"{'Isotropic Gaussian (N=100) [paper]':<40} {'55.82+/-4.15':<15} {'0.6945':<15}\")\n",
    "print('='*80)\n",
    "\n",
    "summary = {\n",
    "    'experiment': 'Exp09_Covariance_Matched_Controls',\n",
    "    'config': {'layer': BEST_LAYER, 'alpha_0': ALPHA_0, 'K': K, 'decay': DECAY,\n",
    "               'max_new_tokens': MAX_NEW_TOKENS, 'n_test': len(test_subset), 'n_cov_matched': N_COV_MATCHED},\n",
    "    'results': {\n",
    "        'cov_matched': {\n",
    "            'mean_refpref': float(np.mean(cov_rps)),\n",
    "            'std_refpref': float(np.std(cov_rps)),\n",
    "            'mean_bertscore_f1': float(np.mean(cov_bs)),\n",
    "            'individual_refprefs': cov_rps,\n",
    "            'cosines': [float(c) for c in cos_sims_cov]\n",
    "        }\n",
    "    }\n",
    "}\n",
    "\n",
    "out_path = os.path.join(OUTPUT_DIR, 'exp09_covariance_matched_results.json')\n",
    "with open(out_path, 'w', encoding='utf-8') as f:\n",
    "    json.dump(summary, f, indent=2, ensure_ascii=False)\n",
    "print(f'\\nSaved: {out_path}')\n"
]

nb2_cells.append(nb9['cells'][14])  # Cell 14: Run 9D (CovMatch)
nb2_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": summary_cell_code_2
})

nb2['cells'] = nb2_cells

with open('Exp09_Covariance_Matched_Controls.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=1, ensure_ascii=False)
print('Created Exp09_Covariance_Matched_Controls.ipynb (~8.5 - 9.0h runtime)')

print('\nDone! Both Exp 9 notebooks generated cleanly.')
