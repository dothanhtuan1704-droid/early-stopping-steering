import json, copy, os

# Generator for 7 Covariance-Matched Notebooks (N=20 total, 3 vectors per notebook)
# Each notebook runs 3 covariance-matched vectors on 500 test samples (~8.5-9.0 hours runtime)

with open('Kaggle_Experiment_09_LabelShuffled_Placebo.ipynb', 'r', encoding='utf-8') as f:
    template = json.load(f)

# Define the 7 batches
batches = [
    (1, 0, 3),   # Vectors 1..3 (indices 0, 1, 2)
    (2, 3, 6),   # Vectors 4..6 (indices 3, 4, 5)
    (3, 6, 9),   # Vectors 7..9 (indices 6, 7, 8)
    (4, 9, 12),  # Vectors 10..12 (indices 9, 10, 11)
    (5, 12, 15), # Vectors 13..15 (indices 12, 13, 14)
    (6, 15, 18), # Vectors 16..18 (indices 15, 16, 17)
    (7, 18, 20), # Vectors 19..20 (indices 18, 19)
]

for batch_idx, v_start, v_end in batches:
    fname = f'Exp09_CovMatch_Batch{batch_idx}.ipynb'
    nb = copy.deepcopy(template)
    
    # Title Markdown (Cell 0)
    nb['cells'][0]['source'] = [
        f"# Exp 9: Covariance-Matched Random Vectors — Batch {batch_idx}/7 (Vectors {v_start+1} to {v_end})\n",
        f"Evaluates Covariance-Matched vectors {v_start+1}..{v_end} on 500 test samples (max_new_tokens=200, greedy).\n",
        f"Runtime: ~{ (v_end - v_start) * 2.9:.1f} hours total (completes 100% cleanly within Kaggle 12h limit)."
    ]
    
    # Cell 9: Covariance vector sampling cell - update to extract only v_start..v_end
    cov_sampling_code = [
        f"# Compute Covariance-Matched Random Vectors for Batch {batch_idx} (Vectors {v_start+1} to {v_end})\n",
        f"print('Computing covariance-matched random vectors for Batch {batch_idx}...')\n",
        "diff_vectors = activations_pos - activations_neg\n",
        "from sklearn.covariance import LedoitWolf\n",
        "print('  Fitting Ledoit-Wolf shrinkage covariance estimator...')\n",
        "lw = LedoitWolf()\n",
        "lw.fit(diff_vectors)\n",
        "cov_matrix = lw.covariance_\n",
        "print(f'  Covariance matrix: {cov_matrix.shape}, shrinkage={lw.shrinkage_:.4f}')\n",
        "try:\n",
        "    L = np.linalg.cholesky(cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0]))\n",
        "except np.linalg.LinAlgError:\n",
        "    eigvals, eigvecs = np.linalg.eigh(cov_matrix)\n",
        "    eigvals = np.maximum(eigvals, 1e-6)\n",
        "    L = eigvecs @ np.diag(np.sqrt(eigvals))\n",
        "\n",
        "cov_matched_vectors = []\n",
        "cos_sims_cov = []\n",
        f"V_START = {v_start}\n",
        f"V_END = {v_end}\n",
        f"BATCH_IDX = {batch_idx}\n",
        "for i in range(V_START, V_END):\n",
        "    np.random.seed(SEED + 3000 + i)\n",
        "    z = np.random.randn(cov_matrix.shape[0])\n",
        "    v_cov = L @ z\n",
        "    v_cov_norm = v_cov / np.linalg.norm(v_cov)\n",
        "    cov_matched_vectors.append(torch.tensor(v_cov_norm, dtype=torch.bfloat16, device='cuda'))\n",
        "    cs = np.dot(v_steer_np, v_cov_norm)\n",
        "    cos_sims_cov.append(cs)\n",
        "    print(f'  CovMatched-{i+1:02d}: cos(v_steer, v_cov)={cs:.4f}')\n",
        "\n",
        f"print(f'Batch {batch_idx}: {{len(cov_matched_vectors)}} vectors computed (seeds {{SEED+3000+V_START}} to {{SEED+3000+V_END-1}})')\n"
    ]
    nb['cells'][9]['source'] = cov_sampling_code
    
    # Cells to keep: 0-5 (install, config, data, model, activations), 6 (v_steer), 9 (cov sampling), 10 (eval engine), 14 (run cov match), 15 (summary)
    keep_cells = [
        nb['cells'][0], nb['cells'][1], nb['cells'][2], nb['cells'][3], nb['cells'][4], nb['cells'][5],
        nb['cells'][6], nb['cells'][9], nb['cells'][10]
    ]
    
    # Update Cell 14 (Run CovMatch) to fix BATCH_IDX and prevent NameError
    run_cov_code = [
        f"# === RUN BATCH {batch_idx}: COVARIANCE MATCHED VECTORS {v_start+1}..{v_end} ===\n",
        f"print('\\n' + '#'*70)\n",
        f"print('# BATCH {batch_idx}: COVARIANCE MATCHED VECTORS {v_start+1}..{v_end}')\n",
        f"print('#'*70)\n",
        "results_cov = []\n",
        "for idx_in_batch, v_c in enumerate(cov_matched_vectors):\n",
        "    abs_v_idx = V_START + idx_in_batch + 1\n",
        "    print(f'\\n--- Running Covariance-Matched Vector {abs_v_idx}/20 ---')\n",
        "    r = generate_outputs(model, tokenizer, test_subset,\n",
        "        v_vector=v_c, alpha=ALPHA_0, K=K, decay=DECAY,\n",
        "        max_new_tokens=MAX_NEW_TOKENS, layer_idx=BEST_LAYER,\n",
        "        name=f'CovMatch_{abs_v_idx:02d}')\n",
        "    results_cov.append(r)\n",
        "    gc.collect(); torch.cuda.empty_cache()\n",
        "    running_rps = [x['refpref_pct'] for x in results_cov]\n",
        f"    print(f'  Batch {batch_idx} running avg ({{len(results_cov)}}/{{len(cov_matched_vectors)}}): RefPref={{np.mean(running_rps):.2f}}%')\n"
    ]
    keep_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": run_cov_code
    })
    
    # Output JSON summary cell
    out_json_name = f'exp09_covmatch_batch{batch_idx}.json'
    summary_code = [
        f"# === SUMMARY & SAVE BATCH {batch_idx} ===\n",
        "cov_rps = [r['refpref_pct'] for r in results_cov]\n",
        "cov_bs = [r['bertscore_f1'] for r in results_cov]\n",
        "print('\\n' + '='*80)\n",
        f"print('COVARIANCE MATCHED BATCH {batch_idx} SUMMARY')\n",
        "print('='*80)\n",
        "for r in results_cov:\n",
        "    print(f\"{r['name']:<20} RefPref: {r['refpref_pct']:.2f}% | BS-F1: {r['bertscore_f1']:.4f}\")\n",
        "print(f'Batch Mean RefPref: {np.mean(cov_rps):.2f}% +/- {np.std(cov_rps):.2f}%')\n",
        "\n",
        "batch_summary = {\n",
        "    'batch_idx': " + str(batch_idx) + ",\n",
        "    'v_start': V_START + 1, 'v_end': V_END,\n",
        "    'results': results_cov,\n",
        "    'mean_refpref': float(np.mean(cov_rps)),\n",
        "    'std_refpref': float(np.std(cov_rps)),\n",
        "    'mean_bertscore_f1': float(np.mean(cov_bs)),\n",
        "    'cosines': [float(c) for c in cos_sims_cov]\n",
        "}\n",
        f"out_path = os.path.join(OUTPUT_DIR, '{out_json_name}')\n",
        "with open(out_path, 'w', encoding='utf-8') as f:\n",
        "    json.dump(batch_summary, f, indent=2, ensure_ascii=False)\n",
        "print(f'\\nSaved: {out_path}')\n"
    ]
    keep_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": summary_code
    })
    
    nb['cells'] = keep_cells
    
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Fixed: {fname} (Batch {batch_idx}, vectors {v_start+1}..{v_end})')

print(f'\nTotal: All 7 Covariance-Matched notebooks regenerated cleanly with fixed BATCH_IDX!')
