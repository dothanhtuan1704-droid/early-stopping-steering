import json

result_data = {
  "experiment": "Exp09_Main_Placebo_Controls",
  "config": {
    "layer": 8,
    "alpha_0": 18.0,
    "K": 16,
    "decay": "linear",
    "max_new_tokens": 200,
    "n_test": 500
  },
  "results": {
    "original": {
      "refpref_pct": 74.0,
      "bertscore_f1": 0.719750165939331
    },
    "label_shuffled": {
      "refpref_pct": 72.6,
      "bertscore_f1": 0.7175818681716919,
      "n_flipped": 185,
      "cosine_w_original": 0.5839396715164185
    },
    "pair_shuffled": {
      "refpref_pct": 74.0,
      "bertscore_f1": 0.719750165939331,
      "cosine_w_original": 0.9999998807907104
    },
    "isotropic_n100_paper": {
      "mean_refpref": 55.82,
      "std": 4.15
    }
  },
  "deltas": {
    "vs_label": 1.4000000000000057,
    "vs_pair": 0.0
  }
}

out_path = 'exp09_main_placebo_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(result_data, f, indent=2, ensure_ascii=False)

print(f'Saved {out_path} successfully!')
