import json

b6_data = {
  "batch_idx": 6,
  "v_start": 16,
  "v_end": 18,
  "results": [
    {
      "name": "CovMatch_16",
      "refpref_pct": 73.8,
      "refpref_count": 369,
      "total": 500,
      "bertscore_f1": 0.7179787158966064,
      "gen_time_min": 184.3488885998726,
      "bs_time_min": 0.5075381755828857,
      "avg_latency": 22.119310017108916,
      "avg_tokens": 199.612
    },
    {
      "name": "CovMatch_17",
      "refpref_pct": 74.4,
      "refpref_count": 372,
      "total": 500,
      "bertscore_f1": 0.7180016040802002,
      "gen_time_min": 184.44883581002554,
      "bs_time_min": 0.34396341641743977,
      "avg_latency": 22.127383293151855,
      "avg_tokens": 199.328
    },
    {
      "name": "CovMatch_18",
      "refpref_pct": 75.2,
      "refpref_count": 376,
      "total": 500,
      "bertscore_f1": 0.7176411151885986,
      "gen_time_min": 184.8784129023552,
      "bs_time_min": 0.394710898399353,
      "avg_latency": 22.17845866012573,
      "avg_tokens": 199.048
    }
  ],
  "mean_refpref": 74.46666666666665,
  "std_refpref": 0.5734883511361772,
  "mean_bertscore_f1": 0.7178738117218018,
  "cosines": [
    -0.05942814063600699,
    -0.1409271468824619,
    0.15762421058245552
  ]
}

out_path = 'exp09_covmatch_batch6.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(b6_data, f, indent=2, ensure_ascii=False)

print(f'Saved {out_path} successfully!')
