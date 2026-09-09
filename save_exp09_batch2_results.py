import json

b2_data = {
  "batch_idx": 2,
  "v_start": 4,
  "v_end": 6,
  "results": [
    {
      "name": "CovMatch_04",
      "refpref_pct": 75.0,
      "refpref_count": 375,
      "total": 500,
      "bertscore_f1": 0.718848466873169,
      "gen_time_min": 186.12662911812464,
      "bs_time_min": 0.4712282220522563,
      "avg_latency": 22.332196831703186,
      "avg_tokens": 199.324
    },
    {
      "name": "CovMatch_05",
      "refpref_pct": 73.2,
      "refpref_count": 366,
      "total": 500,
      "bertscore_f1": 0.7176573276519775,
      "gen_time_min": 185.8655466914177,
      "bs_time_min": 0.3228346467018127,
      "avg_latency": 22.297204191207886,
      "avg_tokens": 199.402
    },
    {
      "name": "CovMatch_06",
      "refpref_pct": 74.4,
      "refpref_count": 372,
      "total": 500,
      "bertscore_f1": 0.7179570198059082,
      "gen_time_min": 185.56802872419357,
      "bs_time_min": 0.31399909655253094,
      "avg_latency": 22.2614648103714,
      "avg_tokens": 199.102
    }
  ],
  "mean_refpref": 74.2,
  "std_refpref": 0.7483314773547874,
  "mean_bertscore_f1": 0.7181542714436849,
  "cosines": [
    -0.24169280999809364,
    -0.006555009347104939,
    -0.05726640833545193
  ]
}

out_path = 'exp09_covmatch_batch2.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(b2_data, f, indent=2, ensure_ascii=False)

print(f'Saved {out_path} successfully!')
