import json

b5_data = {
  "batch_idx": 5,
  "v_start": 13,
  "v_end": 15,
  "results": [
    {
      "name": "CovMatch_13",
      "refpref_pct": 75.4,
      "refpref_count": 377,
      "total": 500,
      "bertscore_f1": 0.7175837159156799,
      "gen_time_min": 187.2525003393491,
      "bs_time_min": 0.41133087476094565,
      "avg_latency": 22.467698546409608,
      "avg_tokens": 199.424
    },
    {
      "name": "CovMatch_14",
      "refpref_pct": 75.2,
      "refpref_count": 376,
      "total": 500,
      "bertscore_f1": 0.718782365322113,
      "gen_time_min": 186.65653836727142,
      "bs_time_min": 0.3122565269470215,
      "avg_latency": 22.39253550195694,
      "avg_tokens": 199.05
    },
    {
      "name": "CovMatch_15",
      "refpref_pct": 75.0,
      "refpref_count": 375,
      "total": 500,
      "bertscore_f1": 0.7175371050834656,
      "gen_time_min": 186.87037277619044,
      "bs_time_min": 0.31105188926060995,
      "avg_latency": 22.418130551338194,
      "avg_tokens": 199.246
    }
  ],
  "mean_refpref": 75.2,
  "std_refpref": 0.16329931618554752,
  "mean_bertscore_f1": 0.7179677287737528,
  "cosines": [
    -0.23618434446571904,
    0.15439861115606074,
    -0.03788627584915537
  ]
}

out_path = 'exp09_covmatch_batch5.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(b5_data, f, indent=2, ensure_ascii=False)

print(f'Saved {out_path} successfully!')
