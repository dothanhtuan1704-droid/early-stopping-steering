import json

b4_data = {
  "batch_idx": 4,
  "v_start": 10,
  "v_end": 12,
  "results": [
    {
      "name": "CovMatch_10",
      "refpref_pct": 76.0,
      "refpref_count": 380,
      "total": 500,
      "bertscore_f1": 0.7179060578346252,
      "gen_time_min": 184.9854145606359,
      "bs_time_min": 0.40003141164779665,
      "avg_latency": 22.195645648479463,
      "avg_tokens": 199.558
    },
    {
      "name": "CovMatch_11",
      "refpref_pct": 72.6,
      "refpref_count": 363,
      "total": 500,
      "bertscore_f1": 0.7189947962760925,
      "gen_time_min": 184.97059312264125,
      "bs_time_min": 0.30173508326212567,
      "avg_latency": 22.19018659877777,
      "avg_tokens": 199.398
    },
    {
      "name": "CovMatch_12",
      "refpref_pct": 76.0,
      "refpref_count": 380,
      "total": 500,
      "bertscore_f1": 0.7195342779159546,
      "gen_time_min": 184.43162992397944,
      "bs_time_min": 0.30580234130223594,
      "avg_latency": 22.12529396390915,
      "avg_tokens": 198.888
    }
  ],
  "mean_refpref": 74.86666666666666,
  "std_refpref": 1.6027753706895105,
  "mean_bertscore_f1": 0.7188117106755575,
  "cosines": [
    -0.32360985077099313,
    0.08182130684322646,
    -0.13054786377669048
  ]
}

out_path = 'exp09_covmatch_batch4.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(b4_data, f, indent=2, ensure_ascii=False)

print(f'Saved {out_path} successfully!')
