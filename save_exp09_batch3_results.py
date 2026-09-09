import json

b3_data = {
  "batch_idx": 3,
  "v_start": 7,
  "v_end": 9,
  "results": [
    {
      "name": "CovMatch_07",
      "refpref_pct": 74.6,
      "refpref_count": 373,
      "total": 500,
      "bertscore_f1": 0.7178974151611328,
      "gen_time_min": 182.0156740864118,
      "bs_time_min": 0.4857204437255859,
      "avg_latency": 21.838972084999085,
      "avg_tokens": 199.496
    },
    {
      "name": "CovMatch_08",
      "refpref_pct": 75.8,
      "refpref_count": 379,
      "total": 500,
      "bertscore_f1": 0.7187582850456238,
      "gen_time_min": 181.84195212125778,
      "bs_time_min": 0.30500902732213336,
      "avg_latency": 21.814389834403993,
      "avg_tokens": 199.286
    },
    {
      "name": "CovMatch_09",
      "refpref_pct": 74.2,
      "refpref_count": 371,
      "total": 500,
      "bertscore_f1": 0.7191793322563171,
      "gen_time_min": 182.3925732254982,
      "bs_time_min": 0.28294280370076497,
      "avg_latency": 21.8797454829216,
      "avg_tokens": 199.336
    }
  ],
  "mean_refpref": 74.86666666666666,
  "std_refpref": 0.6798692684790365,
  "mean_bertscore_f1": 0.7186116774876913,
  "cosines": [
    0.1140964087979572,
    -0.12333332386712818,
    0.07998103343231515
  ]
}

out_path = 'exp09_covmatch_batch3.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(b3_data, f, indent=2, ensure_ascii=False)

print(f'Saved {out_path} successfully!')
