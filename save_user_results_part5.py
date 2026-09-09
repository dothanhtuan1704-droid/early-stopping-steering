import json

json_data = [
{
  "name": "LinearDecay_RepPen_PartA",
  "refpref_pct": 76.4,
  "refpref_n": 191,
  "total": 250,
  "bs_f1": 0.6857866048812866,
  "bs_std": 0.03805113583803177,
  "rep4": 4.277210452821429,
  "rep4_std": 6.6130312942809955,
  "eos_pct": 1.2,
  "eos_n": 3,
  "lat": 43.90560532951355,
  "tok": 412.064,
  "peak_gpu": 1.9340858459472656,
  "rep_pen": 1.15,
  "gen_min": 182.95233718554178,
  "bs_min": 0.42408838669459026,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "params_B": 4.352972288,
  "peak_load_gpu": 2.361870288848877
},
{
  "name": "LinearDecay_RepPen_PartB",
  "refpref_pct": 75.2,
  "refpref_n": 188,
  "total": 250,
  "bs_f1": 0.689116358757019,
  "bs_std": 0.03757086768746376,
  "rep4": 3.817252385856067,
  "rep4_std": 5.201426812051515,
  "eos_pct": 1.2,
  "eos_n": 3,
  "lat": 45.40883103752136,
  "tok": 404.472,
  "peak_gpu": 1.9348134994506836,
  "rep_pen": 1.15,
  "gen_min": 189.21677843729654,
  "bs_min": 0.4242913007736206,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "params_B": 4.352972288,
  "peak_load_gpu": 2.361870288848877
}
]

for item in json_data:
    filename = f"exp10_{item['name'].lower()}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(item, f, indent=2)
    print(f"Saved: {filename}")
