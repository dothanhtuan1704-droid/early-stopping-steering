import json, os, glob

json_files = [
    'exp02_part1_early_results.json',
    'exp02_part2_delayed_results.json',
    'exp02_part3_later_results.json',
    'exp02_part4_continuous_results.json',
    'exp02_part1_early_800tok_results.json',
    'exp02_part2_delayed_800tok_results.json',
    'exp02_part3_later_800tok_results.json',
    'exp02_part4_continuous_800tok_results.json'
]

combined_200 = {}
combined_800 = {}
found = 0

for jf in json_files:
    if os.path.exists(jf):
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
            max_tok = data.get('max_new_tokens', 200)
            window_name = data.get('window_name', jf)
            if max_tok == 800:
                combined_800[window_name] = data
            else:
                combined_200[window_name] = data
            found += 1
            print(f"Loaded {jf} ({max_tok} tok): {data.get('refpref_pct')}% RefPref, {data.get('rep4_pct')}% Rep-4")

output_file_200 = 'exp02_delayed_injection_window_results.json'
output_file_800 = 'exp02_delayed_injection_window_800tok_results.json'

if combined_200:
    with open(output_file_200, 'w', encoding='utf-8') as f:
        json.dump(combined_200, f, indent=2, ensure_ascii=False)
    print(f"\nMerged 200tok results into '{output_file_200}'.")

if combined_800:
    with open(output_file_800, 'w', encoding='utf-8') as f:
        json.dump(combined_800, f, indent=2, ensure_ascii=False)
    print(f"Merged 800tok results into '{output_file_800}'.")

print(f"\nTotal loaded result files: {found}")
