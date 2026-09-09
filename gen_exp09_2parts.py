import json, copy

# Generator for 2 Exp 9 Notebooks: Part A (samples 0-249) & Part B (samples 250-499)
# Each part runs: Original + Label-Shuffled + Pair-Shuffled + N=3 Covariance-Matched vectors
# Total runtime per notebook: ~9.0 hours (completes 100% cleanly within Kaggle 12h limit!)

with open('Kaggle_Experiment_09_LabelShuffled_Placebo.ipynb', 'r', encoding='utf-8') as f:
    template = json.load(f)

for part, s_start, s_end in [('A', 0, 250), ('B', 250, 500)]:
    fname = f'Exp09_Placebo_Part{part}.ipynb'
    nb = copy.deepcopy(template)
    
    # Cell 0: Title
    nb['cells'][0]['source'] = [
        f"# Exp 9: Stronger Placebo Controls — Part {part} (samples {s_start}-{s_end-1})\n",
        f"Runs Original, Label-Shuffled, Pair-Shuffled, & N=3 Covariance-Matched vectors.\n",
        f"max_new_tokens=200, greedy decoding, {s_end-s_start} test samples.\n",
        f"Runtime: ~9.0 hours (completes 100% cleanly within 12h limit)."
    ]
    
    # Cell 2: Config - update N_COV_MATCHED=3 and SLICE
    config_cell = nb['cells'][2]
    new_src = []
    for line in config_cell['source']:
        if 'N_COV_MATCHED =' in line:
            line = 'N_COV_MATCHED = 3  # 3 covariance-matched vectors for 9h runtime limit\n'
        new_src.append(line)
    new_src.append(f"SLICE_START={s_start}; SLICE_END={s_end}\n")
    new_src.append(f"print(f'Part {part} Slice: {{SLICE_START}} to {{SLICE_END-1}}')\n")
    config_cell['source'] = new_src
    
    # Cell 3: Data loading - apply slice
    data_cell = nb['cells'][3]
    new_data_src = []
    for line in data_cell['source']:
        if 'test_subset = test_records[:TEST_LIMIT]' in line:
            line = 'test_subset = test_records[SLICE_START:SLICE_END]\n'
        new_data_src.append(line)
    data_cell['source'] = new_data_src
    
    # Cell 15: Output filename
    last_cell = nb['cells'][-1]
    new_last = []
    for line in last_cell['source']:
        if 'experiment_09_stronger_placebo_results.json' in line:
            line = line.replace('experiment_09_stronger_placebo_results.json', f'exp09_placebo_part{part.lower()}.json')
        new_last.append(line)
    last_cell['source'] = new_last
    
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Created: {fname} (Part {part}, samples {s_start}-{s_end-1}, N_COV_MATCHED=3, ~9.0h runtime)')

print('\nDone! Generated 2 notebooks for Exp 9.')
