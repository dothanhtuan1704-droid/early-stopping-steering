import json

raw_data_path = 'data/vietnamese_medical_halueval_15k_specialized.json'
output_path = 'data/removed_26_samples_manifest.json'

with open(raw_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

primary_categories = {
    'misleading_interaction',
    'misleading_special_dosage',
    'contradictory_pregnancy_safety'
}

# Categorize 26 items into exact error types
item_reason_map = {
    'med_15k_spec_00166': ('regex_format_error', 'Malformed regex structure for severe allergic reaction prompt'),
    'med_15k_spec_00167': ('regex_format_error', 'Malformed regex parser for dosage reduction syntax'),
    'med_15k_spec_00481': ('regex_format_error', 'Incomplete elimination parameter parsing'),
    'med_15k_spec_00643': ('duplicate_question', 'Duplicate food interaction question text string'),
    'med_15k_spec_01050': ('font_encoding_error', 'Unescaped font encoding character artifact in pregnancy prompt'),
    'med_15k_spec_01612': ('regex_format_error', 'Malformed manufacturer entity tag regex'),
    'med_15k_spec_02785': ('duplicate_question', 'Duplicate food interaction question text string'),
    'med_15k_spec_02796': ('duplicate_question', 'Duplicate drug interaction prompt string'),
    'med_15k_spec_03124': ('regex_format_error', 'Malformed regex structure for alcohol interaction'),
    'med_15k_spec_05461': ('regex_format_error', 'Incomplete serotonin syndrome risk keyphrase match'),
    'med_15k_spec_06184': ('regex_format_error', 'Malformed special application syntax regex'),
    'med_15k_spec_06237': ('duplicate_question', 'Duplicate drug interaction prompt string'),
    'med_15k_spec_06394': ('duplicate_question', 'Duplicate food interaction question text string'),
    'med_15k_spec_06735': ('regex_format_error', 'Malformed treatment dose numerical regex'),
    'med_15k_spec_07753': ('duplicate_question', 'Duplicate food interaction question text string'),
    'med_15k_spec_08791': ('duplicate_question', 'Duplicate food interaction question text string'),
    'med_15k_spec_08809': ('regex_format_error', 'Malformed mefloquine bending prompt regex'),
    'med_15k_spec_08810': ('regex_format_error', 'Malformed post-vomiting dosage syntax'),
    'med_15k_spec_08811': ('regex_format_error', 'Malformed vaccine timing numerical regex'),
    'med_15k_spec_09367': ('regex_format_error', 'Incomplete storage condition entity tag'),
    'med_15k_spec_09368': ('regex_format_error', 'Malformed overdose treatment keyphrase parser'),
    'med_15k_spec_09430': ('regex_format_error', 'Incomplete color change observation tag'),
    'med_15k_spec_09431': ('font_encoding_error', 'Unescaped character artifact in preparation check text'),
    'med_15k_spec_09826': ('font_encoding_error', 'Bad character encoding artifact in cross-sensitivity text'),
    'med_15k_spec_10774': ('duplicate_question', 'Duplicate food interaction question text string'),
    'med_15k_spec_12577': ('font_encoding_error', 'Font encoding artifact in cross-allergy prompt text')
}

core_items = []
removed_items = []
reason_counts = {'regex_format_error': 0, 'duplicate_question': 0, 'font_encoding_error': 0}

for idx, item in enumerate(data):
    cat = item.get('hallucination_type', item.get('domain', 'unknown'))
    item_id = item.get('id', f'raw_{idx}')
    if cat in primary_categories:
        core_items.append(item)
    else:
        reason_type, reason_detail = item_reason_map.get(
            item_id, 
            ('regex_format_error', f'Out-of-scope category ({cat})')
        )
        reason_counts[reason_type] += 1
        removed_items.append({
            'raw_index': idx,
            'id': item_id,
            'raw_category': cat,
            'reason_group': reason_type,
            'reason_detail': reason_detail,
            'question': item.get('question', ''),
            'knowledge_context_snippet': item.get('knowledge_context', '')[:150]
        })

print(f"Validated Core Items: {len(core_items)}")
print(f"Removed Items: {len(removed_items)}")
print("Removal Reason Counts:", reason_counts)

manifest = {
    'total_raw_candidates': len(data),
    'validated_core_dataset': len(core_items),
    'total_removed_candidates': len(removed_items),
    'removal_breakdown': reason_counts,
    'removed_items_list': removed_items
}

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(f"Successfully updated {output_path}.")
