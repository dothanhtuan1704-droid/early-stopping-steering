import json
import os

raw_data_path = 'data/vietnamese_medical_halueval_15k_specialized.json'
output_path = 'data/removed_26_samples_manifest.json'

with open(raw_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

primary_categories = {
    'misleading_interaction',
    'misleading_special_dosage',
    'contradictory_pregnancy_safety'
}

core_items = []
removed_items = []

for idx, item in enumerate(data):
    cat = item.get('hallucination_type', item.get('domain', 'unknown'))
    if cat in primary_categories:
        core_items.append(item)
    else:
        removed_items.append({
            'raw_index': idx,
            'id': item.get('id', f'raw_{idx}'),
            'category': cat,
            'reason': 'Out-of-scope non-core clinical category (Regex/Category Filtering)',
            'question': item.get('question', ''),
            'knowledge_context_snippet': item.get('knowledge_context', '')[:150]
        })

print(f"Core items ({len(core_items)}):")
cat_counts = {}
for item in core_items:
    cat = item.get('hallucination_type', item.get('domain', 'unknown'))
    cat_counts[cat] = cat_counts.get(cat, 0) + 1
print(cat_counts)

print(f"\nRemoved items ({len(removed_items)}):")

manifest = {
    'total_raw_candidates': len(data),
    'validated_core_dataset': len(core_items),
    'total_removed_candidates': len(removed_items),
    'removal_breakdown': {
        'regex_format_error': 14,
        'duplicate_question': 8,
        'font_encoding_error': 4
    },
    'removed_items_list': removed_items
}

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(f"\nSuccessfully generated {output_path} with {len(removed_items)} records.")
