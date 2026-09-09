import json, os, glob

# Fix build_exp02_250split_notebooks.py
with open('build_exp02_250split_notebooks.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace direct key lookup with safe fallback lookup
old_block = '''"pos_ref": rec['pos_ref'],
        "neg_ref": rec['neg_ref'],'''

new_block = '''"pos_ref": rec.get('pos_ref', rec.get('right_answer', '')),
        "neg_ref": rec.get('neg_ref', rec.get('hallucinated_answer', '')),'''

code_fixed = code.replace(old_block, new_block)

with open('build_exp02_250split_notebooks.py', 'w', encoding='utf-8') as f:
    f.write(code_fixed)

print("Updated build_exp02_250split_notebooks.py with key fallbacks!")

# Fix build_exp04_split_notebooks.py as well
with open('build_exp04_split_notebooks.py', 'r', encoding='utf-8') as f:
    code04 = f.read()

code04_fixed = code04.replace(old_block, new_block)
code04_fixed = code04_fixed.replace("context = item.get('pos_ref', '')", "context = item.get('pos_ref', item.get('right_answer', ''))")

with open('build_exp04_split_notebooks.py', 'w', encoding='utf-8') as f:
    f.write(code04_fixed)

print("Updated build_exp04_split_notebooks.py with key fallbacks!")
