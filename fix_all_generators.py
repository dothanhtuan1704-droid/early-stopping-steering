import json, os

for fn in ['build_exp02_split_notebooks.py', 'build_exp02_800tok_split_notebooks.py']:
    if os.path.exists(fn):
        with open(fn, 'r', encoding='utf-8') as f:
            c = f.read()
        c = c.replace('"pos_ref": rec[\'pos_ref\'],', '"pos_ref": rec.get(\'pos_ref\', rec.get(\'right_answer\', \'\')),')
        c = c.replace('"neg_ref": rec[\'neg_ref\'],', '"neg_ref": rec.get(\'neg_ref\', rec.get(\'hallucinated_answer\', \'\')),')
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(c)
        print(f"Updated {fn} with key fallbacks!")

