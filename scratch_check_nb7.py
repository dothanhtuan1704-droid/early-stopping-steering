import json

with open('gpu-experiment-7-complete-dense-bge-m3-and-hybr.ipynb', 'r', encoding='utf-8', errors='ignore') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb.get('cells', []))}")

for idx, cell in enumerate(nb.get('cells', [])):
    cell_type = cell.get('cell_type')
    print(f"\n==================== Cell {idx} ({cell_type}) ====================")
    source = "".join(cell.get('source', []))
    print("SOURCE:")
    print(source[:500].encode('ascii', 'replace').decode('ascii'))
    
    if cell_type == 'code':
        outputs = cell.get('outputs', [])
        print(f"\nOUTPUTS COUNT: {len(outputs)}")
        for o_idx, out in enumerate(outputs):
            text = ""
            if 'text' in out:
                if isinstance(out['text'], list):
                    text += "".join(out['text'])
                else:
                    text += str(out['text'])
            if 'data' in out:
                for k, v in out['data'].items():
                    if isinstance(v, list):
                        text += "".join(v)
                    else:
                        text += str(v)
            clean_text = text.encode('ascii', 'replace').decode('ascii')
            print(f"--- Output {o_idx} ---")
            print(clean_text[:4000])
