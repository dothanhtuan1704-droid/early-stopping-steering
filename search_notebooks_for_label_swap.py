"""
Searches all Jupyter Notebook (.ipynb) files in the workspace for label shuffling / permutation code and results.
"""
import glob, json, os

print("================================================================================")
print("SEARCHING ALL JUPYTER NOTEBOOKS FOR LABEL SHUFFLE / PERMUTATION CODE")
print("================================================================================")

ipynb_files = glob.glob('*.ipynb') + glob.glob('*/*.ipynb')
print(f"Found {len(ipynb_files)} notebooks in workspace:")

for nb_path in sorted(ipynb_files):
    try:
        with open(nb_path, 'r', encoding='utf-8', errors='ignore') as f:
            nb = json.load(f)
        
        cells = nb.get('cells', [])
        nb_str = json.dumps(nb)
        
        has_label_shuf = 'shuff' in nb_str.lower() or 'flipped' in nb_str.lower() or 'perm' in nb_str.lower()
        has_185 = '185' in nb_str
        has_50 = '50%' in nb_str or '5145' in nb_str
        
        print(f"\n--- Notebook: {nb_path} (cells: {len(cells)}, size: {os.path.getsize(nb_path)} bytes) ---")
        print(f"  - Contains 'shuffle/flipped/perm': {has_label_shuf}")
        print(f"  - Contains '185': {has_185}")
        print(f"  - Contains '50% / 5145': {has_50}")
        
        # Search cell outputs or source
        for c_idx, cell in enumerate(cells):
            src = ''.join(cell.get('source', []))
            outputs = cell.get('outputs', [])
            out_str = json.dumps(outputs)
            
            if 'shuff' in src.lower() or 'shuff' in out_str.lower() or '185' in src or '185' in out_str:
                print(f"    * Cell {c_idx+1} [{cell.get('cell_type')}]: match found!")
                for line in src.split('\n')[:5]:
                    if line.strip():
                        print(f"      src: {line.strip()[:100]}")
    except Exception as e:
        print(f"  - Error reading {nb_path}: {e}")

print("\nNotebook search complete!")
