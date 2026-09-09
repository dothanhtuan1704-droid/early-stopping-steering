"""
Dumps all cells of notebook948d663962 (1).ipynb into a clean utf-8 text summary.
"""
import json

nb_path = 'notebook948d663962 (1).ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('notebook948d663962_summary.txt', 'w', encoding='utf-8') as f_out:
    f_out.write(f"SUMMARY OF NOTEBOOK: {nb_path}\n")
    f_out.write("=" * 80 + "\n\n")
    for idx, cell in enumerate(nb.get('cells', [])):
        c_type = cell.get('cell_type')
        src = ''.join(cell.get('source', []))
        f_out.write(f"=== CELL {idx+1} [{c_type}] ===\n")
        f_out.write(src + "\n\n")

print("Notebook summary written to notebook948d663962_summary.txt")
