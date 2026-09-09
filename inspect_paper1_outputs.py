import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

def inspect_notebook(nb_path):
    print(f"\n==========================================")
    print(f"INSPECTING: {os.path.basename(nb_path)}")
    print(f"==========================================")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    cells = nb.get("cells", [])
    print(f"Total cells: {len(cells)}")
    
    output_files = []
    text_outputs = []
    
    for idx, c in enumerate(cells):
        cell_type = c.get("cell_type")
        outputs = c.get("outputs", [])
        for out in outputs:
            # Check text outputs
            text = out.get("text", [])
            if isinstance(text, list):
                text_str = "".join(text)
            else:
                text_str = str(text)
            
            if "Saved" in text_str or "results" in text_str or ".json" in text_str or "Accuracy" in text_str or "BERTScore" in text_str or "Score" in text_str or "Steering" in text_str:
                for line in text_str.split("\n"):
                    if any(k in line for k in ["Saved", "json", "Accuracy", "Score", "Steering", "BERTScore", "Evaluation", "Phase", "Complete"]):
                        text_outputs.append(f"Cell {idx+1}: {line.strip()}")
                        
    print("Key Text Output Findings:")
    for t in text_outputs[:20]:
        print("  -", t)

inspect_notebook(r"E:\Paper_Steering_VN_15K\kaggle-phase5a2-greedy-main-200tok.ipynb")
inspect_notebook(r"E:\Paper_Steering_VN_15K\kaggle-phase5b2-greedy-control-200tok.ipynb")
