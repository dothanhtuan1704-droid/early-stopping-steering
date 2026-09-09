import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

dir_path = r"E:\Paper_Steering_VN_15K"

for nb_name in ["kaggle-phase5a2-greedy-main-200tok.ipynb", "kaggle-phase5b2-greedy-control-200tok.ipynb"]:
    nb_path = os.path.join(dir_path, nb_name)
    print(f"\n==========================================")
    print(f"PRINTING CELL OUTPUTS FOR: {nb_name}")
    print(f"==========================================")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    for idx, c in enumerate(nb.get("cells", [])):
        outputs = c.get("outputs", [])
        if outputs:
            print(f"--- Cell {idx+1} Outputs ---")
            for out in outputs:
                text = out.get("text", [])
                if isinstance(text, list):
                    text_str = "".join(text)
                else:
                    text_str = str(text)
                print(text_str.strip())
