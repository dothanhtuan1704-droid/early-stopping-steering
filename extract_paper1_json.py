import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

dir_path = r"E:\Paper_Steering_VN_15K"
nb_a = os.path.join(dir_path, "kaggle-phase5a2-greedy-main-200tok.ipynb")
nb_b = os.path.join(dir_path, "kaggle-phase5b2-greedy-control-200tok.ipynb")

def extract_json_from_nb(nb_path, out_name):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    extracted_data = {}
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            # Check text / execute_result
            text_lines = []
            if "text" in out:
                text_lines = out["text"]
            elif "data" in out and "text/plain" in out["data"]:
                text_lines = out["data"]["text/plain"]
            
            if isinstance(text_lines, str):
                text_lines = [text_lines]
                
            full_text = "".join(text_lines)
            if "{" in full_text and "}" in full_text:
                # Try to parse json blocks
                try:
                    start_idx = full_text.find("{")
                    end_idx = full_text.rfind("}") + 1
                    data = json.loads(full_text[start_idx:end_idx])
                    extracted_data.update(data)
                except Exception:
                    pass
    
    out_file = os.path.join(dir_path, out_name)
    print(f"Extracted keys from {os.path.basename(nb_path)}: {list(extracted_data.keys())}")
    return extracted_data

data_a = extract_json_from_nb(nb_a, "phase5a2_main_200_greedy_results.json")
data_b = extract_json_from_nb(nb_b, "phase5b2_control_200_greedy_results.json")
