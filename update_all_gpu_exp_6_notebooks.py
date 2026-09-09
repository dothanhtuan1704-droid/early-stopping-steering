import json
import glob
import os

target_files = glob.glob(r'E:\Paper_Steering_VN_15K\*6*.ipynb') + glob.glob(r'E:\Paper_Steering_VN_15K\FINAL_SUBMISSION_PACKAGE\*6*.ipynb')

updated_count = 0
for filepath in target_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        modified = False
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                src = ''.join(cell['source'])
                if 'class TeacherForcingActivationTracker' in src or 'def hook_fn' in src:
                    if 'proj = torch.dot(h[0, -1, :], v_steer).item()' in src or 'v_steer = self.v_steer.to(h.device)' in src:
                        # Replace hook_fn implementation with dtype-safe version
                        new_src = src.replace(
                            'v_steer = self.v_steer.to(h.device)',
                            'v_steer = self.v_steer.to(device=h.device, dtype=h.dtype)\n            v_steer_float = self.v_steer.to(device=h.device, dtype=torch.float32)\n            h_last_float = h[0, -1, :].to(torch.float32)'
                        ).replace(
                            'proj = torch.dot(h[0, -1, :], v_steer).item()',
                            'proj = torch.dot(h_last_float, v_steer_float).item()'
                        ).replace(
                            'cos_sim = torch.cosine_similarity(h[0, -1, :].unsqueeze(0), v_steer.unsqueeze(0)).item()',
                            'cos_sim = torch.cosine_similarity(h_last_float.unsqueeze(0), v_steer_float.unsqueeze(0)).item()'
                        )
                        cell['source'] = new_src.splitlines(keepends=True)
                        modified = True
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=2, ensure_ascii=False)
            print(f'[SUCCESS] Updated: {filepath}')
            updated_count += 1
    except Exception as e:
        print(f'[ERROR] {filepath}: {e}')

print(f'Total notebooks updated: {updated_count}')
