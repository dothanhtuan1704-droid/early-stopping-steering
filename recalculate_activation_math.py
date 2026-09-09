import json
import numpy as np
import pandas as pd

with open(r'E:\Paper_Steering_VN_15K\activation_mechanism_trajectories.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

conditions = ["baseline", "continuous", "hard_cutoff", "linear_decay"]
summary_rows = []

for cond in conditions:
    cond_runs = [r for r in results if r["condition"] == cond]
    for t in [1, 10, 16, 50, 100]:
        post_norms = [r["post_hook_norm"][t-1] for r in cond_runs if len(r["post_hook_norm"]) >= t]
        projs = [r["proj_v_steer"][t-1] for r in cond_runs if len(r["proj_v_steer"]) >= t]
        
        # Exact mathematical identity: cos_sim_exact = proj / post_norm
        cosines_exact = [p / n if n > 0 else 0.0 for p, n in zip(projs, post_norms)]
        
        if post_norms:
            summary_rows.append({
                "condition": cond,
                "step_t": t,
                "mean_post_norm": round(float(np.mean(post_norms)), 2),
                "std_post_norm": round(float(np.std(post_norms)), 2),
                "mean_projection": round(float(np.mean(projs)), 2),
                "std_projection": round(float(np.std(projs)), 2),
                "mean_cosine_sim": round(float(np.mean(cosines_exact)), 4),
                "std_cosine_sim": round(float(np.std(cosines_exact)), 4)
            })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(r'E:\Paper_Steering_VN_15K\activation_mechanism_summary_recalculated.csv', index=False, encoding='utf-8')
print("[SUCCESS] Recalculated exact mathematical activation summary:")
print(df_summary.to_string())
