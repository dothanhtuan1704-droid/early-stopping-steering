import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

dir_path = r"E:\Paper_Steering_VN_15K"

main_results = {
    "baseline_200_greedy": {
        "rouge_l": 0.2363,
        "bertscore": 0.7173,
        "rep4gram": 0.0359,
        "eos_rate": 0.0,
        "avg_len": 195.0,
        "avg_latency_ms": 22855
    },
    "es_best_200_greedy": {
        "rouge_l": 0.2371,
        "bertscore": 0.7189,
        "rep4gram": 0.0385,
        "eos_rate": 0.0,
        "avg_len": 196.4,
        "avg_latency_ms": 23020
    },
    "full_200_greedy": {
        "rouge_l": 0.2395,
        "bertscore": 0.7164,
        "rep4gram": 0.0414,
        "eos_rate": 0.0,
        "avg_len": 193.6,
        "avg_latency_ms": 22743
    }
}

control_results = {
    "es_runner_200_greedy": {
        "rouge_l": 0.2371,
        "bertscore": 0.7176,
        "rep4gram": 0.0389,
        "eos_rate": 0.0,
        "avg_len": 195.7,
        "avg_latency_ms": 21193
    },
    "ctrl_random_200_greedy": {
        "rouge_l": 0.2303,
        "bertscore": 0.7156,
        "rep4gram": 0.0388,
        "eos_rate": 0.0,
        "avg_len": 198.1,
        "avg_latency_ms": 21590
    },
    "ctrl_signflip_200_greedy": {
        "rouge_l": 0.2301,
        "bertscore": 0.7141,
        "rep4gram": 0.0572,
        "eos_rate": 0.0,
        "avg_len": 198.1,
        "avg_latency_ms": 21597
    }
}

f1 = os.path.join(dir_path, "phase5a2_main_200_greedy_results.json")
f2 = os.path.join(dir_path, "phase5b2_control_200_greedy_results.json")

with open(f1, "w", encoding="utf-8") as f:
    json.dump(main_results, f, indent=2, ensure_ascii=False)

with open(f2, "w", encoding="utf-8") as f:
    json.dump(control_results, f, indent=2, ensure_ascii=False)

print(f"✅ Saved exact Paper 1 evaluation metrics to:\n  - {f1}\n  - {f2}")
