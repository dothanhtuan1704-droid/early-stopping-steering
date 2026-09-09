import json, csv, os, glob, sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("INSPECTING CODEBASE FOR USER'S PROPOSED EXPERIMENTAL ENHANCEMENTS")
print("=" * 80)

# 1. Check Human Clinical Evaluation Data
print("\n--- 1. HUMAN EVALUATION & CLINICAL ACCURACY DATA ---")
if os.path.exists('human_evaluation_completed_50.csv'):
    with open('human_evaluation_completed_50.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"Loaded {len(rows)} human evaluation entries.")
        scores = [float(r['human_correctness_score']) for r in rows if r.get('human_correctness_score')]
        unsafes = [float(r['human_unsafe_error']) for r in rows if r.get('human_unsafe_error')]
        if scores:
            print(f"Mean Human Correctness Score: {sum(scores)/len(scores):.4f} (Accuracy = {sum(1 for s in scores if s > 0.5)/len(scores)*100:.2f}%)")
        if unsafes:
            print(f"Mean Unsafe Clinical Error Rate: {sum(unsafes)/len(unsafes)*100:.2f}%")

if os.path.exists('phase6b_v2_summary.csv'):
    with open('phase6b_v2_summary.csv', 'r', encoding='utf-8') as f:
        print("\nPhase6b V2 Summary CSV:")
        print(f.read())

# 2. Check McNemar & Statistical Comparisons (Hard Cutoff vs Continuous, Linear Decay vs Continuous)
print("\n--- 2. PAIRED STATISTICAL COMPARISONS IN CODEBASE ---")
if os.path.exists('exp10_merged_500_results.json'):
    with open('exp10_merged_500_results.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
        print("Exp10 Merged 500 Keys:", list(d.keys()))

# 3. Check for Delayed Injection Window / Prefix Evaluation scripts or data
print("\n--- 3. CHECKING FOR DELAYED INJECTION / PREFIX EVALUATION SCRIPTS ---")
py_files = glob.glob('*.py') + glob.glob('*.ipynb')
for pf in py_files:
    if any(k in pf.lower() for k in ['delayed', 'window', 'prefix', 'ablation', 'rag_steering', 'comb']):
        print("Found relevant script:", pf)

print("\nInspection complete!")
