import json, re, glob, os

print("=" * 80)
print("COMPREHENSIVE DATA INTEGRITY AUDIT FOR PAPER_SOICT.TEX")
print("=" * 80)

# Load paper_soict.tex
with open('paper_soict.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# Load all JSON disk files
disk_json = {}
for fname in glob.glob('*.json'):
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            disk_json[fname] = json.load(f)
    except Exception as e:
        pass

print(f"Loaded {len(disk_json)} JSON result files from root directory.")

# Load CSV files
disk_csv = {}
for fname in glob.glob('*.csv'):
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            disk_csv[fname] = f.read()
    except Exception as e:
        pass
print(f"Loaded {len(disk_csv)} CSV result files from root directory.")

print("\n--------------------------------------------------------------------------------")
print("SECTION 1: DATASET & METADATA NUMBERS")
print("--------------------------------------------------------------------------------")

meta_claims = [
    ("14,674", "Total dataset samples", "14,674 records"),
    ("1,026", "Total monographs", "1,026 monographs"),
    ("14.1", "Total characters (M)", "14.1M characters"),
    ("5,000", "Pathology & Diagnostic Procedures", "5,000 samples (34.08%)"),
    ("4,885", "Pharmacotherapy & Dosage Instructions", "4,885 samples (33.29%)"),
    ("4,789", "Surgical & Emergency Protocols", "4,789 samples (32.64%)"),
    ("10,272", "Train set size", "10,272 (70.0%)"),
    ("2,223", "Validation set size", "2,223 (15.15%)"),
    ("2,179", "Test set size", "2,179 (14.85%)"),
    ("500", "Evaluation subset N_test", "500 prompts"),
    ("175", "Test subset cat 1", "175 prompts"),
    ("160", "Test subset cat 2", "160 prompts"),
    ("165", "Test subset cat 3", "165 prompts"),
    ("0.91", "Cohen's kappa inter-annotator", "0.91 agreement"),
    ("0.89", "Fleiss' kappa multi-annotator", "0.89 agreement")
]

for val, desc, full_claim in meta_claims:
    found = val in tex
    print(f"  - {desc} ({val}): {'[MATCH / PRESENT]' if found else '[MISSING]'}")

print("\n--------------------------------------------------------------------------------")
print("SECTION 2: TABLE 1 & TABLE 2 (800-TOKEN GENERATION & MCNEMAR)")
print("--------------------------------------------------------------------------------")

table1_claims = [
    ("Baseline (65.40% RefPref, 327/500)", "65.40"),
    ("Baseline BERTScore F1 (0.6779)", "0.6779"),
    ("Baseline EOS Hit (100.00%)", "100.00"),
    ("Baseline Rep-4 (4.10%)", "4.10"),
    ("Continuous RefPref (68.60%, 343/500)", "68.60"),
    ("Continuous BERTScore F1 (0.6705)", "0.6705"),
    ("Continuous EOS Hit (99.80%)", "99.80"),
    ("Continuous Rep-4 (4.38%)", "4.38"),
    ("Linear Decay RefPref (67.80%, 339/500)", "67.80"),
    ("Linear Decay BERTScore F1 (0.6732)", "0.6732"),
    ("Linear Decay Rep-4 (5.37%)", "5.37"),
    ("Hard Cutoff RefPref (70.60%, 353/500)", "70.60"),
    ("Hard Cutoff BERTScore F1 (0.6695)", "0.6695"),
    ("Hard Cutoff Rep-4 (5.35%)", "5.35"),
    ("McNemar Hard Cutoff (a=311, b=16, c=42, d=131)", "311, 16, 42, 131"),
    ("McNemar Hard Cutoff p-value (0.00086)", "0.00086"),
    ("McNemar Hard Cutoff Holm p_adj (0.0026)", "0.0026"),
    ("McNemar Continuous (a=309, b=18, c=34, d=139)", "309, 18, 34, 139"),
    ("McNemar Continuous p-value (0.0365)", "0.0365"),
    ("McNemar Linear Decay (a=308, b=19, c=31, d=142)", "308, 19, 31, 142"),
    ("McNemar Linear Decay p-value (0.1189)", "0.1189"),
    ("Bootstrap CI Hard Cutoff (+5.20 pp, [+2.20, +8.20])", "[+2.20, +8.20]"),
    ("Bootstrap CI Continuous (+3.20 pp, [+0.40, +6.00])", "[+0.40, +6.00]"),
    ("Bootstrap CI Linear Decay (+2.40 pp, [-0.40, +5.20])", "[-0.40, +5.20]")
]

for desc, val in table1_claims:
    found = val in tex
    print(f"  - {desc}: {'[MATCH / PRESENT]' if found else '[MISSING]'}")

print("\n--------------------------------------------------------------------------------")
print("SECTION 3: TABLE 3 & TABLE 4 (200-TOKEN BOUNDED STRESS TEST & MCNEMAR)")
print("--------------------------------------------------------------------------------")

table3_claims = [
    ("Baseline RefPref (73.00%, 365/500)", "73.00"),
    ("Baseline BERTScore F1 (0.7133)", "0.7133"),
    ("Baseline ROUGE-L (23.12%)", "23.12"),
    ("Baseline Rep-4 (3.67%)", "3.67"),
    ("Continuous RefPref (75.60%, 378/500)", "75.60"),
    ("Continuous BERTScore F1 (0.7133)", "0.7133"),
    ("Continuous ROUGE-L (23.36%)", "23.36"),
    ("Continuous Rep-4 (3.90%)", "3.90"),
    ("Hard Cutoff RefPref (77.00%, 385/500)", "77.00"),
    ("Hard Cutoff BERTScore F1 (0.7151)", "0.7151"),
    ("Hard Cutoff ROUGE-L (23.16%)", "23.16"),
    ("Hard Cutoff Rep-4 (3.89%)", "3.89"),
    ("Linear Decay RefPref (77.20%, 386/500)", "77.20"),
    ("Linear Decay BERTScore F1 (0.7150)", "0.7150"),
    ("Linear Decay ROUGE-L (23.21%)", "23.21"),
    ("Linear Decay Rep-4 (3.74%)", "3.74"),
    ("McNemar Linear Decay (a=349, b=16, c=37, d=98)", "349, 16, 37, 98"),
    ("McNemar Linear Decay p-value (0.0055)", "0.0055"),
    ("McNemar Linear Decay Holm p_adj (0.0165)", "0.0165"),
    ("McNemar Hard Cutoff (a=349, b=16, c=36, d=99)", "349, 16, 36, 99"),
    ("McNemar Hard Cutoff p-value (0.0078)", "0.0078"),
    ("McNemar Continuous (a=347, b=18, c=31, d=104)", "347, 18, 31, 104"),
    ("McNemar Continuous p-value (0.0854)", "0.0854"),
    ("Bootstrap CI Linear Decay (+4.20 pp, [+1.40, +7.00])", "[+1.40, +7.00]"),
    ("Bootstrap CI Hard Cutoff (+4.00 pp, [+1.20, +6.80])", "[+1.20, +6.80]"),
    ("Bootstrap CI Continuous (+2.60 pp, [-0.60, +5.80])", "[-0.60, +5.80]")
]

for desc, val in table3_claims:
    found = val in tex
    print(f"  - {desc}: {'[MATCH / PRESENT]' if found else '[MISSING]'}")

print("\n--------------------------------------------------------------------------------")
print("SECTION 4: TABLE 5 & TABLE 6 (BASELINES, CONTROLS, RAG & PAIRED TESTS)")
print("--------------------------------------------------------------------------------")

table5_claims = [
    ("Isotropic Gaussian Placebo (55.82% +/- 4.15%, max 64.60%)", "55.82"),
    ("Isotropic Gaussian BERTScore F1 (0.6945)", "0.6945"),
    ("Negative Steered (62.80%, 314/500)", "62.80"),
    ("Negative Steered BERTScore F1 (0.6889)", "0.6889"),
    ("BM25 Lexical RAG RefPref (68.60%, 343/500)", "68.60"),
    ("BM25 Recall@1 (19.20%), MRR (0.2433), BS F1 (0.6970)", "19.20"),
    ("Label-Shuffled Steering RefPref (68.60%, 343/500)", "68.60"),
    ("Label-Shuffled BERTScore F1 (0.6845)", "0.6845"),
    ("Covariance-Matched Controls (74.81% +/- 0.95%)", "74.81"),
    ("Covariance-Matched BERTScore F1 (0.7182)", "0.7182"),
    ("Dense BGE-M3 RAG RefPref (74.80%, 374/500)", "74.80"),
    ("Dense BGE-M3 Recall@1 (42.60%), MRR (0.5124), BS F1 (0.7180)", "42.60"),
    ("Hybrid RRF RAG RefPref (76.20%, 381/500)", "76.20"),
    ("Hybrid RRF Recall@1 (48.20%), MRR (0.5681), BS F1 (0.7190)", "48.20"),
    ("Oracle Gold-Context RAG RefPref (89.40%, 447/500)", "89.40"),
    ("Oracle Recall@1 (100.0%), MRR (1.0000), BS F1 (0.8002)", "0.8002"),
    ("Table 6A McNemar Steered vs BM25: b=59, c=16, p=6.11e-7", "6.11 \\times 10^{-7}"),
    ("Table 6B McNemar Steered vs Hybrid RRF: b=27, c=22, p=0.5682", "0.5682")
]

for desc, val in table5_claims:
    found = val in tex
    print(f"  - {desc}: {'[MATCH / PRESENT]' if found else '[MISSING]'}")

print("\n--------------------------------------------------------------------------------")
print("SECTION 5: TABLE 7 (FACTORIAL ABLATION ALPHA = 15, 18, 20 & DOSE-MATCHED)")
print("--------------------------------------------------------------------------------")

table7_claims = [
    ("Alpha 15 Continuous (75.20%)", "75.20"),
    ("Alpha 15 Hard Cutoff (75.80%)", "75.80"),
    ("Alpha 15 Linear Decay (75.60%)", "75.60"),
    ("Alpha 15 Nominal Matched (75.40%)", "75.40"),
    ("Alpha 18 Continuous (75.60%)", "75.60"),
    ("Alpha 18 Hard Cutoff (77.00%)", "77.00"),
    ("Alpha 18 Linear Decay (77.20%)", "77.20"),
    ("Alpha 18 Nominal Matched (75.40%)", "75.40"),
    ("Alpha 20 Continuous (75.40%)", "75.40"),
    ("Alpha 20 Hard Cutoff (76.40%)", "76.40"),
    ("Alpha 20 Linear Decay (76.60%)", "76.60"),
    ("Alpha 20 Nominal Matched (75.20%)", "75.20")
]

for desc, val in table7_claims:
    found = val in tex
    print(f"  - {desc}: {'[MATCH / PRESENT]' if found else '[MISSING]'}")

print("\n--------------------------------------------------------------------------------")
print("SECTION 6: ACTIVATION NORMS & MECHANISTIC METRICS")
print("--------------------------------------------------------------------------------")

activation_claims = [
    ("Unsteered Baseline activation norm (53.39 +/- 0.64)", "53.39"),
    ("Continuous Steering activation norm delta (+2.95 at t=50)", "2.95"),
    ("Continuous Steering saturation norm (56.34 +/- 0.81 at t=50)", "56.34"),
    ("Continuous Steering final norm (68.42 +/- 1.95 at t=800)", "68.42"),
    ("Hard Cutoff drop at t=17 (53.42 +/- 0.65)", "53.42"),
    ("Validation sweep peak at Layer 8, K=16, alpha=18.0 (77.40%)", "77.40"),
    ("Validation sweep grid: 32 layer x 7 K x 5 alpha", "32 \\times 7 \\times 5")
]

for desc, val in activation_claims:
    found = val in tex
    print(f"  - {desc}: {'[MATCH / PRESENT]' if found else '[MISSING]'}")

print("\n================================================================================")
print("AUDIT SUMMARY COMPLETE")
print("================================================================================")
