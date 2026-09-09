"""
Audits synchronization between paper.tex, FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex, and FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.md.
"""
import os, re

files = {
    'paper.tex': 'paper.tex',
    'full_tex': 'FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex',
    'full_md': 'FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.md'
}

data = {}
for name, fpath in files.items():
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            data[name] = f.read()

print("Checking presence of key tables and metrics across all 3 manuscript files...\n")

# Check 1: Table baselines / Placebo N=20 vs N=100
for name, content in data.items():
    print(f"=== {name} ===")
    print("  - Contains 'Cov-Matched':", 'Cov-Matched' in content or 'Covariance-Matched' in content)
    print("  - Contains 'Label-Shuffled':", 'Label-Shuffled' in content or 'Label-shuffled' in content or 'label_shuffled' in content)
    print("  - Contains 'Pair-Shuffled':", 'Pair-Shuffled' in content or 'Pair-shuffled' in content or 'pair_shuffled' in content)
    print("  - Contains 'Isotropic Gaussian':", 'Isotropic Gaussian' in content or 'isotropic' in content or 'Multi-Random Placebo' in content)
    print()

print("Audit complete!")
