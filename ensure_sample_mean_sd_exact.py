"""
Ensures exact explicit phrasing ($53.39 \pm 0.64$, sample SD across prompts) in paper.tex and FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex.
"""
import os

files = ['paper.tex', 'FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex']

for fpath in files:
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()

    text = text.replace(
        "($53.39 \\pm 0.64$, $\\Delta = 0.00$)",
        "($53.39 \\pm 0.64$, sample SD across prompts, $\\Delta = 0.00$)"
    )

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Explicitly defined sample SD in {fpath}")

print("Sample SD update finished!")
