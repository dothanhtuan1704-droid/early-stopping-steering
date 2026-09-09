"""
Refines directional specificity narrative and Monte Carlo tail probability phrasing in paper.tex and FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex.
"""
import os

files = ['paper.tex', 'FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex']

for fpath in files:
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Replace 'strictly direction-specific' with nuanced directional specificity narrative
    text = text.replace(
        "strictly direction-specific rather than an artifact of non-specific residual perturbation.",
        "directionally superior over both off-manifold isotropic Gaussian noise (+5.15\\sigma) and on-manifold covariance-matched activation variation (+2.52\\sigma / +2.39~pp)."
    )

    # Frame p = 1/101 as one-sided Monte Carlo empirical tail probability
    text = text.replace(
        "empirical randomization $p=1/101=0.0099 < 0.01$",
        "one-sided Monte Carlo empirical tail probability $p=1/101=0.0099 < 0.01$"
    )
    text = text.replace(
        "empirical randomization test $p=1/101=0.0099 < 0.01$",
        "one-sided Monte Carlo empirical tail probability $p=1/101=0.0099 < 0.01$"
    )

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Refined directional specificity narrative in {fpath}")

print("Directional specificity refinement finished!")
