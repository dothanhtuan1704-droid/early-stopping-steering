"""
Ensures 100% strict adherence to on-disk empirical JSON result files:
Restores the exact disk JSON values for Label-Shuffled Steering (N_flipped = 185, cos = 0.5839, 72.60% RefPref)
from exp09_main_placebo_results.json across paper.tex and FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex.
"""
import os

files = ['paper.tex', 'FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex']

for fpath in files:
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Exact phrasing matching exp09_main_placebo_results.json on disk
    old_phrase = r"(ii) Full Balanced Label-Permutation Null Control ($v_{\text{perm}}$), constructed by randomly swapping 50\% of contrastive training labels ($y_i^+ \leftrightarrow y_i^-$ across $N_{\text{flipped}}=5,145 / 10,290$ pairs), yielding an orthogonal null vector ($\cos(v_{\text{steer}}, v_{\text{perm}})=0.0124$) with baseline-level null accuracy ($57.40\% \pm 3.12\%$);"
    new_phrase = r"(ii) Label-Shuffled Steering ($v_{\text{shuf}}$), constructed by randomly swapping contrastive prompt labels ($y_i^+ \leftrightarrow y_i^-$) across $N_{\text{flipped}}=185$ training pairs ($\cos(v_{\text{steer}}, v_{\text{shuf}})=0.5839$), achieving 72.60\% RefPref;"

    if old_phrase in text:
        text = text.replace(old_phrase, new_phrase)

    text = text.replace("Full Label-Permutation Control ($v_{\text{perm}}$, 50\\% swap)", "Label-Shuffled Steering ($v_{\text{shuf}}$, $N_{\text{flipped}}=185$)")
    text = text.replace("Full Label-Permutation Control", "Label-Shuffled Steering")

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Restored exact empirical disk JSON values in {fpath}")

print("Exact disk JSON value restoration complete!")
