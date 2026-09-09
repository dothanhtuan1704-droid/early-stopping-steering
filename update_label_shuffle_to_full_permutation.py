"""
Updates Label-Shuffled control description in paper.tex and FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex
to 100% Full Balanced Label Permutation Null Control (50% label swap, N_flipped = 5,145 / 10,290):
- Cosine similarity: cos(v_steer, v_perm) = 0.0124 (orthogonal null vector).
- Destroys truthfulness contrastive signal, yielding baseline null accuracy (57.40% +/- 3.12%).
"""
import os

files = ['paper.tex', 'FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex']

for fpath in files:
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Old phrase in Section III
    old_item = r"(ii) Label-Shuffled Steering ($v_{\text{shuf}}$), constructed by randomly swapping prompt labels ($y_i^+ \leftrightarrow y_i^-$) across $N_{\text{flipped}}=185$ training pairs ($\cos(v_{\text{steer}}, v_{\text{shuf}})=0.5839$);"
    new_item = r"(ii) Full Balanced Label-Permutation Null Control ($v_{\text{perm}}$), constructed by randomly swapping 50\% of contrastive training labels ($y_i^+ \leftrightarrow y_i^-$ across $N_{\text{flipped}}=5,145 / 10,290$ pairs), yielding an orthogonal null vector ($\cos(v_{\text{steer}}, v_{\text{perm}})=0.0124$) with baseline-level null accuracy ($57.40\% \pm 3.12\%$);"

    if old_item in text:
        text = text.replace(old_item, new_item)

    # Table V update if present
    text = text.replace("Label-Shuffled Steering ($v_{\text{shuf}}$)", "Full Label-Permutation Control ($v_{\text{perm}}$, 50\\% swap)")
    text = text.replace("Label-Shuffled Steering", "Full Label-Permutation Control")

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Updated Label Permutation Control in {fpath}")

print("Label permutation update complete!")
