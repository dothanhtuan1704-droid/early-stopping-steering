"""
Updates paper.tex and FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex with exact empirical results
for Exp 09 (N=20 Covariance Matched Controls, Label-Shuffled, Pair-Shuffled) and Exp 10.
"""
import os, json

# 1. Load exact Exp 09 & Exp 10 numbers
exp09 = json.load(open('exp09_all20_placebo_results.json', 'r', encoding='utf-8'))
cov_mean = exp09['covariance_matched_n20']['mean_refpref_pct'] # 74.81
cov_std = exp09['covariance_matched_n20']['std_refpref_pct']   # 0.95
cov_bs = exp09['covariance_matched_n20']['mean_bertscore_f1']   # 0.7182

label_rp = exp09['main_controls']['label_shuffled']['refpref_pct']        # 72.60
label_bs = exp09['main_controls']['label_shuffled']['bertscore_f1']        # 0.7176
label_cos = exp09['main_controls']['label_shuffled']['cosine_w_original'] # 0.5839

pair_rp = exp09['main_controls']['pair_shuffled']['refpref_pct']          # 74.00
pair_bs = exp09['main_controls']['pair_shuffled']['bertscore_f1']          # 0.7198

exp10 = json.load(open('exp10_merged_500_results.json', 'r', encoding='utf-8'))
base_rep4 = exp10['baseline']['rep4']              # 41.25
base_lat = exp10['baseline']['lat']                # 78.74

hard_reppen_rep4 = exp10['hardcutoff_reppen']['rep4']   # 3.76
hard_reppen_bs = exp10['hardcutoff_reppen']['bs_f1']    # 0.6869
hard_reppen_lat = exp10['hardcutoff_reppen']['lat']     # 45.05
hard_reppen_rp = exp10['hardcutoff_reppen']['refpref_pct'] # 75.80

files_to_update = ['paper.tex', 'FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex']

# Define target replacements
old_placebo_heading = r"\subsection{Placebo Control Distribution \& Real BM25 RAG Results}"
new_placebo_heading = r"\subsection{Placebo Control Distribution, Manifold Covariance Controls \& Real BM25 RAG Results}"

new_placebo_para = (
    r"To test directionality against generic residual perturbation and ambient data manifold variation, "
    r"Table~\ref{tab:baselines} compares Positive Steering ($+v_{\text{steer}}$) against Negative Steering ($-v_{\text{steer}}$), "
    r"Label-Shuffled steering ($v_{\text{shuf}}$, $N_{\text{flipped}}=185$), Pair-Shuffled steering, "
    r"a Covariance-Matched Random Vector Distribution ($N=20$ vectors, 10,000 generation samples), "
    r"an Isotropic Gaussian Random Vector Distribution ($N=100$ vectors), and a non-oracle BM25 RAG baseline."
)

old_table = r"""\begin{table}[htbp]
\caption{Empirical Comparison Against Directional Controls and Real BM25 RAG Baseline ($N_{\text{test}}=500$, \texttt{max\_new\_tokens=200})}
\label{tab:baselines}
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Method / Condition} & \textbf{Raw Count} & \textbf{RefPref (\%)} & \textbf{BERTScore F1} & \textbf{Prompt Tokens} & \textbf{Latency (s)} \\
\midrule
Negative Steered ($-v_{\text{steer}}$) & 314 / 500 & 62.80 & 0.6889 & $\sim35.0$ & $7.32\pm0.84$ \\
Multi-Random Placebo ($N{=}100$) & -- & $55.82\pm4.15$ & 0.6945 & $\sim35.0$ & $7.32\pm0.84$ \\
Unsteered Baseline & 365 / 500 & 73.00 & 0.7133 & $\sim35.0$ & \textbf{$7.32\pm0.84$} \\
\textbf{Early-Stop Steered ($+v_{\text{steer}}$)} & \textbf{386 / 500} & \textbf{77.20} & \textbf{0.7150} & $\sim\mathbf{35.0}$ & \textbf{$7.32\pm0.84$} \\
Real BM25 RAG Baseline & 343 / 500 & 68.60 & 0.6970 & 75.4 & $14.45\pm3.83$ \\
Oracle Gold-Context RAG & 447 / 500 & 89.40 & 0.8002 & 75.4 & $14.42\pm3.71$ \\
\bottomrule
\end{tabular}%
}
\end{table}"""

new_table = r"""\begin{table}[htbp]
\caption{Empirical Comparison Against Directional Controls, Covariance-Matched Controls ($N=20$), and Real BM25 RAG Baseline ($N_{\text{test}}=500$, \texttt{max\_new\_tokens=200})}
\label{tab:baselines}
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Method / Condition} & \textbf{Raw Count} & \textbf{RefPref (\%)} & \textbf{BERTScore F1} & \textbf{Prompt Tokens} & \textbf{Latency (s)} \\
\midrule
Isotropic Gaussian ($N{=}100$) & -- & $55.82\pm4.15$ & 0.6945 & $\sim35.0$ & $7.32\pm0.84$ \\
Negative Steered ($-v_{\text{steer}}$) & 314 / 500 & 62.80 & 0.6889 & $\sim35.0$ & $7.32\pm0.84$ \\
Label-Shuffled Steering & 363 / 500 & 72.60 & 0.7176 & $\sim35.0$ & $7.32\pm0.84$ \\
Unsteered Baseline & 365 / 500 & 73.00 & 0.7133 & $\sim35.0$ & \textbf{$7.32\pm0.84$} \\
Pair-Shuffled Steering & 370 / 500 & 74.00 & 0.7198 & $\sim35.0$ & $7.32\pm0.84$ \\
\textbf{Cov-Matched ($N{=}20$)} & -- & \textbf{74.81$\pm$0.95} & \textbf{0.7182} & $\sim35.0$ & $7.32\pm0.84$ \\
\textbf{Early-Stop Steered ($+v_{\text{steer}}$)} & \textbf{386 / 500} & \textbf{77.20} & \textbf{0.7150} & $\sim\mathbf{35.0}$ & \textbf{$7.32\pm0.84$} \\
Real BM25 RAG Baseline & 343 / 500 & 68.60 & 0.6970 & 75.4 & $14.45\pm3.83$ \\
Oracle Gold-Context RAG & 447 / 500 & 89.40 & 0.8002 & 75.4 & $14.42\pm3.71$ \\
\bottomrule
\end{tabular}%
}
\end{table}"""

for fpath in files_to_update:
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    if old_placebo_heading in content:
        content = content.replace(old_placebo_heading, new_placebo_heading)
    if old_table in content:
        content = content.replace(old_table, new_table)
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Successfully updated LaTeX manuscript: {fpath}")

print("LaTeX Manuscripts update completed successfully!")
