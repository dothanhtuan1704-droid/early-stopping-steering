"""
Restores paper.tex, FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex, and FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.md
back to their exact pre-edit state.
"""
import os

new_placebo_heading = r"\subsection{Placebo Control Distribution, Manifold Covariance Controls \& Real BM25 RAG Results}"
old_placebo_heading = r"\subsection{Placebo Control Distribution \& Real BM25 RAG Results}"

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

files_to_restore = ['paper.tex', 'FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex']

for fpath in files_to_restore:
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    if new_placebo_heading in content:
        content = content.replace(new_placebo_heading, old_placebo_heading)
    if new_table in content:
        content = content.replace(new_table, old_table)

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Restored file: {fpath}")

# Also restore Markdown manuscript
md_path = 'FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.md'
if os.path.exists(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_md_heading = '### 4. Direct Comparison Against Placebo Controls, Covariance-Matched Controls (N=20), and Real BM25 RAG Baseline'
    old_md_heading = '### 4. Direct Comparison Against Placebo Controls and Real BM25 RAG Baseline'

    if new_md_heading in content:
        content = content.replace(new_md_heading, old_md_heading)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Restored file: {md_path}")

print("Rollback completed 100%!")
