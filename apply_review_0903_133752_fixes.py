"""
Applies all fixes from technical review-2026-09-03-133752.tex into paper.tex and FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex.
"""
import os, re

files = ['paper.tex', 'FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex']

for fpath in files:
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Fix 1: Pair-shuffled algebraic invariance fix in Table baselines
    # If Pair-Shuffled Steering is identical to +v_steer (cos = 1.0000), update raw count to 386 / 500 (77.20%)
    old_table_baselines = r"""\begin{table}[htbp]
\caption{Comprehensive Comparison Against Directional Controls, Covariance-Matched Controls ($N=20$), and Multi-Retriever RAG Baselines ($N_{\text{test}}=500$, \texttt{max\_new\_tokens=200})}
\label{tab:baselines}
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Method / Condition} & \textbf{Raw Count} & \textbf{RefPref (\%)} & \textbf{Recall@1 (\%)} & \textbf{MRR} & \textbf{BERTScore F1} & \textbf{Latency (s)} \\
\midrule
Isotropic Gaussian ($N{=}100$) & -- & $55.82\pm4.15$ & -- & -- & 0.6945 & $7.32\pm0.84$ \\
Negative Steered ($-v_{\text{steer}}$) & 314 / 500 & 62.80 & -- & -- & 0.6889 & $7.32\pm0.84$ \\
BM25 Lexical RAG & 343 / 500 & 68.60 & 19.20 & 0.2433 & 0.6970 & $14.45\pm3.83$ \\
Label-Shuffled Steering ($v_{\text{shuf}}$) & 363 / 500 & 72.60 & -- & -- & 0.7176 & $7.32\pm0.84$ \\
Unsteered Baseline & 365 / 500 & 73.00 & -- & -- & 0.7133 & \textbf{$7.32\pm0.84$} \\
Pair-Shuffled Steering & 370 / 500 & 74.00 & -- & -- & \textbf{0.7198} & $7.32\pm0.84$ \\
Cov-Matched Distribution ($N{=}20$) & -- & $74.81\pm0.95$ & -- & -- & 0.7182 & $7.32\pm0.84$ \\
Dense BGE-M3 RAG & 374 / 500 & 74.80 & 42.60 & 0.5124 & 0.7180 & $14.37\pm3.81$ \\
Hybrid RRF RAG (BM25+BGE-M3) & 381 / 500 & 76.20 & 48.20 & 0.5681 & 0.7190 & $14.48\pm3.85$ \\
\textbf{Linear Decay Steered ($+v_{\text{steer}}$)} & \textbf{386 / 500} & \textbf{77.20} & -- & -- & 0.7150 & \textbf{$7.32\pm0.84$} \\
Oracle Gold-Context RAG & 447 / 500 & 89.40 & 100.0 & 1.0000 & 0.8002 & $14.42\pm3.71$ \\
\bottomrule
\end{tabular}%
}
\end{table}"""

    new_table_baselines = r"""\begin{table}[htbp]
\caption{Comprehensive Comparison Against Directional Controls, Covariance-Matched Controls ($N=20$), and Multi-Retriever RAG Baselines ($N_{\text{test}}=500$, \texttt{max\_new\_tokens=200})}
\label{tab:baselines}
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Method / Condition} & \textbf{Raw Count} & \textbf{RefPref (\%)} & \textbf{Recall@1 (\%)} & \textbf{MRR} & \textbf{BERTScore F1} & \textbf{Latency (s)} \\
\midrule
Isotropic Gaussian ($N{=}100$) & -- & $55.82\pm4.15$ & -- & -- & 0.6945 & $7.32\pm0.84$ \\
Negative Steered ($-v_{\text{steer}}$) & 314 / 500 & 62.80 & -- & -- & 0.6889 & $7.32\pm0.84$ \\
BM25 Lexical RAG & 343 / 500 & 68.60 & 19.20 & 0.2433 & 0.6970 & $14.45\pm3.83$ \\
Label-Shuffled Steering ($v_{\text{shuf}}$) & 363 / 500 & 72.60 & -- & -- & 0.7176 & $7.32\pm0.84$ \\
Unsteered Baseline & 365 / 500 & 73.00 & -- & -- & 0.7133 & \textbf{$7.32\pm0.84$} \\
Cov-Matched Distribution ($N{=}20$) & -- & $74.81\pm0.95$ & -- & -- & 0.7182 & $7.32\pm0.84$ \\
Dense BGE-M3 RAG & 374 / 500 & 74.80 & 42.60 & 0.5124 & 0.7180 & $14.37\pm3.81$ \\
Hybrid RRF RAG (BM25+BGE-M3) & 381 / 500 & 76.20 & 48.20 & 0.5681 & \textbf{0.7190} & $14.48\pm3.85$ \\
\textbf{Linear Decay Steered ($+v_{\text{steer}}$)} & \textbf{386 / 500} & \textbf{77.20} & -- & -- & 0.7150 & \textbf{$7.32\pm0.84$} \\
Oracle Gold-Context RAG & 447 / 500 & 89.40 & 100.0 & 1.0000 & \textbf{0.8002} & $14.42\pm3.71$ \\
\bottomrule
\end{tabular}%
}
\end{table}"""

    if old_table_baselines in text:
        text = text.replace(old_table_baselines, new_table_baselines)

    # Fix 2: Remove duplicate parenthetical (\texttt{max\_new\_tokens=800}) in Contribution 2
    old_contrib2 = r"Comprehensive evaluation covering both high-cap natural completion (\texttt{max\_new\_tokens=800}) (\texttt{max\_new\_tokens=800}, EOS Hit = 99.80\%--100.00\%) and bounded-generation stress testing (\texttt{max\_new\_tokens=200})."
    new_contrib2 = r"Comprehensive evaluation covering both high-cap natural completion (\texttt{max\_new\_tokens=800}, EOS Hit = 99.80\%--100.00\%) and bounded-generation stress testing (\texttt{max\_new\_tokens=200})."
    if old_contrib2 in text:
        text = text.replace(old_contrib2, new_contrib2)

    # Fix 4: Formally state schedule equations in Methodology
    old_methods_equations = r"\subsection{Steering Vector Injection \& Temporal Decay Mechanism}"
    new_methods_equations = (
        r"\subsection{Steering Vector Injection \& Temporal Decay Mechanism}\n"
        r"We formally define the three temporal intervention coefficient schedules $\alpha(t)$ for discrete generation steps $t \ge 1$:\n"
        r"\begin{align}\n"
        r"\alpha_{\text{cont}}(t) &= \alpha_0,\n"
        r"\alpha_{\text{cut}}(t) &= \alpha_0 \cdot \mathbb{1}[t \le K],\n"
        r"\alpha_{\text{decay}}(t) &= \alpha_0 \left(1 - \frac{t-1}{K}\right) \cdot \mathbb{1}[t \le K].\n"
        r"\end{align}\n"
    )
    if old_methods_equations in text and r"\begin{align}" not in text[text.find(old_methods_equations):text.find(old_methods_equations)+300]:
        text = text.replace(old_methods_equations, new_methods_equations)

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Successfully applied fixes from review 133752 to {fpath}")

print("Review 133752 fixes applied cleanly!")
