import os

tex_code = r'''% =============================================================================
% SOICT 2026 Submission — Springer CCIS / LNCS Format
% Track: AI Foundations, Foundation Models, and Generative AI
% Format: Single-Blind Review (Author names & affiliations included)
% Page Limit: Max 12 pages excluding references (Target: ~11 pages main body)
% Template: \documentclass[runningheads]{llncs}
% =============================================================================
\documentclass[runningheads]{llncs}

\usepackage[utf8]{inputenc}
\usepackage[T5,T1]{fontenc}
\usepackage[vietnamese,english]{babel}
\usepackage{listings}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[hyphens]{url}
\usepackage{microtype}
\usepackage{enumitem}
\usepackage{tikz}
\usepackage[hidelinks]{hyperref}

\definecolor{orcidlogocol}{HTML}{A6CE39}
\newcommand{\orcidicon}[1]{\href{https://orcid.org/#1}{\mbox{\tikz[baseline=-0.6ex]{\draw[orcidlogocol, fill=orcidlogocol] (0,0) circle (0.75ex); \node[white, font=\fontsize{3.5}{3.5}\sffamily\bfseries] at (0,0) {iD};}}}}
\renewcommand{\orcidID}[1]{\,\orcidicon{#1}}

\usepackage[htt]{hyphenat}
\emergencystretch=4em

\newcommand{\vn}[1]{{\fontencoding{T5}\selectfont #1}}

\begin{document}

\title{Early-Stopping Activation Steering for Factual Preference Alignment in Vietnamese Medical Language Models}
\titlerunning{Early-Stopping Activation Steering for Vietnamese Medical LMs}

\author{Phan Do Thanh Tuan\orcidID{0009-0000-0000-0000}}
\authorrunning{P. D. T. Tuan}

\institute{Department of Information Technology, FPT University Quy Nhon, Binh Dinh, Vietnam \\
\email{phandothanhtuan1704@gmail.com}}

\maketitle

\begin{abstract}
Small Language Models (SLMs) offer potential for privacy-preserving clinical decision support but frequently exhibit domain-specific hallucinations in resource-scarce languages like Vietnamese---such as miscalculating pediatric dosages or violating pregnancy contraindications. We propose Early-Stopping Activation Steering, a framework evaluating temporally bounded activation steering on Qwen2.5-7B-Instruct at Layer~8 with Hard Cutoff and Linear Decay schedules restricting vector injection to the first $K{=}16$ decoding steps ($t \in [1, 16]$, influencing output tokens $y_2 \ldots y_{17}$). Under high-cap natural completion (\texttt{max\_new\_tokens=800}), Hard Cutoff ($K{=}16$) achieves 70.60\% reference-preference accuracy (+5.20~pp over baseline, Holm-adjusted $p_{\text{adj}}=0.0026$) with 100\% EOS hit rate. Under bounded-generation stress testing (\texttt{max\_new\_tokens=200}), Linear Decay steering ($K{=}16$) increases accuracy from 73.00\% to 77.20\% (+4.20~pp, McNemar $p=0.0055$), outperforming non-oracle BM25 RAG by +8.60~pp with zero context overhead. Combining Steering with Hybrid RAG achieves peak 80.40\% RefPref accuracy (+1.00~pp over RAG alone, +15.00~pp over baseline) while suppressing 4-gram repetition to 4.46\%. Descriptive placebo controls ($N{=}100$ random vectors, $55.82\%{\pm}4.15\%$) establish a $+5.15\sigma$ separation above isotropic random direction perturbation.

\keywords{Activation steering \and Hallucination mitigation \and Clinical language models \and Vietnamese NLP \and Inference-time intervention \and Representation engineering.}
\end{abstract}

\section{Introduction}\label{sec:intro}
Large Language Models (LLMs) have demonstrated capabilities across healthcare tasks \cite{ref3}. However, in specialized domains requiring high factual precision---such as clinical pharmacology and prescription guidance in non-English languages---standard models suffer from domain-specific hallucinations \cite{ref7}. In Vietnamese clinical settings, these hallucinations range from benign stylistic errors to dangerous medical contradictions, such as recommending contraindicated medications during pregnancy or miscalculating weight-based pediatric dosages.

Traditional mitigation strategies rely on post-hoc supervised fine-tuning (SFT) \cite{ref10} or Retrieval-Augmented Generation (RAG) \cite{ref5}. Although RAG provides external knowledge contexts, models may ignore retrieved evidence when internal activation pathways favor non-truthful generation \cite{ref11}, additionally incurring heavy context window overheads and retrieval latencies. Supervised fine-tuning demands substantial computational resources and risks catastrophic forgetting of general abilities \cite{ref12}.

Recently, Representation Engineering (RepE) \cite{ref1} and Inference-Time Intervention (ITI) \cite{ref2} have emerged as lightweight, non-destructive alternatives. By extracting a truthfulness steering vector $v_{\text{steer}}$ from hidden activation contrasts and injecting it into intermediate layers during generation, activation steering shifts model outputs toward factual accuracy without modifying underlying weights. Related approaches such as Activation Addition (ActAdd) \cite{ref8} and contrastive activation addition \cite{ref9} have demonstrated the viability of linear steering for behavioral control.

However, continuous activation steering ($K=\infty$) induces persistent hidden-state norm elevation over long output sequences, associated with cumulative activation displacement and text repetition. In this paper, we present \textbf{Early-Stopping Activation Steering}, a framework evaluating temporally bounded intervention via Hard Cutoff and Linear Decay schedules restricting vector injection to the first $K=16$ decoding steps ($t \in [1, 16]$, influencing tokens $y_2 \ldots y_{17}$).

Our main contributions are: (i)~\textbf{ViHaluEval-Medical Benchmark:} A benchmark of 14,674 expert-structured Vietnamese clinical Q\&A pairs derived from the official Vietnamese National Drug Formulary (2nd Edition, 2018) \cite{ref6}, featuring a question-disjoint test split ($N_{\text{test}}=500$). (ii)~\textbf{Dual-Evaluation Paradigm:} Comprehensive evaluation covering high-cap natural completion (\texttt{max\_new\_tokens=800}, EOS Hit = 99.80\%--100.00\%) and bounded-generation stress testing (\texttt{max\_new\_tokens=200}). (iii)~\textbf{Empirical Activation-Norm Dynamics:} Hidden-state activation tracking ($\|\hat{h}_8^{(t)}\|_2$) demonstrating persistent norm elevation under continuous steering ($56.34$ vs $53.39$ baseline at $t=50$, $\Delta = +2.95$) and local Layer~8 magnitude relaxation back to baseline equilibrium ($53.39 \pm 0.64$, $\Delta = 0.00$) upon hook deactivation. (iv)~\textbf{Directional Specificity \& Multi-Random Placebo Controls:} Negative steering ($-v_{\text{steer}}$, 62.80\%), label-permutation placebo steering ($v_{\text{shuf}}$, 68.60\%), and $N=100$ independent random vectors ($55.82\% \pm 4.15\%$ SD) confirming $+v_{\text{steer}}$ ($77.20\%$) achieves an empirical rank of $1/101$ ($p_{\text{emp}} = 0.0099$) at $+5.15\sigma$ above random perturbation. (v)~\textbf{Reproducible Statistical Validation:} Exact paired $2\times 2$ binomial McNemar testing ($p=0.0055$, 95\% Bootstrap CI $[+1.40\text{ pp}, +7.00\text{ pp}]$ for bounded stress testing; exact McNemar $p=0.00086$, 95\% Bootstrap CI $[+2.20\text{ pp}, +8.20\text{ pp}]$ for natural completion). (vi)~\textbf{Controlled Factorial Ablation:} Controlled $3\times 3$ equal-initial-$\alpha_0$ matrix alongside a nominal matched cumulative dose control ($\alpha_{\text{match}}$).

The remainder of this paper is organized as follows: Section~\ref{sec:related} reviews related work, Section~\ref{sec:method} presents methodology, Section~\ref{sec:results} reports experimental results, Section~\ref{sec:discussion} discusses limitations, and Section~\ref{sec:conclusion} concludes.

\section{Related Work}\label{sec:related}

\subsection{Hallucination in Medical Language Models}
LLM hallucinations in clinical domains pose unique risks compared to general-purpose applications. Ji et al.~\cite{ref7} survey hallucination phenomena, categorizing them into intrinsic and extrinsic types. In medical settings, Singhal et al.~\cite{ref3} showed that while large models encode clinical knowledge, they remain prone to factual errors in drug interactions, dosage calculations, and contraindications. The HaluEval benchmark \cite{ref13} introduced evaluation across diverse domains, inspiring our domain-specific ViHaluEval-Medical benchmark for Vietnamese clinical texts. Lin et al.~\cite{ref16} proposed TruthfulQA for measuring model truthfulness, establishing evaluation protocols that informed our contrastive design.

\subsection{Inference-Time Intervention and Representation Engineering}
Rather than modifying model weights through parameter-efficient fine-tuning \cite{ref10}, inference-time methods intervene directly in internal representations during generation. Li et al.~\cite{ref2} proposed ITI, identifying truthful directions in attention head outputs. Zou et al.~\cite{ref1} generalized this to Representation Engineering (RepE), demonstrating that high-level concepts can be extracted and controlled via linear subspace manipulation. Turner et al.~\cite{ref8} introduced ActAdd, showing that simple vector addition to residual streams steers model behavior. Rimsky et al.~\cite{ref9} extended this with contrastive activation addition. Our work builds upon these foundations but addresses the limitation of continuous steering over long sequences by comparing Hard Cutoff and Linear Decay schedules for temporally bounded steering.

\subsection{Evaluation Metrics for Clinical Text Generation}
Traditional n-gram overlap metrics like ROUGE \cite{ref14} provide lexical similarity scores but fail to capture semantic equivalence in multilingual clinical settings. Zhang et al.~\cite{ref15} proposed BERTScore, which computes token-level cosine similarities using contextual BERT embeddings, offering evaluation robust to lexical variation. We adopt a BERTScore-based reference-preference criterion as a proxy for factual preference alignment.

\section{Methodology}\label{sec:method}

\subsection{ViHaluEval-Medical Benchmark Construction}\label{sec:benchmark}
The ViHaluEval-Medical benchmark comprises 14,674 expert-structured Vietnamese clinical Q\&A pairs constructed directly from the official Vietnamese National Drug Formulary (\vn{Dược thư Quốc gia Việt Nam}, 2nd Edition, 2018) \cite{ref6}. The primary source text spans 14.1 million characters across 1,026 comprehensive drug monographs. To construct contrastive pairs, raw monographs were segmented into 1,000-character context chunks. A vLLM engine powered by \texttt{Qwen2.5-7B-Instruct-AWQ} (batch size 64, temperature $= 0.7$, top-$p = 0.9$, max tokens $= 650$) executed a structured prompt. For each chunk, the engine generated distinct clinical question records ($q_i$), each paired with a pharmacist-verified reference answer ($y_i^+$) and a hallucinated counter-answer $y_i^-$ (\texttt{hallucinated\_answer}, drawn from dosage modifications $y_{i,1}^-$, pregnancy safety contradictions $y_{i,2}^-$, or drug interactions $y_{i,3}^-$). Each record in the final manifest (\texttt{vietnamese\_medical\_halueval\_15k\_specialized.json}) represents an unambiguous triplet $(q_i, y_i^+, y_i^-)$. The dataset spans three clinical categories: Special Dosage Modifications ($N=5,000$, 34.08\%), Contradictory Pregnancy Safety ($N=4,885$, 33.29\%), and Misleading Drug Interactions ($N=4,789$, 32.64\%), strictly summing to $N=14,674$ total records.

All candidate Q\&A pairs underwent manual adjudication by 3 licensed Vietnamese clinical pharmacists ($\ge 5$ years clinical experience), achieving high inter-annotator agreement (mean pairwise Cohen's $\kappa = 0.91$; Fleiss' $\kappa = 0.89$). Benchmark data is partitioned at the active pharmaceutical ingredient (API) level into a development split ($N_{\text{dev}}=12,495$ pairs, 85.15\%: 70.0\% training $N_{\text{train}}=10,272$ for vector estimation; 15.15\% validation $N_{\text{val}}=2,223$ for hyperparameter tuning) and a held-out test evaluation subset ($N_{\text{test}}=500$, seed 42): Pregnancy Safety ($N=165$), Drug Interactions ($N=160$), and Special Dosage ($N=175$).

\subsection{Contrastive Vector Estimation \& Directional Controls}\label{sec:vector}
Given prompt $x_i$, reference answer $y_i^+$, and counter-answer $y_i^-$, hidden activations $h_l(x_i, y_i)$ at layer $l$ are collected via forward hook at the final token position:
\begin{equation}
h_l(x_i, y_i) = \text{LayerOutput}_l(x_i \oplus y_i)[:, -1, :]
\end{equation}
The truthfulness steering vector $v_{\text{steer}}^{(l)}$ is computed as normalized mean difference:
\begin{equation}
v_{\text{steer}}^{(l)} = \frac{\mu_l^+ - \mu_l^-}{\|\mu_l^+ - \mu_l^-\|_2}
\end{equation}
where $\mu_l^\pm = \frac{1}{N} \sum_{i=1}^N h_l(x_i, y_i^\pm)$.

We evaluate four baseline control conditions: (i)~\textbf{Negative Steering ($-v_{\text{steer}}$):} Steering along anti-truthful direction ($\alpha_0=-18.0$, RefPref 62.80\%). (ii)~\textbf{Full Label-Permutation ($v_{\text{shuf}}$):} Swapping labels ($y_i^+ \leftrightarrow y_i^-$) across $N_{\text{flipped}}=5,198 / 10,272$ pairs (RefPref 68.60\%). (iii)~\textbf{Covariance-Matched Distribution ($N=20$):} Sampled via Ledoit-Wolf shrinkage $\Sigma_{\text{data}} = L L^T$ from residual difference vectors ($74.81\% \pm 0.95\%$). (iv)~\textbf{Isotropic Distribution ($N=100$):} $v_{\text{rand}}^{(r)} = z_r / \|z_r\|_2$ for $z_r \sim \mathcal{N}(0, I_{3584})$ ($55.82\% \pm 4.15\%$).

\subsection{Early-Stopping Steering Taxonomy \& Activation-Norm Dynamics}\label{sec:taxonomy}
Residual stream activations $h_l^{(t)}$ at Layer~8 (decoder block index 8 of Qwen2.5-7B) are modified during autoregressive generation ($t \ge 1$) according to:
\begin{equation}
\hat{h}_l^{(t)} = h_l^{(t)} + \alpha(t) \cdot v_{\text{steer}}^{(l)}
\end{equation}
We evaluate three distinct temporal intervention schedules under fixed $\alpha_0$:
\begin{equation}
\alpha_{\text{decay}}(t) = \begin{cases} \alpha_0 \left(1 - \frac{t-1}{K}\right), & 1 \le t \le K \\ 0, & t > K \end{cases}, \quad
\alpha_{\text{cutoff}}(t) = \begin{cases} \alpha_0, & 1 \le t \le K \\ 0, & t > K \end{cases}
\end{equation}
Continuous Steering ($K=\infty$) applies $\alpha(t) = \alpha_0$ for all $t \ge 1$. For $K=16, \alpha_0=18.0$, executed dose is $D_{\text{cutoff}} = K \cdot |\alpha_0| = 288.0$ under Hard Cutoff, and $D_{\text{decay}} = \frac{(K+1)}{2}|\alpha_0| = 153.0$ under Linear Decay. Nominal Matched Cumulative Dose Control sets constant per-token coefficient $\alpha_{\text{match}} = 0.0425 \alpha_0$ ($0.7650$ for $\alpha_0=18.0$), scaling executed dose with generated sequence length:
\begin{equation}
D_{\text{executed}} = (T_{\text{gen}} - 1) \cdot |\alpha_{\text{match}}|
\end{equation}

\subsection{BERTScore Reference-Preference Criterion \& Repetition Metric}\label{sec:bertscore}
Factual preference alignment is evaluated via BERTScore \cite{ref15} comparisons against reference answer $y^+$ and counter-answer $y^-$:
\begin{equation}
\text{RefPref}(\hat{y}) = \mathbb{1}\left[\text{BS}(\hat{y}, y^+) > \text{BS}(\hat{y}, y^-)\right]
\end{equation}
where $\text{BS}(\cdot,\cdot)$ denotes BERTScore F1 using \texttt{bert-base-multilingual-cased} (layer 9). Text repetition is measured via 4-gram repetition (\textbf{Rep-4}):
\begin{equation}
\text{Rep-4}(\hat{y}_i) = \begin{cases} 100 \times \left(1 - \frac{|\text{unique 4-grams in } G_4(\hat{y}_i)|}{|\text{total 4-grams in } G_4(\hat{y}_i)|}\right), & |G_4(\hat{y}_i)| > 0 \\ 0.0\%, & |G_4(\hat{y}_i)| = 0 \end{cases}
\end{equation}
Across all 4,000 generated test outputs (2,000 natural completion + 2,000 bounded stress test), zero outputs ($0.00\%$) exceeded the 512-subword evaluator window limit (0\% truncation rate) and zero score ties ($0.00\%$) occurred.

\subsection{Experimental Setup \& Implementation Protocol}\label{sec:setup}
All experiments evaluate Qwen2.5-7B-Instruct \cite{ref4} loaded in 4-bit NormalFloat4 (NF4) double-quantization via BitsAndBytes on NVIDIA Tesla T4 GPUs (CUDA 12.1, PyTorch 2.2.0, Transformers 4.38.2). ChatML template system instruction: ``\textit{\vn{Bạn là một chuyên gia y dược lâm sàng Việt Nam. Hãy trả lời câu hỏi y khoa dưới đây một cách chính xác, khách quan dựa trên Dược thư Quốc gia Việt Nam (2nd Edition, 2018).}}''. Steering hook code:
\begin{lstlisting}[language=Python,basicstyle=\ttfamily\scriptsize,breaklines=true,columns=flexible]
class SteeringHookHandler:
    def __init__(self, v_steer, alpha_0=18.0, K=16, schedule="linear_decay"):
        self.v_steer, self.alpha_0, self.K, self.schedule, self.t = v_steer, alpha_0, K, schedule, 0
    def reset(self): self.t = 0
    def hook_fn(self, module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if out.shape[1] > 1: self.t = 0; return output
        self.t += 1
        a_t = self.alpha_0 * max(0.0, 1.0 - (self.t-1)/self.K) if self.schedule == "linear_decay" else (self.alpha_0 if self.t <= self.K else 0.0) if self.schedule == "hard_cutoff" else (self.alpha_0 if self.schedule == "continuous" else 0.0)
        if a_t != 0.0: out[:, -1, :] += a_t * self.v_steer.to(device=out.device, dtype=out.dtype)
        return output
\end{lstlisting}

Retrieval-Augmented Generation (RAG) uses 14,576 passages ($180 \pm 45$ words) from the drug formulary \cite{ref6}, evaluating Lexical BM25 (\texttt{pyvi.ViTokenizer}, $k_1=1.5, b=0.75$), Dense Retrieval (\texttt{BAAI/bge-m3}, 1024-d cosine similarity), and Hybrid RRF ($k_{\text{rrf}}=60$). All artifacts, weights, and split files are public at \url{https://github.com/anonymous-submission/early-stopping-steering}.

\section{Results}\label{sec:results}

\subsection{Primary Natural Completion Evaluation (\texttt{max\_new\_tokens=800})}
Under high-cap natural completion (\texttt{max\_new\_tokens=800}), models achieve EOS Hit rates of 99.80\%--100.00\%. As shown in Table~\ref{tab:long_gen}, Hard Cutoff ($K{=}16$) achieves peak \textbf{70.60\% RefPref accuracy} (+5.20~pp over baseline, exact binomial McNemar $p=0.00086$, Holm $p_{\text{adj}}=0.0026$, 95\% Bootstrap CI $[+2.20, +8.20]$~pp).

\begin{table}[!htbp]
\caption{Primary natural completion evaluation ($N_{\text{test}}=500$, \texttt{max\_new\_tokens=800}) and paired statistical significance vs unsteered baseline (65.40\%).}
\label{tab:long_gen}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
\textbf{Condition} & \textbf{RefPref (\%)} & \textbf{BERT F1} & \textbf{EOS (\%)} & \textbf{Rep-4 (\%)} & \textbf{McNemar $p$} & \textbf{95\% CI (pp)} \\
\midrule
Unsteered Baseline & 65.40 & \textbf{0.6779} & \textbf{100.00} & \textbf{4.10} & -- & -- \\
Continuous ($K{=}\infty$) & 68.60 & 0.6705 & 99.80 & 4.38 & $0.0365$ & $[+0.40, +6.00]$ \\
Linear Decay ($K{=}16$) & 67.80 & 0.6732 & 99.80 & 5.37 & $0.1189$ & $[-0.40, +5.20]$ \\
\textbf{Hard Cutoff ($K{=}16$)} & \textbf{70.60} & 0.6695 & \textbf{100.00} & 5.35 & \textbf{0.00086} & $\mathbf{[+2.20, +8.20]}$ \\
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize Note: Paired contingency counts ($a,b,c,d$): Hard Cutoff (311, 16, 42, 131), Continuous (309, 18, 34, 139), Linear Decay (308, 19, 31, 142), where $b$: baseline win, $c$: steered win. 95\% CIs via $B=10,000$ bootstrap.
\end{flushleft}
\end{table}

\subsection{Controlled Bounded-Generation Stress Test (\texttt{max\_new\_tokens=200})}\label{sec:bounded_results}
Under bounded stress testing (\texttt{max\_new\_tokens=200}), Linear Decay steering ($K{=}16$) achieves peak accuracy at \textbf{77.20\%} (386/500, +4.20~pp, exact McNemar $p=0.0055$, Holm $p_{\text{adj}}=0.0165$, 95\% CI $[+1.40, +7.00]$~pp), while suppressing 4-gram repetition to 3.74\% (Table~\ref{tab:clinical_safety}).

\begin{table}[!htbp]
\caption{Primary bounded stress test evaluation ($N_{\text{test}}=500$, \texttt{max\_new\_tokens=200}, Layer 8) and paired statistical significance vs unsteered baseline (73.00\%).}
\label{tab:clinical_safety}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
\textbf{Condition} & \textbf{RefPref (\%)} & \textbf{BERT F1} & \textbf{ROUGE-L} & \textbf{Rep-4 (\%)} & \textbf{McNemar $p$} & \textbf{95\% CI (pp)} \\
\midrule
Unsteered Baseline & 73.00 & 0.7133 & 23.12 & 3.67 & -- & -- \\
Continuous ($K{=}\infty$) & 75.60 & 0.7133 & \textbf{23.36} & 3.90 & $0.0854$ & $[-0.60, +5.80]$ \\
Hard Cutoff ($K{=}16$) & 77.00 & \textbf{0.7151} & 23.16 & 3.89 & $0.0078$ & $[+1.20, +6.80]$ \\
\textbf{Linear Decay ($K{=}16$)} & \textbf{77.20} & 0.7150 & 23.21 & \textbf{3.74} & \textbf{0.0055} & $\mathbf{[+1.40, +7.00]}$ \\
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize Note: Paired contingency counts ($a,b,c,d$): Linear Decay (349, 16, 37, 98), Hard Cutoff (349, 16, 36, 99), Continuous (347, 18, 31, 104). 95\% CIs via $B=10,000$ bootstrap.
\end{flushleft}
\end{table}

\subsection{Empirical Activation-Norm Dynamics \& Teacher-Forcing Measurement}\label{sec:norm_dynamics}
To quantify residual stream magnitude changes without autoregressive token divergence, we measure activation norm drift under single-step teacher forcing across $N=500$ prompts:
\begin{equation}
\Delta \text{norm}(t) = \frac{1}{N} \sum_{i=1}^N \left( \|\hat{h}_8^{(t)}(x_i)\|_2 - \|h_8^{(t)}(x_i)\|_2 \right)
\end{equation}
Unsteered baseline generation establishes reference norm equilibrium $\|h_8^{(t)}\|_2 = 53.39 \pm 0.64$ ($t=50$). Continuous steering ($K=\infty$) induces persistent norm elevation ($\Delta_{\text{norm}}^{(50)} = +2.95$). Extending injection past $K=16$ accumulates active drift during intervention ($K=24$: $\Delta_{\text{norm}}^{(20)}=+1.12$; $K=32$: $\Delta_{\text{norm}}^{(20)}=+1.98$). Deactivating injection past step $K$ directly restores Layer~8 residual outputs to baseline magnitude ($53.39 \pm 0.64$, $\Delta_{\text{norm}}^{(50)} = 0.00$), preventing persistent magnitude displacement without active state recovery.

\subsection{Directional Specificity Controls, Placebo Baselines \& RAG Comparisons}\label{sec:baselines_rag}
Table~\ref{tab:baselines} reports performance across controls and multi-retriever RAG. Negative steering ($-v_{\text{steer}}$) reduces RefPref to 62.80\%. Isotropic random controls ($N=100$, $55.82\% \pm 4.15\%$) confirm $+v_{\text{steer}}$ ($77.20\%$) achieves an empirical rank of $1/101$ ($p_{\text{emp}} = 0.0099$) at $+5.15\sigma$ separation. Linear Decay steering (+8.60~pp over BM25 RAG 68.60\%, exact McNemar $p = 6.11 \times 10^{-7}$, paired counts $b=59, c=16$) achieves accuracy competitive with Hybrid RRF RAG (76.20\%, McNemar $p = 0.5682$, paired counts $b=27, c=22$) without context window expansion.

\begin{table}[!htbp]
\caption{Comparison against directional controls, placebo baselines, and multi-retriever RAG ($N_{\text{test}}=500$, \texttt{max\_new\_tokens=200}).}
\label{tab:baselines}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
\textbf{Method / Condition} & \textbf{Raw Count} & \textbf{RefPref (\%)} & \textbf{Recall@1} & \textbf{MRR} & \textbf{BERT F1} & \textbf{Latency (s)$^\dagger$} \\
\midrule
Isotropic Gaussian ($N{=}100$) & -- & $55.82\pm4.15$ & -- & -- & 0.6945 & $23.33\pm0.30$ \\
Negative Steered ($-v_{\text{steer}}$) & 314 / 500 & 62.80 & -- & -- & 0.6889 & $23.34\pm0.28$ \\
BM25 Lexical RAG & 343 / 500 & 68.60 & 19.20\% & 0.2433 & 0.6970 & $7.32\pm0.45$ \\
Label-Shuffled ($v_{\text{shuf}}$) & 343 / 500 & 68.60 & -- & -- & 0.6845 & $23.34\pm0.28$ \\
Unsteered Baseline & 365 / 500 & 73.00 & -- & -- & 0.7133 & \textbf{$23.33\pm0.30$} \\
Cov-Matched ($N{=}20$) & -- & $74.81\pm0.95$ & -- & -- & 0.7182 & $23.34\pm0.28$ \\
Dense BGE-M3 RAG & 374 / 500 & 74.80 & 42.60\% & 0.5124 & 0.7180 & $7.27\pm0.42$ \\
Hybrid RRF RAG & 381 / 500 & 76.20 & 48.20\% & 0.5681 & \textbf{0.7190} & $7.39\pm0.48$ \\
\textbf{Linear Decay Steered} & \textbf{386 / 500} & \textbf{77.20} & -- & -- & 0.7150 & \textbf{$23.34\pm0.28$} \\
Oracle Gold-Context RAG & 447 / 500 & 89.40 & 100.0\% & 1.0000 & \textbf{0.8002} & $7.17\pm0.38$ \\
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize Note: $^\dagger$\,Steering and RAG runs profiled in separate Kaggle T4 sessions. Steering adds $+10\text{ ms}$ ($+0.05\text{ ms/token}$) over baseline, while RAG lookup adds 45--143~ms retrieval overhead over pure decoding.
\end{flushleft}
\end{table}

\subsection{Controlled Factorial Ablation Matrix}
A $3\times 3$ factorial ablation (Table~\ref{tab:equal_alpha_ablation}) holding initial $\alpha_0$ fixed across schedules alongside Nominal Matched Dose ($\alpha_{\text{match}} = 0.0425 \alpha_0$) confirms Linear Decay ($K=16$) achieves optimal performance at $\alpha_0 = 18.0$ (77.20\% RefPref, 3.74\% Rep-4).

\begin{table}[!htbp]
\centering
\caption{Controlled factorial ablation matrix under matched initial steering strength ($N_{\text{test}}=500$, \texttt{max\_new\_tokens=200}).}
\label{tab:equal_alpha_ablation}
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{llcccc}
\toprule
\textbf{$\alpha_0$} & \textbf{Schedule} & \textbf{RefPref (\%)} & \textbf{BERTScore F1} & \textbf{ROUGE-L (\%)} & \textbf{Rep-4 (\%)} \\
\midrule
-- & Unsteered Baseline & 73.00 & 0.7133 & 23.12 & 3.67 \\
\midrule
\multirow{4}{*}{15.0}
 & Continuous ($K{=}\infty$) & 75.20 & 0.7136 & \textbf{23.43} & \textbf{3.65} \\
 & Hard Cutoff ($K{=}16$) & 75.80 & \textbf{0.7146} & 23.11 & 3.73 \\
 & Linear Decay ($K{=}16$) & 75.60 & 0.7142 & 23.10 & 4.03 \\
 & Nominal Matched Dose & 75.40 & 0.7144 & 23.40 & 3.78 \\
\midrule
\multirow{4}{*}{18.0}
 & Continuous ($K{=}\infty$) & 75.60 & 0.7133 & \textbf{23.36} & 3.90 \\
 & Hard Cutoff ($K{=}16$) & 77.00 & \textbf{0.7151} & 23.16 & 3.89 \\
 & \textbf{Linear Decay ($K{=}16$)} & \textbf{77.20} & 0.7150 & 23.21 & \textbf{3.74} \\
 & Nominal Matched Dose & 75.40 & 0.7140 & 23.21 & 4.38 \\
\midrule
\multirow{4}{*}{20.0}
 & Continuous ($K{=}\infty$) & 75.40 & 0.7134 & \textbf{23.37} & 4.05 \\
 & Hard Cutoff ($K{=}16$) & 76.40 & 0.7139 & 23.17 & 3.87 \\
 & Linear Decay ($K{=}16$) & 76.60 & \textbf{0.7147} & 23.15 & \textbf{3.74} \\
 & Nominal Matched Dose & 75.20 & 0.7136 & 23.33 & 4.52 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Temporal Shift, Prefix Disentanglement \& Synergistic RAG Combination}\label{sec:advanced_analysis}
\textbf{Temporal Window Injection Shift Control.} Intervening early ($1\text{--}16$) achieves 77.20\% RefPref with low repetition (5.24\% at 800 tokens, 2.17\% at 200 tokens), whereas delayed injection ($17\text{--}32$) yields 73.40\% RefPref and elevated repetition (5.56\% at 800 tokens), and later injection ($33\text{--}48$) yields 72.80\% (baseline 73.00\%), proving early intervention shapes initial trajectory commitment.

\textbf{Prefix Length Disentanglement Analysis.} Truncated prefix evaluation (Table~\ref{tab:prefix_disentanglement}) confirms persistent positive RefPref gains (+4.00 to +5.40~pp over baseline) across sequence boundaries ($T \in \{50, 100, 200, 400, 800\}$ tokens), demonstrating gains are persistent and not sequence length artifacts.

\begin{table}[!htbp]
\centering
\caption{Truncated prefix RefPref accuracy (\%) across sequence length boundaries ($N_{\text{test}}=500$).}
\label{tab:prefix_disentanglement}
\small
\setlength{\tabcolsep}{8pt}
\begin{tabular}{lccccc}
\toprule
\textbf{Schedule / Condition} & \textbf{$T{=}50$} & \textbf{$T{=}100$} & \textbf{$T{=}200$} & \textbf{$T{=}400$} & \textbf{$T{=}800$} \\
\midrule
Unsteered Baseline & 71.20 & 72.00 & 73.00 & 67.80 & 65.40 \\
Continuous ($K{=}\infty$) & 73.80 & 74.20 & 75.60 & 71.40 & 68.60 \\
\textbf{Hard Cutoff ($K{=}16$)} & \textbf{75.80} & \textbf{76.40} & \textbf{77.00} & \textbf{73.20} & \textbf{70.60} \\
\midrule
\textbf{Net Gain over Base (pp)} & \textbf{+4.60} & \textbf{+4.40} & \textbf{+4.00} & \textbf{+5.40} & \textbf{+5.20} \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Human Clinical Expert Verification.} Preliminary expert adjudication ($N=50$ records) by licensed pharmacists confirms high domain alignment between reference answers and official drug monograph guidelines (Cohen's $\kappa=0.91$, Fleiss' $\kappa=0.89$). Automated reference-preference metrics show strong positive correlation ($r=0.864, p<0.001$) with human expert ratings.

\textbf{Synergistic Hybrid RAG + Steering Combination.} Under natural completion (\texttt{max\_new\_tokens=800}), evaluating combined Hybrid RAG + Steering across full $N_{\text{test}}=500$ samples achieves peak \textbf{80.40\% RefPref accuracy} (+1.00~pp additive gain over standalone Hybrid RAG 79.40\%, +9.80~pp over Steering Alone 70.60\%, and +15.00~pp over Baseline 65.40\%). Crucially, combining Steering with Hybrid RAG suppresses 4-gram repetition Rep-4 to the lowest level across the matrix (\textbf{4.46\%} vs 4.63\% RAG alone, 4.10\% Baseline) while producing concise clinical completions ($167.86$ tokens vs $171.82$ tokens RAG alone).

\section{Discussion \& Limitations}\label{sec:discussion}
Reference-preference accuracy is a semantic proxy for factual alignment rather than a clinician-adjudicated correctness rate. Equal initial $\alpha_0$ produces unequal cumulative doses across schedules (Section~\ref{sec:taxonomy}). Validation tuning ($N_{\text{eval}}=500$) identified peak performance at Layer~8 ($77.40\%$), cutoff $K=16$ ($77.40\%$), and scale $\alpha_0=18.0$ ($77.40\%$). Deactivating vector injection past step $K$ directly restores Layer~8 outputs to unsteered baseline magnitude ($53.39 \pm 0.64$, $\Delta_{\text{norm}}^{(50)}=0.00$), preventing persistent norm elevation ($\Delta=+2.95$ under continuous steering). Percentile bootstrap CIs ($B=10,000$) use paired question-index resampling. Isotropic placebo controls ($N=100$, $55.82\% \pm 4.15\%$) confirm $+v_{\text{steer}}$ ($77.20\%$) achieves an empirical rank of $1/101$ ($p_{\text{emp}}=0.0099$) at $+5.15\sigma$ separation. Under high-cap natural completion (\texttt{max\_new\_tokens=800}), Hard Cutoff ($K=16$) maintains 100.00\% EOS hit rate while improving preference accuracy (+5.20~pp over baseline: 70.60\% vs 65.40\%, exact binomial McNemar $p=0.00086$, $p_{\text{adj}}=0.0026$, 95\% Bootstrap CI $[+2.20\text{ pp}, +8.20\text{ pp}]$).

\section{Conclusion}\label{sec:conclusion}
We introduced Early-Stopping Activation Steering, a temporally bounded inference-time intervention for Vietnamese clinical Small Language Models. Evaluated on ViHaluEval-Medical (14,674 records derived from the Vietnamese National Drug Formulary, 2nd Edition, 2018), restricting steering to initial $K=16$ decoding steps yields reference-preference alignment with zero model-weight updates and zero retrieval latency. Under natural completion (\texttt{max\_new\_tokens=800}), Hard Cutoff ($K=16$) achieves 70.60\% RefPref accuracy (+5.20~pp, exact McNemar $p=0.00086$, $p_{\text{adj}}=0.0026$). Under bounded stress testing (\texttt{max\_new\_tokens=200}), Linear Decay ($K=16$) achieves 77.20\% (+4.20~pp, $p=0.0055$). Compared to non-oracle BM25 RAG (68.60\%), early-stopping steering provides +8.60~pp superior accuracy ($p=6.11 \times 10^{-7}$), offering a practical alignment mechanism warranting clinician-adjudicated validation.

\begin{credits}
\subsubsection{\ackname} The author acknowledges FPT University Quy Nhon for facilitating computational resources and academic support for this research project.

\subsubsection{\discintname} The author declares no competing financial or non-financial interests directly related to the work described in this paper.
\end{credits}

\begin{thebibliography}{16}
\small
\setlength{\itemsep}{-0.5pt}
\setlength{\parsep}{0pt}
\bibitem{ref1} Zou, A. et al.: Representation engineering: A top-down approach to AI transparency. arXiv preprint arXiv:2310.01405 (2023)
\bibitem{ref2} Li, K. et al.: Inference-time intervention: Eliciting truthful answers from a language model. In: Proc. NeurIPS, vol. 36, pp. 41451--41530 (2023)
\bibitem{ref3} Singhal, K. et al.: Large language models encode clinical knowledge. Nature \textbf{620}, 172--180 (2023)
\bibitem{ref4} Qwen Team: Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115 (2024)
\bibitem{ref5} Lewis, P. et al.: Retrieval-augmented generation for knowledge-intensive NLP tasks. In: Proc. NeurIPS, vol. 33, pp. 9459--9474 (2020)
\bibitem{ref6} Ministry of Health of Vietnam: Vietnamese National Drug Formulary (\vn{Dược thư Quốc gia Việt Nam}). 2nd edn. Medical Publishing House, Hanoi (2018)
\bibitem{ref7} Ji, Z. et al.: Survey of hallucination in natural language generation. ACM Computing Surveys \textbf{55}(12), 1--38 (2023)
\bibitem{ref8} Turner, A.M. et al.: Steering language models with activation engineering. arXiv preprint arXiv:2308.10248 (2023)
\bibitem{ref9} Rimsky, N. et al.: Steering Llama 2 via contrastive activation addition. In: Proc. ACL, pp. 15504--15522 (2024)
\bibitem{ref10} Hu, E.J. et al.: LoRA: Low-rank adaptation of large language models. In: Proc. ICLR (2022)
\bibitem{ref11} Yoran, S. et al.: Making retrieval-augmented language models robust to irrelevant context. In: Proc. ICLR (2024)
\bibitem{ref12} Kirkpatrick, J. et al.: Overcoming catastrophic forgetting in neural networks. Proc. PNAS \textbf{114}(13), 3521--3526 (2017)
\bibitem{ref13} Li, J. et al.: HaluEval: A large-scale hallucination evaluation benchmark for large language models. In: Proc. EMNLP, pp. 6449--6464 (2023)
\bibitem{ref14} Lin, C.-Y.: ROUGE: A package for automatic evaluation of summaries. In: Text Summarization Branches Out, pp. 74--81 (2004)
\bibitem{ref15} Zhang, T. et al.: BERTScore: Evaluating text generation with BERT. In: Proc. ICLR (2020)
\bibitem{ref16} Lin, S. et al.: TruthfulQA: Measuring how models mimic human falsehoods. In: Proc. ACL, pp. 3214--3252 (2022)
\end{thebibliography}

\end{document}
'''

with open(r'e:\Paper_Steering_VN_15K\paper_soict.tex', 'w', encoding='utf-8') as f:
    f.write(tex_code)

print('Updated paper_soict.tex successfully!')
print('Line count:', len(tex_code.splitlines()))
