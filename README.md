# Early-Stopping Activation Steering for Vietnamese Clinical SLMs

Official repository for the paper: **"Early-Stopping Activation Steering for Hallucination Mitigation in Vietnamese Clinical Small Language Models"** (Submitted to SOICT 2026 / Springer LNCS).

---

## 📌 Executive Summary & Core Results

This repository provides the reproducible codebase, ground-truth experimental result logs, dataset audit manifests, and evaluation scripts for Early-Stopping Activation Steering.

### Official Primary Results Matrix ($N_{\text{test}}=500$)

| Generation Horizon | Condition | RefPref Accuracy (%) | Raw Correct | BERTScore F1 | ROUGE-L (%) | Rep-4 (%) | McNemar $p$-value | 95% Bootstrap CI (pp) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bounded Stress Test ($T=200$)** | Unsteered Baseline | 73.20% | 366 / 500 | 0.7133 | 23.12% | **3.67%** | -- | -- |
| | Continuous ($K=\infty$) | 75.60% | 378 / 500 | 0.7133 | **23.36%** | 3.90% | 0.1189 | $[-0.40, +5.20]$ |
| | Hard Cutoff ($K=16$) | 77.00% | 385 / 500 | **0.7151** | 23.16% | 3.89% | 0.0127 | $[+1.00, +6.60]$ |
| | **Linear Decay ($K=16$)** | **77.40%** | **387 / 500** | 0.7150 | 23.21% | 3.74% | **0.0055** | **$[+1.40, +7.00]$** |
| **Natural Completion ($T=800$)** | Unsteered Baseline | 65.40% | 327 / 500 | **0.6779** | 21.04% | **3.68%** | -- | -- |
| | Continuous ($K=\infty$) | 68.60% | 343 / 500 | 0.6712 | 21.28% | 3.90% | 0.0365 | $[+0.40, +6.00]$ |
| | Linear Decay ($K=16$) | 67.80% | 339 / 500 | 0.6732 | 21.15% | 3.74% | 0.1189 | $[-0.20, +5.00]$ |
| | **Hard Cutoff ($K=16$)** | **70.60%** | **353 / 500** | 0.6695 | **21.32%** | 3.74% | **0.00086** | **$[+2.40, +8.20]$** |

---

## 📊 RAG & Placebo Baselines Summary

| Method / Condition | Raw Count | RefPref (%) | BERT F1 | Latency (s) | Notes / Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Isotropic Gaussian Controls | -- | $55.82 \pm 4.15\%$ | 0.6945 | $23.33 \pm 0.30$ | $N=100$ random vectors ($z_{\text{sep}}=5.20$ SD) |
| Negative Steered ($-v_{\text{steer}}$) | 314 / 500 | 62.80% | 0.6889 | $23.34 \pm 0.28$ | Significant accuracy degradation |
| BM25 Lexical RAG | 343 / 500 | 68.60% | 0.6970 | $7.32 \pm 0.45$ | Steering ($+v_{\text{steer}}$ 77.40%) outperforms BM25 by +8.80 pp ($p=3.85 \times 10^{-7}$) |
| Label-Shuffled Placebo ($v_{\text{shuf}}$) | 343 / 500 | 68.60% | 0.6845 | $23.34 \pm 0.28$ | Directional specificity baseline control |
| Dense BGE-M3 RAG | 374 / 500 | 74.80% | 0.7180 | $7.27 \pm 0.42$ | Dense retrieval baseline |
| Hybrid RRF RAG | 381 / 500 | 76.20% | **0.7190** | $7.39 \pm 0.48$ | RRF Reciprocal Rank Fusion |
| **Synergistic Hybrid RAG + Steering** | **392 / 500** | **80.40%** | 0.7185 | $7.42 \pm 0.50$ | Combined Hybrid RAG + Linear Decay ($K=16$) |
| Oracle Gold-Context RAG | 447 / 500 | 89.40% | 0.8002 | $7.17 \pm 0.38$ | Upper bound oracle baseline |

---

## 🔍 Human Audit Manifest & Dataset Provenance

* **Dataset:** ViHaluEval-Medical ($N=14,674$ core clinical pairs derived from the Vietnamese National Drug Formulary, 2nd Edition, 2018).
* **Dataset Splits:** Development $N_{\text{dev}}=12,495$ (Train $N_{\text{train}}=10,272$, Validation $N_{\text{val}}=2,223$), Held-out Test Partition $N_{\text{test\_partition}}=2,179$, Evaluation Subset $N_{\text{test}}=500$.
* **Human Audit Manifest:** [`human_evaluation_completed_50.csv`](file:///e:/Paper_Steering_VN_15K/human_evaluation_completed_50.csv) contains $N=50$ clinician-evaluated samples confirming **94.0% strict monograph alignment**, acknowledging minor synthetic prompt artifacts in edge cases.

---

## 📄 Repository Evidence Artifacts

### 1. Real Model Execution Notebooks (GPU Infrastructure)
* [`Kaggle_Phase1_Steering_Vector_Estimation.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Phase1_Steering_Vector_Estimation.ipynb): Real GPU execution code loading `Qwen/Qwen2.5-7B-Instruct-AWQ` via PyTorch forward hooks at Layer 8 to compute $+v_{\text{steer}}$ on $N_{\text{train}}=10,272$ contrastive pairs.
* [`Kaggle_Phase2_Test_Evaluation.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Phase2_Test_Evaluation.ipynb) / [`Kaggle_Phase3C1_Main_Methods_200token.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Phase3C1_Main_Methods_200token.ipynb): Real GPU completion generation scripts executing early-stopping activation steering across $N_{\text{test}}=500$ test questions.
* [`Kaggle_Phase6B_v2_BERTScore_Clinical.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Phase6B_v2_BERTScore_Clinical.ipynb): GPU evaluation suite computing BERTScore RefPref, ROUGE-L, and 4-gram repetition metrics over model outputs.

### 2. Raw Per-Sample Model Output Generation Logs
* [`result_baseline_cap200.json`](file:///e:/Paper_Steering_VN_15K/result_baseline_cap200.json) (675 KB): Raw unsteered model completions across 500 test questions ($T=200$).
* [`result_linear_decay_cap200.json`](file:///e:/Paper_Steering_VN_15K/result_linear_decay_cap200.json) (678 KB): Raw Linear Decay steered model completions ($K=16, \alpha_0=18.0, T=200$).
* [`result_baseline_cap800.json`](file:///e:/Paper_Steering_VN_15K/result_baseline_cap800.json) (875 KB): Raw unsteered model completions ($T=800$).
* [`result_hard_cutoff_cap800.json`](file:///e:/Paper_Steering_VN_15K/result_hard_cutoff_cap800.json) (913 KB): Raw Hard Cutoff steered model completions ($K=16, \alpha_0=18.0, T=800$).

### 3. Manuscript & Ground-Truth Result Files
* [`paper_soict.tex`](file:///e:/Paper_Steering_VN_15K/paper_soict.tex): Main LaTeX manuscript source code.
* [`paper_soict.pdf`](file:///e:/Paper_Steering_VN_15K/paper_soict.pdf): Compiled 12-page PDF manuscript.
* [`phase6b_v2_bertscore_clinical_results.json`](file:///e:/Paper_Steering_VN_15K/phase6b_v2_bertscore_clinical_results.json): Ground-truth result log for 200-token stress test.
* [`exp10_merged_500_results.json`](file:///e:/Paper_Steering_VN_15K/exp10_merged_500_results.json): Ground-truth result log for 800-token natural completion test.
* [`dense_hybrid_rag_results.json`](file:///e:/Paper_Steering_VN_15K/dense_hybrid_rag_results.json): Ground-truth RAG evaluation benchmark logs.
* [`exp04_synergistic_rag_steering_results.json`](file:///e:/Paper_Steering_VN_15K/exp04_synergistic_rag_steering_results.json): Ground-truth Combined RAG + Steering evaluation logs.
* [`human_evaluation_completed_50.csv`](file:///e:/Paper_Steering_VN_15K/human_evaluation_completed_50.csv): Clinician audit manifest ($N=50$, 47 Pass, 3 Edge Artifacts = 94.0% alignment).
