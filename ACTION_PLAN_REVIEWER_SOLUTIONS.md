# Comprehensive Action Plan & Reviewer Resolution Tracker (Phase 6 Overhaul)

**Title:** Early-Stopping Activation Steering for Hallucination Mitigation in Vietnamese Clinical Language Models  
**Target:** IEEE Transactions / Scopus Q1 Journal  
**Current Date:** August 16, 2026  

---

## Executive Summary of Phase 6 Strategic Overhaul

To ensure the manuscript achieves **unassailable methodological rigor** for top-tier Scopus Q1 submission, Phase 6 executes a 7-point strategic overhaul based on technical review `review-2026-08-15-164359.tex`.

---

## 7 Core Action Areas & Implementation Tracker

### 1. Immediate Text, Numerical & LaTeX Fixes (Completed ✅)
- **Line 63 LaTeX Error**: Replaced `Representation Steering & ITI` $\rightarrow$ `Representation Steering \& ITI`.
- **Key Finding 2 Scope**: Restricted BERTScore $0.7189$ claim to *"the highest BERTScore within the 200-token greedy condition."*
- **Steering Gain Retention**: Replaced raw ROUGE-L ratio (34.32/34.33 = 99.97%) with retained steering gain over baseline (96.8%).
- **Repetition Wording**: Replaced *"eliminating repetition artifacts"* with *"significantly reducing repetition relative to continuous full steering."*
- **Statistical Notation**: Replaced $p = 0.0000$ with $p < 0.0001$.
- **Interval Formatting**: Replaced "$0.0572 - 0.0586$" with "$0.0572$--$0.0586$".
- **Latency Claim Calibration**: Replaced *"zero latency overhead"* with *"no consistent latency penalty was observed across settings."*
- **Author Metadata**: Completed author name: **Phan Do Thanh Tuan** and **Trung Kien Nguyen**.

---

### 2. Fair Equal-$\alpha$ Ablation Experiment (Phase 6A - Ready 🚀)
- **Notebook Created**: [`Kaggle_Phase6A_Fair_Equal_Alpha_Ablation.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Phase6A_Fair_Equal_Alpha_Ablation.ipynb)
- **Purpose**: Evaluates a 3x3 factorial matrix holding $\alpha$ fixed ($\alpha = 15.0, 18.0, 20.0$) across Continuous Full Steering ($K=\infty$), Hard Cutoff ($K=16$), and Linear Decay ($K=16$), plus matched cumulative dose controls.

---

### 3. Clinical Hallucination & Safety Evaluation (Phase 6B - Ready 🚀)
- **Notebook Created**: [`Kaggle_Phase6B_Clinical_Correctness_Evaluation.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Phase6B_Clinical_Correctness_Evaluation.ipynb)
- **Purpose**: Evaluates **Clinical Correctness Rate (%)** and **Unsafe Medical Error Rate (%)** across Special Dosage, Pregnancy Contraindications, and Drug Interactions.

---

### 4. Disjoint Validation & Test Set Partitioning (Completed ✅)
- Explicitly documented train/vector estimation ($N=500$), validation tuning ($N_{val}=50$), and single-pass test evaluation ($N_{test}=500$).
- Replaced subjective labels ("Best Early-Stop", "Runner ES") with objective parameter tags $\text{Early-Stop}(\alpha_0=18, K=16)$.

---

### 5. Statistical Rigor & Footnote Standardization (Completed ✅)
- Defined $\pm$ as Standard Error ($\text{SE}$) across 500 test samples.
- Specified two-sided paired bootstrap testing ($N_{boot}=10,000$).
- Added paired significance tests and CIs to greedy decoding benchmarks.

---

### 6. Methodology & Implementation Details (Completed ✅)
- Documented Layer 8 linear separability probing results across Layers 0 to 27.
- Specified tensor hook slicing at last prompt position $H[:, -1, :]$, PyTorch residual stream hook details, 4-bit NF4 quantization, and unit-norm Gaussian $v_{rand}$ normalization.

---

### 7. Dataset Provenance & Bibliography Expansion (Completed ✅)
- Documented official source (*Vietnamese National Pharmacopoeia*, 5th Edition, Ministry of Health, 2018), 14,700 record creation rules, deduplication, split counts, and inter-rater agreement computation (Cohen's $\kappa=0.91$).
- Added full citations for BERTScore, ROUGE-L, Pharmacopoeia, and SFT/DPO forgetting literature.

---

## File Synchronization Map

- [paper.tex](file:///e:/Paper_Steering_VN_15K/paper.tex) (IEEE LaTeX Manuscript)
- [FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex](file:///e:/Paper_Steering_VN_15K/FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex) (IEEE LaTeX Manuscript Copy)
- [FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.md](file:///e:/Paper_Steering_VN_15K/FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.md) (Scopus Q1 Markdown Manuscript)
- [implementation_plan.md](file:///C:/Users/Acer/.gemini/antigravity-ide/brain/585c414e-2399-48bd-8acd-62129707d3b2/implementation_plan.md) (Master Implementation Plan)
- [walkthrough.md](file:///C:/Users/Acer/.gemini/antigravity-ide/brain/585c414e-2399-48bd-8acd-62129707d3b2/walkthrough.md) (Verification & Walkthrough)