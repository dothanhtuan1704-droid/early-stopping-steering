# Early-Stopping Activation Steering for Vietnamese Clinical SLMs

Official repository for the paper: **"Early-Stopping Activation Steering for Hallucination Mitigation in Vietnamese Clinical Small Language Models"** (Submitted to SOICT 2026 / Springer LNCS).

---

## 🗺️ Provenance Mapping: Experimental Evidence, Notebook Sources & Output Logs

This section provides an **explicit 1-to-1 provenance map** linking every table, figure, metric, and experimental claim in [`paper_soict.tex`](file:///e:/Paper_Steering_VN_15K/paper_soict.tex) directly to its source Kaggle GPU notebook, dataset split, raw sample outputs, and summary evidence log.

> [!NOTE]
> **Execution Environment Distinction:**
> - **GPU Notebook Execution:** All model forward passes, activation layer extractions, KV-cache steering vector injections ($v_{\text{steer}} = \mu_{\text{correct}} - \mu_{\text{incorrect}}$ at Layer 8), and text generation were executed on **Kaggle T4/P100 GPUs** using PyTorch and vLLM.
> - **Local CPU Scripts:** Lightweight repository Python scripts (e.g., `eval_prefix_disentanglement.py`, `verify_all_numbers_exact.py`) are post-hoc deterministic aggregators that process raw model output JSON files to compute statistical metrics (McNemar tests, Holm-Bonferroni $p_{\text{adj}}$, bootstrap CIs).

---

### 🔗 Kaggle GPU Execution Provenance

For 1-click cloud re-execution on Kaggle GPU instances, key representative notebook links are indexed below (see [`KAGGLE_NOTEBOOK_LINKS.md`](file:///e:/Paper_Steering_VN_15K/KAGGLE_NOTEBOOK_LINKS.md) for the full experiment manifest):

* **Primary Benchmark ($T=200, 800$)**: [Hard Cutoff Steering $K=16$](https://www.kaggle.com/code/duuykiu/kaggle-sub-notebook-hard-cutoff-steering-k-16) | [Continuous Steering $K=\infty$](https://www.kaggle.com/code/thanhtranguyn/kaggle-sub-notebook-continuous-steering-alpha)
* **Factorial Strength Ablation Sweep ($\alpha \in \{15, 18, 20\}$)**: [Phase 6A Factorial Sweep](https://www.kaggle.com/code/trungkiennnn/kaggle-phase6a1-part1-alpha15-ipynb)
* **Placebo & Directional Controls ($N=100$)**: [Exp 09 Label Permutation & Placebo](https://www.kaggle.com/code/duuykiu/kaggle-experiment-09-full-50-balanced-label-p)
* **Delayed Injection Window Controls ($K_0$)**: [Exp 02 Delayed Window Controls](https://www.kaggle.com/code/duuykiu/experiment-2-part-1b-delayed-injection-windo)

---


### 1. Table 1: Primary Benchmark Matrix ($N_{\text{test}}=500$)

#### **Panel A: Bounded Stress Test ($\text{max\_new\_tokens}=200$)**
* **Target Paper Table:** Table 1, Panel A (Lines 160–185 in `paper_soict.tex`)
* **Execution Environment:** Kaggle Dual T4 GPU (PyTorch / vLLM, bfloat16)
* **Source Notebook:** [`Kaggle_Phase6B_v2_BERTScore_Clinical.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Phase6B_v2_BERTScore_Clinical.ipynb) / [`phase-6b-v2-bertscore-based-clinical-correctness.ipynb`](file:///e:/Paper_Steering_VN_15K/phase-6b-v2-bertscore-based-clinical-correctness.ipynb)
* **Dataset Partition:** Held-out Test Partition ($N_{\text{test}}=500$, `test_ids.json`, seed 42, question-disjoint from $N_{\text{train}}=10,272$)
* **Raw Per-Sample Output JSONs:**
  * Baseline ($73.20\%$): [`result_baseline_cap200.json`](file:///e:/Paper_Steering_VN_15K/result_baseline_cap200.json)
  * Continuous ($75.60\%$): [`result_continuous_cap200.json`](file:///e:/Paper_Steering_VN_15K/result_continuous_cap200.json)
  * Hard Cutoff ($77.00\%$): [`result_hard_cutoff_cap200.json`](file:///e:/Paper_Steering_VN_15K/result_hard_cutoff_cap200.json)
  * Linear Decay ($77.40\%$): [`result_linear_decay_cap200.json`](file:///e:/Paper_Steering_VN_15K/result_linear_decay_cap200.json)
* **Summary Metrics Evidence Log:** [`phase6b_v2_bertscore_clinical_results.json`](file:///e:/Paper_Steering_VN_15K/phase6b_v2_bertscore_clinical_results.json)

#### **Panel B: High-Cap Natural Completion ($\text{max\_new\_tokens}=800$)**
* **Target Paper Table:** Table 1, Panel B (Lines 186–210 in `paper_soict.tex`)
* **Execution Environment:** Kaggle Dual T4 GPU (PyTorch / vLLM, bfloat16)
* **Source Notebooks:** [`Exp10_Part1_Baseline.ipynb`](file:///e:/Paper_Steering_VN_15K/Exp10_Part1_Baseline.ipynb), [`Exp10_Part2_HardCutoff_NoPen.ipynb`](file:///e:/Paper_Steering_VN_15K/Exp10_Part2_HardCutoff_NoPen.ipynb), [`Kaggle_Phase3C_Extended_Generation.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Phase3C_Extended_Generation.ipynb)
* **Dataset Partition:** Held-out Test Partition ($N_{\text{test}}=500$, `test_ids.json`, seed 42)
* **Raw Per-Sample Output JSONs:**
  * Baseline ($65.40\%$): [`result_baseline_cap800.json`](file:///e:/Paper_Steering_VN_15K/result_baseline_cap800.json)
  * Continuous ($68.60\%$): [`result_continuous_cap800.json`](file:///e:/Paper_Steering_VN_15K/result_continuous_cap800.json)
  * Linear Decay ($67.80\%$): [`result_linear_decay_cap800.json`](file:///e:/Paper_Steering_VN_15K/result_linear_decay_cap800.json)
  * Hard Cutoff ($70.60\%$): [`result_hard_cutoff_cap800.json`](file:///e:/Paper_Steering_VN_15K/result_hard_cutoff_cap800.json)
* **Summary Metrics Evidence Log:** [`exp10_merged_500_results.json`](file:///e:/Paper_Steering_VN_15K/exp10_merged_500_results.json)

---

### 2. Table 2: Directional Controls, Placebo Baselines & Multi-Retriever RAG ($N_{\text{test}}=500$)

#### **Directional Controls & Isotropic Placebo Distribution ($N=100$ Random Vectors)**
* **Target Paper Table:** Table 2 Top Section (Lines 220–240 in `paper_soict.tex`)
* **Execution Environment:** Kaggle P100 / T4 GPU
* **Source Notebook:** [`Exp09_Main_Placebo_Controls.ipynb`](file:///e:/Paper_Steering_VN_15K/Exp09_Main_Placebo_Controls.ipynb) / [`Kaggle_Experiment_09_LabelShuffled_Placebo.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Experiment_09_LabelShuffled_Placebo.ipynb)
* **Summary Evidence Logs:** [`exp09_main_placebo_results.json`](file:///e:/Paper_Steering_VN_15K/exp09_main_placebo_results.json), [`expanded_100_placebo_results.csv`](file:///e:/Paper_Steering_VN_15K/expanded_100_placebo_results.csv), [`directional_controls_outputs.csv`](file:///e:/Paper_Steering_VN_15K/directional_controls_outputs.csv)

#### **Multi-Retriever RAG Baselines (BM25, Dense BGE-M3, Hybrid RRF, Oracle Gold)**
* **Target Paper Table:** Table 2 Middle Section (Lines 241–260 in `paper_soict.tex`)
* **Execution Environment:** Kaggle GPU / Local Vector Index
* **Source Notebook:** [`07_dense_bge_m3_hybrid_rag_eval.ipynb`](file:///e:/Paper_Steering_VN_15K/07_dense_bge_m3_hybrid_rag_eval.ipynb) / [`Kaggle_Phase9_Placebo_and_RAG.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Phase9_Placebo_and_RAG.ipynb)
* **Summary Evidence Logs:** [`dense_hybrid_rag_results.json`](file:///e:/Paper_Steering_VN_15K/dense_hybrid_rag_results.json), [`rag_evaluation_outputs.csv`](file:///e:/Paper_Steering_VN_15K/rag_evaluation_outputs.csv)

#### **Synergistic Combined Hybrid RAG + Steering ($80.40\%$ RefPref)**
* **Target Paper Table:** Table 2 Bottom Line (Lines 261–275 in `paper_soict.tex`)
* **Execution Environment:** Kaggle Dual T4 GPU
* **Source Notebook:** [`Kaggle_Exp04_Synergistic_RAG_Plus_Steering.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Exp04_Synergistic_RAG_Plus_Steering.ipynb)
* **Summary Evidence Log:** [`exp04_synergistic_rag_steering_results.json`](file:///e:/Paper_Steering_VN_15K/exp04_synergistic_rag_steering_results.json)

---

### 3. Table 3: Factorial Steering Strength Ablation ($\alpha_0 \in \{15.0, 18.0, 20.0\}$)

* **Target Paper Table:** Table 3 (Lines 280–305 in `paper_soict.tex`)
* **Execution Environment:** Kaggle Dual T4 GPU
* **Source Notebook:** [`kaggle-phase-6a2-fair-equal-alpha-ablation-experi.ipynb`](file:///e:/Paper_Steering_VN_15K/kaggle-phase-6a2-fair-equal-alpha-ablation-experi.ipynb)
* **Summary Evidence Log:** [`exp05_factorial_ablation_results.json`](file:///e:/Paper_Steering_VN_15K/exp05_factorial_ablation_results.json)

---

### 4. Table 4 & Section 4.2: Temporal Shift & Injection Window Disentanglement ($K_0$)

* **Target Paper Section/Table:** Table 4 & Section 4.2 (Lines 310–340 in `paper_soict.tex`)
* **Execution Environment:** Kaggle Dual T4 GPU
* **Source Notebook:** [`Kaggle_Exp02_Delayed_Injection_Window_Control.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Exp02_Delayed_Injection_Window_Control.ipynb)
* **Summary Evidence Log:** [`exp02_delayed_injection_window_800tok_results.json`](file:///e:/Paper_Steering_VN_15K/exp02_delayed_injection_window_800tok_results.json)

---

### 5. Section 3.3: Activation Norm Dynamics & Representation Saturation (Layer 8)

* **Target Paper Section/Figure:** Section 3.3 & Figure 3 (Lines 135–158 in `paper_soict.tex`)
* **Execution Environment:** Kaggle T4 GPU (Teacher-Forcing Hooking)
* **Source Notebooks:** [`06_independent_activation_norm_benchmark.ipynb`](file:///e:/Paper_Steering_VN_15K/06_independent_activation_norm_benchmark.ipynb), [`06_activation_mechanism_teacher_forcing.ipynb`](file:///e:/Paper_Steering_VN_15K/06_activation_mechanism_teacher_forcing.ipynb)
* **Summary Evidence Logs:** [`independent_activation_trajectories.json`](file:///e:/Paper_Steering_VN_15K/independent_activation_trajectories.json), [`activation_mechanism_trajectories_exact.json`](file:///e:/Paper_Steering_VN_15K/activation_mechanism_trajectories_exact.json)

---

### 6. Section 3.1 & 4.3: Clinician Monograph Audit Manifest (94.0% Alignment)

* **Target Paper Section:** Section 3.1 & Section 4.3 (Lines 110–130 in `paper_soict.tex`)
* **Manifest File:** [`human_evaluation_completed_50.csv`](file:///e:/Paper_Steering_VN_15K/human_evaluation_completed_50.csv)
* **Audit Breakdown:** $N=50$ double-blind clinical evaluations by licensed medical practitioners:
  * $47$ Strict Pass (`Score: 2`)
  * $3$ Edge-case prompt formatting artifacts (`Score: 1` on IDs 4, 8, 17)
  * Calculation: $47 / 50 = \mathbf{94.0\%}$ strict clinical monograph compliance.

---

## 📊 Summary Benchmark Matrix ($N_{\text{test}}=500$)

| Generation Horizon | Condition | RefPref Accuracy (%) | Raw Correct | BERTScore F1 | ROUGE-L (%) | Rep-4 (%) | McNemar $p$-value | 95% Bootstrap CI (pp) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bounded Stress Test ($T=200$)** | Unsteered Baseline | 73.20% | 366 / 500 | 0.7134 | 23.12% | **3.67%** | -- | -- |
| | Continuous ($K=\infty$) | 75.60% | 378 / 500 | 0.7133 | **23.36%** | 3.90% | 0.1189 | $[-0.40, +5.20]$ |
| | Hard Cutoff ($K=16$) | 77.00% | 385 / 500 | 0.7151 | 23.16% | 3.89% | 0.0127 | $[+1.00, +6.60]$ |
| | **Linear Decay ($K=16$)** | **77.40%** | **387 / 500** | **0.7157** | **23.33%** | 3.88% | **0.0055** | **$[+1.40, +7.00]$** |
| **Natural Completion ($T=800$)** | Unsteered Baseline | 65.40% | 327 / 500 | **0.6779** | 21.04% | **3.68%** | -- | -- |
| | Continuous ($K=\infty$) | 68.60% | 343 / 500 | 0.6712 | 21.28% | 3.90% | 0.0365 | $[+0.40, +6.00]$ |
| | Linear Decay ($K=16$) | 67.80% | 339 / 500 | 0.6732 | 21.15% | 3.74% | 0.1189 | $[-0.20, +5.00]$ |
| | **Hard Cutoff ($K=16$)** | **70.60%** | **353 / 500** | 0.6695 | **21.32%** | 3.74% | **0.00086** | **$[+2.40, +8.20]$** |

---

## 🛠️ Verification & Reproduction Commands

To independently verify that all numbers reported in [`paper_soict.tex`](file:///e:/Paper_Steering_VN_15K/paper_soict.tex) exactly match the raw JSON evidence files, run:

```bash
# Verify 1:1 numerical consistency across all tables and manuscript text
python verify_all_numbers_exact.py

# Check LaTeX syntax, page bounds, and reference integrity
python check_latex.py
```
