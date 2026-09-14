# Early-Stopping Activation Steering for Vietnamese Clinical SLMs

Official repository for the paper: **"Early-Stopping Activation Steering for Hallucination Mitigation in Vietnamese Clinical Small Language Models"** (Submitted to SOICT 2026 / Springer LNCS).

---

## 🗺️ Provenance Mapping: Paper Results to Notebooks & Raw Log Files

This section maps **every single table, figure, and experimental claim in `paper_soict.tex`** directly to its corresponding Kaggle GPU Notebook, input split, and raw output log file.

### 1. Table 1: Primary Benchmark Matrix ($N_{\text{test}}=500$)
* **Panel A: Bounded Stress Test ($\text{max\_new\_tokens}=200$)**
  * **Source Notebook:** [`Kaggle_Phase6B_v2_BERTScore_Clinical.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Phase6B_v2_BERTScore_Clinical.ipynb) / [`phase-6b-v2-bertscore-based-clinical-correctness.ipynb`](file:///e:/Paper_Steering_VN_15K/phase-6b-v2-bertscore-based-clinical-correctness.ipynb)
  * **Input Split:** Held-out Test Partition ($N_{\text{test}}=500$, `test_ids.json`, seed 42)
  * **Raw Sample Outputs:** [`result_baseline_cap200.json`](file:///e:/Paper_Steering_VN_15K/result_baseline_cap200.json), [`result_linear_decay_cap200.json`](file:///e:/Paper_Steering_VN_15K/result_linear_decay_cap200.json), [`result_hard_cutoff_cap200.json`](file:///e:/Paper_Steering_VN_15K/result_hard_cutoff_cap200.json), [`result_continuous_cap200.json`](file:///e:/Paper_Steering_VN_15K/result_continuous_cap200.json)
  * **Summary Evidence Log:** [`phase6b_v2_bertscore_clinical_results.json`](file:///e:/Paper_Steering_VN_15K/phase6b_v2_bertscore_clinical_results.json)
* **Panel B: High-Cap Natural Completion ($\text{max\_new\_tokens}=800$)**
  * **Source Notebook:** [`Exp10_Part1_Baseline.ipynb`](file:///e:/Paper_Steering_VN_15K/Exp10_Part1_Baseline.ipynb), [`Exp10_Part2_HardCutoff_NoPen.ipynb`](file:///e:/Paper_Steering_VN_15K/Exp10_Part2_HardCutoff_NoPen.ipynb), [`Kaggle_Phase3C_Extended_Generation.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Phase3C_Extended_Generation.ipynb)
  * **Input Split:** Held-out Test Partition ($N_{\text{test}}=500$)
  * **Raw Sample Outputs:** [`result_baseline_cap800.json`](file:///e:/Paper_Steering_VN_15K/result_baseline_cap800.json), [`result_hard_cutoff_cap800.json`](file:///e:/Paper_Steering_VN_15K/result_hard_cutoff_cap800.json), [`result_linear_decay_cap800.json`](file:///e:/Paper_Steering_VN_15K/result_linear_decay_cap800.json), [`result_continuous_cap800.json`](file:///e:/Paper_Steering_VN_15K/result_continuous_cap800.json)
  * **Summary Evidence Log:** [`exp10_merged_500_results.json`](file:///e:/Paper_Steering_VN_15K/exp10_merged_500_results.json)

---

### 2. Table 2: Directional Controls, Placebo Baselines & Multi-Retriever RAG ($N_{\text{test}}=500$)
* **Directional Controls & Isotropic Placebo Distribution ($N=100$ Random Vectors):**
  * **Source Notebook:** [`Exp09_Main_Placebo_Controls.ipynb`](file:///e:/Paper_Steering_VN_15K/Exp09_Main_Placebo_Controls.ipynb) / [`Kaggle_Experiment_09_LabelShuffled_Placebo.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Experiment_09_LabelShuffled_Placebo.ipynb)
  * **Summary Evidence Log:** [`exp09_main_placebo_results.json`](file:///e:/Paper_Steering_VN_15K/exp09_main_placebo_results.json), [`expanded_100_placebo_results.csv`](file:///e:/Paper_Steering_VN_15K/expanded_100_placebo_results.csv)
* **Multi-Retriever RAG Baselines (BM25, Dense BGE-M3, Hybrid RRF, Oracle Gold):**
  * **Source Notebook:** [`07_dense_bge_m3_hybrid_rag_eval.ipynb`](file:///e:/Paper_Steering_VN_15K/07_dense_bge_m3_hybrid_rag_eval.ipynb) / [`Kaggle_Phase9_Placebo_and_RAG.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Phase9_Placebo_and_RAG.ipynb)
  * **Summary Evidence Log:** [`dense_hybrid_rag_results.json`](file:///e:/Paper_Steering_VN_15K/dense_hybrid_rag_results.json), [`rag_evaluation_outputs.csv`](file:///e:/Paper_Steering_VN_15K/rag_evaluation_outputs.csv)
* **Synergistic Combined Hybrid RAG + Steering (80.40% RefPref):**
  * **Source Notebook:** [`Kaggle_Exp04_Synergistic_RAG_Plus_Steering.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Exp04_Synergistic_RAG_Plus_Steering.ipynb)
  * **Summary Evidence Log:** [`exp04_synergistic_rag_steering_results.json`](file:///e:/Paper_Steering_VN_15K/exp04_synergistic_rag_steering_results.json)

---

### 3. Table 3: Controlled Steering Strength Ablation ($\alpha_0 \in \{15.0, 18.0, 20.0\}$)
* **Source Notebook:** [`kaggle-phase-6a2-fair-equal-alpha-ablation-experi.ipynb`](file:///e:/Paper_Steering_VN_15K/kaggle-phase-6a2-fair-equal-alpha-ablation-experi.ipynb)
* **Summary Evidence Log:** [`exp05_factorial_ablation_results.json`](file:///e:/Paper_Steering_VN_15K/exp05_factorial_ablation_results.json)

---

### 4. Table 4 & Section 4.2: Temporal Shift & Prefix Length Disentanglement
* **Source Notebook:** [`Kaggle_Exp02_Delayed_Injection_Window_Control.ipynb`](file:///e:/Paper_Steering_VN_15K/Kaggle_Exp02_Delayed_Injection_Window_Control.ipynb)
* **Summary Evidence Log:** [`exp02_delayed_injection_window_800tok_results.json`](file:///e:/Paper_Steering_VN_15K/exp02_delayed_injection_window_800tok_results.json)

---

### 5. Section 3.3: Empirical Activation-Norm Dynamics (Layer 8)
* **Source Notebook:** [`06_independent_activation_norm_benchmark.ipynb`](file:///e:/Paper_Steering_VN_15K/06_independent_activation_norm_benchmark.ipynb) / [`06_activation_mechanism_teacher_forcing.ipynb`](file:///e:/Paper_Steering_VN_15K/06_activation_mechanism_teacher_forcing.ipynb)
* **Summary Evidence Log:** [`independent_activation_trajectories.json`](file:///e:/Paper_Steering_VN_15K/independent_activation_trajectories.json), [`activation_mechanism_trajectories_exact.json`](file:///e:/Paper_Steering_VN_15K/activation_mechanism_trajectories_exact.json)

---

### 6. Section 3.1 & 4.3: Clinician Monograph Audit Manifest (94.0% Alignment)
* **Manifest File:** [`human_evaluation_completed_50.csv`](file:///e:/Paper_Steering_VN_15K/human_evaluation_completed_50.csv)
* **Audit Breakdown:** $N=50$ stratified test samples ($47$ Strict Pass `Score: 2`, $3$ Edge-case prompt artifacts `Score: 1` for IDs 4, 8, 17 $\implies 47/50 = 94.0\%$ strict monograph alignment).

---

## 📊 Summary Results Matrix ($N_{\text{test}}=500$)

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
