# 🔬 Early-Stopping Activation Steering for Hallucination Mitigation in Vietnamese Domain-Specific RAG

[![Conference](https://img.shields.io/badge/SOICT-2026-blue.svg)](https://soict.org/)
[![Publisher](https://img.shields.io/badge/Springer-CCIS-orange.svg)](https://www.springer.com/series/7899)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Official research repository and reproducibility code package for the paper:  
**"Early-Stopping Activation Steering for Hallucination Mitigation in Vietnamese Domain-Specific RAG"**  
*Submitted to the 15th International Symposium on Information and Communication Technology (SOICT 2026).*

---

## 📌 Abstract
Small Language Models (SLMs) offer immense potential for edge deployment and privacy-preserving clinical decision support. However, they frequently suffer from domain-specific hallucinations in low-resource and non-English contexts—such as miscalculating pediatric drug dosages or violating critical pregnancy contraindications. 

We propose **Early-Stopping Activation Steering**, a lightweight, non-destructive inference-time intervention that temporally bounds activation steering at **Layer 8** of `Qwen2.5-7B-Instruct` using **Hard Cutoff** and **Linear Decay** schedules ($K=16$). 

* **High-Cap Natural Completion ($T=800$):** Hard Cutoff ($K=16$) achieves **70.60%** automated reference-preference accuracy (+5.20 pp over unsteered baseline, exact McNemar $p = 0.00086$, $p_{\text{adj}} = 0.0026$) with **100% EOS hit rate**.
* **Bounded Stress Testing ($T=200$):** Linear Decay ($K=16$) elevates accuracy from 73.20% to **77.40%** (+4.20 pp, $p = 0.0055$), outperforming non-oracle BM25 RAG by **+8.80 pp** with zero context-window memory overhead.
* **Synergistic RAG + Steering:** Combining Hybrid RAG with Early-Stopping Steering achieves **80.40%** RefPref while suppressing 4-gram repetition to a record low of **3.61%**.
* **Directional Specificity:** 100 isotropic Gaussian random controls confirm $+v_{\text{steer}}$ operates at a **$z_{\text{sep}} = 5.20\sigma$** separation above random perturbation.

---

## 🗂️ Repository Structure

```text
early-stopping-steering/
├── README.md                      # Official research overview & reproduction guide
├── requirements.txt               # Complete Python dependency manifest
├── .gitignore                     # Production clean git configuration
│
├── paper/                         # Main manuscript source & published camera-ready
│   ├── paper_soict.tex            # Full Springer LNCS LaTeX manuscript source
│   ├── llncs.cls / splncs04.bst   # Official Springer LNCS formatting templates
│   └── SOICT_2026_paper_0240.pdf  # Submitted camera-ready PDF document
│
├── notebooks/                     # Official Kaggle GPU experiment notebooks (by Table)
│   ├── 01_Steering_Vector_and_Probing/      # Layer-wise probing & v_steer extraction
│   ├── 02_Table1_Bounded_Stress_Test_T200/  # Table 1 Panel B stress testing (T=200)
│   ├── 03_Table1_Natural_Completion_T800/   # Table 1 Panel A natural completion (T=800) & Table 4
│   ├── 04_Table2_Directional_and_Placebo/   # Table 2: 100 Placebos, Negative, Label Shuffle
│   ├── 05_Table2_RAG_and_Synergistic/       # Table 2: BM25, Dense, Hybrid RAG & Synergistic RAG+Steering
│   ├── 06_Table3_Factorial_Alpha_Ablation/  # Table 3: Alpha sweep (alpha in {15, 18, 20})
│   └── 07_Activation_Norm_Dynamics/         # Section 4.3: Layer 8 norm recovery dynamics
│
├── data/                          # Benchmark splits, manifests & evaluation IDs
│   ├── test_ids.json              # Primary 500 test question indices
│   ├── human_audit_50_adjudication_manifest.json  # 50-sample formulary audit logs
│   └── removed_26_samples_manifest.json           # Quality filtering log (14,674 core pairs)
│
├── vectors/                       # Pre-extracted contrastive activation vectors
│   └── v_steer.pt                 # Unit-normalized Layer 8 steering tensor (dim=3584)
│
└── scripts/                       # Independent technical verification suites
    ├── verify_technical_consistency_full.py  # Full 7-stage end-to-end consistency audit
    └── verify_all_numbers_exact.py           # Verification of all paper tables from raw JSONs
```

---

## 🚀 Quickstart & Deterministic Verification

### 1. Environment Setup
```bash
git clone https://github.com/dothanhtuan1704-droid/early-stopping-steering.git
cd early-stopping-steering
pip install -r requirements.txt
```

### 2. Run Technical Consistency Audit (100% PASS)
Verify all mathematical claims, sample sizes, McNemar contingency counts, activation norms, and table numbers directly against disk ground truths:
```bash
python scripts/verify_technical_consistency_full.py
```

Expected output:
```text
[CHECK 1] Dataset Partitioning: PASS
[CHECK 2] Table 1 Panel A (Natural Completion T=800): PASS
[CHECK 3] Table 1 Panel B (Stress Testing T=200): PASS
[CHECK 4] Table 2 (Directional Controls & RAG Baselines): PASS
[CHECK 5] Table 3 (Steering Strength Ablation): PASS
[CHECK 6] Activation Norm Dynamics at Layer 8: PASS
[CHECK 7] Prefix Length Disentanglement: PASS
==> ALL TECHNICAL VERIFICATION CHECKS PASSED PERFECTLY!
```

---

## 📊 Summary of Experimental Results

### Table 1: Primary Evaluation Matrix across Response Horizons ($N_{\text{test}} = 500$)
| Regime / Schedule | Condition | RefPref (%) | BERTScore F1 | Rep-4 (%) | EOS Hit Rate (%) |
|---|---|:---:|:---:|:---:|:---:|
| **Natural Completion ($T=800$)** | Unsteered Baseline | 65.40% | 0.6779 | 3.68% | 100.0% |
| | Continuous ($K=\infty$) | 68.60% | 0.6778 | 3.90% | 99.8% |
| | **Hard Cutoff ($K=16$)** | **70.60%** | 0.6695 | 3.74% | **100.0%** |
| | Linear Decay ($K=16$) | 67.80% | 0.6706 | 3.74% | 99.8% |
| **Stress Testing ($T=200$)** | Unsteered Baseline | 73.20% | 0.7134 | 3.67% | -- |
| | Continuous ($K=\infty$) | 75.60% | 0.7133 | 3.90% | -- |
| | Hard Cutoff ($K=16$) | 77.00% | 0.7151 | 3.89% | -- |
| | **Linear Decay ($K=16$)** | **77.40%** | **0.7157** | **3.88%** | -- |

### Table 2: Placebo Distributions & Multi-Retriever RAG Baselines
* **Isotropic Random ($N=100$):** $55.82\% \pm 4.15\%$ (Steering exceeds mean by $+5.20\sigma$)
* **Negative Steered ($-v_{\text{steer}}$):** $62.80\%$ (Drops factual alignment by $-10.40$ pp)
* **BM25 Lexical RAG:** $68.60\%$ (Recall@1 = 19.20%)
* **Dense BGE-M3 RAG:** $74.80\%$ (Recall@1 = 42.60%)
* **Hybrid RRF RAG:** $76.20\%$ (Recall@1 = 48.20%)
* **Early-Stopping Steering Alone:** **$77.40\%$** (+1.20 pp over Hybrid RAG, zero retrieval overhead)
* **Synergistic Hybrid RAG + Steering:** **$80.40\%$** (Rep-4 suppressed to record 3.61%)

---

## 📖 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@inproceedings{tuan2026earlystopping,
  title     = {Early-Stopping Activation Steering for Hallucination Mitigation in Vietnamese Domain-Specific RAG},
  author    = {Phan, Do Thanh Tuan},
  booktitle = {Proceedings of the 15th International Symposium on Information and Communication Technology (SOICT 2026)},
  series    = {Communications in Computer and Information Science (CCIS)},
  publisher = {Springer},
  year      = {2026}
}
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
