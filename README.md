# 🔬 Early-Stopping Activation Steering for Factual Preference Alignment in Vietnamese Medical Language Models

[![Conference](https://img.shields.io/badge/SOICT-2026-blue.svg)](https://soict.org/)
[![Publisher](https://img.shields.io/badge/Springer-CCIS-orange.svg)](https://www.springer.com/series/7899)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Official research repository and reproducibility code package for the paper:  
**"Early-Stopping Activation Steering for Factual Preference Alignment in Vietnamese Medical Language Models"**  
*Submitted to the 15th International Symposium on Information and Communication Technology (SOICT 2026).*

---

## 📌 Abstract
Small Language Models (SLMs) offer immense potential for edge deployment and privacy-preserving clinical decision support. However, they frequently suffer from domain-specific hallucinations in low-resource and non-English contexts—such as miscalculating pediatric drug dosages or violating critical pregnancy contraindications. 

We propose **Early-Stopping Activation Steering**, a lightweight, non-destructive inference-time intervention that temporally bounds activation steering at **Layer 8** of `Qwen2.5-7B-Instruct` using **Hard Cutoff** and **Linear Decay** schedules ($K=16$). 

* **High-Cap Natural Completion ($T=800$):** Hard Cutoff ($K=16$) achieves **70.60%** automated reference-preference accuracy (+5.20 pp over unsteered baseline, exact McNemar $p = 0.00086$, $p_{\text{adj}} = 0.0026$) with **100% EOS hit rate** (BERTScore F1 0.6695 vs 0.6779; Rep-4 5.35% vs 4.10% baseline).
* **Bounded Stress Testing ($T=200$):** Linear Decay ($K=16$) elevates accuracy from 73.20% to **77.40%** (+4.20 pp, $p = 0.0055$), showing no statistically significant difference relative to multi-retriever Hybrid RRF RAG (76.20%, $p=0.4709$) and exceeding lexical BM25 RAG by **+8.80 pp** ($p = 3.85 \times 10^{-7}$) with zero context prompt expansion.
* **Combined RAG + Steering:** Combining Hybrid RAG with post-hoc selected Hard Cutoff achieves **80.40%** RefPref (+1.00 pp over RAG alone, $p=0.3018$, non-significant marginal gain) while maintaining 4-gram repetition at **4.46%** (vs 4.63% RAG alone).
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
│   ├── paper_soict.tex            # Canonical Springer LNCS LaTeX manuscript source
│   ├── llncs.cls / splncs04.bst   # Official Springer LNCS formatting templates
│   └── SOICT_2026_paper_0240.pdf  # Historical initial submission snapshot (see paper_soict.tex for camera-ready source)
│
├── notebooks/                     # Official Kaggle GPU experiment notebooks (by Table)
│   ├── 01_Steering_Vector_and_Probing/             # Layer-wise probing & v_steer extraction
│   ├── 02_Table1_Bounded_Stress_Test_T200/         # Table 1 Panel A stress testing (T=200)
│   ├── 03_Table1_Natural_Completion_T800/          # Table 1 Panel B natural completion (T=800) & Table 4
│   ├── 04_Table2_Directional_and_Placebo_Controls/ # Table 2: 100 Placebos, Negative, Label Shuffle
│   ├── 05_Table2_RAG_and_Synergistic_Steering/     # Table 2: BM25, Dense, Hybrid RAG & Synergistic RAG+Steering
│   ├── 06_Table3_Factorial_Alpha_Ablation/         # Table 3: Alpha sweep (alpha in {15, 18, 20})
│   ├── 07_Activation_Norm_Dynamics/                # Section 4.3: Layer 8 norm recovery dynamics
│   ├── 08_Temporal_Window_Controls/                # Section 4.5: Delayed & later injection controls
│   └── README_NOTEBOOK_MAPPING.md                  # Comprehensive mapping of all 27 notebooks
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
[CHECK 2] Table 1 Panel A (Bounded Stress Testing T=200): PASS
[CHECK 3] Table 1 Panel B (High-Cap Natural Completion T=800): PASS
[CHECK 4] Table 2 (Directional Controls & RAG Baselines): PASS
[CHECK 5] Table 3 (Steering Strength Ablation): PASS
[CHECK 6] Activation Norm Dynamics at Layer 8: PASS
[CHECK 7] Table 1 Footnote Matrices & Holm p_adj: PASS
==> ALL TECHNICAL VERIFICATION CHECKS PASSED PERFECTLY!
```

---

## 📊 Summary of Experimental Results

### Table 1: Primary Evaluation Matrix across Response Horizons ($N_{\text{test}} = 500$)
| Regime / Schedule | Condition | RefPref (%) | BERTScore F1 | Rep-4 (%) | EOS Hit Rate (%) |
|---|---|:---:|:---:|:---:|:---:|
| **Natural Completion ($T=800$)** | Unsteered Baseline | 65.40% | 0.6779 | **4.10%** | 100.0% |
| | Continuous ($K=\infty$) | 68.60% | 0.6712 | 4.38% | 99.8% |
| | **Hard Cutoff ($K=16$)** | **70.60%** | 0.6695 | 5.35% | **100.0%** |
| | Linear Decay ($K=16$) | 67.80% | 0.6732 | 5.37% | 99.8% |
| **Stress Testing ($T=200$)** | Unsteered Baseline | 73.20% | 0.7134 | 3.67% | -- |
| | Continuous ($K=\infty$) | 75.60% | 0.7133 | 3.90% | -- |
| | Hard Cutoff ($K=16$) | 77.00% | 0.7151 | 3.89% | -- |
| | **Linear Decay ($K=16$)** | **77.40%** | **0.7157** | **3.88%** | -- |

### Table 2: Comparison against Directional Controls, Placebo Baselines, and Multi-Retriever RAG ($N_{\text{test}}=500$, $T=200$)
| Method / Condition | Raw Count | RefPref (%) | Recall@1 | MRR@10 | BERT F1 | Latency (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Isotropic Gaussian ($N{=}100$) | -- | $55.82 \pm 4.15$ | -- | -- | 0.6945 | $23.33 \pm 0.30$ |
| Negative Steered ($-18.0 \cdot v_{\text{steer}}$) | 314 / 500 | 62.80% | -- | -- | 0.6889 | $23.34 \pm 0.28$ |
| BM25 Lexical RAG | 343 / 500 | 68.60% | 19.20% | 0.2433 | 0.6970 | $7.32 \pm 0.45$ |
| Label-Shuffled ($v_{\text{shuf}}$) | 343 / 500 | 68.60% | -- | -- | 0.6845 | $23.34 \pm 0.28$ |
| Unsteered Baseline | 366 / 500 | 73.20% | -- | -- | 0.7134 | $23.33 \pm 0.30$ |
| Cov-Matched ($N{=}20$) | -- | $74.81 \pm 0.95$ | -- | -- | 0.7182 | $23.34 \pm 0.28$ |
| Dense BGE-M3 RAG | 374 / 500 | 74.80% | 42.60% | 0.5124 | 0.7180 | **7.27 $\pm$ 0.42** |
| Hybrid RRF RAG | 381 / 500 | 76.20% | **48.20%** | **0.5681** | **0.7190** | $7.39 \pm 0.48$ |
| **Linear Decay Steered** | **387 / 500** | **77.40%** | -- | -- | 0.7157 | $23.34 \pm 0.28$ |
| Oracle Gold-Context RAG | 447 / 500 | 89.40% | 100.0% | 1.0000 | 0.8002 | $7.17 \pm 0.38$ |
| *Exploratory Combined Hybrid RAG + Steering ($T=800$)* | 402 / 500 | 80.40% | -- | -- | 0.6695 | $23.34 \pm 0.28$ |

### Table 3: Initial Steering Strength Ablation ($N_{\text{test}} = 500$, $T=200$)
| $\alpha_0$ | Schedule | RefPref (%) | BERTScore F1 | ROUGE-L (%) | Rep-4 (%) |
|---|---|:---:|:---:|:---:|:---:|
| -- | Unsteered Baseline | 73.20% | 0.7134 | 23.12% | 3.67% |
| -- | Continuous Reference ($\alpha_0{=}18.0$) | 75.60% | 0.7133 | 23.36% | 3.90% |
| 15.0 | Hard Cutoff ($K{=}16$) | **75.80%** | **0.7146** | 23.11% | 3.73% |
| | Linear Decay ($K{=}16$) | 75.60% | 0.7142 | 23.10% | 4.03% |
| | Nominal Matched Dose | 75.40% | 0.7144 | **23.40%** | 3.78% |
| 18.0 | Hard Cutoff ($K{=}16$) | 77.00% | 0.7151 | 23.16% | 3.89% |
| | **Linear Decay ($K{=}16$)** | **77.40%** | **0.7157** | 23.33% | **3.88%** |
| | Nominal Matched Dose | 75.40% | 0.7140 | 23.21% | 4.38% |
| 20.0 | Hard Cutoff ($K{=}16$) | 76.40% | 0.7139 | 23.17% | 3.87% |
| | Linear Decay ($K{=}16$) | **76.60%** | **0.7147** | 23.15% | **3.74%** |
| | Nominal Matched Dose | 75.20% | 0.7136 | 23.33% | 4.52% |


---

## 📖 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@inproceedings{tuan2026earlystopping,
  title     = {Early-Stopping Activation Steering for Factual Preference Alignment in Vietnamese Medical Language Models},
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
