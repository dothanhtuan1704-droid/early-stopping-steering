# 🔗 Official Kaggle Notebook Links for Paper Experiments

All experimental conditions, directional controls, ablation sweeps, and delayed injection experiments have been executed on Kaggle GPU (T4 x2 / P100) across the held-out test set $N_{\text{test}} = 500$.

---

## 📌 Main Benchmark & Early-Stopping Steering Links

| Experiment | Method / Condition | Kaggle Notebook Public Link | Account / Author | Status |
|---|---|---|---|:---:|
| **Hard Cutoff ($K=16$)** | Primary Hard Cutoff Steering | [Kaggle Link](https://www.kaggle.com/code/duuykiu/kaggle-sub-notebook-hard-cutoff-steering-k-16) | `duuykiu` | ✅ Completed |
| **Phase 3C1 (Main Methods)** | Main Methods 200tok | [Kaggle Link](https://www.kaggle.com/code/thanhtranguyn/kaggle-phase3c1-main-methods-200token?scriptVersionId=342250616) | `thanhtranguyn` | ✅ Completed |
| **Phase 3C2 (Controls)** | Control Methods 200tok | [Kaggle Link](https://www.kaggle.com/code/thanhtranguyn/kaggle-phase3c2-control-methods-200tok?scriptVersionId=342250913) | `thanhtranguyn` | ✅ Completed |
| **Phase 6 Baseline** | Unsteered Baseline Control | `notebookb58962698a.ipynb` | `anhemgithom` | ✅ Completed |

---

## 📌 Exp 02: Delayed Injection Window Controls ($K_0$)

| Experiment | Injection Window | Kaggle Notebook Public Link | Account | Status |
|---|:---:|---|---|:---:|
| **Exp 02 (Part 1B)** | $K_0 \in [1, 16]$ | [Kaggle Link](https://www.kaggle.com/code/duuykiu/experiment-2-part-1b-delayed-injection-windo) | `duuykiu` | ✅ Completed |
| **Exp 02 (Part 2A)** | $K_0 \in [17, 32]$ | [Kaggle Link](https://www.kaggle.com/code/duuykiu/experiment-2-part-2a-delayed-injection-windo) | `duuykiu` | ✅ Completed |
| **Exp 02 (Part 2B)** | $K_0 \in [17, 32]$ | [Kaggle Link](https://www.kaggle.com/code/duuykiu/experiment-2-part-2b-delayed-injection-windo) | `duuykiu` | ✅ Completed |

---

## 📌 Exp 09: Isotropic & Covariance-Matched Placebo Baselines

| Experiment | Control Type | Kaggle Notebook Public Link | Account | Status |
|---|---|---|---|:---:|
| **Exp 09 Label Permutation** | Full 50% Balanced Permutation | [Kaggle Link](https://www.kaggle.com/code/duuykiu/kaggle-experiment-09-full-50-balanced-label-p) | `duuykiu` | ✅ Completed |
| **Exp 09 Covariance Match (B1)** | Random Vector Batch 1 ($N=100$) | [Kaggle Link](https://www.kaggle.com/code/duuykiu/exp-9-covariance-matched-random-vectors-batch-1) | `duuykiu` | ✅ Completed |
| **Exp 09 Covariance Match (B2)** | Random Vector Batch 2 ($N=100$) | [Kaggle Link](https://www.kaggle.com/code/duuykiu/exp-9-covariance-matched-random-vectors-batch-2) | `duuykiu` | ✅ Completed |

---

## 📌 Phase 6A: Equal-$\alpha$ Factorial Ablation Experiment Links

| Experiment | Alpha ($\alpha$) | Part | Kaggle Notebook Link | Account |
|---|:---:|:---:|---|---|
| **Phase 6A1 (Part 1)** | 15.0 | Part 1 (`Continuous`, `Hard Cutoff`) | [Kaggle Link](https://www.kaggle.com/code/trungkiennnn/kaggle-phase6a1-part1-alpha15-ipynb) | `trungkiennnn` |
| **Phase 6A1 (Part 2)** | 15.0 | Part 2 (`Linear Decay`, `Matched Dose`) | [Kaggle Link](https://www.kaggle.com/code/trungkiennnn/kaggle-phase6a1-part2-alpha15-ipynb) | `trungkiennnn` |
| **Phase 6A2 (Part 1)** | 18.0 | Part 1 (`Continuous`, `Hard Cutoff`) | [Kaggle Link](https://www.kaggle.com/code/tunthanh66/kaggle-phase6a2-part1-alpha18-ipynb) | `tunthanh66` |
| **Phase 6A2 (Part 2)** | 18.0 | Part 2 (`Linear Decay`, `Matched Dose`) | [Kaggle Link](https://www.kaggle.com/code/tunthanh66/kaggle-phase6a2-part2-alpha18-ipynb) | `tunthanh66` |
| **Phase 6A3 (Part 1)** | 20.0 | Part 1 (`Continuous`, `Hard Cutoff`) | [Kaggle Link](https://www.kaggle.com/code/anhemgithom/kaggle-phase6a3-part1-alpha20-ipynb) | `anhemgithom` |
| **Phase 6A3 (Part 2)** | 20.0 | Part 2 (`Linear Decay`, `Matched Dose`) | [Kaggle Link](https://www.kaggle.com/code/anhemgithom/kaggle-phase6a3-part2-alpha20-ipynb) | `anhemgithom` |
| **Diagnostics & Probing** | -- | Probing & Norm Trajectories | [Kaggle Link](https://www.kaggle.com/code/duuykiu/notebook83db094084) | `duuykiu` |
