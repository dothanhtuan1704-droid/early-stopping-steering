# 📚 DANH MỤC NOTEBOOK THỰC NGHIỆM CHÍNH THỨC CỦA BÀI BÁO (SOICT 2026)

Thư mục này tổng hợp toàn bộ **27 Notebook Jupyter chính thức** đã dùng để trích xuất vector, chạy thực nghiệm GPU (Kaggle T4/P100), và tạo ra toàn bộ số liệu cho bài báo:
**"Early-Stopping Activation Steering for Hallucination Mitigation in Vietnamese Domain-Specific RAG"**

---

## 🗂️ BẢNG ÁNH XẠ TỪNG NOTEBOOK ĐẾN BẢNG BIỂU & NỘI DUNG BÀI BÁO

| Thư mục / Nhóm thực nghiệm | File Notebook chính | Bảng biểu / Phần trong bài báo | Giá trị chứng minh |
| :--- | :--- | :--- | :--- |
| **`01_Steering_Vector_and_Probing/`** | • `Kaggle_Phase1_Steering_Vector_Estimation.ipynb`<br>• `gpu-experiment-1-layer-wise-probing.ipynb`<br>• `Kaggle_Phase7_Layer_Probing.ipynb` | **Section 3.1 & 3.2**<br>(Trích xuất vector $\mathbf{v}_{steer}$ & Probing tầng) | Trích xuất vector Layer 8 từ 10.272 cặp train; xác định Layer 8 là đỉnh phân tách ngữ nghĩa y tế. |
| **`02_Table1_Bounded_Stress_Test_T200/`** | • `phase-6b-v2-bertscore-based-clinical-correctness.ipynb`<br>• `Kaggle_Phase6B_v2_BERTScore_Clinical.ipynb` | **Table 1 (Panel B)**<br>(Stress testing $T=200$) | Baseline 73.20% $\rightarrow$ Linear Decay **77.40%** (+4.20 pp, McNemar $p=0.0055$). |
| **`03_Table1_Natural_Completion_T800/`** | • `gpu-experiment-2a-primary-natural-completion-eval.ipynb`<br>• `Exp10_Baseline_PartA/B.ipynb`<br>• `Exp10_HardCutoff_NoPen_PartA/B.ipynb`<br>• `Exp10_LinearDecay_NoPen_PartA/B.ipynb` | **Table 1 (Panel A)**<br>(Natural completion $T=800$) | Baseline 65.40% $\rightarrow$ Hard Cutoff **70.60%** (+5.20 pp, McNemar $p=0.00086$, 100% EOS hit rate). |
| **`04_Table2_Directional_and_Placebo_Controls/`** | • `03b_expanded_100_placebo_controls.ipynb`<br>• `03-directional-controls-multi-random-placebo-di.ipynb`<br>• `Kaggle_Experiment_09_LabelShuffled_Placebo.ipynb` | **Table 2**<br>(Directional & Placebo controls) | $N=100$ isotropic random vectors ($55.82\% \pm 4.15\%$, $z_{\text{sep}}=5.20\sigma$); Negative steering (62.80%); Label permutation (68.60%). |
| **`05_Table2_RAG_and_Synergistic_Steering/`** | • `04-bm25-oracle-context-rag-evaluation-suite.ipynb`<br>• `07_dense_bge_m3_hybrid_rag_eval.ipynb`<br>• `Kaggle_Exp04_Synergistic_RAG_Plus_Steering.ipynb` | **Table 2 & Section 4.5**<br>(RAG baselines & RAG+Steering) | BM25 RAG (68.60%), Dense BGE-M3 (74.80%), Hybrid RAG (76.20%), Oracle (89.40%). **Synergistic RAG + Steering đạt 80.40%**, Rep-4 giảm còn 3.61%. |
| **`06_Table3_Factorial_Alpha_Ablation/`** | • `kaggle-phase-6a2-fair-equal-alpha-ablation-experi.ipynb`<br>• `Kaggle_Phase6A_Fair_Equal_Alpha_Ablation.ipynb` | **Table 3**<br>(Ablation hệ số $\alpha_0 \in \{15, 18, 20\}$) | Chứng minh $\alpha_0 = 18.0$ đạt đỉnh 77.40% RefPref và cân bằng tối ưu Rep-4. |
| **`07_Activation_Norm_Dynamics/`** | • `02-activation-saturation-drift-diagnostics.ipynb`<br>• `06_independent_activation_norm_benchmark.ipynb`<br>• `06_activation_mechanism_teacher_forcing.ipynb` | **Section 4.3**<br>(Động học Norm Layer 8) | Triệt tiêu độ dời norm sau cutoff ($\Delta_{\text{norm}}=0.00$); paired $t$-test so với baseline $p=0.1050 > 0.05$ (không lệch norm). |
| **`08_Temporal_Window_Controls/`** | • `Kaggle_Exp02_Delayed_Injection_Window_Control.ipynb` | **Section 4.5**<br>(Thử nghiệm dịch chuyển cửa sổ tiêm) | Delayed injection $t \in [17, 32]$ và later injection $t \in [33, 48]$ (72.80%, không hiệu quả bằng can thiệp sớm). |

---

## 🔒 Cam kết tính toàn vẹn
- Tất cả các notebook trên đều được lưu trữ nguyên bản, dùng để làm minh chứng tái lập kết quả (reproducibility) cho hội nghị hoặc ban phản biện khi được yêu cầu.
