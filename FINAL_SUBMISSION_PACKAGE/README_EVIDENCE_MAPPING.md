# 🗺️ BẢNG ÁNH XẠ CHÍNH THỨC: TỪ CON SỐ BÀI BÁO ĐẾN NOTEBOOK VÀ DỮ LIỆU THỰC NGHIỆM GỐC

**Dự án:** Early-Stopping Activation Steering for Factual Preference Alignment in Vietnamese Medical Language Models  
**Thư mục chứa:** `E:\Paper_Steering_VN_15K\FINAL_SUBMISSION_PACKAGE\`  

---

## 📊 1. BẢNG TỔNG HỢP ÁNH XẠ THỰC NGHIỆM CHÍNH (PRIMARY EXPERIMENTAL MATRIX)

| # | Hạng mục / Số liệu trong Bài báo (Manuscript Metric) | Giá trị thực nghiệm | File Notebook / Dữ liệu chứng minh (Proof Artifact) | Vị trí Cell / Dữ liệu gốc |
|---|-----------------------------------------------------|:-------------------:|-----------------------------------------------------|---------------------------|
| **1** | **Bounded Test Baseline Accuracy** | 73.00% (365/500) | [`phase-6b-v2-bertscore-based-clinical-correctness.ipynb`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/phase-6b-v2-bertscore-based-clinical-correctness.ipynb)<br>[`phase6b_v2_bertscore_clinical_results.json`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/phase6b_v2_bertscore_clinical_results.json) | Cell 3 & 4 |
| **2** | **Bounded Test Steered Accuracy (+v_steer)** | 77.20% (386/500) | Same as above | Cell 3 & 4 |
| **3** | **Bounded Test McNemar Contingency Matrix** | $a=349, b=16, c=37, d=98$<br>(Exact $p=0.0055$) | Same as above | JSON field `mcnemar_contingency_matrix` |
| **4** | **Natural Completion Accuracy (800 tokens)** | Baseline: 65.40% (327)<br>Hard Cutoff: 70.60% (353)<br>Continuous: 68.60% (343)<br>Linear Decay: 67.80% (339) | [`gpu-experiment-2a-primary-natural-completion-eval.ipynb`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/gpu-experiment-2a-primary-natural-completion-eval.ipynb) | Cell 4 & 5 |
| **5** | **Natural Completion Paired Cells (a,b,c,d)** | Hard Cutoff: (311,16,42,131)<br>Continuous: (309,18,34,139)<br>Linear Decay: (308,19,31,142) | Same as above | Cell 5 (`binomtest`) |
| **6** | **Placebo Controls Distribution (N=100)** | Mean: $55.82\% \pm 4.15\%$ SD<br>Distance: $+5.15\sigma$<br>Empirical $p=0.0099$ | [`03b_expanded_100_placebo_controls.ipynb`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/03b_expanded_100_placebo_controls.ipynb)<br>[`expanded_100_placebo_results.csv`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/expanded_100_placebo_results.csv) | CSV rows 1--100 |
| **7** | **Negative Steering (-v_steer)** | 62.80% (314/500) | [`03-directional-controls-multi-random-placebo-di.ipynb`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/03-directional-controls-multi-random-placebo-di.ipynb) | Cell 3 |
| **8** | **BM25 Retrieval Performance** | Recall@1 = 19.20%<br>Recall@3 = 28.40%<br>Recall@5 = 32.80%<br>MRR = 0.2433 | [`04-bm25-oracle-context-rag-evaluation-suite (1).ipynb`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/04-bm25-oracle-context-rag-evaluation-suite%20(1).ipynb) | Cell 4 |
| **9** | **RAG Generation Accuracy** | Non-oracle BM25: 68.60%<br>Oracle RAG: 89.40% | Same as above | Cell 5 & 6 |
| **10** | **Activation L2 Norm Trajectories (Layer 8)** | Baseline: $t=10: 53.64, t=50: 54.21$<br>Continuous: $t=10: 56.28, t=50: 56.18$<br>Linear Decay: $t=10: 52.95, t=50: 51.20$<br>Hard Cutoff: $t=10: 56.28, t=50: 52.75$ | [`02-activation-saturation-drift-diagnostics.ipynb`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/02-activation-saturation-drift-diagnostics.ipynb) | Cell 3 & 4 |
| **11** | **Layer Probing (Layers 4--16)** | Peak at Layer 8 ($\Delta \le 0.0100$) | [`gpu-experiment-1-layer-wise-probing (1).ipynb`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/gpu-experiment-1-layer-wise-probing%20(1).ipynb) | Cell 4 |
| **12** | **Human Evaluation Pilot (N=50)** | 50/50 correctness score 2 | [`human_evaluation_completed_50.csv`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/human_evaluation_completed_50.csv) | All 50 rows |

---

## 🚀 2. BỘ MODULE MỞ RỘNG THỰC NGHIỆM ĐẮT GIÁ (EXTENDED RESEARCH MODULES)

### A. Extended Module 1: Teacher Forcing Activation Trajectory Probing
* **Mã nguồn:** [`run_activation_mechanism_suite.py`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/run_activation_mechanism_suite.py)
* **Kết quả:** [`activation_mechanism_summary.csv`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/activation_mechanism_summary.csv), [`activation_mechanism_trajectories.json`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/activation_mechanism_trajectories.json)
* **Thông số đo đạc:**
  - Fixed-token teacher forcing trên chuỗi 100 steps cho cả 4 điều kiện.
  - Đo pre/post-hook norm, vector projection $\langle \hat{h}_8^{(t)}, v_{\text{steer}} \rangle$, cosine similarity, và Shannon logit entropy.

### B. Extended Module 2: Dense (BGE-M3) & Hybrid RAG + Profiling
* **Mã nguồn:** [`run_dense_hybrid_rag_suite.py`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/run_dense_hybrid_rag_suite.py)
* **Kết quả:** [`dense_hybrid_rag_summary.csv`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/dense_hybrid_rag_summary.csv), [`dense_hybrid_rag_results.json`](file:///E:/Paper_Steering_VN_15K/FINAL_SUBMISSION_PACKAGE/dense_hybrid_rag_results.json)
* **Thông số đo đạc:**
  - **Lexical BM25:** Recall@1 = 19.20%, Recall@5 = 32.80%, Accuracy = 68.60%
  - **Dense (BGE-M3):** Recall@1 = 42.60%, Recall@5 = 64.20%, Accuracy = 74.80%
  - **Hybrid (RRF BM25 + Dense):** Recall@1 = 48.20%, Recall@5 = 69.40%, Accuracy = 77.60%
  - **Oracle RAG:** Recall@1 = 100.0%, Accuracy = 89.40%
  - **Profiling 6 chi tiết hệ thống:** Retrieval Latency (ms), Prefill Latency (ms), Decoding Latency (ms), Total Pipeline Latency (ms), Peak VRAM Allocation (MB), Prompt Context Tokens.

---

📁 Tất cả 2 module mã nguồn mở rộng và các file kết quả tương ứng **đã được lưu trọn vẹn trong thư mục `E:\Paper_Steering_VN_15K\FINAL_SUBMISSION_PACKAGE\`**. Tác giả có thể chạy trực tiếp hoặc trích dẫn bổ sung vào bài báo bất cứ lúc nào!
