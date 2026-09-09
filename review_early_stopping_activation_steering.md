# Peer Review — *Early-Stopping Activation Steering for Hallucination Mitigation in Vietnamese Clinical Language Models*

**Tác giả:** Phan Do Thanh Tuan (FPT University Gia Lai)
**Độ dài:** 4 trang, định dạng IEEE conference
**Ngày review:** 23/08/2026
**Bộ tiêu chí áp dụng:** ACL Rolling Review (top-tier NLP) làm trục chính, đối chiếu thêm với chuẩn tạp chí Q1 y-tin học (JAMIA / JBI / AI in Medicine) và chuẩn IEEE/Scopus conference khu vực.

---

## 0. Lưu ý thuật ngữ trước khi vào review

"Q1" là phân vị **tạp chí** (Scopus SJR / JCR), không áp dụng cho hội nghị. Thứ hạng tương đương cho hội nghị là **CORE A\*/A** (ACL, EMNLP, NAACL, NeurIPS, ICLR, AAAI). Trong ngữ cảnh học thuật Việt Nam, "hội nghị Q1" thường được hiểu là một trong hai:

1. Hội nghị top-tier CORE A\*/A — chuẩn cao nhất, và
2. Hội nghị có proceedings được index Scopus, hoặc có special issue chuyển vào tạp chí Q1.

Review này chấm theo (1) làm chuẩn chính, vì đó là mức khắt khe nhất, rồi hạ dần cho (2) ở **Mục 10**.

---

## 1. Tóm tắt đóng góp theo cách hiểu của reviewer

Bài đề xuất **Early-Stopping Activation Steering**: trích một vector "truthfulness" `v_steer` từ hiệu trung bình activation giữa cặp câu trả lời đúng `y⁺` và câu trả lời ảo giác `y⁻` (Eq. 2), rồi cộng vector này vào residual stream tại **layer 8** của Qwen2.5-7B-Instruct, nhưng **chỉ trong K = 16 token đầu** với hệ số suy giảm tuyến tính (Eq. 4), thay vì steer liên tục.

Bài kèm theo benchmark tự xây **ViHaluEval-Medical** (14.700 bản ghi từ Dược thư Quốc gia Việt Nam, bản 5, 2018), và đánh giá bằng tiêu chí *BERTScore reference-preference*: đếm tỉ lệ output gần `y⁺` hơn `y⁻` về mặt ngữ nghĩa.

**Kết quả chính được tuyên bố:** độ chính xác reference-preference tăng 73,15% → 77,35% (+4,21 pp, McNemar p = 0,006) trên N = 499.

---

## 2. Bảng điểm tổng hợp

| Hạng mục (thang ACL ARR 1–5) | Điểm | Ghi chú |
|---|:---:|---|
| **Soundness** (luận cứ có đỡ nổi kết luận không) | **2,0** | Metric đánh giá bị vòng lặp logic với phương pháp; thí nghiệm robustness của chính tác giả bác bỏ kết quả chính |
| **Excitement / Novelty** | **2,0** | Chủ đề quan trọng, nhưng đóng góp kỹ thuật là một tinh chỉnh lịch trình nhỏ; hiệu ứng ở mức nhiễu |
| **Reproducibility** | **1,5** | Không code, không data, không prompt template, thiếu siêu tham số then chốt |
| **Substance / khối lượng thực nghiệm** | **2,0** | 1 model, 1 layer, 1 ngôn ngữ, 1 seed, 3 giá trị α, 0 baseline đối thủ |
| **Clarity / Presentation** | **3,0** | Văn phong rõ, toán học đúng, nhưng 0 hình vẽ, 0 ví dụ định tính |
| **Ethics & Responsible NLP** | **2,5** | Có thừa nhận giới hạn (đáng khen) nhưng thiếu ethics statement, thiếu bàn về rủi ro triển khai lâm sàng, thiếu xử lý bản quyền Dược thư |
| **Meta-review Overall** | **2,0 / 5** | |

**Khuyến nghị:** **Reject** ở mức top-tier NLP (ACL/EMNLP/NeurIPS/ICLR) và tạp chí Q1 y-tin học. **Major Revision** cho IEEE/Scopus conference khu vực. Chi tiết lộ trình sửa ở **Mục 9–10**.

---

## 3. Điểm mạnh (nêu trước, và đây là những điểm thật)

**3.1. Vấn đề nghiên cứu có giá trị thực và bị bỏ trống.** Ảo giác của LLM trên dữ liệu dược lâm sàng tiếng Việt là bài toán thật, ít người làm, có tác động xã hội rõ ràng. Chọn 3 nhóm rủi ro (Pregnancy Safety, Drug Interaction, Special Dosage) là chọn đúng chỗ đau.

**3.2. Tính trung thực học thuật cao hơn mặt bằng.** Đây là điểm nổi bật nhất của bài. Tác giả **tự gọi metric của mình là proxy**, tự nêu confound về cumulative dose, tự nói effect size nhỏ, và tự chạy thí nghiệm natural-termination để kiểm tra artifact. Rất nhiều bài activation steering ở hội nghị lớn không làm được như vậy. Phẩm chất này cần được giữ, và thực ra nó chính là lý do reviewer có thể chỉ ra được các lỗi bên dưới — tác giả đã đưa đủ dữ liệu để tự bác mình.

**3.3. Thiết kế kiểm soát cumulative dose là ý tưởng tốt.** Eq. 5–7 và hàng *Matched Dose Control* thể hiện tư duy đúng: nhận ra rằng so sánh lịch trình steering ở cùng α₀ là so sánh không công bằng về tổng liều can thiệp. Reviewer đã kiểm chứng lại toàn bộ số học và **các công thức đều đúng**:

| α₀ | Continuous (200α₀) | Cutoff (16α₀) | Decay ((K+1)/2·α₀) | α_match/token |
|---|---|---|---|---|
| 15,0 | 3000 | 240 | 127,5 ✓ | 0,6375 |
| 18,0 | 3600 | 288 | 153,0 ✓ | 0,7650 |
| 20,0 | 4000 | 320 | 170,0 ✓ | 0,8500 |

Ghi chú ở Eq. 4 về việc hệ số cuối tại `t = K` bằng `α₀/K` chứ không bằng 0 cũng chính xác — đây là mức độ cẩn thận đáng ghi nhận.

**3.4. Phương pháp rẻ và không phá hoại trọng số.** Zero-training, không catastrophic forgetting, chi phí tính toán O(d) mỗi token. Nếu hiệu ứng là thật, giá trị ứng dụng sẽ cao.

**3.5. Có phản xạ kiểm tra đúng.** Chạy layer sweep (L4–L15) và chạy natural-termination check là hai phản xạ nghiên cứu tốt. Vấn đề nằm ở chỗ **diễn giải** kết quả của chúng, chứ không phải ở việc có làm hay không.

---

## 4. Điểm yếu nghiêm trọng (blocking — xếp theo mức độ)

### ★ W1. Metric đánh giá bị vòng lặp logic với chính phương pháp can thiệp (lỗi nghiêm trọng nhất)

Đây là lỗi khiến kết quả chính không diễn giải được, và nó là lý do đủ để reject ở mọi venue top-tier.

- Vector steering được xây từ: `v_steer ∝ μ⁺ − μ⁻`, tức **hiệu activation giữa `y⁺` và `y⁻`** (Eq. 2).
- Metric đánh giá là: `RefPref(ŷ) = 1[BS(ŷ, y⁺) > BS(ŷ, y⁻)]` (Eq. 8), tức **so ŷ gần `y⁺` hay `y⁻` hơn**.

Cùng một cặp đối lập `(y⁺, y⁻)` vừa định nghĩa hướng can thiệp, vừa định nghĩa thước đo thành công. Đẩy model theo hướng `μ⁺ − μ⁻` rồi đo xem output có gần `y⁺` hơn `y⁻` không thì **gần như bảo đảm sẽ ra số dương** — bất kể vector đó có mã hóa "tính đúng sự thật" hay chỉ mã hóa *phong cách/template bề mặt* của tập `y⁺`.

Test set là question-disjoint, nhưng điều đó **không** giải quyết vấn đề: `y⁺` và `y⁻` của toàn bộ 14.700 bản ghi nhiều khả năng được sinh theo cùng một template, nên vector có thể chỉ đang học "văn phong của câu trả lời đúng" chứ không phải nội dung y khoa. +4,21 pp có thể hoàn toàn là hiệu ứng template-matching.

> **Cách sửa bắt buộc:** một metric độc lập với cặp `(y⁺, y⁻)`. Tối thiểu: cho **2–3 dược sĩ/bác sĩ chấm mù** 200–300 output (đúng/sai lâm sàng, có/không lỗi nguy hiểm), báo cáo Cohen's κ, rồi báo cáo **sensitivity/specificity của BERTScore-RefPref so với nhãn người**. Nếu tương quan yếu, toàn bộ Bảng II phải bị rút xuống thành phân tích phụ.

---

### ★ W2. Thí nghiệm robustness của chính tác giả bác bỏ kết quả chính, nhưng bài lại diễn giải ngược

Đây là lỗi diễn giải nặng nhất trong bài.

Ở Mục V-C, tác giả nới `max_new_tokens` từ 200 lên 800, đạt EOS hit 100%. Kết quả:

| Chế độ | Baseline | Steered | Δ |
|---|---|---|---|
| Truncated 200 token | 73,15% | 77,35% | **+4,21 pp** |
| Natural termination 800 token | 64,80% | 65,80% | **+1,00 pp** |

Bài viết: *"confirming that steering benefits persist in un-truncated long-form generation."*

Nhưng con số nói điều ngược lại — hiệu ứng **giữ lại chưa tới 1/4 độ lớn**, và reviewer tính lại thấy nó **không còn có ý nghĩa thống kê**:

- +1,00 pp trên N = 500 tương đương **net +5 item** (324 → 329).
- McNemar với `|b−c| = 5`, giả định tỉ lệ đổi kết quả tương tự nhánh 200-token (~10,6%, tức b+c ≈ 53): χ² ≈ 0,30 → **p ≈ 0,58**.
- Quét toàn dải giả định discordant từ 8% đến 20%: **p ∈ [0,53 ; 0,69]**. Không có giả định hợp lý nào cho ra p < 0,05.

Đồng thời BERTScore F1 tụt từ ~0,715 xuống 0,6804/0,6834 — tức **chênh lệch do đổi chế độ sinh (0,0316) lớn gấp ~18,6 lần chênh lệch mà bài coi là phát hiện chính (0,0017)**.

**Hệ quả:** kết quả headline +4,21 pp gần như chắc chắn là **artifact của việc cắt cứng ở 200 token**, không phải hiệu quả chống ảo giác. Bài đang trình bày điều kiện thí nghiệm thuận lợi nhất làm kết quả chính, và trình bày điều kiện thực tế hơn ở phần Discussion với một câu diễn giải sai chiều. Reviewer top-tier sẽ coi đây là vấn đề nghiêm trọng về mặt trình bày kết quả.

> **Bắt buộc:** đảo vai trò hai thí nghiệm. Chế độ natural-termination phải là kết quả chính; chế độ 200-token là phụ lục. Và phải báo cáo McNemar cho nhánh 800-token — kể cả khi p không đẹp.

---

### ★ W3. Thiếu hoàn toàn control giả dược (placebo) và kiểm tra dose–response

Trong văn hóa nghiên cứu activation steering, hai control này là **bắt buộc**, và bài không có cái nào:

1. **Random-direction control.** Cộng một vector ngẫu nhiên (hoặc trực giao với `v_steer`) cùng norm, cùng α, cùng layer 8, cùng K = 16. Nếu control này cũng cho +4 pp thì hiệu ứng đến từ *việc nhiễu loạn residual stream*, không phải từ *hướng truthfulness*. Không có control này thì **không có cơ sở nào để quy hiệu ứng cho ngữ nghĩa của vector**.
2. **Negative steering (−α).** Steer ngược phải làm giảm accuracy một cách hệ thống. Nếu không giảm, giả thuyết "trục truthfulness tuyến tính" sụp đổ.
3. **Đường cong α.** Bài chỉ thử α ∈ {15, 18, 20} — một dải hẹp bao quanh giá trị đã chọn. Cần α ∈ {0, 2, 5, 10, 15, 20, 30, 50} để thấy đường cong dose–response có hình chuông đúng như lý thuyết dự đoán không.

---

### ★ W4. Không có bất kỳ baseline đối thủ nào

Bài dựa hoàn toàn trên ITI [2], RepE [1], ActAdd [8], CAA [9] — nhưng **không so sánh với bất kỳ phương pháp nào trong số đó**. So sánh duy nhất là giữa các lịch trình steering của chính bài với nhau, cộng baseline unsteered.

Bảng baseline tối thiểu mà một venue top-tier sẽ đòi:

| Nhóm | Baseline cần có |
|---|---|
| Inference-time intervention | ITI (per-head), CAA, ActAdd, **DoLa** (Chuang et al. 2024), Contrastive Decoding |
| Không cần train | Prompt engineering ("chỉ trả lời khi chắc chắn"), self-consistency, chain-of-verification |
| Có train | LoRA SFT trên 70% training split (bài đã có sẵn data này) |
| Retrieval | RAG trên chính Dược thư — đây là baseline hiển nhiên nhất cho domain có nguồn tri thức đóng, và Mục I đã nêu RAG rồi lại không thử |

Thiếu RAG đặc biệt khó biện minh: toàn bộ ground truth đến từ **một cuốn sách**. RAG trên một corpus đóng, cố định, có cấu trúc là giải pháp mạnh nhất và rẻ nhất cho bài toán này, và bài chỉ bác nó bằng một câu trích dẫn [11].

---

### ★ W5. Hiệu ứng nằm dưới sàn nhiễu của chính hệ thống đo

Reviewer tính lại effect size từ CI mà bài báo cáo (Wilcoxon, 95% CI cho mean difference = [+0,0006, +0,0041]):

- SE ≈ 0,00089 → SD của hiệu ≈ 0,0200 → **Cohen's dz ≈ 0,12**.

Theo quy ước Cohen, d = 0,2 đã là "small"; **0,12 là dưới ngưỡng "small"**. Với N = 500, p = 0,017 chỉ chứng minh hiệu ứng khác 0, không chứng minh nó đáng kể. Cận trên của CI là +0,0041 BERTScore — vô nghĩa về mặt thực tiễn.

So sánh thang độ lớn cho thấy rõ vấn đề:

| Nguồn biến thiên | Độ lớn (BERTScore F1) | So với "phát hiện" |
|---|---|---|
| **Hiệu ứng bài tuyên bố** (Decay vs Continuous @ α=18) | 0,0017 | 1,0× |
| Biến thiên do chọn layer (L4–L15) | 0,0100 | **5,9×** |
| Biến thiên do đổi chế độ sinh (200 vs 800 token) | 0,0316 | **18,6×** |

Nói cách khác: **hai lựa chọn thiết kế tùy tiện gây ra biến thiên lớn hơn hiệu ứng nghiên cứu từ 6 đến 19 lần.** Không có CI nào trong Bảng I, không có seed lặp lại, không hiệu chỉnh đa so sánh (12 cấu hình × 4 metric = 48 con số; α Bonferroni cho 12 so sánh cặp là 0,0042).

---

### ★ W6. Benchmark — đóng góp số 1 của bài — hoàn toàn không được mô tả

ViHaluEval-Medical được nêu là contribution đầu tiên, nhưng bài **không nói một chữ nào** về:

- Quy trình xây dựng: rule-based? LLM sinh? người viết? Nếu dùng LLM thì model nào, prompt gì?
- **Ai viết `y⁻`?** Đây là điểm sống còn — nếu `y⁻` do LLM sinh theo template thì vector steering học template, quay lại W1.
- Kiểm định chất lượng: có người rà không? Bao nhiêu %? Inter-annotator agreement?
- Phân bố category trên toàn bộ 14.700 bản ghi (bài chỉ cho thấy phân bố của test split).
- **Bất thường cần giải thích:** category *Misleading Storage Conditions* chỉ có **N = 1** trong 500 mẫu test. Một benchmark 14.700 bản ghi mà có nhóm chiếm 0,2% test set là dấu hiệu pipeline sinh dữ liệu rất mất cân bằng. Việc loại nhóm này đi cần được giải thích, không chỉ ghi chú.
- Link công bố, license, format, datasheet.
- **Vấn đề bản quyền:** Dược thư Quốc gia Việt Nam là tài liệu có bản quyền của Bộ Y tế. Phát hành một benchmark phái sinh 14.700 bản ghi cần nêu rõ cơ sở pháp lý (fair use? xin phép? chỉ phát hành ID + script tái tạo?). Tạp chí Q1 y-tin học sẽ chặn bài ở đúng điểm này.

Không có phần này, contribution #1 **không tồn tại** dưới góc nhìn reviewer.

---

### ★ W7. Contribution #2 (chọn Layer 8) tự mâu thuẫn

Bài liệt kê *"Layer Selection: Identification of Layer 8 in Qwen2.5-7B-Instruct as a suitable layer"* là đóng góp riêng. Nhưng chính Mục V-C viết: quét L4–L15 cho BERTScore F1 dao động trong **dải hẹp 0,7040–0,7140**, và kết luận *"truthfulness representations are distributed across multiple intermediate layers rather than concentrated in a single layer."*

Nếu biểu diễn phân tán và Δ < 0,01 giữa các layer, thì **việc "xác định Layer 8" không phải phát hiện** — nó chỉ là một siêu tham số được tune trên validation set. Contribution #2 phải bị gỡ bỏ, hoặc được viết lại thành phát hiện âm tính thú vị hơn nhiều: *"truthfulness không định vị được ở một layer đơn lẻ trong Qwen2.5-7B cho tiếng Việt y khoa"*.

---

### ★ W8. Phương pháp được đề xuất không phải phương pháp thắng trong bảng của chính bài

Bài đặt tên và quảng bá **Linear Decay**. Nhưng đọc Bảng I:

| α₀ | Hard Cutoff (BS F1) | Linear Decay (BS F1) | Ai thắng? |
|---|---|---|---|
| 15,0 | **0,7146** | 0,7142 | Hard Cutoff |
| 18,0 | **0,7151** | 0,7150 | Hard Cutoff |
| 20,0 | 0,7139 | **0,7147** | Linear Decay |

**Hard Cutoff thắng 2/3 cấu hình**, và nó đơn giản hơn (không cần lịch trình suy giảm). Bài không hề thừa nhận điều này. Cần hoặc (a) trình bày Hard Cutoff là phương pháp chính, hoặc (b) chứng minh Linear Decay vượt trội ở khía cạnh khác với bằng chứng có CI. Hiện tại việc chọn Linear Decay làm "phương pháp của chúng tôi" trông giống chọn hậu nghiệm.

Đồng thời, **không có sweep trên K**. K = 16 xuất hiện không kèm biện minh thực nghiệm. Thí nghiệm hiển nhiên nhất — K ∈ {2, 4, 8, 16, 32, 64, 128, ∞} — bị bỏ trống, dù đó chính là biến trung tâm của bài.

---

### ★ W9. Cơ chế được giả định ("representation saturation") không hề được chứng minh

Toàn bộ động lực của bài là: steering liên tục gây *representation saturation*, biểu hiện thành *repetitive text loops và degraded syntax*. Nhưng bài **không đưa ra một bằng chứng nào** cho hiện tượng này:

- Không có biểu đồ `‖h_l^(t)‖` theo bước sinh t.
- Không có đo độ trôi của activation so với phân phối không steer.
- Không có ví dụ định tính về loop.
- Bằng chứng gián tiếp duy nhất là Rep-4, và nó **đổi chiều theo α**:

| α₀ | Continuous | Linear Decay | Giả thuyết saturation có đúng không? |
|---|---|---|---|
| 15,0 | **3,65** | 4,03 | ✗ Sai chiều — continuous lặp ÍT hơn |
| 18,0 | 3,90 | **3,74** | ✓ Đúng chiều |
| 20,0 | 4,05 | **3,74** | ✓ Đúng chiều |

Hai trong ba mức ủng hộ, một mức bác bỏ, biên độ 0,1–0,3 pp không có CI. Đây là dao động ngẫu nhiên, không phải xu hướng. Chính bài cũng thừa nhận điều này ở Mục V-A (*"At α₀ = 15 … the pattern is less consistent"*) nhưng không rút ra hệ quả cho giả thuyết trung tâm.

Tín hiệu **thú vị nhất trong Bảng I lại bị bỏ qua hoàn toàn**: hàng *Matched Dose Control* — cùng tổng liều với Linear Decay nhưng rải đều 200 token — có Rep-4 cao nhất bảng ở α = 18 và 20 (**4,38% và 4,52%** vs 3,74% của Decay). Nếu có thật, điều này nói rằng *rải liều mỏng và lâu* gây lặp nhiều hơn *dồn liều vào đầu*, tức là bằng chứng trực tiếp nhất cho luận điểm của bài. Tác giả nên khai thác quan sát này (kèm CI), thay vì tập trung vào chênh lệch BERTScore 0,0017.

Giả thuyết trung tâm của bài hiện đang ở trạng thái **được khẳng định chứ không được chứng minh**.

---

## 5. Điểm yếu mức trung bình

**W10. Bảng I thiếu hàng baseline α₀ = 0.** Bảng I không có dòng unsteered. Vì vậy từ Bảng I **không thể kết luận steering có tốt hơn không steering hay không** trên ROUGE-L / BERTScore / Rep-4. Baseline chỉ xuất hiện ở Bảng II với metric khác. Đây là lỗi thiết kế bảng cơ bản.

**W11. α = 18 không diễn giải được nếu không có tỉ lệ α/‖h_l‖.** `v_steer` được chuẩn hóa về norm 1 (Eq. 2), nên α = 18 nghĩa là cộng một vector độ dài 18 vào residual stream. Nhưng **‖h_8‖ điển hình của Qwen2.5-7B là bao nhiêu?** Nếu ~30 thì đây là can thiệp cực mạnh (60% norm); nếu ~500 thì gần như không đáng kể. Không có con số này, α = 18 vô nghĩa với người đọc và **không thể chuyển sang model khác** — điều này mâu thuẫn trực tiếp với tuyên bố "framework is architecture-agnostic" ở Mục V-C. Chuẩn báo cáo hiện nay là dùng α tương đối (`α · ‖h_l‖`) chứ không phải tuyệt đối.

**W12. Confound lượng tử hóa 4-bit NF4 không được xử lý.** Vector steering được trích và áp dụng trên model đã lượng tử hóa 4-bit. NF4 làm méo hình học activation. Không rõ hiệu ứng (hoặc sự vắng mặt của hiệu ứng) có tồn tại ở FP16 hay không. Cần ít nhất một run FP16 đối chứng.

**W13. Vấn đề độ dài — nghi ngờ toàn bộ dải động của metric bị nén.** Bài ghi `EOS Hit ≈ 0%` ở chế độ 200 token: model **không bao giờ dừng tự nhiên**, luôn bị cắt cứng ở 200 token. Nếu `y⁺` là một mục Dược thư ngắn (~30–60 token) còn `ŷ` luôn dài 200 token, thì BERTScore F1 bị chi phối bởi hình phạt precision trên phần đuôi thừa — điều này giải thích vì sao **mọi con số trong Bảng I đều nằm trong khoảng 0,7133–0,7151** (dải rộng 0,0018 trên 12 cấu hình). Metric đang bị bão hòa bởi độ dài, không phải bởi chất lượng.
Bài **không báo cáo độ dài trung bình của `y⁺`, `y⁻`, `ŷ`** ở bất kỳ đâu. Đây là thông tin bắt buộc.

**W14. Cấu hình BERTScore không phù hợp tiếng Việt.**
- Dùng `bert-base-multilingual-cased` (layer 9). mBERT là encoder **yếu** cho tiếng Việt. **PhoBERT** hoặc **XLM-R large** là lựa chọn chuẩn và sẽ cho tín hiệu ngữ nghĩa tốt hơn nhiều.
- Không dùng `rescale_with_baseline`. Đây là lý do trực tiếp khiến mọi điểm dồn quanh 0,71 và mọi khác biệt trông như 0,001. Rescale sẽ giãn dải động và cho thấy rõ khác biệt là nhiễu hay thật.

**W15. Các tuyên bố theo từng category không có ý nghĩa thống kê riêng.** Bài viết *"Early-stopping steering raises reference-preference accuracy across all three displayed categories"* như thể là ba phát hiện. Quy về số item tuyệt đối:

| Category | N | Net item | McNemar ước tính (discordant ~11%) |
|---|---|---|---|
| Pregnancy Safety | 168 | +10 | p ≈ 0,03 (yếu, mất khi hiệu chỉnh đa so sánh) |
| Drug Interaction | 157 | +6 | **p ≈ 0,22 — không ý nghĩa** |
| Special Dosage | 174 | +5 | **p ≈ 0,35 — không ý nghĩa** |

Không có kiểm định nào được báo cáo cho từng nhóm, và không có hiệu chỉnh Holm/Bonferroni. Cần hoặc bỏ tuyên bố per-category, hoặc báo cáo kiểm định + hiệu chỉnh đầy đủ.

**W16. Tuyên bố về latency không được số liệu đỡ.** *"no consistent latency penalty"*. Latency trong Bảng I dao động 21.613–23.414 ms (biên độ ~8%), không repeat, không CI, không nêu warm-up / batch size / số lần đo. Biến thiên đo lường lớn hơn mọi khác biệt hệ thống. Ngoài ra đây là tuyên bố tầm thường về mặt lý thuyết: hook là phép cộng O(d) mỗi token — hiển nhiên không tốn gì. Nên viết là "chi phí tính toán không đáng kể về mặt lý thuyết; phép đo của chúng tôi không đủ độ phân giải để phát hiện khác biệt nhỏ."

**W17. Chỉ greedy decoding.** Triển khai thực tế dùng sampling (temperature > 0, top-p). Hành vi của steering dưới sampling hoàn toàn chưa được thử, và steering có thể tương tác mạnh với entropy phân phối.

**W18. Không có negative control về năng lực tổng quát.** Mục I chê SFT vì catastrophic forgetting. Vậy bài phải chứng minh steering **không** làm hỏng năng lực chung — chạy model đã steer trên một benchmark tổng quát tiếng Việt (ViMMRC, ViQuAD, MMLU-vi) và cho thấy không tụt. Hiện không có.

**W19. Không có phân tích định tính, không có một ví dụ nào.** Một bài về ảo giác y khoa mà **không in ra dù chỉ một cặp output trước/sau steering**. Ở bất kỳ venue NLP nào đây là thiếu sót không chấp nhận được. Cần: 5–10 ví dụ thành công, và quan trọng hơn — **ví dụ thất bại**, gồm cả trường hợp steering *tạo ra* lỗi mới.

**W20. Thiếu chỉ số an toàn.** Với domain lâm sàng, metric quan trọng nhất không phải accuracy trung bình mà là **tỉ lệ lỗi nguy hiểm** (unsafe error rate): bao nhiêu % output còn khuyến nghị thuốc chống chỉ định thai kỳ, bao nhiêu % tính sai liều nhi khoa quá X%. Một phương pháp tăng accuracy 4 pp nhưng tăng lỗi nguy hiểm thì tệ hơn không làm gì. Bài không đo.

**W21. Chi tiết trích xuất vector chưa đủ.**
- Eq. 1 lấy activation tại **token cuối** của `x ⊕ y`. Không có ablation so với mean-pooling, hay lấy tại các vị trí token của phần answer (CAA và ITI dùng cách khác). Lựa chọn này ảnh hưởng lớn đến chất lượng vector.
- Không nói rõ hook áp lên **chỉ token mới sinh** hay cả context; không nói KV cache có bị ảnh hưởng không.
- Không nói `v_steer` được ước lượng trên bao nhiêu cặp (N trong Eq. 2), và độ ổn định của nó qua các subsample (bootstrap CI cho hướng vector).

---

## 6. Trình bày & định dạng

**P1. Độ dài 4 trang không đỡ nổi 5 contribution.** ACL long = 8 trang, NeurIPS = 9, ICLR = 9, tạp chí Q1 = không giới hạn thực tế. Bài 4 trang tuyên bố cả một benchmark 14.700 bản ghi + phát hiện layer + cơ chế mới + ablation 3×4 + giao thức đánh giá mới. Đây là mật độ của một **short paper/workshop paper** đang mang tham vọng của full paper. Cần hoặc mở rộng lên 8 trang với thực nghiệm tương xứng, hoặc thu hẹp tuyên bố xuống 1–2 contribution.

**P2. Không có hình vẽ nào.** Bài về layer probing, lịch trình theo thời gian và động lực học activation mà 0 figure. Tối thiểu cần 4 hình:
1. Đường cong layer sweep (L4–L15) kèm error bar.
2. Đường cong K sweep (biến trung tâm của bài, hiện chưa có).
3. `‖h_8^(t)‖` và độ trôi activation theo bước sinh — bằng chứng cho giả thuyết saturation (W9).
4. Sơ đồ minh họa 4 lịch trình α(t) (Continuous / Cutoff / Decay / Matched) — hình này sẽ làm Eq. 4–7 dễ hiểu ngay lập tức.

**P3. Nhất quán số.** Abstract ghi N_test = 500, đánh giá thực tế N = 499. Nên thống nhất "N = 500, phân tích trên 499 sau khi loại 1 mẫu đơn lẻ" ngay từ abstract.

**P4. Nên báo cáo ô discordant của McNemar.** Chỉ đưa p = 0,006 là không đủ. Reviewer phải tự suy ngược: từ 365 → 386 (net +21) và p = 0,006, ước tính **b ≈ 16, c ≈ 37, tổng discordant ≈ 53 (10,6% số mẫu đổi kết quả)**. Con số này nên nằm trong bài — nó cho thấy can thiệp chỉ chạm tới ~11% mẫu, và trong số đó có **~16 mẫu bị steering làm sai đi**. Đó là thông tin quan trọng cho domain lâm sàng và hiện đang bị ẩn.

**P5. Thiếu tuyên bố bắt buộc.** Không có Ethics Statement, Limitations section riêng (đang lồng trong Discussion), Reproducibility Statement, Data/Code Availability, Conflict of Interest. ACL bắt buộc Limitations riêng; tạp chí Q1 bắt buộc cả 5.

**P6. Đơn tác giả cho một bài y-tin học lâm sàng.** Không có đồng tác giả là chuyên gia y/dược. Với tạp chí Q1 y-tin học, đây gần như là điều kiện loại. Kể cả ở venue NLP, có một dược sĩ đồng hành sẽ giải quyết luôn W1 và W20.

---

## 7. Khoảng trống trong Related Work

**7.1. Thiếu các baseline decoding cùng họ:**
- **DoLa** (Chuang et al., ICLR 2024) — decoding by contrasting layers, cực kỳ liên quan.
- Contrastive Decoding (Li et al. 2023), SelfCheckGPT, Chain-of-Verification, TruthX.

**7.2. Thiếu tài liệu phê phán chính activation steering.** Có một dòng nghiên cứu cho thấy steering vector thường bắt **đặc trưng giả/bề mặt** thay vì khái niệm, và khái quát hóa kém ngoài phân phối (ví dụ các nghiên cứu về độ tin cậy/khả năng khái quát của steering vector, 2024). Vì đây chính là mối đe dọa lớn nhất với bài (W1), việc không trích dẫn và không phản biện dòng này là khoảng trống học thuật nghiêm trọng.

**7.3. Thiếu toàn bộ mảng NLP tiếng Việt.** Bài tuyên bố đóng góp một benchmark y khoa tiếng Việt nhưng **không trích dẫn một công trình tiếng Việt nào**: PhoBERT, PhoGPT, VinaLLaMA, ViGPT, SeaLLM, ViMedAQA, ViHealthQA, ViMQ, các dataset QA y tế tiếng Việt hiện có. Không thể khẳng định tính mới của benchmark khi không đối chiếu với những gì đã tồn tại. Đây là điểm reviewer khu vực (RIVF/KSE/NICS/PACLIC) sẽ bắt ngay lập tức.

**7.4. Thiếu công trình về lập lịch/định vị can thiệp.** Ý tưởng "chỉ can thiệp ở K token đầu" chắc chắn có tiền lệ trong các nghiên cứu về vị trí token và prompt-position trong steering/prefix-tuning. Cần khảo sát để xác định biên giới của tính mới.

---

## 8. Câu hỏi trực tiếp cho tác giả

1. `y⁻` được sinh ra như thế nào, bởi ai hoặc bởi model nào, và có bao nhiêu % được chuyên gia y/dược rà soát?
2. Vì `v_steer` được xây từ `μ⁺ − μ⁻` còn metric là `BS(ŷ,y⁺) > BS(ŷ,y⁻)`, tác giả loại trừ khả năng vòng lặp logic (W1) bằng cách nào?
3. Kết quả của **random-direction control** cùng norm và cùng α là bao nhiêu?
4. Steering ngược (−α) có làm giảm accuracy không? Giảm bao nhiêu?
5. Tỉ lệ `α / ‖h_8‖` là bao nhiêu? Con số này quyết định khả năng chuyển giao sang model khác.
6. Ở nhánh natural-termination 800 token, giá trị p của McNemar là bao nhiêu? Tác giả có coi +1,00 pp là bằng chứng ủng hộ phương pháp không?
7. Độ dài trung bình (token) của `y⁺`, `y⁻` và `ŷ` ở cả hai chế độ 200 và 800 token?
8. Trong ~53 mẫu đổi kết quả, **~16 mẫu bị steering làm sai đi** (ước tính từ p = 0,006) là những loại câu hỏi gì? Có mẫu nào chuyển từ an toàn sang nguy hiểm không?
9. Vì sao chọn Linear Decay làm phương pháp chính khi Hard Cutoff cho BERTScore cao hơn ở 2/3 mức α?
10. Vì sao K = 16? Có sweep nào trên K không?
11. Benchmark có được công bố không, theo license nào, và tác giả xử lý bản quyền Dược thư Quốc gia ra sao?
12. Kết quả có giữ nguyên ở FP16 (không lượng tử hóa 4-bit) không?

---

## 9. Lộ trình sửa — theo thứ tự ưu tiên

### Bắt buộc (không có thì không qua được bất kỳ venue nghiêm túc nào)

| # | Việc cần làm | Ước tính |
|---|---|---|
| 1 | **Random-direction + negative-steering control.** Rẻ nhất, đắt giá nhất. Nếu placebo cũng ra +4 pp thì bài phải viết lại từ đầu — biết sớm tốt hơn. | 1–2 ngày |
| 2 | **Chuyển natural-termination (800 token) thành thí nghiệm chính**, báo cáo McNemar đầy đủ, và sửa câu diễn giải ở Mục V-C. | 1 ngày |
| 3 | **Human validation trên 200–300 mẫu**, 2 chuyên gia y/dược chấm mù, báo cáo Cohen's κ + sensitivity/specificity của BERTScore-RefPref. | 1–2 tuần |
| 4 | **Datasheet cho ViHaluEval-Medical**: pipeline, QC, phân bố category, license, link công bố. | 3–5 ngày |
| 5 | **Ít nhất 3 baseline**: RAG trên Dược thư, LoRA SFT, và một trong ITI/CAA/DoLa. | 1 tuần |
| 6 | **Sweep K** ∈ {2,4,8,16,32,64,128,∞} kèm hình. | 2 ngày |
| 7 | **CI + nhiều seed** cho mọi ô trong Bảng I (bootstrap trên test set, ≥1000 lần lấy mẫu lại), cộng hiệu chỉnh Holm. | 2 ngày |

### Rất nên có

| # | Việc cần làm |
|---|---|
| 8 | Đổi BERTScore sang **PhoBERT/XLM-R + `rescale_with_baseline`**; báo cáo cả hai cấu hình |
| 9 | Thêm **unsafe error rate** — chỉ số quan trọng nhất cho domain lâm sàng |
| 10 | Bằng chứng thực nghiệm cho "representation saturation": biểu đồ `‖h_8^(t)‖` theo t |
| 11 | Báo cáo tỉ lệ `α/‖h‖`; chuyển sang α tương đối |
| 12 | Thêm 1 model thứ hai (Qwen2.5-1.5B/3B để đúng nghĩa "SLM", hoặc SeaLLM/VinaLLaMA cho tiếng Việt) |
| 13 | Negative control năng lực tổng quát trên benchmark tiếng Việt phổ thông |
| 14 | 4 hình vẽ (Mục P2) + 8–10 ví dụ định tính gồm cả ca thất bại |
| 15 | Bổ sung Related Work tiếng Việt + dòng phê phán steering vector |
| 16 | Mở rộng lên 8 trang; tách Limitations và Ethics Statement riêng |
| 17 | Mời đồng tác giả là dược sĩ/bác sĩ |

### Nên cân nhắc thay đổi khung bài

Nếu (1) cho thấy placebo control cũng tạo hiệu ứng, hoặc (3) cho thấy BERTScore-RefPref tương quan yếu với nhãn chuyên gia, thì **bài vẫn có giá trị xuất bản cao** — nhưng dưới dạng khác:

> *"Activation steering không mang lại lợi ích đo được cho việc chống ảo giác y khoa tiếng Việt: một nghiên cứu có control đầy đủ"* — kèm benchmark ViHaluEval-Medical được ghi chép chuẩn làm đóng góp chính.

Đây là bài mạnh hơn bài hiện tại. Nghiên cứu âm tính có control tốt, trên một benchmark mới, trong một ngôn ngữ ít tài nguyên, là đóng góp thật và đang thiếu trong cộng đồng. Bộ dữ liệu và pipeline đã có sẵn; chỉ cần đổi câu hỏi nghiên cứu và bổ sung control.

---

## 10. Đối chiếu theo từng loại venue

### A. Top-tier NLP (ACL / EMNLP / NAACL / NeurIPS / ICLR)

**Phán quyết: Reject.**

| Tiêu chí ARR | Đánh giá |
|---|---|
| Soundness | 2 — W1 (vòng lặp metric) và W2 (robustness tự bác) là hai lỗi chặn độc lập nhau |
| Excitement | 2 — đóng góp là một tinh chỉnh lịch trình, hiệu ứng dưới sàn nhiễu |
| Reproducibility | 1,5 — không artifact, thiếu siêu tham số then chốt |
| Ethics | cần Ethics + Limitations riêng, cần xử lý license Dược thư |

**Rào chắn cụ thể:** thiếu baseline ITI/CAA/DoLa (W4), thiếu placebo control (W3), độ dài 4 trang (P1), thiếu ví dụ định tính (W19), thiếu Related Work tiếng Việt (7.3).
**Cần tối thiểu để có cửa:** hạng mục 1–7 ở Mục 9, mở rộng lên 8 trang, và effect size phải sống sót qua placebo control.

### B. Tạp chí Q1 y-tin học (JAMIA / JBI / Artificial Intelligence in Medicine)

**Phán quyết: Reject / Reject-and-resubmit.**

Ở nhóm này, **clinical validation không phải mục "nên có" mà là điều kiện tồn tại**. Một bài tuyên bố giảm ảo giác trong "clinical decision support" mà không có một nhãn nào do bác sĩ/dược sĩ chấm sẽ bị desk-reject.

| Yêu cầu | Trạng thái |
|---|---|
| Clinician-adjudicated ground truth | ✗ Không có |
| Unsafe / harmful error rate | ✗ Không có |
| Ethics approval / IRB, hoặc tuyên bố miễn trừ | ✗ Không có |
| Datasheet cho dataset y khoa | ✗ Không có |
| Nguồn gốc & license dữ liệu y tế | ✗ Không có |
| Đồng tác giả có chuyên môn y/dược | ✗ Không có |
| Effect size có ý nghĩa lâm sàng | ✗ dz ≈ 0,12 |
| Data/Code availability statement | ✗ Không có |

Ngoài ra tiêu đề và abstract ("privacy-preserving clinical decision support in local hospital infrastructure") **hứa nhiều hơn nội dung chứng minh** — reviewer y khoa rất nhạy với overclaim kiểu này, kể cả khi phần Limitations đã đính chính.

### C. IEEE / Scopus conference khu vực (RIVF, KSE, NICS, ICCAIS, ACOMP, PACLIC…)

**Phán quyết: Major Revision — khả thi, và đây là đích thực tế nhất hiện nay.**

Bài đã có nền tốt cho nhóm này: định dạng IEEE đúng, toán học chính xác, ablation có cấu trúc, phần Limitations trung thực.

**Việc tối thiểu để được nhận:**

1. Thêm **random-direction control** (Mục 9, mục 1) — không tốn kém, và là điều reviewer kỹ thuật sẽ hỏi đầu tiên.
2. Thêm **1–2 baseline** (RAG trên Dược thư là dễ nhất và thuyết phục nhất).
3. Thêm **sweep K** + ít nhất 2 hình vẽ.
4. **Mô tả pipeline xây dựng benchmark** — nửa cột là đủ ở mức này.
5. **Sửa câu diễn giải natural-termination** (W2) và hạ tông kết quả headline. Reviewer khu vực có thể bỏ qua effect size nhỏ, nhưng sẽ không bỏ qua việc diễn giải sai dữ liệu của chính mình.
6. Bổ sung **Related Work tiếng Việt** — bắt buộc ở venue khu vực.
7. Gỡ hoặc viết lại contribution #2 (Layer 8) cho khỏi tự mâu thuẫn.
8. Mở rộng lên 6 trang IEEE.

Với 7–10 ngày làm việc tập trung cho các mục trên, bài có cơ hội tốt ở nhóm C.

---

## 11. Kết luận

Bài này có một chủ đề đúng, một tác giả trung thực, và một bộ dữ liệu có tiềm năng. Vấn đề nằm ở tầng luận chứng, không ở tầng ý tưởng.

Hai lỗi chặn cần được giải quyết trước mọi thứ khác:

1. **Thước đo không độc lập với can thiệp** (W1) — nên +4,21 pp chưa thể được đọc là "giảm ảo giác".
2. **Chính thí nghiệm robustness của tác giả cho thấy hiệu ứng gần như biến mất khi bỏ cắt cứng 200 token** (W2) — +1,00 pp, p ước tính ≈ 0,58, và bài đang diễn giải nó theo chiều ngược lại.

Cộng thêm việc thiếu placebo control (W3), một thí nghiệm chỉ tốn 1–2 ngày, hiện chưa có cơ sở nào để khẳng định hiệu ứng đến từ *hướng truthfulness* thay vì từ *bất kỳ nhiễu loạn nào* cùng độ lớn tại layer 8.

Điều đáng nói: **thí nghiệm rẻ nhất trong danh sách lại là thí nghiệm quyết định số phận bài báo.** Chạy random-direction control trước, rồi hãy quyết định viết bài theo hướng nào. Nếu hiệu ứng sống sót, bài có một phát hiện thật và đáng đầu tư đủ 7 hạng mục bắt buộc. Nếu không sống sót, bài chuyển thành một nghiên cứu âm tính có control tốt kèm benchmark tiếng Việt mới — vẫn xuất bản được, và trung thực hơn.

Phẩm chất tự phê bình đã thể hiện trong Mục V-C là tài sản lớn nhất của tác giả. Việc cần làm là để phẩm chất đó chi phối cả abstract và phần kết luận, chứ không chỉ nằm ở phần Discussion.

---

*Review dựa trên toàn văn 4 trang của bản thảo. Các con số ở W2, W5, W15 và P4 do reviewer tính lại từ dữ liệu bài báo cáo (McNemar với hiệu chỉnh liên tục, Cohen's dz suy ra từ khoảng tin cậy Wilcoxon); các ước tính p theo category và cho nhánh 800-token phụ thuộc giả định về tỉ lệ mẫu đổi kết quả, đã được quét trên dải 8–20% và kết luận không đổi trong toàn dải.*
