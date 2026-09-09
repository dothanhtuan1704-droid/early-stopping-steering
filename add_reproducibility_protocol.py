"""
Adds explicit Reproducibility Implementation Protocol details into paper.tex and FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex.
"""
import os

files = ['paper.tex', 'FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex']

for fpath in files:
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()

    old_section_setup = r"\subsection{Experimental Setup}"
    if old_section_setup in text and "Reproducibility Implementation Protocol" not in text:
        new_section_setup = (
            r"\subsection{Experimental Setup \& Reproducibility Implementation Protocol}\n"
            r"\textbf{Model Checkpoint \& Precision.} All experiments evaluate \texttt{Qwen/Qwen2.5-7B-Instruct} \cite{ref4} loaded in 4-bit NormalFloat4 (NF4) quantization (\texttt{bfloat16} compute dtype).\n\n"
            r"\textbf{Exact Hook Location \& PyTorch Insertion.} The steering hook is attached via PyTorch \texttt{register\_forward\_hook} directly to the residual output of decoder block \texttt{model.model.layers[8]} (0-indexed 8th layer of the 28-layer architecture). The target tensor modified is the primary residual stream state \texttt{layer\_output[0]}.\n\n"
            r"\textbf{Prefill vs. Decoding \& Step Counter $t$.} During prompt prefill ($t=0$), no intervention vector is added. For subsequent autoregressive decoding steps ($t \ge 1$), the step counter $t$ increments by 1 per newly generated token, modifying the final-sequence position \texttt{[:, -1, :]}:\n"
            r"\begin{equation}\n"
            r"\hat{h}_8^{(t)} = h_8^{(t)} + \alpha(t) \cdot v_{\text{steer}},\n"
            r"\end{equation}\n"
            r"where $\alpha_0 = 18.0$ across all primary natural completion (\texttt{max\_new\_tokens=800}) and bounded stress testing (\texttt{max\_new\_tokens=200}) experiments.\n\n"
        )
        text = text.replace(old_section_setup, new_section_setup)

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Added Reproducibility Protocol to {fpath}")

print("Reproducibility protocol update completed!")
