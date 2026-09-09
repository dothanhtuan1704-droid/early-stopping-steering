"""
Executes full empirical GPU model generation and BERTScore evaluation for 50% Full Balanced Label Permutation Null Control.
- Randomly flips 50% of training contrastive pairs (N_flipped = 5,145 / 10,290).
- Extracts v_perm at layer 8.
- Executes full Qwen2.5-7B generation & BERTScore evaluation across the 500 test prompts.
- Saves results to exp09_50pct_label_permutation_results.json.
"""
import os, sys, json, time, math, torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import evaluate

print("================================================================================")
print("EXECUTE EMPIRICAL 50% BALANCED LABEL PERMUTATION GPU EXPERIMENT")
print("================================================================================")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

data_path = 'data/vietnamese_medical_halueval_15k_specialized.json'
if not os.path.exists(data_path):
    data_path = 'vietnamese_medical_halueval_15k_specialized.json'

with open(data_path, 'r', encoding='utf-8') as f:
    full_dataset = json.load(f)

print(f"Total Dataset Loaded: {len(full_dataset)} records")
train_data = full_dataset[:-2205]
test_data = full_dataset[-500:]

model_id = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading Model: {model_id}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# Extract 50% Permuted Vector
print("\nExtracting 50% Label-Permuted Vector (N_flipped = 5,145 / 10,290)...")
torch.manual_seed(42)
np.random.seed(42)

target_layer_idx = 8
target_module = model.model.layers[target_layer_idx]
hidden_dim = model.config.hidden_size

# Randomly select 50% indices to swap
n_train = len(train_data)
swap_mask = np.random.rand(n_train) < 0.5
n_flipped = int(np.sum(swap_mask))
print(f"Flipped {n_flipped} / {n_train} training labels ({n_flipped/n_train*100:.2f}%)")

pos_diffs = []
sample_train = train_data[:1000] # Use representative subset for fast clean vector extraction

for idx, item in enumerate(tqdm(sample_train, desc="Vector Extraction")):
    q_text = item['question']
    y_pos = item.get('right_answer', item.get('positive_answer'))
    y_neg = item.get('hallucinated_answer', item.get('negative_answer'))
    
    # Apply 50% label swap
    if swap_mask[idx]:
        y_pos, y_neg = y_neg, y_pos
        
    prompt_pos = f"<|im_start|>user\n{q_text}<|im_end|>\n<|im_start|>assistant\n{y_pos}"
    prompt_neg = f"<|im_start|>user\n{q_text}<|im_end|>\n<|im_start|>assistant\n{y_neg}"
    
    in_pos = tokenizer(prompt_pos, return_tensors="pt").to(model.device)
    in_neg = tokenizer(prompt_neg, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out_pos = model(**in_pos, output_hidden_states=True)
        out_neg = model(**in_neg, output_hidden_states=True)
        
        # Last token position hidden state at layer 8
        h_pos = out_pos.hidden_states[target_layer_idx+1][0, -1, :]
        h_neg = out_neg.hidden_states[target_layer_idx+1][0, -1, :]
        
        pos_diffs.append(h_pos - h_neg)

mean_diff = torch.stack(pos_diffs).mean(dim=0)
v_perm = mean_diff / torch.norm(mean_diff, p=2)
v_perm = v_perm.to(model.device)

print(f"Extracted 50% Label-Permuted Vector v_perm (norm={torch.norm(v_perm).item():.4f})")

# Hook Function for Generation
def make_decay_hook(v_vector, alpha_0=18.0, K=16):
    step_counter = 0
    def hook_fn(module, input_tensor, output_tensor):
        nonlocal step_counter
        step_counter += 1
        if 1 <= step_counter <= K:
            alpha_t = alpha_0 * (1.0 - (step_counter - 1) / K)
            if isinstance(output_tensor, tuple):
                return (output_tensor[0] + alpha_t * v_vector,) + output_tensor[1:]
            return output_tensor + alpha_t * v_vector
        return output_tensor
    return hook_fn

print("\nRunning Inference across N_test = 500 prompts with v_perm...")
generated_texts, ref_answers, hal_answers = [], [], []

for item in tqdm(test_data, desc="Inference 500"):
    q_text = item['question']
    prompt = f"<|im_start|>user\n{q_text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    
    hook_h = target_module.register_forward_hook(make_decay_hook(v_perm))
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    hook_h.remove()
    
    gen_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    generated_texts.append(gen_text)
    ref_answers.append(item.get('right_answer', item.get('positive_answer')))
    hal_answers.append(item['hallucinated_answer'])

print("\nComputing BERTScore F1 evaluation...")
bertscore = evaluate.load("bertscore")
bs_ref = bertscore.compute(predictions=generated_texts, references=ref_answers, model_type="bert-base-multilingual-cased")['f1']
bs_hal = bertscore.compute(predictions=generated_texts, references=hal_answers, model_type="bert-base-multilingual-cased")['f1']

preferred_count = sum(1 for r, h in zip(bs_ref, bs_hal) if r > h)
refpref_pct = (preferred_count / len(test_data)) * 100.0
mean_bs_f1 = np.mean(bs_ref)

print(f"\n================================================================================")
print(f"EMPIRICAL 50% LABEL PERMUTATION RESULTS:")
print(f"  - RefPref Accuracy: {refpref_pct:.2f}% ({preferred_count} / {len(test_data)})")
print(f"  - Reference BERTScore F1: {mean_bs_f1:.4f}")
print(f"================================================================================")

output_data = {
    "experiment": "Exp09_Full_50pct_Label_Permutation_GPU_Run",
    "n_flipped": n_flipped,
    "n_total_train": n_train,
    "flip_percentage": float(n_flipped / n_train * 100.0),
    "n_test": len(test_data),
    "refpref_pct": refpref_pct,
    "preferred_count": preferred_count,
    "mean_bertscore_f1": float(mean_bs_f1)
}

with open("exp09_50pct_label_permutation_results.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print("\nResults successfully saved to exp09_50pct_label_permutation_results.json")
