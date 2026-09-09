import json, os

def create_exp02_subsplit_notebook(part_num, part_name, sub_part, start_idx, end_idx, window_start, window_end, max_tokens, output_json, filename):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# 🔬 Experiment 2 - Part {part_num}{sub_part}: Delayed Injection Window ({part_name}) [{start_idx}-{end_idx} samples, {max_tokens}tok]\n",
                f"**Mục tiêu:** Đánh giá can thiệp steering cửa sổ **{part_name}** (`window_start={window_start}`, `window_end={window_end}`) với $\\alpha_0=18.0$ ở độ dài `max_new_tokens={max_tokens}` trên tập con **{start_idx}-{end_idx}** mẫu Test.\n",
                f"**Output file:** `{output_json}`\n",
                "---"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Cài đặt Dependencies\n",
                "!pip install -q bitsandbytes accelerate transformers torch rouge-score bert-score tqdm\n",
                "print('✅ Dependencies installed successfully!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Khởi tạo Environment, Seeds & Config\n",
                "import os, json, glob, random, time, math, gc\n",
                "import numpy as np\n",
                "import torch\n",
                "from tqdm import tqdm\n",
                "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
                "from bert_score import score as bert_score_eval\n",
                "from rouge_score import rouge_scorer\n",
                "\n",
                "SEED = 42\n",
                "random.seed(SEED)\n",
                "np.random.seed(SEED)\n",
                "torch.manual_seed(SEED)\n",
                "if torch.cuda.is_available():\n",
                "    torch.cuda.manual_seed_all(SEED)\n",
                "\n",
                "MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'\n",
                "PART_NUM = " + str(part_num) + "\n",
                "PART_NAME = '" + part_name + "'\n",
                "SUB_PART = '" + sub_part + "'\n",
                "START_IDX = " + str(start_idx) + "\n",
                "END_IDX = " + str(end_idx) + "\n",
                "ALPHA_0 = 18.0\n",
                "WINDOW_START = " + str(window_start) + "\n",
                "WINDOW_END = " + str(window_end) + "\n",
                "MAX_NEW_TOKENS = " + str(max_tokens) + "\n",
                "OUTPUT_JSON = '" + output_json + "'\n",
                "\n",
                "print(f'Config Loaded: Window=[{WINDOW_START}, {WINDOW_END}], Alpha={ALPHA_0}, Sample Range=[{START_IDX}, {END_IDX}], MaxTokens={MAX_NEW_TOKENS}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Tải Dataset (Split 250 mẫu từ 500 Test Records)\n",
                "DATA_FILENAME = 'vietnamese_medical_halueval_15k_specialized.json'\n",
                "search_paths = [\n",
                "    f'/kaggle/input/**/{DATA_FILENAME}',\n",
                "    f'/kaggle/input/{DATA_FILENAME}',\n",
                "    f'data/{DATA_FILENAME}', f'./{DATA_FILENAME}'\n",
                "]\n",
                "data_path = None\n",
                "for pattern in search_paths:\n",
                "    matches = glob.glob(pattern, recursive=True)\n",
                "    if matches:\n",
                "        data_path = matches[0]\n",
                "        break\n",
                "if not data_path:\n",
                "    raise FileNotFoundError(f'❌ {DATA_FILENAME} not found!')\n",
                "\n",
                "with open(data_path, 'r', encoding='utf-8') as f:\n",
                "    raw_dataset = json.load(f)\n",
                "\n",
                "shuffled_records = list(raw_dataset)\n",
                "random.seed(SEED)\n",
                "random.shuffle(shuffled_records)\n",
                "n_total = len(shuffled_records)\n",
                "n_train = int(n_total * 0.70)\n",
                "n_val = int(n_total * 0.15)\n",
                "all_test = shuffled_records[n_train + n_val:]\n",
                "test_500 = all_test[:500]\n",
                "test_records = test_500[START_IDX:END_IDX]\n",
                "print(f'📊 Loaded Sub-Split [{START_IDX}:{END_IDX}]: {len(test_records):,} records')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Load Qwen2.5 Model & Vector Steer\n",
                "bnb_config = BitsAndBytesConfig(\n",
                "    load_in_4bit=True,\n",
                "    bnb_4bit_quant_type='nf4',\n",
                "    bnb_4bit_compute_dtype=torch.bfloat16\n",
                ")\n",
                "\n",
                "print(f'Loading model {MODEL_ID}...')\n",
                "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)\n",
                "tokenizer.padding_side = 'left'\n",
                "if tokenizer.pad_token is None:\n",
                "    tokenizer.pad_token = tokenizer.eos_token\n",
                "\n",
                "model = AutoModelForCausalLM.from_pretrained(\n",
                "    MODEL_ID,\n",
                "    quantization_config=bnb_config,\n",
                "    device_map='auto',\n",
                "    trust_remote_code=True\n",
                ")\n",
                "model.eval()\n",
                "print('✅ Model loaded successfully!')\n",
                "\n",
                "# Load or compute steering vector v_steer\n",
                "v_steer_paths = glob.glob('/kaggle/input/**/v_steer.pt', recursive=True)\n",
                "if v_steer_paths:\n",
                "    v_steer = torch.load(v_steer_paths[0], map_location='cpu')\n",
                "    print('✅ Loaded v_steer.pt from Kaggle dataset input')\n",
                "else:\n",
                "    print('⚠️ v_steer.pt not found on input, generating synthetic normalized steering vector...')\n",
                "    v_steer = torch.randn(model.config.hidden_size, dtype=torch.bfloat16)\n",
                "    v_steer = v_steer / torch.norm(v_steer)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Dynamic Multi-GPU Device Safe Hook\n",
                "class WindowedSteeringHook:\n",
                "    def __init__(self, vector, alpha, start_step, end_step):\n",
                "        self.vector = vector.detach().clone()\n",
                "        self.alpha = alpha\n",
                "        self.start_step = start_step\n",
                "        self.end_step = end_step\n",
                "        self.t = 0\n",
                "        self.handle = None\n",
                "\n",
                "    def hook_fn(self, module, input, output):\n",
                "        if isinstance(output, tuple):\n",
                "            out_tensor = output[0]\n",
                "        else:\n",
                "            out_tensor = output\n",
                "\n",
                "        # Prefill vs Decoding check\n",
                "        if out_tensor.shape[1] > 1:\n",
                "            self.t = 0\n",
                "            return output\n",
                "\n",
                "        self.t += 1\n",
                "        if self.start_step <= self.t <= self.end_step:\n",
                "            # Dynamic device & dtype casting for Multi-GPU compatibility\n",
                "            steer_vec = self.vector.to(device=out_tensor.device, dtype=out_tensor.dtype)\n",
                "            out_tensor[:, -1, :] = out_tensor[:, -1, :] + (self.alpha * steer_vec)\n",
                "\n",
                "        if isinstance(output, tuple):\n",
                "            return (out_tensor,) + output[1:]\n",
                "        return out_tensor\n",
                "\n",
                "    def register(self, layer_module):\n",
                "        self.handle = layer_module.register_forward_hook(self.hook_fn)\n",
                "\n",
                "    def remove(self):\n",
                "        if self.handle:\n",
                "            self.handle.remove()\n",
                "\n",
                "print(f'Multi-GPU Safe WindowedSteeringHook prepared for [{WINDOW_START}, {WINDOW_END}] with alpha={ALPHA_0}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Thực Thi Sinh 250 Câu với Hook Can Thiệp Cửa Sổ\n",
                "hook = WindowedSteeringHook(v_steer, ALPHA_0, WINDOW_START, WINDOW_END)\n",
                "target_layer = model.model.layers[8]\n",
                "hook.register(target_layer)\n",
                "\n",
                "generated_texts = []\n",
                "eos_hits = 0\n",
                "gen_lengths = []\n",
                "\n",
                "print(f'🚀 Starting Generation for {len(test_records)} prompts [{START_IDX}:{END_IDX}]...')\n",
                "start_time = time.time()\n",
                "\n",
                "for item in tqdm(test_records):\n",
                "    prompt = item['question']\n",
                "    messages = [{'role': 'user', 'content': prompt}]\n",
                "    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
                "    inputs = tokenizer(input_text, return_tensors='pt').to(model.device)\n",
                "\n",
                "    with torch.no_grad():\n",
                "        outputs = model.generate(\n",
                "            **inputs,\n",
                "            max_new_tokens=MAX_NEW_TOKENS,\n",
                "            do_sample=False,\n",
                "            pad_token_id=tokenizer.pad_token_id,\n",
                "            eos_token_id=tokenizer.eos_token_id\n",
                "        )\n",
                "\n",
                "    gen_tokens = outputs[0][inputs['input_ids'].shape[1]:]\n",
                "    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)\n",
                "    generated_texts.append(gen_text)\n",
                "    gen_lengths.append(len(gen_tokens))\n",
                "    if tokenizer.eos_token_id in gen_tokens:\n",
                "        eos_hits += 1\n",
                "\n",
                "hook.remove()\n",
                "total_time = time.time() - start_time\n",
                "print(f'✅ Generation Completed in {total_time:.2f}s! EOS Hit Rate: {eos_hits/len(test_records)*100:.2f}%')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7: Lưu Trữ Raw Output để Merger Gộp Chỉnh Sửa Sau\n",
                "raw_results = []\n",
                "for idx, (rec, text, gen_len) in enumerate(zip(test_records, generated_texts, gen_lengths)):\n",
                "    raw_results.append({\n",
                "        \"global_idx\": START_IDX + idx,\n",
                "        \"question\": rec['question'],\n",
                "        \"pos_ref\": rec['pos_ref'],\n",
                "        \"neg_ref\": rec['neg_ref'],\n",
                "        \"gen_text\": text,\n",
                "        \"gen_length\": gen_len\n",
                "    })\n",
                "\n",
                "output_data = {\n",
                "    \"experiment\": f\"Exp02_Part{PART_NUM}{SUB_PART}_{PART_NAME}_{MAX_NEW_TOKENS}tok\",\n",
                "    \"window_name\": PART_NAME,\n",
                "    \"sub_part\": SUB_PART,\n",
                "    \"start_idx\": START_IDX,\n",
                "    \"end_idx\": END_IDX,\n",
                "    \"window_start\": WINDOW_START,\n",
                "    \"window_end\": WINDOW_END,\n",
                "    \"alpha_0\": ALPHA_0,\n",
                "    \"max_new_tokens\": MAX_NEW_TOKENS,\n",
                "    \"eos_hits\": eos_hits,\n",
                "    \"total_samples\": len(test_records),\n",
                "    \"records\": raw_results\n",
                "}\n",
                "\n",
                "with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:\n",
                "    json.dump(output_data, f, indent=2, ensure_ascii=False)\n",
                "print(f'✅ Sub-split raw results saved to {OUTPUT_JSON}')"
            ]
        }
    ]

    nb_dict = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "gpuClass": "standard",
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb_dict, f, indent=2, ensure_ascii=False)
    print(f"Created 250-sample split notebook: {filename}")

windows = [
    (1, "Early_1_16", 1, 16),
    (2, "Delayed_17_32", 17, 32),
    (3, "Later_33_48", 33, 48),
    (4, "Continuous", 1, 9999)
]

# Generate Part A (0-250) and Part B (250-500) for both 200tok and 800tok
for tok in [200, 800]:
    for part_num, name, start, end in windows:
        # Part A: 0-250
        create_exp02_subsplit_notebook(
            part_num, name, "A", 0, 250, start, end, tok,
            f"exp02_part{part_num}A_{name.lower()}_{tok}tok_0_250.json",
            f"Exp02_Part{part_num}A_{name}_0_250_{tok}tok.ipynb"
        )
        # Part B: 250-500
        create_exp02_subsplit_notebook(
            part_num, name, "B", 250, 500, start, end, tok,
            f"exp02_part{part_num}B_{name.lower()}_{tok}tok_250_500.json",
            f"Exp02_Part{part_num}B_{name}_250_500_{tok}tok.ipynb"
        )

print("\nALL 250-SAMPLE SPLIT NOTEBOOKS CREATED SUCCESSFULLY!")
