# You Only Judge Once: Multi-response Reward Modeling in a Single Forward Pass

Official code release for **"You Only Judge Once: Multi-response Reward Modeling in a Single Forward Pass"**.

<p align="center">
  <a href="https://arxiv.org/abs/2604.10966"><img src="https://img.shields.io/badge/arXiv-2604.10966-b31b1b.svg"></a>
  <a href="https://huggingface.co/yinuoy/MR2-Molmo2-4B-RM"><img src="https://img.shields.io/badge/🤗_Model-MR2--Molmo2--4B--RM-yellow"></a>
  <a href="https://huggingface.co/datasets/yinuoy/MR2Bench"><img src="https://img.shields.io/badge/🤗_Dataset-MR2Bench-blue"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg"></a>
</p>

## Overview

We present a discriminative multimodal reward model that scores all N candidate responses in a **single forward pass**, achieving up to N× wall-clock speedup and FLOPs reduction over conventional single-response scoring, and state-of-the-art accuracy across six benchmarks with only 4B parameters.

**Key Idea**: Cross-response attention under the causal mask enables direct comparative reasoning between candidates, yielding both higher accuracy and greater efficiency than approaches that score responses independently (discriminative) or compare them pairwise (generative).

## Released Artifacts

| Artifact | Link |
|---|---|
| 📃 Paper | [arXiv:2604.10966](https://arxiv.org/abs/2604.10966) |
| 🤖 Reward Model | [yinuoy/MR2-Molmo2-4B-RM](https://huggingface.co/yinuoy/MR2-Molmo2-4B-RM) |
| 📊 Benchmarks | [yinuoy/MR2Bench](https://huggingface.co/datasets/yinuoy/MR2Bench) |
| 💻 Code | [yinuoyang01/multi-response-rm](https://github.com/yinuoyang01/multi-response-rm) |

## Installation

```bash
git clone https://github.com/yinuoyang01/multi-response-rm.git
cd multi-response-rm
pip install -r requirements.txt
```

## Quick Start: Inference

Uses `MultiResponseRewardModel` from `mr2rm/`, consistent with training:

```python
import json
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import hf_hub_download
from PIL import Image

from mr2rm.models.reward_model import MultiResponseRewardModel
from mr2rm.data.dataset import add_resp_sep_token, RESP_SEP_TOKEN

model_id = "yinuoy/MR2-Molmo2-4B-RM"

# 1. Load processor and register separator token
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, dtype="auto", device_map="auto")
add_resp_sep_token(processor.tokenizer)

# 2. Load base model
base_model = AutoModelForImageTextToText.from_pretrained(
    model_id, trust_remote_code=True, dtype="auto", device_map="auto"
)

# 3. Create reward model (config from reward_model_config.json)
config_path = hf_hub_download(repo_id=model_id, filename="reward_model_config.json")
with open(config_path) as f:
    rm_config = json.load(f)

reward_model = MultiResponseRewardModel(
    base_model=base_model,
    value_head_type=rm_config.get("value_head_type", "mlp"),
    value_head_hidden_dim=rm_config.get("value_head_hidden_dim", 1024),
    value_head_activation=rm_config.get("value_head_activation", "silu"),
    resp_repr_mode=rm_config.get("resp_repr_mode", "last"),
)

# 4. Load value head weights
vh_path = hf_hub_download(repo_id=model_id, filename="value_head.pt")
reward_model.value_head.load_state_dict(torch.load(vh_path, map_location="cpu"))
device = next(reward_model.base_model.parameters()).device
dtype = next(reward_model.base_model.parameters()).dtype
reward_model.value_head = reward_model.value_head.to(device=device, dtype=dtype)
reward_model.eval()

# 5. Score N candidate responses
image = Image.open("example.jpg").convert("RGB")
responses = [
    "A golden retriever sitting on grass.",
    "A dog in a park on a sunny day.",
    "There is an animal outside.",
    "I don't know.",
]

# Build input: concatenate responses with separator (same format as training)
sep = f"\n\n{RESP_SEP_TOKEN}\n\n"
text = "Describe this image." + "\n\n" + sep.join(responses)
messages = [{"role": "user", "content": [dict(type="image", image=image), dict(type="text", text=text)]}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt", return_dict=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Find response boundary positions
sep_token_id = processor.tokenizer.convert_tokens_to_ids(RESP_SEP_TOKEN)
input_ids = inputs["input_ids"][0]
sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0].tolist()
end_positions = [p - 1 for p in sep_positions] + [input_ids.size(0) - 1]
resp_indices = torch.tensor([end_positions], device=device)

# Forward pass — single pass scores all N responses
with torch.inference_mode():
    scores = reward_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        resp_indices=resp_indices,
        **{k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}
    )
    rewards = scores[0][0].tolist()  # [N] scores

print("Scores:", rewards)
print("Best response:", responses[rewards.index(max(rewards))])
```

## Training

### 1. Prepare Data

JSONL, one sample per line:

```json
{
  "prompt": "...",
  "responses": ["A", "B", "C", "D"],
  "rankings": [3, 1, 2, 4],
  "image": "path/to/img.jpg",
  "video": "path/to/video.mp4"
}
```

- `rankings`: list of ints, **1 = best**, higher = worse; ties allowed (e.g. `[1, 1, 3, 4]`). Must match `responses` length.
- `image` / `video`: optional. Accepts absolute paths or paths relative to `--image_base_dir` / `--video_base_dir`.

The paper's training mixture is drawn from several public preference datasets (MM-RLHF, LLaVA-Critic-113K, RLAIF-V, VLFeedback, POVID, WildVision, Tulu, Skywork-Reward, Nectar, PKU-SafeRLHF). Convert each dataset to the JSONL schema above using its native downloader; tooling for this is not bundled here.

### 2. Train + Merge

Edit paths in [`scripts/train.sh`](scripts/train.sh) (or override via env vars), then:

```bash
bash scripts/train.sh
```

Defaults (single node, 8 GPUs, GBS 64, 3 epochs, LoRA r=128, MLP-SiLU head, last-token pooling):

| Flag | Default |
|---|---|
| `BASE_MODEL_PATH` | `allenai/Molmo2-4B` |
| `BATCH_SIZE × NPROC × NUM_NODES × GRAD_ACCUM` | `1 × 8 × 1 × 8 = 64` |
| `LEARNING_RATE` | `1e-4` |
| `NUM_EPOCHS` | `3` |
| `MAX_LENGTH` | `24576` |
| `LORA_R / ALPHA / DROPOUT` | `128 / 16 / 0.05` |
| `VALUE_HEAD_TYPE / HIDDEN / ACT` | `mlp / 1024 / silu` |
| `RESP_REPR_MODE` | `last` |

Merge LoRA into the base model for deployment:

```bash
python scripts/merge_lora.py \
    --adapter_path ./checkpoints/mr2rm \
    --output_dir ./checkpoints/mr2rm-merged \
    --torch_dtype bfloat16
```

For multi-node training, set `NUM_NODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` before launching `scripts/train.sh` on each node.

## Evaluation

We evaluate on six multimodal reward benchmarks using each benchmark's official evaluation protocol:

- **[VL-RewardBench](https://huggingface.co/datasets/MMInstruction/VL-RewardBench)**
- **[Multimodal RewardBench](https://huggingface.co/datasets/MMRB/Multimodal-RewardBench)**
- **[MM-RLHF RewardBench](https://github.com/YangRui2015/MM-RLHF)**
- **[VideoRewardBench](https://huggingface.co/datasets/Dragonriser/VideoRewardBench)**
- **MR²Bench-Image / MR²Bench-Video (Ours)**: [yinuoy/MR2Bench](https://huggingface.co/datasets/yinuoy/MR2Bench)

## Results

### Main Results on Six Multimodal Reward Benchmarks

**Proprietary models (as judge)**

| Model | Size | VL-RB | MM-RB | MMRLHF | MR²B-I | VRB | MR²B-V | Avg |
|---|---|---|---|---|---|---|---|---|
| GPT-5 | -- | **75.0** | 64.6 | **71.8** | **87.1** | **68.2** | **50.1** | **69.5** |
| Claude-Sonnet-4.5 | -- | 68.6 | 78.2 | 70.0 | 72.9 | 67.5 | 49.1 | 67.7 |
| Gemini-2.5-Pro | -- | 70.5 | **82.4** | 70.6 | 71.2 | 63.2 | 49.7 | 67.9 |

**Open-source VLMs (as judge)**

| Model | Size | VL-RB | MM-RB | MMRLHF | MR²B-I | VRB | MR²B-V | Avg |
|---|---|---|---|---|---|---|---|---|
| InternVL3-8B | 8B | 56.6 | 66.9 | 69.4 | 55.4 | 57.9 | 40.4 | 57.8 |
| Qwen2.5-VL-7B | 7B | 66.7 | 62.6 | 77.6 | 52.5 | 55.3 | 44.4 | 59.9 |
| Qwen3-VL-4B | 4B | 61.4 | 65.9 | 80.0 | 60.8 | 64.9 | 47.9 | 63.5 |
| Qwen3-VL-32B | 32B | 67.1 | **79.0** | 78.8 | 60.8 | **65.8** | **49.9** | **66.9** |
| Molmo2-4B | 4B | 59.6 | 61.8 | 73.5 | 61.7 | 58.2 | 43.2 | 59.7 |
| InternVL3-78B | 78B | 61.9 | 75.7 | **81.8** | **65.0** | 58.5 | 47.7 | 65.1 |

**Generative reward models**

| Model | Size | VL-RB | MM-RB | MMRLHF | MR²B-I | VRB | MR²B-V | Avg |
|---|---|---|---|---|---|---|---|---|
| R1-Reward | 7B | **71.4** | **82.2** | 80.6 | **58.8** | **61.2** | **44.9** | **66.5** |
| MM-RLHF-Reward | 7B | 51.0 | 67.1 | **85.0** | 45.0 | 52.2 | 36.6 | 56.1 |
| LLaVA-Critic | 7B | 44.0 | 62.2 | 77.6 | 56.3 | 14.7 | 40.2 | 49.2 |

**Discriminative reward models (including ours)**

| Model | Size | VL-RB | MM-RB | MMRLHF | MR²B-I | VRB | MR²B-V | Avg |
|---|---|---|---|---|---|---|---|---|
| Skywork-VL-Reward | 7B | 69.0 | **74.2** | 72.4 | 52.9 | 62.9 | 46.7 | 63.0 |
| IXC-2.5-Reward | 7B | 70.0 | 66.6 | 71.2 | 55.0 | 57.1 | 48.7 | 61.4 |
| **Molmo2-4B RM (Ours)** | **4B** | **82.2** | 73.2 | **92.4** | **62.5** | **66.3** | **50.7** | **71.2** |
| **Qwen3-VL-4B RM (Ours)** | **4B** | 63.3 | 71.2 | 84.7 | 58.8 | 64.9 | 47.5 | 65.1 |

### Downstream Policy Optimization with GRPO

When used as the scoring function for [GRPO](https://arxiv.org/abs/2402.03300) policy optimization on Molmo2-4B, our multi-response RM substantially improves open-ended generation quality while preserving performance on 24 standard image and video benchmarks:

| Model | WildVision (win%) | LLaVA-Bench | MMHal (0-6) |
|---|---|---|---|
| Molmo2-4B (base) | 54.6 | 92.4 | 3.98 |
| + GRPO (Single-RM) | 55.8 (+1.2) | 91.6 (-0.8) | 4.17 (+0.19) |
| **+ GRPO (Multi-RM, Ours)** | **60.2 (+5.6)** | **97.0 (+4.6)** | **4.25 (+0.27)** |

The multi-response RM provides a steadily increasing validation reward signal during GRPO training, whereas the single-response RM's reward remains flat — see the paper for details.

## Citation

```bibtex
@misc{yang2026judgeoncemultiresponsereward,
      title={You Only Judge Once: Multi-response Reward Modeling in a Single Forward Pass},
      author={Yinuo Yang and Zixian Ma and Manasi Ganti and Jieyu Zhang and Ranjay Krishna},
      year={2026},
      eprint={2604.10966},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.10966},
}
```

## Acknowledgments

This work was partially supported by a grant from DSO National Laboratories, the Qualcomm Innovation Fellowship, OpenAI Superalignment Fellowship, and Apple AI/ML PhD Fellowship.

## License

This code is licensed under **Apache 2.0**. See [LICENSE](LICENSE) for details.
