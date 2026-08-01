import os
import torch
from safetensors.torch import load_file
from transformers import AutoConfig
import sys
from flash.model.cnets import Model
from flash.model.configs import EConfig
save_dir = "path/to/checkpoint"

config = EConfig.from_pretrained('path/to/checkpoint/config.json')

model = Model(config, load_emb=True, path='path/to/Qwen2.5-VL-7B-Instruct')


safetensors_files = sorted(
    [f for f in os.listdir(save_dir) if f.startswith("model_") and f.endswith(".safetensors")]
)


merged_state_dict = {}
for f in safetensors_files:
    shard = load_file(os.path.join(save_dir, f))
    merged_state_dict.update(shard)



model.load_state_dict(merged_state_dict)

torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))


